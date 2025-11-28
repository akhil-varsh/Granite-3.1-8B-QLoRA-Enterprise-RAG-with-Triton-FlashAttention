import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.vllm_server import VLLMServer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    max_tokens: int = 512
    temperature: float = 0.7


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: List[Dict]
    retrieval_score: float


class Document(BaseModel):
    id: str
    title: str
    content: str
    category: str
    metadata: Dict


class VectorDatabase:
    """Vector database using FAISS and sentence-transformers"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
    ):
        """
        Initialize vector database
        
        Args:
            embedding_model: Sentence transformer model name
            index_path: Path to saved FAISS index
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        self.documents = []
        self.doc_ids = []
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        
        logger.info(f" Vector DB initialized (dimension: {self.dimension})")
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to vector database
        
        Args:
            documents: List of document dicts
        """
        logger.info(f"Adding {len(documents)} documents to vector DB...")
        
        # Extract text for embedding
        texts = []
        for doc in documents:
            # Combine title and content for better retrieval
            text = f"{doc['title']}\n\n{doc['content']}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        self.doc_ids.extend([doc['id'] for doc in documents])
        
        logger.info(f" Added {len(documents)} documents (total: {len(self.documents)})")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of documents with scores
        """
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
        )
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def save_index(self, output_path: str):
        """Save FAISS index and documents"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        
        # Save documents
        with open(output_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved index to {output_path}")
    
    def load_index(self, index_path: str):
        """Load FAISS index and documents"""
        index_dir = Path(index_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        
        # Load documents
        with open(index_dir / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        self.doc_ids = [doc['id'] for doc in self.documents]
        
        logger.info(f"Loaded index from {index_path} ({len(self.documents)} documents)")


class RAGSystem:
    """RAG system combining retrieval and generation"""
    
    def __init__(
        self,
        model_path: str,
        documents_path: str,
        index_path: Optional[str] = None,
    ):
        """
        Initialize RAG system
        
        Args:
            model_path: Path to LLM model
            documents_path: Path to documents directory or metadata file
            index_path: Path to saved vector index
        """
        logger.info("="*60)
        logger.info("INITIALIZING RAG SYSTEM")
        logger.info("="*60)
        
        # Initialize vector database
        self.vector_db = VectorDatabase(index_path=index_path)
        
        # Load documents if needed
        if not index_path or not Path(index_path).exists():
            self._load_documents(documents_path)
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        self.llm = VLLMServer(
            model_path=model_path,
            max_model_len=2048,
            use_custom_kernel=True,
        )
        
        logger.info("RAG system initialized")
    
    def _load_documents(self, documents_path: str):
        """Load documents from path"""
        doc_path = Path(documents_path)
        
        # Check if metadata file exists
        if (doc_path / "documents_metadata.json").exists():
            logger.info(f"Loading documents from {doc_path / 'documents_metadata.json'}")
            with open(doc_path / "documents_metadata.json", 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            self.vector_db.add_documents(documents)
        
        # Otherwise load individual .txt files
        elif doc_path.is_dir():
            logger.info(f"Loading documents from {doc_path}")
            documents = []
            
            for txt_file in doc_path.glob("*.txt"):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse metadata from content
                lines = content.split('\n')
                title = lines[0].replace("Title:", "").strip() if lines else txt_file.stem
                category = lines[1].replace("Category:", "").strip() if len(lines) > 1 else "general"
                
                doc = {
                    "id": txt_file.stem,
                    "title": title,
                    "content": content,
                    "category": category,
                    "metadata": {}
                }
                documents.append(doc)
            
            if documents:
                self.vector_db.add_documents(documents)
        
        else:
            logger.warning(f"No documents found at {documents_path}")
    
    def query(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Process RAG query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        
        Returns:
            Response dict with answer and retrieved documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_db.search(query, top_k=top_k)
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[Document {i+1}]")
            context_parts.append(f"Title: {doc['title']}")
            context_parts.append(f"Category: {doc['category']}")
            context_parts.append(f"Content: {doc['content'][:500]}...")  # Truncate long docs
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Build prompt with retrieved context
        prompt = f"""Below is an instruction that describes a task, paired with context from relevant documents. Write a response that appropriately completes the request using the provided context.

### Context:
{context}

### Instruction:
{query}

### Response:
"""
        
        # Generate response
        response = self.llm.generate([prompt], max_tokens=max_tokens, temperature=temperature)[0]
        
        # Calculate retrieval quality score (average of top scores)
        retrieval_score = np.mean([doc['score'] for doc in retrieved_docs]) if retrieved_docs else 0.0
        
        return {
            "query": query,
            "answer": response,
            "retrieved_docs": retrieved_docs,
            "retrieval_score": float(retrieval_score),
        }


# FastAPI application
app = FastAPI(
    title="Enterprise RAG API",
    description="RAG system with Granite-3.1-8B-Instruct fine-tuned model",
    version="1.0.0"
)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    model_path = os.getenv("MODEL_PATH", "outputs/merged_model")
    documents_path = os.getenv("DOCUMENTS_PATH", "data/rag_documents")
    index_path = os.getenv("INDEX_PATH", None)
    
    logger.info("Starting RAG API server...")
    rag_system = RAGSystem(
        model_path=model_path,
        documents_path=documents_path,
        index_path=index_path,
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Enterprise RAG API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "documents": len(rag_system.vector_db.documents) if rag_system else 0,
        "vector_dim": rag_system.vector_db.dimension if rag_system else 0,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process RAG query
    
    Args:
        request: Query request with query text and parameters
    
    Returns:
        Query response with answer and retrieved documents
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            query=request.query,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the vector database"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "total": len(rag_system.vector_db.documents),
        "documents": [
            {
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"]
            }
            for doc in rag_system.vector_db.documents
        ]
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI RAG Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--model_path", type=str, default="outputs/merged_model",
                       help="Path to model")
    parser.add_argument("--documents_path", type=str, default="data/rag_documents",
                       help="Path to documents")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["DOCUMENTS_PATH"] = args.documents_path
    
    # Run server
    uvicorn.run(
        "fastapi_rag:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
