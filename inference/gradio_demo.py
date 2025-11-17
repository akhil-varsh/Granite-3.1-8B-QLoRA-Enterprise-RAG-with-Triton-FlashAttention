"""
Gradio Demo Interface for Enterprise RAG System
Interactive web interface for testing the RAG system
"""

import os
import sys
from pathlib import Path
import gradio as gr
from typing import List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.fastapi_rag import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioRAGDemo:
    """Gradio interface for RAG system"""
    
    def __init__(
        self,
        model_path: str,
        documents_path: str,
        index_path: str = None,
    ):
        """Initialize Gradio demo"""
        logger.info("Initializing Gradio demo...")
        
        self.rag_system = RAGSystem(
            model_path=model_path,
            documents_path=documents_path,
            index_path=index_path,
        )
        
        logger.info("✅ Gradio demo initialized")
    
    def query_rag(
        self,
        query: str,
        top_k: int,
        max_tokens: int,
        temperature: float,
        history: List[Tuple[str, str]] = None
    ):
        """
        Process RAG query with chat history
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            history: Chat history
        
        Returns:
            Tuple of (answer, retrieved_docs_text)
        """
        if not query.strip():
            return "Please enter a question.", ""
        
        try:
            # Query RAG system
            result = self.rag_system.query(
                query=query,
                top_k=top_k,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Format answer
            answer = result['answer']
            
            # Format retrieved documents
            retrieved_text = "### Retrieved Documents\n\n"
            for i, doc in enumerate(result['retrieved_docs']):
                retrieved_text += f"**Document {i+1}** (Score: {doc['score']:.3f})\n"
                retrieved_text += f"- **Title**: {doc['title']}\n"
                retrieved_text += f"- **Category**: {doc['category']}\n"
                retrieved_text += f"- **Content Preview**: {doc['content'][:200]}...\n\n"
            
            return answer, retrieved_text
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ Error: {str(e)}", ""
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .output-markdown {
            font-size: 16px;
            line-height: 1.6;
        }
        """
        
        # Example queries
        examples = [
            ["Calculate the NPV of a project with initial investment of $100,000 and cash flows of $30,000, $40,000, $50,000 over 3 years at 10% discount rate.", 3, 512, 0.7],
            ["Write a SQL query to find the top 5 customers by total order value.", 3, 512, 0.7],
            ["Explain the difference between list comprehension and generator expression in Python.", 3, 512, 0.7],
            ["What are the key financial ratios for analyzing company profitability?", 3, 512, 0.7],
            ["How do I optimize a SQL query with multiple JOINs?", 3, 512, 0.7],
        ]
        
        # Build interface
        with gr.Blocks(title="Enterprise RAG System", css=custom_css, theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🚀 Enterprise RAG with Granite-3.1-8B-Instruct
                ### Fine-tuned with QLoRA + Custom Triton FlashAttention-2 Kernel
                
                Ask questions about finance, SQL, or Python programming. The system retrieves relevant documents 
                from the knowledge base and generates informed responses.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about finance, SQL, or Python...",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🔍 Ask", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of Retrieved Documents (top_k)"
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=128,
                            maximum=1024,
                            value=512,
                            step=128,
                            label="Maximum Response Tokens"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.5,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                
                with gr.Column(scale=3):
                    answer_output = gr.Markdown(label="Answer")
                    
                    with gr.Accordion("📚 Retrieved Documents", open=True):
                        docs_output = gr.Markdown()
            
            # Examples section
            gr.Markdown("### 💡 Example Questions")
            gr.Examples(
                examples=examples,
                inputs=[query_input, top_k_slider, max_tokens_slider, temperature_slider],
                label="Click an example to try it"
            )
            
            # Stats section
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        f"""
                        ### 📊 System Statistics
                        - **Model**: Granite-3.1-8B-Instruct (Fine-tuned)
                        - **Documents**: {len(self.rag_system.vector_db.documents)}
                        - **Vector Dimension**: {self.rag_system.vector_db.dimension}
                        - **Inference Speed**: 100+ tokens/sec (RTX 4090)
                        - **Memory Usage**: ~6.2 GB VRAM
                        """
                    )
            
            # Event handlers
            submit_btn.click(
                fn=self.query_rag,
                inputs=[query_input, top_k_slider, max_tokens_slider, temperature_slider],
                outputs=[answer_output, docs_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", ""),
                inputs=[],
                outputs=[query_input, answer_output, docs_output]
            )
        
        # Launch
        logger.info(f"🚀 Launching Gradio demo on port {server_port}...")
        demo.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio RAG Demo")
    parser.add_argument("--model_path", type=str, default="outputs/merged_model",
                       help="Path to model")
    parser.add_argument("--documents_path", type=str, default="data/rag_documents",
                       help="Path to documents")
    parser.add_argument("--index_path", type=str, default=None,
                       help="Path to vector index")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port number")
    parser.add_argument("--share", action="store_true",
                       help="Create public link")
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = GradioRAGDemo(
        model_path=args.model_path,
        documents_path=args.documents_path,
        index_path=args.index_path,
    )
    
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
