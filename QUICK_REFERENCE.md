# 🚀 QUICK REFERENCE GUIDE

## Essential Commands

### Setup
```bash
# Initial setup
python setup.py

# Or manual setup
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add HF_TOKEN
```

### Data Preparation
```bash
# Prepare 18K training dataset (~10 min)
python scripts/prepare_dataset.py

# Generate 200 RAG documents (~2 min)
python scripts/generate_rag_docs.py

# Create evaluation benchmark (~1 min)
python scripts/evaluate_rag.py --create_benchmark
```

### Training
```bash
# Full training (~18 hours on A100)
python scripts/train_qlora.py \
    --model_name ibm-granite/granite-3.1-8b-instruct \
    --dataset_path data/enterprise_dataset.json \
    --output_dir outputs/qlora_model \
    --num_epochs 3

# Or use one-click script
./train_pipeline.sh  # Linux/Mac
train_pipeline.bat   # Windows
```

### Inference
```bash
# Merge LoRA adapters
python scripts/merge_and_push.py \
    --adapter_path outputs/qlora_model \
    --output_path outputs/merged_model

# Launch Gradio demo
python inference/gradio_demo.py \
    --model_path outputs/merged_model \
    --port 7860

# Or FastAPI server
python inference/fastapi_rag.py \
    --model_path outputs/merged_model \
    --port 8080
```

### Evaluation & Benchmarking
```bash
# Benchmark Triton kernel
python scripts/benchmark_triton.py \
    --device cuda \
    --output_dir results

# Evaluate model
python scripts/evaluate_rag.py \
    --model_path outputs/merged_model \
    --base_model_path ibm-granite/granite-3.1-8b-instruct \
    --output_dir results
```

### Docker
```bash
# Build
docker build -t enterprise-rag:latest -f docker/Dockerfile .

# Run
docker run --gpus all -p 8080:8080 -p 7860:7860 \
    -v $(pwd)/outputs:/workspace/outputs \
    enterprise-rag:latest
```

---

## File Locations

### Input Data
- Training dataset: `data/enterprise_dataset.json`
- RAG documents: `data/rag_documents/*.txt`
- Evaluation benchmark: `data/enterprise_benchmark.json`

### Models
- LoRA adapters: `outputs/qlora_model/`
- Merged model: `outputs/merged_model/`
- Base model: Downloaded to `~/.cache/huggingface/`

### Results
- Training logs: `outputs/qlora_model/logs/`
- Benchmark results: `results/benchmark_results.json`
- Evaluation results: `results/evaluation_*.json`
- Graphs: `results/*.png`

---

## Configuration Files

### .env
```bash
HF_TOKEN=your_token_here
WANDB_API_KEY=your_key_here
BASE_MODEL=ibm-granite/granite-3.1-8b-instruct
OUTPUT_DIR=./outputs
MAX_SEQ_LENGTH=2048
```

### Key Hyperparameters
```python
# QLoRA
lora_rank = 64
lora_alpha = 16
lora_dropout = 0.1
quantization = "4bit-nf4"

# Training
batch_size = 4
gradient_accumulation = 4  # Effective batch = 16
learning_rate = 2e-4
epochs = 3
max_seq_length = 2048

# Inference
temperature = 0.7
top_p = 0.9
max_tokens = 512
```

---

## API Examples

### Python
```python
from inference import RAGSystem

rag = RAGSystem(
    model_path="outputs/merged_model",
    documents_path="data/rag_documents"
)

result = rag.query(
    query="Calculate ROE for company with $500K net income, $2M equity",
    top_k=3,
    max_tokens=512
)

print(result['answer'])
print(f"Retrieved {len(result['retrieved_docs'])} documents")
```

### cURL
```bash
curl -X POST "http://localhost:8080/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Write SQL to find top 5 customers by revenue",
       "top_k": 3,
       "max_tokens": 512,
       "temperature": 0.7
     }'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8080/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: "Explain Python decorators with example",
    top_k: 3,
    max_tokens: 512,
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.answer);
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 2 --gradient_accumulation_steps 8

# Reduce sequence length
--max_seq_length 1024

# Enable CPU offloading
--offload_folder offload/
```

### Slow Training
```bash
# Enable TF32 (A100)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Use Flash Attention 2
pip install flash-attn --no-build-isolation

# Check GPU utilization
nvidia-smi -l 1
```

### Import Errors
```bash
# Reinstall Triton
pip uninstall triton -y
pip install triton==2.1.0 --no-cache-dir

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Port Already in Use
```bash
# Find process
lsof -i :8080  # Linux/Mac
netstat -ano | findstr :8080  # Windows

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows

# Or use different port
--port 8081
```

---

## Performance Tips

### Training
- Use `bf16` on A100/H100 (better than `fp16`)
- Enable gradient checkpointing (saves 40% memory)
- Use `paged_adamw_8bit` optimizer (saves memory)
- Monitor with WandB for early stopping
- Save checkpoints every 500 steps

### Inference
- Use vLLM for batched inference (5-10x faster)
- Enable KV cache for multi-turn conversations
- Quantize to 4-bit for deployment (2x faster, 4x less memory)
- Use custom Triton kernel (42% speedup)
- Profile with `torch.profiler` to find bottlenecks

### RAG
- Batch encode documents (10x faster than one-by-one)
- Use GPU for encoding if available
- Cache embeddings (don't re-encode every query)
- Tune `top_k` (3-5 usually optimal)
- Filter by category for domain-specific queries

---

## Testing Checklist

- [ ] Dataset preparation completes without errors
- [ ] Training runs for at least 1 epoch
- [ ] Model generates coherent responses
- [ ] Triton kernel passes correctness test
- [ ] Benchmark shows >35% speedup
- [ ] FastAPI endpoints return 200 status
- [ ] Gradio interface loads successfully
- [ ] RAG retrieves relevant documents
- [ ] Evaluation metrics look reasonable
- [ ] Docker image builds and runs

---

## Production Deployment

### Before Deployment
1. Run full evaluation on test set
2. Benchmark inference speed
3. Load test API endpoints
4. Profile memory usage
5. Check logs for errors
6. Test edge cases
7. Document API changes
8. Create backup of model

### Monitoring
```python
# Add to inference code
import time
import logging

start = time.time()
response = model.generate(...)
latency = time.time() - start

logging.info(f"Latency: {latency:.3f}s")
logging.info(f"Tokens: {len(response)}")
logging.info(f"Tokens/sec: {len(response)/latency:.1f}")
```

### Scaling
- Horizontal: Multiple replicas behind load balancer
- Vertical: Larger GPU (A100 → H100)
- Batching: Vllm automatic batching
- Caching: Redis for frequent queries
- CDN: Static assets (Gradio UI)

---

## Useful Links

- [Triton Language](https://triton-lang.org/main/index.html)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Granite Models](https://huggingface.co/ibm-granite)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

---

## Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/repo/issues)
- Discussions: [Ask questions](https://github.com/yourusername/repo/discussions)
- Email: your.email@example.com

---

**Last Updated**: November 2025
**Version**: 1.0.0
