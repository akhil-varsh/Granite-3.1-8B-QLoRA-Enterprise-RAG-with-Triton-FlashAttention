# Architecture Diagrams

## System Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTERPRISE RAG SYSTEM                            │
│                    Granite-3.1-8B-Instruct (QLoRA)                      │
└─────────────────────────────────────────────────────────────────────────┘

                                 USER LAYER
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Gradio    │  │   FastAPI   │  │    cURL     │  │  Python SDK │
│   Web UI    │  │   REST API  │  │   Client    │  │   Client    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FastAPI Backend   │
                    │  /query endpoint   │
                    └─────────┬──────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  Query Encoder   │           │   Vector Search  │
    │ SentenceT5 Model │           │   FAISS Index    │
    └────────┬─────────┘           │  200 Documents   │
             │                     └────────┬─────────┘
             │ Embedding                    │
             │                              │ Top-K Retrieval
             └──────────────┬───────────────┘
                            │
                  ┌─────────▼──────────┐
                  │ Retrieved Context  │
                  │   (3-5 documents)  │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────────────────────────┐
                  │    Granite-3.1-8B-Instruct Model      │
                  │    (4-bit Quantized + LoRA)           │
                  │                                        │
                  │  ┌──────────────────────────────────┐ │
                  │  │  Input: Query + Context          │ │
                  │  └─────────────┬────────────────────┘ │
                  │                │                       │
                  │  ┌─────────────▼────────────────────┐ │
                  │  │  32 Transformer Layers           │ │
                  │  │  • Custom Triton FlashAttention  │ │
                  │  │  • LoRA Adapters (rank=64)       │ │
                  │  │  • 4-bit NF4 Quantization        │ │
                  │  └─────────────┬────────────────────┘ │
                  │                │                       │
                  │  ┌─────────────▼────────────────────┐ │
                  │  │  Output: Generated Response      │ │
                  │  └──────────────────────────────────┘ │
                  └────────────────┬───────────────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │  Final Response    │
                         │  + Retrieved Docs  │
                         │  + Confidence      │
                         └────────────────────┘
```

## Training Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Dolly-15k │────▶│            │     │  Finance   │
│  10,000    │     │            │     │  Alpaca    │
└────────────┘     │   Dataset  │◀────│  3,000     │
                   │  Combiner  │     └────────────┘
┌────────────┐     │            │     ┌────────────┐
│  ConvFinQA │────▶│            │     │   Spider   │
│  2,000     │     │            │◀────│   SQL      │
└────────────┘     └─────┬──────┘     │   1,500    │
                         │            └────────────┘
┌────────────┐           │            ┌────────────┐
│CodeAlpaca  │───────────┴───────────▶│ Combined   │
│  1,500     │                        │  Dataset   │
└────────────┘                        │  18,000    │
                                      └─────┬──────┘
                                            │
                                  ┌─────────▼──────────┐
                                  │  QLoRA Training    │
                                  │  • 4-bit quantize  │
                                  │  • LoRA rank=64    │
                                  │  • 3 epochs        │
                                  │  • ~18 hours       │
                                  └─────────┬──────────┘
                                            │
                          ┌─────────────────┼─────────────────┐
                          │                 │                 │
                    ┌─────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐
                    │  Epoch 1   │   │  Epoch 2   │   │  Epoch 3   │
                    │  Loss:1.23 │   │  Loss:0.89 │   │  Loss:0.72 │
                    └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                          └─────────────────┴─────────────────┘
                                            │
                                  ┌─────────▼──────────┐
                                  │  Merge LoRA        │
                                  │  • Combine weights │
                                  │  • Save 4-bit      │
                                  └─────────┬──────────┘
                                            │
                                  ┌─────────▼──────────┐
                                  │  Fine-tuned Model  │
                                  │  Ready for Deploy  │
                                  └────────────────────┘
```

## Custom Triton Kernel Architecture

```
STANDARD PYTORCH ATTENTION (Slow - 4 HBM roundtrips)
═══════════════════════════════════════════════════

GPU SRAM (Fast)              HBM (Slow)
┌──────────┐                ┌──────────┐
│    Q     │───── Load ────▶│    Q     │
│    K     │───── Load ────▶│    K     │
│    V     │───── Load ────▶│    V     │
└──────────┘                └──────────┘
     │                            │
     │ Compute QK^T              │
     ▼                            │
┌──────────┐                     │
│  Scores  │───── Store ────────▶│ Scores   │◀─┐
└──────────┘                     └──────────┘  │
                                       │ Load   │
                                       ▼        │
                                  ┌──────────┐ │
                                  │ Softmax  │─┘
                                  └─────┬────┘
                                        │ Store
                                        ▼
                                  ┌──────────┐
                                  │   Attn   │◀─┐
                                  └──────────┘  │
                                        │ Load  │
                                  ┌─────▼────┐  │
                                  │  Attn@V  │──┘
                                  └─────┬────┘
                                        │ Store
                                        ▼
                                  ┌──────────┐
                                  │  Output  │
                                  └──────────┘

TRITON FLASHATTENTION-2 (Fast - All in SRAM)
════════════════════════════════════════════

GPU SRAM (Fast)              HBM (Slow)
┌─────────────────────────┐  ┌──────────┐
│  Tile Q (128×64)        │◀─│    Q     │ Load once
│  Tile K (128×64)        │◀─│    K     │ Load in chunks
│  Tile V (128×64)        │◀─│    V     │ Load in chunks
├─────────────────────────┤  └──────────┘
│  Compute:               │
│  1. QK^T (fused)        │
│  2. Online Softmax      │
│  3. Attn @ V (fused)    │
│  4. Accumulate          │
├─────────────────────────┤
│  Output Tile (128×64)   │──▶ Store once
└─────────────────────────┘
         ▲         │
         │         │
         └─────────┘
    Repeat for all tiles

PERFORMANCE GAIN: 42% faster, 15% less memory
```

## Memory Layout

```
MODEL MEMORY BREAKDOWN (24GB GPU)
═════════════════════════════════

Base Model (FP16):           16.0 GB  ████████████████░░░░░░░░
4-bit Quantized:              4.0 GB  ████░░░░░░░░░░░░░░░░░░░░
LoRA Adapters:                0.2 GB  ░░░░░░░░░░░░░░░░░░░░░░░░
KV Cache:                     1.5 GB  █░░░░░░░░░░░░░░░░░░░░░░░
Activations:                  0.5 GB  ░░░░░░░░░░░░░░░░░░░░░░░░
                             ────────
Total Used:                   6.2 GB  ██████░░░░░░░░░░░░░░░░░░
Available:                   17.8 GB  ██████████████████░░░░░░

TRAINING MEMORY (24GB GPU)
══════════════════════════

Model Weights (4-bit):        4.0 GB  ████░░░░░░░░░░░░░░░░░░░░
LoRA Weights:                 0.2 GB  ░░░░░░░░░░░░░░░░░░░░░░░░
Optimizer State (8-bit):      0.8 GB  █░░░░░░░░░░░░░░░░░░░░░░░
Gradients (checkpointed):     2.5 GB  ██░░░░░░░░░░░░░░░░░░░░░░
Activations:                  8.0 GB  ████████░░░░░░░░░░░░░░░░
Batch Data:                   2.0 GB  ██░░░░░░░░░░░░░░░░░░░░░░
                             ────────
Total Used:                  17.5 GB  █████████████████░░░░░░░
Available:                    6.5 GB  ██████░░░░░░░░░░░░░░░░░░
```

## Inference Flow

```
USER QUERY: "Calculate NPV with cash flows $30K, $40K, $50K at 10%"
│
├─▶ 1. Query Encoding (Sentence Transformer)
│   └─▶ Embedding: [0.23, -0.45, 0.67, ...] (384 dim)
│
├─▶ 2. Vector Search (FAISS)
│   ├─▶ Document 1: "NPV Calculation Guide" (score: 0.89)
│   ├─▶ Document 2: "Time Value of Money" (score: 0.82)
│   └─▶ Document 3: "Financial Formulas" (score: 0.78)
│
├─▶ 3. Context Building
│   └─▶ Prompt = Query + Retrieved Documents (1024 tokens)
│
├─▶ 4. Model Inference (vLLM + Triton Kernel)
│   ├─▶ Tokenization: [128, 456, 789, ...] (1024 tokens)
│   ├─▶ Forward Pass:
│   │   ├─▶ Layer 1-32: Custom Triton FlashAttention
│   │   │   └─▶ 19.6ms per layer (vs 45.2ms PyTorch)
│   │   └─▶ LoRA Adapters: +2ms per layer
│   └─▶ Generation: 512 tokens @ 102 tok/s = 5.0s
│
└─▶ 5. Response Formatting
    └─▶ "NPV = (30K/1.1) + (40K/1.21) + (50K/1.331) = $101,700..."
    
TOTAL LATENCY: ~5.5 seconds
```

## Deployment Architecture

```
PRODUCTION DEPLOYMENT
════════════════════

                    ┌──────────────┐
                    │ Load Balancer│
                    │   (Nginx)    │
                    └───────┬──────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼─────┐       ┌────▼─────┐       ┌────▼─────┐
   │ Pod 1    │       │ Pod 2    │       │ Pod 3    │
   │ GPU: A100│       │ GPU: A100│       │ GPU: A100│
   └────┬─────┘       └────┬─────┘       └────┬─────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ┌───────▼──────┐
                    │ Redis Cache  │
                    │ (Embeddings) │
                    └───────┬──────┘
                            │
                    ┌───────▼──────┐
                    │ FAISS Index  │
                    │ (Vector DB)  │
                    └───────┬──────┘
                            │
                    ┌───────▼──────┐
                    │ PostgreSQL   │
                    │ (Metadata)   │
                    └──────────────┘

MONITORING
═════════

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Prometheus │───▶│   Grafana   │◀───│   WandB     │
│  (Metrics)  │    │ (Dashboard) │    │  (Training) │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

These diagrams provide visual understanding of:
- System architecture and data flow
- Training pipeline stages
- Triton kernel optimizations
- Memory layout and usage
- Inference process
- Production deployment setup
