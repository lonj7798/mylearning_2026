---
chapter: ch-18
course: llm-inference
phase: read
excerpt_of: "Hugging Face Text Generation Inference — production server architecture"
source_url: https://github.com/huggingface/text-generation-inference
created_at: "2026-05-21"
---

# Excerpt: TGI — Rust router + Python shards + Hugging Face ecosystem

**Authors:** Hugging Face
**Year:** 2022–present
**URLs:** https://github.com/huggingface/text-generation-inference / https://huggingface.github.io/text-generation-inference/
**Raw-data source:** [[raw-data/hf-tgi]] + [[raw-data/huggingface-inference]]

---

## The architectural bet

**Rust router for HTTP / TLS / streaming, Python shards for model execution, Docker-first ops.**

```
┌──────────────────────────────────────────────────────────────┐
│ text-generation-launcher (CLI, Python)                       │
│  • Spawns router (Rust)                                       │
│  • Spawns N shard processes (Python, one per TP rank)         │
└──────────────────────────────────────────────────────────────┘
                              ↓ launches
┌──────────────────────────────────────────────────────────────┐
│ Router (Rust, axum HTTP server)                              │
│  Endpoints:                                                   │
│   • POST /generate, /generate_stream                          │
│   • POST /v1/chat/completions (OpenAI-compatible)             │
│   • POST /tokenize                                            │
│   • GET  /info, /metrics, /health                             │
│  Pipeline:                                                    │
│   • Validate request (max_input_length, max_total_tokens)     │
│   • Tokenize (optionally; can defer to shards)                │
│   • Enqueue to scheduler                                      │
│   • Continuous batching: assemble batches each step           │
│   • Forward to model shards via gRPC                          │
│   • Stream tokens back over SSE                               │
└──────────────────────────────────────────────────────────────┘
                              ↓ gRPC
┌──────────────────────────────────────────────────────────────┐
│ Model shard processes (Python, one per TP rank)              │
│  • Load weights via safetensors                               │
│  • Run model via Transformers + PagedAttention kernels        │
│  • Apply continuous-batching update each iteration            │
│  • Return generated logits / tokens to router                 │
└──────────────────────────────────────────────────────────────┘
```

Why Rust for the router: HTTP, TLS, SSE streaming, request validation, and queueing are concurrency-heavy and benefit from Tokio's async runtime. Python is reserved for what it's good at — model execution.

---

## Launch and configuration

```bash
text-generation-launcher \
    --model-id meta-llama/Llama-3-8B-Instruct \
    --num-shard 1 \
    --port 8080 \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 16384 \
    --max-batch-total-tokens 32768 \
    --max-waiting-tokens 20 \
    --max-concurrent-requests 128 \
    --quantize bitsandbytes-nf4 \
    --trust-remote-code \
    --hostname 0.0.0.0
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--max-batch-prefill-tokens` | Max tokens of prefill work in a single batch |
| `--max-batch-total-tokens` | Max KV-cache tokens active at once |
| `--max-waiting-tokens` | Hard cap on tokens waiting in queue |
| `--max-concurrent-requests` | Request-count cap |
| `--quantize` | One of: `awq`, `gptq`, `bitsandbytes`, `bitsandbytes-nf4`, `fp8`, `eetq`, `marlin` |
| `--speculate` | Number of tokens to speculate per step (for spec decoding) |
| `--num-shard` | TP degree |

---

## The Docker-first deployment story

```bash
docker run --gpus all -p 8080:80 \
    -v /data:/data \
    -e HF_TOKEN=$HF_TOKEN \
    ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id meta-llama/Llama-3-8B-Instruct \
    --max-batch-prefill-tokens 16384
```

The official image bundles a working router + shard environment with all kernel dependencies (FlashAttention, paged attention, quantization backends). For most teams this is a near-zero-friction deployment.

---

## OpenAPI surface — schema is enforced

TGI exposes its full API as OpenAPI / Swagger:

```bash
curl http://localhost:8080/info
curl http://localhost:8080/docs/openapi.json    # full schema
```

Client SDKs in Python, JS, Rust auto-generated from the schema.

---

## Streaming over SSE

```bash
curl -N http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "llama3",
      "messages": [{"role":"user","content":"hi"}],
      "stream": true,
      "max_tokens": 64
    }'

# Output:
# data: {"id":"...","choices":[{"delta":{"content":"Hi"}}]}
# data: {"id":"...","choices":[{"delta":{"content":"!"}}]}
# data: [DONE]
```

---

## Hugging Face ecosystem integration

| Surface | Integration |
|---------|-------------|
| `safetensors` checkpoints | Native; no conversion |
| Chat templates from tokenizer | Applied automatically; OpenAI-compatible endpoint uses them |
| Hub revisions | `--revision <sha>` pins a specific commit |
| Model cards | Metadata exposed via `/info` endpoint |
| HF Inference Endpoints (managed) | TGI is the engine under the hood |

---

## Strengths

- Best ecosystem fit for HuggingFace-shaped orgs
- Production-quality Docker, metrics, health, OpenAPI
- Rust router is genuinely fast for concurrency
- Managed deployment available via HF Inference Endpoints

## Limits

- 10–20 % slower than vLLM on the same workload (router + shard runtime overhead)
- Scheduler internals less central as learning artifact
- Advanced research features lag specialized stacks
- Deep customization spans Rust router + Python shards

---

## Connections

- [[excerpts/vllm]] (ch-16) — direct OSS comparison.
- [[excerpts/continuous-batching]] (ch-04) — what the router implements.
- [[ch-18]] — parent decision rubric.
