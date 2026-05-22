---
chapter: ch-06
course: llm-inference
phase: read
excerpt_of: "vLLM — open-source LLM serving engine"
source_url: https://docs.vllm.ai/en/latest/
created_at: "2026-05-21"
---

# Excerpt: vLLM — the engine PagedAttention ships inside

**Authors:** vLLM project / UC Berkeley Sky Computing Lab and community
**Year:** 2023-present
**URLs:** https://docs.vllm.ai/en/latest/ ; https://github.com/vllm-project/vllm
**Raw-data source:** [[raw-data/vllm]]

---

## Why vLLM is the canonical paged-KV implementation

PagedAttention was published in SOSP 2023 ([[pagedattention]]) and shipped in the same authors' open-source serving engine: **vLLM**. The paper and the implementation co-evolved. Reading the paper without referencing vLLM's code understates how much of the win comes from the engine's surrounding choices (continuous batching, chunked prefill, OpenAI-compatible frontend, prefix caching, structured outputs, multi-modal support).

vLLM is what most operators mean when they say "PagedAttention."

---

## Architecture overview

```
┌────────────────────────────────────────────────────────────────────┐
│  API server (FastAPI / Uvicorn)                                    │
│   - OpenAI-compatible: /v1/chat/completions, /v1/completions, ...  │
│   - Streaming SSE, request validation, tokenization                │
└────────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────────┐
│  Engine (vllm/v1/engine)                                           │
│   - Receives requests, hands to scheduler                          │
│   - Pumps step → scheduler → worker → output → client              │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────┬──────────────────┬───────────────────────────────┐
│   Scheduler     │  KV Cache Mgr    │  Other managers               │
│ (ch-05 excerpt) │  (ch-06 excerpt) │  (encoder cache, struct out,  │
│                 │                  │   speculative dec, KV         │
│                 │                  │   connector)                  │
└─────────────────┴──────────────────┴───────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────────┐
│  Worker / model runner (vllm/v1/worker)                            │
│   - Consumes SchedulerOutput                                       │
│   - Runs the forward pass (paged attention kernel, FlashAttention) │
│   - Returns logits + sampled tokens                                │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       NCCL / TP / PP / DP
```

Three things this layout buys:

- **Scheduler is policy, worker is mechanism.** Swapping speculative decoding in/out, or enabling chunked prefill, only changes the scheduler — never the kernel.
- **API surface is OpenAI-compatible.** Any client that talks to OpenAI's API can talk to vLLM with one URL change.
- **KV cache is a first-class manager.** Memory is not a global variable; it's a service that allocates, frees, and caches blocks.

---

## What ships *with* PagedAttention

The paper's contribution is one kernel + one allocator. The engine adds:

| Feature | What it does | Where it lives |
|---|---|---|
| Continuous batching | Iteration-level scheduling | `Scheduler.schedule()` (ch-04, ch-05) |
| Chunked prefill | Token-budget mixed batches | Same scheduler (ch-05) |
| Automatic Prefix Cache (APC) | Block-hash prefix matching + LRU | `KVCacheManager.get_computed_blocks` (ch-07) |
| Structured output | xgrammar / outlines integration | `StructuredOutputManager` (ch-16) |
| Speculative decoding | K-token drafts + verify | Speculation managers (ch-14, ch-15) |
| LoRA adapter serving | Per-request LoRA selection | `LoRAManager` |
| Quantization | AWQ, GPTQ, FP8, FP4 weights | `quantization/` |
| Tensor parallelism | Multi-GPU shard execution | `parallel_state.py` |
| Pipeline parallelism | Multi-GPU layer split | `worker_base.py` |

Each of these composes with paged KV because the scheduler reasons about token budgets and KV-block availability uniformly.

---

## Public API surface

**Offline (Python):**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct",
          gpu_memory_utilization=0.9, enable_prefix_caching=True)
out = llm.generate(["What is paged attention?"], SamplingParams(max_tokens=200))
```

**Online (server):**

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching
```

Then any OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
client.chat.completions.create(model="...", messages=[...])
```

Extra vLLM-specific knobs (`presence_penalty`, `top_k`, `min_p`, `guided_json`, etc.) flow through OpenAI's `extra_body` parameter.

---

## How `num_gpu_blocks` is sized

The most-important silent computation in vLLM startup:

```python
free_hbm = gpu_total_memory - model_weights - activation_buffers
num_gpu_blocks = (free_hbm * gpu_memory_utilization) // bytes_per_block

bytes_per_block = 2 (K+V) × layers × kv_heads × head_dim × block_size × dtype_bytes
```

Worked example, Llama-3-8B (bf16) on H100 80GB:

- Weights: ~16 GB
- Activation buffer headroom: ~4 GB (varies with `max_num_batched_tokens`)
- Free HBM: ~60 GB
- `bytes_per_block` for Llama-3-8B (L=32, H_kv=8, d_head=128, block=16, bf16): `2 × 32 × 8 × 128 × 16 × 2 = 2,097,152 bytes ≈ 2 MB`
- `num_gpu_blocks = 60 GB × 0.9 / 2 MB ≈ 27,000 blocks ≈ 430,000 tokens of KV`

That's ~430k tokens of total KV capacity — enough for 50 concurrent 8k-context requests or 400 concurrent 1k-context requests. The block-pool size is what every other engine knob ultimately bottoms out on.

---

## When to use vLLM (vs alternatives)

The decision matrix (ch-18 expands this):

| Workload | Best choice |
|---|---|
| Mixed OSS models, OpenAI API parity, moderate scale | **vLLM** |
| Heavy shared-prefix workloads (agents, multi-QA over docs) | SGLang (RadixAttention) |
| NVIDIA-only, max throughput, FP8/FP4 | TensorRT-LLM |
| Edge / CPU / Mac inference | llama.cpp |

vLLM is the safe default for production OSS LLM serving in 2026. The combination of paged KV, continuous batching, prefix caching, and broad model support gives you ~80 % of the performance of specialized engines with ~10 % of the integration cost.

---

## Connections

- [[excerpts/pagedattention]] — the algorithm.
- [[excerpts/vllm-kv-cache-manager]] — the block-pool implementation.
- [[excerpts/vllm-scheduler]] (ch-05) — the scheduler driving it.
- [[ch-06]] — parent synthesis.
- Forward to [[ch-16]] — full vLLM internals deep dive.
