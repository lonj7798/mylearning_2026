---
chapter: ch-12
course: llm-inference
phase: read
excerpt_of: "CUDA Graphs for Inference — synthesis card across PyTorch / vLLM / FlashInfer docs"
source_url: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
created_at: "2026-05-21"
---

# Excerpt: CUDA Graphs — capture + replay for inference

**Authors:** NVIDIA CUDA + PyTorch documentation; representative serving use in vLLM, SGLang, FlashInfer
**Year:** 2021–2025 (synthesis)
**URL:** https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
**Raw-data source:** [[raw-data/cuda-graphs-inference]]

---

## The launch-overhead arithmetic (the load-bearing observation)

Per-kernel CPU launch cost on a modern CPU + CUDA driver: **5–10 µs**.

Per-token decode kernel count for Llama-3-70B (80 layers × ~10 kernels/layer):
- QKV projection: 1
- RoPE: 1
- Attention + KV append: 1–2
- Output projection: 1
- Two RMS norms: 2
- Gate + up + down FFN matmuls: 2–3
- SiLU/SwiGLU + multiply: 1
- Residual adds: 2
- **per layer: ~12** ; × 80 layers = **~960 kernels per token**

At 7 µs/kernel: **6.7 ms/token of pure CPU launch overhead**.

Compare: H100 decode TPOT for Llama-3-70B is ~12–18 ms total. Half is launch, half is GPU compute.

---

## Capture + replay

```python
import torch

# Static buffers (graph captures pointers)
static_in  = torch.empty(B, D, device="cuda")
static_out = torch.empty(B, V, device="cuda")

# Warmup (autotuning, allocator priming)
for _ in range(3):
    static_out.copy_(model(static_in))

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out.copy_(model(static_in))

# Replay (one cudaGraphLaunch — ~2 µs CPU)
for step in range(N):
    static_in.copy_(real_input)
    g.replay()
    use(static_out)
```

The graph holds:
- Kernel sequence + grid/block dims
- Stream and event dependencies
- Memory addresses of all tensors involved

Replay re-issues all kernels in one driver call.

---

## The static-shape constraint

CUDA graphs capture **pointer values**, not data. The graph runs against the *exact tensors* it was recorded with. Implication:

```text
- Cannot change tensor shapes between replays.
- Cannot reallocate captured tensors.
- Cannot use Python-side conditionals on tensor data (no graph branches).
- New data must be `.copy_()`'d into the recorded buffer.
```

For serving, batch size and KV length vary per iteration. **Naive CUDA graphs don't work for dynamic serving.**

---

## Piecewise capture — the production fix

Capture one graph per batch-size bucket (typically powers of 2 from 1 to max_num_seqs):

```python
graphs = {}
for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    static_in  = torch.empty(B, ..., device="cuda")
    static_out = torch.empty(B, ..., device="cuda")
    for _ in range(3):
        static_out.copy_(model(static_in))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out.copy_(model(static_in))
    graphs[B] = (g, static_in, static_out)

# At serve time: pad to nearest bucket, replay
def step(actual_batch):
    B_bucket = next_pow2(actual_batch.size(0))
    g, sin, sout = graphs[B_bucket]
    sin[:actual_batch.size(0)].copy_(actual_batch)
    g.replay()
    return sout[:actual_batch.size(0)]
```

---

## Cost model

| Cost | Magnitude | Notes |
|---|---|---|
| Capture time | 5–10 s per bucket | one-time, at engine init |
| Static buffer memory | (batch × seq × hidden) bytes per graph | grows with bucket count |
| Padding overhead | ~25% on uniform-random batch sizes | bucket-density tradeoff |
| Replay launch overhead | ~2 µs | down from ~7 ms eager |

**Net TPOT improvement: 30–50% in production engines.**

---

## What graphs do not help

- **Prefill**: variable prompt length defeats bucketing. Run eagerly with FlashInfer/FA.
- **Compute-bound ops**: launch overhead is a small fraction of GPU time; graphs save nothing here.
- **Models with data-dependent control flow** (mixture-of-depths, dynamic routing): graphs can't capture branches.

---

## Production usage

| Framework | CUDA Graph integration |
|---|---|
| vLLM | piecewise; `--enforce-eager` to disable |
| SGLang | piecewise + scheduler maintains bucket stability |
| TensorRT-LLM | AOT engine build with internal graphs |
| FlashInfer | `plan()` API designed for graph capture |

---

## Connections

- [[ch-12]] — parent chapter.
- [[continuous-batching]] (ch-04) — produces the dynamic batches that piecewise capture handles.
- [[excerpts/vllm-piecewise-graphs]] — vLLM's specific implementation.
- [[excerpts/tensorrt-llm-aot]] — TRT-LLM's ahead-of-time alternative.
- [[flashinfer]] (ch-11) — attention engine designed to be graph-safe.
