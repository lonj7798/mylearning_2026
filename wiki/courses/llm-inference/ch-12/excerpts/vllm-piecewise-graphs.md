---
chapter: ch-12
course: llm-inference
phase: read
excerpt_of: "vLLM — piecewise CUDA graph implementation"
source_url: https://docs.vllm.ai/en/latest/
created_at: "2026-05-21"
---

# Excerpt: vLLM's piecewise CUDA graphs

**Authors:** vLLM project (UC Berkeley Sky Computing Lab + community)
**Year:** 2023–present
**URL:** https://docs.vllm.ai/en/latest/
**Raw-data source:** [[raw-data/vllm]]

---

## Where the graph capture lives

vLLM v1 captures graphs in `vllm/v1/worker/gpu_model_runner.py` via the `vllm/compilation/cuda_graph.py` helper. Each worker captures one graph per batch-size bucket.

Default bucket list (when `max_num_seqs=128`):

```python
CUDA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, ..., 128]
```

Roughly: small powers of 2 at the low end, +8 increments at the high end (because padding from 70 to 80 wastes less proportionally than padding 1 to 8).

---

## Capture flow at engine init

```text
1. Engine loads model, allocates KV-cache blocks.
2. For each B in CUDA_GRAPH_CAPTURE_SIZES:
   a. Build a dummy batch of size B (fake token IDs, fake block tables).
   b. Run 3 warmup forward passes to JIT and prime allocator.
   c. capture_into(g_B) running one forward pass.
3. Engine ready to serve.
```

Total capture time: ~30–60 s for a 70B model with 30 buckets.

---

## Per-step dispatch

```text
On each scheduler iteration:
   selected = scheduler.pick_running_batch()   # size N (variable)
   B_bucket = next_captured_size(N)            # nearest captured >= N
   pad selected up to B_bucket with dummy tokens
   copy padded inputs into g_bucket.static_input_buffer
   g_bucket.replay()
   read first N tokens from g_bucket.static_output_buffer
   scheduler.handle_finished(N)
```

The dummy-token padding is benign: the dummy outputs are ignored. Compute is wasted on padding rows but launch overhead is amortized.

---

## What runs eagerly (not in graph)

- **Prefill steps**: variable prompt length means too many bucket-shape pairs to capture. Prefill uses FlashInfer/FA directly in eager mode.
- **Mixed prefill+decode steps** (when chunked prefill is enabled): vLLM detects the mixed case and falls back to eager for that iteration only.
- **Speculative-decode draft step**: variable draft length defeats bucketing.

The serving cost of eager fallback: the eager TPOT regression (typically +3–5 ms) appears only when the mixed condition triggers. For pure-decode iterations, graphs are always used.

---

## Knobs

| Flag | Effect |
|---|---|
| `--enforce-eager` | Disable all graph capture. Debug only. Typical regression: 25–40% TPOT. |
| `--max-num-seqs N` | Bounds the largest captured graph. Memory ∝ N. |
| `--cuda-graph-sizes "1,2,4,8,16,32"` | Override bucket list. |
| `--gpu-memory-utilization 0.92` | Leaves room for graph buffers. Below 0.85 risks OOM at capture. |

---

## Memory footprint

For Llama-3-70B with 30 buckets:
- Per-bucket static activation buffers: ~200 MiB at bucket=128.
- Total static buffer memory across all buckets: ~3–4 GiB.
- KV cache pool: ~30–40 GiB (depends on `gpu_memory_utilization`).
- Model weights: 140 GiB (FP16, sharded across 4-way TP = 35 GiB per GPU).

The graph buffers are not negligible; on tight-memory deployments, prune the bucket list.

---

## Reported numbers (from vLLM benchmarks)

| Setting | TPOT (Llama-3-70B, H100×4) |
|---|---|
| Eager mode | 28 ms |
| Piecewise CUDA graphs | 17 ms |
| Improvement | **~40% lower TPOT** |

For Llama-3-8B (smaller, more launch-dominant per token): 50% TPOT improvement.

---

## Connections

- [[ch-12]] — parent chapter.
- [[excerpts/cuda-graphs-inference]] — the underlying CUDA mechanism.
- [[ch-16]] / [[vllm-scheduler]] — scheduler decisions that affect bucket-hit rate.
- [[flashinfer]] (ch-11) — vLLM's preferred graph-safe attention backend.
- [[continuous-batching]] (ch-04) — the dynamic batches that drive bucket selection.
