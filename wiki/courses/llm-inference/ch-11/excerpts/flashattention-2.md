---
chapter: ch-11
course: llm-inference
phase: read
excerpt_of: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
source_url: https://arxiv.org/abs/2307.08691
created_at: "2026-05-21"
---

# Excerpt: FlashAttention-2 — closing the gap to GEMM efficiency

**Authors:** Tri Dao
**Year:** 2023
**Venue:** arXiv 2307.08691
**URL:** https://arxiv.org/abs/2307.08691
**Raw-data source:** [[raw-data/flashattention-2]]

---

## The unfinished business after FA1

FA1 made attention IO-optimal. But measured FLOPs/sec on A100 was only ~25–40% of GEMM peak — far below cuBLAS's 80%+. The gap was three things:

1. **Too much non-matmul work** in the inner softmax-rescale loop.
2. **Low GPU occupancy** at small batch × few heads (one thread block per `(b, h)`).
3. **Excessive shared-memory traffic** between warps within a block.

FA2 fixes all three.

---

## Improvement 1 — non-matmul work reduction

FA1's recurrence divides by `ℓ_new` every block; the divide is non-matmul and slow. FA2 reorders so the final scaling is applied **once per output row** instead of every block:

```text
Inner loop: just compute exp(S_ij - m) and accumulate without dividing.
Post-loop:  apply 1/ℓ_final to the accumulated output once.
```

Cuts non-matmul ops by ~2×. Since tensor cores are unused for non-matmul work, this directly recovers wall time.

---

## Improvement 2 — sequence-dimension parallelism

FA1 parallelizes over `(batch, head)`. For LLaMA-7B at batch=1, head=32: 32 thread blocks across an A100's 108 SMs — 70% idle.

FA2 also parallelizes across **rows of Q** (sequence dim):

```text
For each (batch, head):
    For each Q-row-block:
        One thread block, runs the full K/V iteration
```

At long context, occupancy is restored. Critical for serving where batch is small and context is large.

---

## Improvement 3 — warp work partitioning

FA1 has 4 warps per block all doing similar work; partial results are exchanged through shared memory at every iteration.

FA2 specializes:

```text
Warps 0–1: load + Q @ K^T matmul
Warps 2–3: compute exp(...) + matmul with V
```

Reduces shared-memory writes by ~50%. The remaining barrier is lighter.

---

## Headline numbers (Table 1 of the paper)

| Setting | A100 TFLOPS | % of GEMM peak |
|---|---|---|
| Standard PyTorch attention | ~40 | 13% |
| FlashAttention 1 | ~135 | 44% |
| **FlashAttention 2** | **~225** | **73%** |

End-to-end GPT-style training: **1.8× faster** than FA1.

---

## Supported features (production-relevant)

- Head dimensions up to 256 (covers Llama-3 d=128, Qwen d=128, DeepSeek MLA d=512 with TP)
- Causal mask, ALiBi, custom score bias
- Variable-length packed batches (for ragged batches in serving)
- **MQA and GQA** native support — important because production models since Llama-2 use GQA
- Backward pass for training

---

## Why FA2 is the Ampere production default

PyTorch's `F.scaled_dot_product_attention` dispatches to FA2 on Ampere when shapes are compatible. vLLM, SGLang, TGI all use FA2 (directly or through FlashInfer) for prefill on A100/H100 in FP16/BF16. FA3 only matters for Hopper FP16+ workloads.

---

## Connections

- [[ch-11]] — parent chapter.
- [[excerpts/flashattention]] — the IO-aware foundation FA2 builds on.
- [[excerpts/flashattention-3]] — Hopper-era follow-on; needs FA2's work partitioning as a base.
- [[excerpts/flashdecoding]] — orthogonal decode-time variant; FA2 is for prefill, FlashDecoding for decode.
- [[excerpts/flashinfer]] — FA2 is one of FlashInfer's dispatch targets.
