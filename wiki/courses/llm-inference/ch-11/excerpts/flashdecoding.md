---
chapter: ch-11
course: llm-inference
phase: read
excerpt_of: "Flash-Decoding for Long-Context Inference"
source_url: https://pytorch.org/blog/flash-decoding/
created_at: "2026-05-21"
---

# Excerpt: FlashDecoding — split-K for the decode case

**Authors:** Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov
**Year:** 2023
**Venue:** PyTorch blog + FlashAttention repository
**URL:** https://pytorch.org/blog/flash-decoding/
**Raw-data source:** [[raw-data/flashdecoding]]

---

## The problem FA2 doesn't solve at decode

At decode, query length is 1. FA2 parallelizes one thread block per `(batch, head)`. For a single user chatting (batch=1) with a model having 32 heads (Llama-2-7B):

```text
thread blocks active = 1 · 32 = 32
SMs available (A100) = 108
GPU utilization      = 32 / 108 = 30%
```

At batch=1, head=8 (GQA on Llama-3-8B): **7%** GPU utilization. The other 93% of the GPU is idle.

This is exactly the workload that long-context single-sequence decode produces — chat with 32k history, code generation in a deep context. The kernel is **memory-bandwidth-bound waiting on KV reads**, but the SMs that could be issuing those reads are inactive.

---

## The split-K reformulation

Partition the KV cache along the sequence dimension into chunks. Each chunk gets its own thread block:

```text
KV cache (length L_kv = 32768)
   └── Chunk 0  (tokens 0..1023)     → thread block 0
   └── Chunk 1  (tokens 1024..2047)  → thread block 1
   ...
   └── Chunk 31 (tokens 31744..32767) → thread block 31
```

Combined with `(batch, head)` parallelism: 1 batch · 32 heads · 32 KV chunks = **1024 thread blocks**, fully saturating the GPU.

---

## The exact-combine step

Each chunk `k` computes a local partial output `O_k` and the log-sum-exp normalizer `lse_k`:

```math
O_k = \text{partial\_softmax}(Q K_k^\top / \sqrt{d}) V_k
```

```math
\text{lse}_k = \log\!\sum_{i \in \text{chunk }k} \exp(Q K_{k,i}^\top / \sqrt{d})
```

After all chunks finish, combine using the same log-sum-exp identity FA1 uses:

```math
O = \sum_k \frac{e^{\text{lse}_k}}{e^{\text{lse}_{\text{global}}}} O_k,
\qquad \text{lse}_{\text{global}} = \log \sum_k e^{\text{lse}_k}
```

Mathematically equivalent to global softmax over the full KV; computationally distributable across many SMs.

---

## When FlashDecoding helps

| Workload | Speedup over FA2 |
|---|---|
| batch=1, L_kv=1k | 1.0× (FA2 already enough) |
| batch=1, L_kv=8k | ~3× |
| batch=1, L_kv=32k | ~6× |
| batch=1, L_kv=128k | ~8× |
| batch=32, any L_kv | 1.0× (batch already saturates) |

The crossover is roughly `batch × kv_heads ≥ n_SMs`. Above that, split-K is unnecessary; below, it's essential.

---

## Production dispatch

vLLM dispatches based on `(batch_size, kv_length)`:

```python
if batch_size * kv_heads < n_sms / 2 and kv_length > 4096:
    use FlashDecoding (split-K)
else:
    use FA2 / FA3
```

FlashInfer (next excerpt) generalizes this dispatch with JIT specialization.

---

## Connections

- [[ch-11]] — parent chapter.
- [[excerpts/flashattention-2]] — the prefill kernel FA2 left a decode gap that FlashDecoding fills.
- [[excerpts/flashinfer]] — packages split-K decode as one of its dispatch options.
- [[ch-08]] — long-context inference techniques; FlashDecoding makes them latency-feasible.
- [[ch-20]] — long-CoT decode in DeepSeek R1 / Qwen 3 thinking mode depends on FlashDecoding.
