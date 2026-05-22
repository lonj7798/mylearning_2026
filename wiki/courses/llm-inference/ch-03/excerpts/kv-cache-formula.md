---
chapter: ch-03
course: llm-inference
phase: read
excerpt_of: "KV Cache Memory Formula (synthesis card; rooted in Vaswani 2017, Shazeer 2019 MQA, Ainslie 2023 GQA)"
source_url: https://arxiv.org/abs/2309.06180
created_at: "2026-05-21"
---

# Excerpt: KV cache memory — the formula and worked examples

**Authors:** Vaswani et al. 2017 (attention); Shazeer 2019 (MQA); Ainslie et al. 2023 (GQA); Kwon et al. 2023 (PagedAttention/vLLM, popularized the memory-management framing)
**Year:** 2017–2023
**URLs:** https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/1911.02150 ; https://arxiv.org/abs/2305.13245 ; https://arxiv.org/abs/2309.06180
**Raw-data source:** [[raw-data/kv-cache-memory-formula]]

---

## The one-line formula

For a decoder-only transformer at inference, the KV cache for **one sequence** at context length `T`:

```math
\text{KV bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot T \cdot b
```

Where:
- `2` = one for K, one for V
- `L` = number of transformer layers
- `H_kv` = number of KV heads (NOT query heads; see ch-02)
- `d_head` = head dimension
- `T` = current context length in tokens (prompt + generated so far)
- `b` = bytes per element (2 for bf16, 1 for int8/fp8, 0.5 for int4)

For a **batch** of `B` independent sequences each at context `T`:

```math
\text{KV bytes}_{\text{batch}} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot T \cdot B \cdot b
```

That's the entire memory model. Every later optimization (PagedAttention, KV quantization, compression, MLA) modifies one of these terms.

---

## Per-token KV cost — the number to memorize per model

For each model, the per-token KV cost is `2 · L · H_kv · d_head · b`:

| Model | `L` | `H_kv` | `d_head` | bytes/token (bf16) |
|---|---:|---:|---:|---:|
| Llama-1-7B (MHA) | 32 | 32 | 128 | 524 KB |
| Llama-2-7B (MHA) | 32 | 32 | 128 | 524 KB |
| Llama-2-13B (MHA) | 40 | 40 | 128 | 819 KB |
| Llama-2-70B (GQA-8) | 80 | 8 | 128 | 320 KB |
| **Llama-3-8B** (GQA-8) | 32 | 8 | 128 | **128 KB** |
| **Llama-3-70B** (GQA-8) | 80 | 8 | 128 | **320 KB** |
| Llama-3-405B (GQA-8) | 126 | 8 | 128 | 504 KB |
| Qwen-3-32B (GQA-8) | 64 | 8 | 128 | 256 KB |
| Mistral-7B (GQA-8) | 32 | 8 | 128 | 128 KB |
| Mixtral-8x7B (GQA-8) | 32 | 8 | 128 | 128 KB |
| PaLM-540B (MQA) | 118 | 1 | 256 | 118 KB |
| DeepSeek V3 (MLA, fp8) | 61 | (MLA latent) | — | ~70 KB |

**Llama-3-70B's 320 KB/token** is the canonical reference number. Multiply by your context length to get per-request KV.

---

## Worked example 1: Llama-3-70B at 8k context

```
per-token KV   =  2 · 80 · 8 · 128 · 2 bytes  =  327,680 bytes ≈ 320 KB
per-request    =  320 KB · 8192                ≈  2.5 GB
```

On 8×H100 (640 GB HBM) with weights = 140 GB (bf16, replicated then sharded TP=8 ≈ 18 GB/GPU) and overhead = 80 GB (activations, scratch, CUDA graphs):

```
KV budget        =  640 − 140 − 80          =  420 GB
batch capacity   =  420 / 2.5                =  168 concurrent 8k-context requests
```

A common production number: ~150 concurrent requests on a single 8×H100 node.

---

## Worked example 2: same model at 128k context

```
per-request    =  320 KB · 131072             ≈  40 GB per request
batch capacity =  420 / 40                    =  10.5 concurrent requests
```

**16× context, ~16× fewer concurrent requests, ~16× lower throughput.** Long context isn't free — it eats into batch capacity linearly.

This is why long-context production deployments often:
1. Use KV quantization (fp8/int8) → 2× more capacity
2. Use KV compression (H2O, SnapKV) → another 2–4×
3. Use disaggregation → decouple prefill (one-time cost) from decode (the binding cost)

---

## Worked example 3: DeepSeek V3 with MLA + fp8

DeepSeek V3 uses Multi-head Latent Attention (MLA): instead of caching `K, V` directly, it caches a learned low-rank latent `c_kv ∈ ℝ^{d_latent}` and reconstructs K, V at attention time. The latent dim is ~512–1024.

```
per-token KV (MLA, fp8)  ≈  70 KB
per-request at 32k       ≈  70 KB · 32k  =  2.3 GB
```

This is *less than Llama-3-70B at 8k*. MLA + fp8 is the single reason DeepSeek V3 671B serving is practical on the same hardware that hosts Llama-3-70B.

---

## What's not in the formula (but should be in your budget)

The raw `2·L·H_kv·d_head·T·B·b` is a lower bound. Real allocators add:

- **Block tables / page tables**: ~1–2% overhead in PagedAttention (ch-06).
- **Fragmentation**: 5–15% in naive contiguous allocators; <5% in PagedAttention.
- **Pre-allocated headroom**: vLLM reserves blocks for swap-out under preemption.
- **CUDA graph KV reserves**: piecewise CUDA graphs (ch-12) bucket batch sizes; each bucket reserves cache.

Rule of thumb: multiply the formula by 1.15 to get real-world KV consumption.

---

## Common pitfalls

- **Using `n_heads` instead of `n_kv_heads`**. Off by 4× to 32× on modern GQA/MQA models. Always read `num_key_value_heads` from the model config.
- **Forgetting that `T` includes the prompt**. A 4k-prompt, 4k-output sequence has `T = 8192`, not 4096.
- **Ignoring dtype changes**. fp8 KV halves the bytes; int4 KV quarters them. KIVI/KVQuant (ch-08) explicitly target this term.
- **Treating KV budget as fixed across batch**. KV grows during decode — at decode step `t`, every active sequence's cache is `t` tokens deep. Schedulers must reserve for max-length, not current length.
- **Replicating KV across TP shards**. KV is sharded along the `H_kv` dimension under tensor parallelism; each GPU holds `H_kv / TP` heads. Make sure your accounting matches your TP setup.

---

## Connections

- [[excerpts/prefill-vs-decode]] — the cache is populated by prefill, consumed by decode; two phases drive two cost models.
- [[excerpts/batching-strategies]] — the cache is the binding constraint on batch size.
- [[raw-data/multi-query-attention]] / [[raw-data/grouped-query-attention]] — the `H_kv` term and its history.
- [[ch-06]] — PagedAttention turns this static formula into a paged virtual memory system.
- [[ch-08]] — KV compression attacks `T` directly (only cache "important" tokens).
- [[ch-20]] — MLA breaks the formula entirely by caching a learned latent instead of K, V.
