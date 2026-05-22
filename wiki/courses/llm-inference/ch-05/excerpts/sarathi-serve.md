---
chapter: ch-05
course: llm-inference
phase: read
excerpt_of: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (Agrawal et al. 2024)"
source_url: https://arxiv.org/abs/2403.02310
created_at: "2026-05-21"
---

# Excerpt: Sarathi-Serve — chunked prefill for stall-free LLM serving

**Authors:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee
**Year:** 2024 (SOSP)
**URL:** https://arxiv.org/abs/2403.02310
**Raw-data source:** [[raw-data/sarathi-serve]]

---

## The problem statement (paper §2 — "the throughput-latency tradeoff")

Pure continuous batching faces three structural conflicts:

1. **Prefill is compute-heavy, decode is memory-bandwidth-heavy.** They have different ideal batch shapes and saturate different hardware resources.
2. **Prefill takes 10–100× longer than decode per step.** Mixing them via simple alternation means *every* decode step shares latency with the longest in-flight prefill.
3. **Bigger batches help decode (amortize KV-cache reads) but the moment a long prefill enters, the whole batch waits.**

The paper's term for the resulting tail-latency disaster is **generation stalls**: moments where running decodes pause for an unrelated prefill. Under vanilla continuous batching at 50 % offered load on Llama-13B, the paper measures stalls accounting for >40 % of p99 TPOT.

---

## The mechanism (paper §3 — "chunked prefills + stall-free scheduling")

Two combined ideas:

**(1) Chunked prefill.** Split a prompt of length `L_p` into `ceil(L_p / C)` chunks of size `C`. Each chunk performs a partial prefill — extending the request's KV cache by `C` positions. Because attention is causal and earlier chunks' K/V have already been cached, the chunk-`k` forward pass is computationally identical to "a request with a `(k-1)C`-token context, now extending by `C` tokens." No new kernel; just variable-length attention with the already-supported `cu_seqlens` packing.

**(2) Stall-free mixed-batch scheduling.** Every step is a single forward pass over a *mixed* batch:

```
step_batch = [prefill chunks] + [one decode token per running request]
sum(step_batch tokens) ≤ T_max
```

Decodes always get their token. The remaining budget goes to prefill chunks. The token budget `T_max` is set to keep the kernel compute-bound — typically 2048–8192 on modern GPUs.

> "The key insight is that decode latency is determined by the total number of tokens processed per iteration, not by the *kind* of tokens. We can therefore include prefill work alongside decodes without hurting decode latency, as long as we cap the per-iteration token count." (paraphrase, §3.2)

---

## The token-budget formula

For a step with `D` running decodes and `P` admitted prefill-chunk tokens:

```math
P + D \;\le\; T_{\max}
```

with the additional per-request cap

```math
\text{chunk}_i \;\le\; \min(\, \text{remaining prefill}_i, \; T_{\max} - D - \sum_{j<i} \text{chunk}_j, \; C_{\max} \,)
```

`T_max = max_num_batched_tokens` is the kernel-throughput knob; `C_max = max_chunk_size` is an optional per-chunk cap. The Sarathi-Serve paper recommends choosing `T_max` as the largest value where one step's latency is still under the deployment's TPOT SLO (typically `T_max × per-token cost ≤ 0.8 × TPOT_SLO`).

---

## Empirical numbers (paper §5, Figures 6–10)

| Metric | Vanilla continuous batching | Sarathi-Serve | Ratio |
|---|---|---|---|
| p99 TPOT (Mistral-7B, 16 req/s) | ~210 ms | ~55 ms | 3.8× lower |
| p99 TTFT (Mistral-7B, 16 req/s) | ~1.4 s | ~1.6 s | 1.14× worse |
| Capacity at SLO (Yi-34B, A100) | 1.0× | 2.6× | 2.6× higher |
| Tokens/sec (mean, mixed trace) | 1.0× | 1.5–2× | up to 2× |

The headline: **2.6× higher request capacity at a fixed SLO**, with TTFT regression of about 10–20 %. The win compounds as prompt-length variance grows in the trace.

---

## Why the kernel cost is essentially free

The marginal cost of running one mixed-batch forward pass with `T = D + P` tokens, versus a decode-only forward pass with `D` tokens, is small on modern attention kernels. From the paper's profiling:

- FFN cost is linear in `T` — no surprise.
- Attention cost grows sub-linearly because the prefill chunk and decode tokens share the dequant-and-load of weights; only the attention itself differs.
- The kernel-launch overhead is paid once per step regardless of `T`, so larger steps amortize it better.

Empirically on H100 + Llama-13B, going from `T=32` (decode only) to `T=512` (32 decodes + a 480-token prefill chunk) adds ~3 ms of step latency. The same 480-token prefill done as a separate step costs ~70 ms.

---

## Tuning recipe (paper Table 4 + §5.4)

| Knob | Typical value | What it controls |
|---|---|---|
| `max_num_batched_tokens` | 2048–8192 (H100), 1024–4096 (A100) | Per-step compute ceiling; raise until step latency ≥ 0.8 × TPOT SLO |
| `max_chunk_size` | 512–2048 | Per-prefill-chunk cap; bound only if you want hard per-step latency |
| `enable_chunked_prefill` | True | Toggle; off ⇒ vanilla continuous batching |
| KV utilization target | 90–95 % | Set via `gpu_memory_utilization` in vLLM |

Default in vLLM V1 since v0.6: chunked prefill on, `max_num_batched_tokens` auto-derived from model + GPU.

---

## What Sarathi-Serve does *not* solve

The paper is candid about the limit (§7 "Limitations"):

- **Decode-batch saturation.** If running decode count alone consumes the entire `T_max` budget, new prefills can't be admitted that step. Either raise `T_max` (TPOT regresses) or accept the TTFT regression.
- **Hard TTFT SLOs.** If your application demands sub-200 ms TTFT for 4k+ prompts, chunked prefill alone can't deliver — you need [[distserve]]-style disaggregation.
- **Long-context preemption.** A request mid-prefill that gets preempted under KV pressure loses its partial KV cache and must restart. The paper recommends sizing KV cache to avoid this, not solving it.

---

## Connections

- [[excerpts/vllm-scheduler]] — production realization in vLLM V1; the `Scheduler.schedule()` loop is a direct implementation of §3.2.
- [[excerpts/continuous-batching]] — Orca-derived iteration-level scheduling that chunked prefill upgrades.
- [[ch-05]] — parent synthesis.
- Forward to [[distserve]] / ch-09 — the disaggregation answer for the cases chunked prefill can't reach.
