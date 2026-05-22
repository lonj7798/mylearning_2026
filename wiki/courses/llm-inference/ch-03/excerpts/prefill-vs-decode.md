---
chapter: ch-03
course: llm-inference
phase: read
excerpt_of: "SARATHI (Agrawal et al. 2023) + DistServe (Zhong et al. 2024) framings of prefill vs decode"
source_url: https://arxiv.org/abs/2308.16369
created_at: "2026-05-21"
---

# Excerpt: Prefill vs decode — the two-phase cost asymmetry

**Authors:** Amey Agrawal et al. (SARATHI, 2023); Yinmin Zhong et al. (DistServe, 2024)
**Year:** 2023 / 2024
**URLs:** https://arxiv.org/abs/2308.16369 ; https://arxiv.org/abs/2401.09670
**Raw-data source:** [[raw-data/prefill-vs-decode]]

---

## The two phases formalized

```
PREFILL  (one forward pass, ingests entire prompt)
  input:    prompt tokens [x_1, ..., x_S]
  work:     compute Q, K, V for all S positions in parallel
            full S × S attention per layer (or FlashAttention equivalent)
            FFN + residuals + layernorms for all S positions
  output:   logits[S-1, :] (used to sample first output token)
  side eff: write K, V for all S positions to KV cache
  cost:     ~7 B S d² + 2 B S² d FLOPs
  latency:  TTFT (time-to-first-token)
  regime:   COMPUTE-BOUND (intensity ~1000+ FLOPs/byte)

DECODE  (repeated; one forward pass per output token)
  input:    one new token + KV cache for positions 1..t
  work:     compute Q, K_t, V_t for position t
            attention is [1 × t] per layer
            FFN + residuals + layernorms for one position
  output:   logits[0, :] (used to sample next output token)
  side eff: append K_t, V_t to cache
  cost:     ~7 B d² + 2 B t d FLOPs/step
  latency:  TPOT (time-per-output-token)
  regime:   BANDWIDTH-BOUND (intensity ~0.3 FLOPs/byte at B=1)
```

---

## Arithmetic intensity — quantified on H100

H100 SXM: 1979 TFLOPS bf16, 3.35 TB/s HBM. Roofline knee: 590 FLOPs/byte.

| Operation | Intensity | Regime | Throughput limit |
|---|---:|---|---|
| Prefill at 4k tokens | ~3100 | compute-bound | peak FLOPs |
| Prefill at 512 tokens | ~600 | mixed | mixed |
| Decode B=1 | ~0.3 | bandwidth-bound | peak HBM bw |
| Decode B=32 | ~9 | bandwidth-bound | peak HBM bw |
| Decode B=256 (short L) | ~70 | mixed | mixed |
| Decode B=256 (long L=32k) | ~3 | bandwidth-bound | peak HBM bw |

**Three orders of magnitude separate prefill and decode arithmetic intensity.** Putting them in the same batch means at least one is starving — either decode wastes 99% of compute or prefill wastes 99% of bandwidth.

This is the fundamental motivation for:
- **Chunked prefill** (ch-05): split long prefill into small chunks that fit a decode batch's compute budget.
- **Disaggregation** (ch-09): physically separate prefill and decode onto different GPU pools.

---

## TTFT and TPOT as separate SLOs

Production SLOs almost always specify two numbers:

```
TTFT (time-to-first-token):  user-visible "the model is alive" signal
                              dominant component: prefill FLOPs
                              typical SLO: p99 < 1000 ms

TPOT (time-per-output-token): user-visible "smooth typing" signal
                              dominant component: KV-bandwidth + weight-bandwidth
                              typical SLO: median < 50 ms
```

A request might have:
- Prompt 2000 tokens, output 200 tokens
- TTFT = 800 ms (prefill 2000 tokens)
- TPOT = 30 ms (200 decode steps)
- Total latency = 800 + 200 · 30 = 6.8 s

The 800 ms of TTFT is overhead the user pays *before* any output appears. The 30 ms/token TPOT is what determines whether streaming feels responsive.

---

## The prefill-decode interference problem (the SARATHI motivation)

If a long prefill (e.g. 4000 tokens) runs in the same forward pass as ongoing decodes, the decode tokens wait `4000-token-prefill-time ≈ 200 ms` before getting their next token. This breaks the TPOT SLO catastrophically.

```
naïve batching timeline:
  decode steps 1..50 (50 ms each)  ← steady 50 ms TPOT
  long prefill arrives             ← 200 ms blip
  decode step 51                   ← waited 200 ms; user sees a hang
  decode steps 52..100             ← back to 50 ms
```

SARATHI's solution: chunked prefill. Split the 4000-token prefill into 16 chunks of 256 tokens; each chunk takes ~12 ms; interleave with decodes. TTFT for the new request goes up slightly (waits for chunks to complete); TPOT for existing requests stays at 50 ms.

DistServe's solution: don't mix prefill and decode on the same GPU at all. Run prefill on a dedicated pool; ship KV cache to decode pool over NVLink/RDMA; run decode at full bandwidth. TTFT and TPOT are now decoupled.

Both work; both have tradeoffs. ch-05 and ch-09 cover the details.

---

## Operational signal: "why is my TTFT slow?"

Decision tree:

```
TTFT > SLO?
├── Is prompt long (> 4k tokens)?
│   ├── Yes → prefill FLOPs dominant
│   │   ├── Try FlashAttention (ch-11) — eliminate L² materialization
│   │   ├── Try prefix caching (ch-07) — reuse shared prompt
│   │   └── Try disaggregation (ch-09) — dedicate prefill GPUs
│   └── No → check scheduler queueing
│       ├── Is decode batch saturated? → admit-control + add capacity
│       └── Is request waiting for KV blocks? → enlarge KV pool
└── No → check TPOT (separate problem)
```

---

## Operational signal: "why is my TPOT slow?"

```
TPOT > SLO?
├── Is batch size small (< 16)?
│   ├── Yes → weight bandwidth dominant; batch up
│   │   └── Continuous batching (ch-04) + larger max_num_seqs
│   └── No → KV bandwidth dominant
│       ├── Is KV per token large (MHA)? → migrate to GQA model (ch-02)
│       ├── Is context very long? → KV compression (ch-08)
│       └── Are kernels fully fused? → CUDA Graphs (ch-12)
├── Are prefills interleaving with decodes?
│   └── Yes → chunked prefill (ch-05) or disaggregation (ch-09)
└── Speculative decoding (ch-14/15) gives 1.5-3× TPOT speedup nearly for free
```

---

## Common pitfalls

- **Reporting "throughput" as one number.** A workload with 2000-token prompts and 50-token outputs has a totally different prefill-to-decode ratio than one with 50-token prompts and 2000-token outputs. Always break out prefill tok/s and decode tok/s separately.
- **Tuning batch size from prefill perspective.** Prefill saturates compute at small batch (1–4); decode saturates bandwidth only at large batch (32–256). The two phases want different batch sizes simultaneously.
- **Assuming "compute-bound" forever**. Prefill is compute-bound at long prompts (≥ 1k tokens). At 50-token prompts it's bandwidth-bound for the weight loads.

---

## Connections

- [[excerpts/kv-cache-formula]] — the cache populated by prefill is read by decode.
- [[excerpts/batching-strategies]] — how the scheduler mixes (or separates) the two phases.
- [[raw-data/sarathi-serve]] — chunked prefill paper.
- [[raw-data/distserve]] — disaggregation paper.
- [[ch-04]] — continuous batching: where the two phases meet at iteration granularity.
- [[ch-19]] — TTFT/TPOT/goodput as the canonical inference metrics.
