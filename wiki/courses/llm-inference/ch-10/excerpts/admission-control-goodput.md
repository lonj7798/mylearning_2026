---
chapter: ch-10
course: llm-inference
phase: read
excerpt_of: "Admission Control and Goodput in LLM Serving — design-pattern synthesis"
source_url: synthesis; references arXiv 2401.09670, 2407.00079, 2410.14257, 2505.23022
created_at: "2026-05-21"
---

# Excerpt: Admission control + goodput as the policy frame

**Authors:** Synthesis card across DistServe, Mooncake, VTC, Sarathi-Serve, SCORPIO, Revisiting SLO/Goodput
**Year:** 2024–present
**URLs:** see ch-10 sources
**Raw-data source:** [[raw-data/admission-control-goodput]]

---

## The throughput-vs-goodput gap

```math
\text{throughput}(t) \;=\; \frac{|\text{completed}(t)|}{t}
```

```math
\text{goodput}(t, S) \;=\; \frac{|\{r : r \in \text{completed}(t),\; r \text{ met SLO } S\}|}{t}
```

Under overload, throughput keeps rising (you complete more requests) while goodput collapses (almost none of them met SLO). The peak of `goodput(λ)` defines the **goodput-feasible QPS** — the actual sustainable capacity.

Empirically, on a typical vLLM deployment serving Llama-3-8B at 500 ms TTFT SLO:
- Throughput-feasible QPS: ~28 req/s (machine saturates at this load).
- Goodput-feasible QPS: ~18 req/s (above this, p95 TTFT exceeds 500 ms).

The 10 req/s gap is the cost of throughput-first scheduling.

---

## The admission cost model

```text
On request R arriving:
    inputs = (
        R.prompt_len,
        R.max_tokens,
        R.SLO,
        queue,
        active_batch,
        KV_free,
        prefill_rate,
        decode_rate,
        output_length_predictor(R.tenant)
    )

    est_queue_delay  = sum(prefill_cost(r) for r in queue) / prefill_rate
    est_prefill_time = R.prompt_len / prefill_rate
    est_TTFT         = est_queue_delay + est_prefill_time

    est_decode_batch = active_batch_size + 1
    est_TPOT         = decode_time(est_decode_batch, KV_pressure_after_admit)

    est_KV_blocks    = ceil((R.prompt_len + predicted_output_len) / block_size)
    KV_fits          = est_KV_blocks <= KV_free

    decision = ADMIT if (est_TTFT <= R.SLO.TTFT_max and
                         est_TPOT <= R.SLO.TPOT_max and
                         KV_fits)
               else REJECT
```

---

## The output-length-prediction problem

The cost model needs `predicted_output_len` and you don't have it. Three strategies:

1. **Per-tenant empirical p95**: `predicted_output_len = ema_p95(tenant.recent_outputs)`. Robust but slow to adapt to a new workload.
2. **Per-prompt-class p95**: classify by prompt shape (chat vs RAG vs code), use the class p95. Better for shared APIs.
3. **Learned predictor**: small model on `(prompt_embedding, request_metadata) → output_len_quantiles`. Mooncake uses this; calibrates on production traces.

All three are conservative — they over-predict to leave KV headroom. Over-prediction reduces admit rate (lower throughput); under-prediction causes mid-flight KV exhaustion (forces preemption, hurts goodput more).

---

## Rejection vs queueing — the binary that matters

A queue is not "soft admission"; an unbounded queue **causes** SLO violation. Two correct designs:

- **Hard reject at the door**: return 429 immediately. Client retries with backoff. SLO is preserved at cost of higher reject rate.
- **Deadline-bounded queue**: admit to a queue with a hard deadline = SLO budget; drop on deadline expiry. Smoother for clients but harder to predict.

The wrong design: unbounded queue with no deadline. This is the default in most stacks and is the single biggest source of SLO violations under load.

---

## SCORPIO + heterogeneous-SLO follow-ups

[[admission-control-goodput]] references SCORPIO and "Revisiting SLO and Goodput Metrics" as 2024–25 follow-ups:

- **SCORPIO**: separate "TTFT guard" and "TPOT guard" because the two SLOs have different feasibility patterns. Admits a request only if both guards pass.
- **Revisiting SLO/Goodput**: argues for **smooth goodput** (partial credit for "just barely missed SLO" requests) instead of binary pass/fail. Useful for tuning, controversial for user-facing reporting.

---

## Connections

- [[ch-10]] — parent chapter.
- [[excerpts/vtc]] — fair scheduling among admitted requests.
- [[excerpts/niyama]] — QoS-class-aware admission and preemption.
- [[ch-09]] / Mooncake — production deployment of predictive early rejection.
- [[ch-19]] — how to actually measure goodput in benchmarks.
