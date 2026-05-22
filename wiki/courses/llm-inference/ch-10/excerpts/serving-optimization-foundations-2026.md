---
chapter: ch-10
course: llm-inference
phase: read
excerpt_of: "Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics"
source_url: https://arxiv.org/abs/2605.01280
created_at: "2026-05-21"
---

# Excerpt: The 2026 position paper — serving has outgrown FIFO/LRU

**Authors:** Zijie Zhou
**Year:** 2026
**Venue:** arXiv 2605.01280 (position paper)
**URL:** https://arxiv.org/abs/2605.01280
**Raw-data source:** [[raw-data/serving-optimization-foundations-2026]]

---

## The thesis (the load-bearing claim)

LLM serving systems still use general-purpose distributed-systems heuristics:

| Layer | Default heuristic | Why it's wrong for LLM |
|---|---|---|
| Request routing | join-shortest-queue | Ignores prefix-cache locality |
| Scheduling | FIFO + size-based | Ignores prefill/decode asymmetry |
| Cache eviction | LRU per-block | Ignores request-level reuse structure |
| Admission | rate limit + max-concurrent | Ignores per-request cost (1000× variance) |
| Load balancing | round-robin | Ignores hardware-tier specialization |

Each was designed for workloads with relatively uniform per-request cost and independent memory. LLM inference violates both assumptions.

---

## The four pieces of LLM-specific structure

1. **Prefill/decode phase asymmetry**: prefill is compute-bound and prompt-length-dependent; decode is memory-bandwidth-bound and output-length-dependent. One queue for both is provably suboptimal.

2. **KV cache grows during execution**: memory occupancy is a function of `(elapsed_decode_steps, prompt_length)`. Admission decisions made at arrival are made on stale state by the time the request runs.

3. **Unknown output length**: service time is fundamentally uncertain. Classic shortest-job-first becomes "shortest-expected-job-first under output-length distribution", which is a stochastic scheduling problem.

4. **Continuous-batching coupling**: every active request's TPOT depends on every other active request. There's no per-request latency; only joint per-batch latency.

---

## The optimization framings the paper proposes

For each layer, replace the heuristic with a formal optimization:

**Routing** (mixed-integer programming):
```
min sum_{r in requests} w_r · (queue_delay_r + transfer_cost_r)
s.t.   sum_r x_{r,w} <= worker_capacity_w
       prefix_cache_match_bonus(r, w) included in objective
```

**Scheduling** (stochastic deadline scheduling):
```
Each request r has:
  - arrival a_r, deadline d_r = a_r + SLO_r
  - prefill cost p_r (known)
  - decode cost D_r ~ distribution F_r(N_output)
Pick a policy that maximizes expected count of requests served by deadline.
```

**Cache eviction** (Belady approximation):
```
Predict, for each cached block b, the time-to-next-use τ_b.
Evict argmax_b τ_b.  (LRU is a degenerate predictor: τ_b = -last_use.)
```

**Admission** (utility-maximizing):
```
Admit r iff:
  E[goodput contribution of r] > E[goodput cost to in-flight requests]
```

---

## What the paper is NOT claiming

- Not claiming heuristics are unusable: at very low load they're fine.
- Not claiming optimal scheduling is computable in real-time at scale.
- Not claiming any single optimization formulation is the final answer.

The claim is that **the field has crystallized enough that researchers should now be working on bounded-suboptimality approximations of these formal problems** rather than tuning yet another FIFO variant.

---

## Connection to the chapter

This paper is the methodological frame for ch-10. The chapter introduces VTC and Niyama as policies; the 2026 paper situates them as early instantiations of a larger optimization-theoretic shift.

For the learner: when designing your own serving policy, start by writing down the formal optimization problem (objective + constraints + uncertainty). Then choose a heuristic that approximates it. Don't start with a heuristic and rationalize.

---

## Connections

- [[ch-10]] — parent chapter; this paper is its methodological frame.
- [[excerpts/vtc]] — early instance of formal fair-queueing approach.
- [[excerpts/niyama]] — early instance of QoS-aware optimization.
- [[excerpts/admission-control-goodput]] — the admission layer the paper calls for.
- [[ch-16]] / [[ch-17]] — concrete frameworks still implementing largely heuristic schedulers.
