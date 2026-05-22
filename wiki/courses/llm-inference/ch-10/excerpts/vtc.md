---
chapter: ch-10
course: llm-inference
phase: read
excerpt_of: "Fairness in Serving Large Language Models (VTC)"
source_url: https://arxiv.org/abs/2401.00588
created_at: "2026-05-21"
---

# Excerpt: VTC — fair queueing for continuous-batched LLM serving

**Authors:** Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica
**Year:** 2024
**Venue:** arXiv 2401.00588
**URL:** https://arxiv.org/abs/2401.00588
**Raw-data source:** [[raw-data/vtc]]

---

## Why per-request rate limits fail

API-gateway rate limits ("100 req/min per key") assume requests are comparable. LLM requests vary by **3 orders of magnitude in cost**:

| Request shape | Cost (relative) |
|---|---|
| 50-token chat prompt, 50-token reply | 1× |
| 500-token prompt, 200-token reply | 8× |
| 8k RAG context, 500-token reply | 70× |
| 200k-token document summarization | 2000× |

A user spamming chat messages and a user submitting one 200k-token document are both "1 request". Count-based fairness mis-prices them by 2000×.

---

## The service function (the load-bearing definition)

VTC tracks per-client cumulative *served work*:

```math
V_c \;=\; \sum_{t \in \text{served steps}} \bigl(w_{\text{in}} \cdot n_{\text{in},c}^{(t)} \;+\; w_{\text{out}} \cdot n_{\text{out},c}^{(t)}\bigr)
```

Weights `w_in / w_out` calibrate the per-token cost ratio. Typical empirical values from the paper and production:

| Scenario | `w_in : w_out` |
|---|---|
| Prefill-cheap, decode-expensive (decode-bound) | 1 : 4 |
| Long-context dominant | 1 : 2 |
| Tiny prompts only | 1 : 8 |

Calibrate from `(prefill_time, decode_time)` measurements on your hardware.

---

## The scheduling rule

```text
At each iteration of the continuous-batching loop:
    active = {c : c has a pending or in-flight request}
    while batch_has_room() and active not empty:
        c* = argmin_{c in active} V_c
        admit c*'s next runnable request into batch
```

This is the LLM equivalent of **weighted fair queueing**: serve the client whose virtual finish time is smallest first. Because we're at iteration granularity (not request granularity), the scheduler can interleave clients within the same batch.

---

## Join / leave handling — the credit-hoarding bug

Naive lowest-`V` selection has a known failure: a client that's idle for 10 minutes accumulates "credit" (their `V_c` doesn't grow while everyone else's does). When they return, they get prioritized for the next 10 minutes — monopolizing the system.

VTC's fix:

```text
On client c rejoining after idle:
    V_c = max(V_c, min_{c' active} V_{c'})
```

A rejoiner cannot have a counter lower than the lowest *currently-active* client. Their long absence does not earn them credit.

---

## The fairness guarantee

The paper proves: for any two continuously-backlogged clients `c1`, `c2` over any time window `T`,

```math
\bigl| \text{served}_{c_1}(T) - \text{served}_{c_2}(T) \bigr| \;\le\; 2 \cdot \max_{c}(\text{cost of one request})
```

i.e., the served-work difference is bounded by 2× the cost of a single request. This is **tight** — the constant 2 cannot be improved while remaining work-conserving (because the scheduler cannot preempt an in-flight prefill mid-step).

---

## Empirical results

On LMSYS production traces:
- FCFS (first-come-first-served): max served ratio between clients ≈ 8× — extreme unfairness during burst from one user.
- Requests-per-minute limit: 4× — better, but coarse.
- **VTC**: 1.5–2× — at the proved bound.

VTC throughput vs FCFS is **within 3%** — fair scheduling does not cost throughput because the scheduler remains work-conserving.

---

## Connections

- [[ch-10]] — parent chapter; VTC is the fair-scheduling layer above continuous batching.
- [[continuous-batching]] — the substrate VTC schedules over.
- [[excerpts/admission-control-goodput]] — VTC handles **who runs**; admission handles **who enters**.
- [[excerpts/niyama]] — different problem (QoS classes), composable with VTC (per-class VTC).
