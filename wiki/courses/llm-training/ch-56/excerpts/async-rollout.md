---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/async-rollout.md
source_url: https://arxiv.org/abs/2405.11143 (OpenRLHF), https://arxiv.org/abs/2409.19256 (HybridFlow/verl)
created_at: "2026-04-23"
---

# Excerpt: Async rollout — the queue, the lock, and the 1.9x throughput

**Source library:** `wiki/raw-data/llm-training/papers/async-rollout.md`
**Authors:** Hu et al. 2024 (OpenRLHF paper §3.3); ByteDance Seed team 2024 (HybridFlow §4); IMPALA ancestry (Espeholt 2018)

---

## Why this source anchors ch-56

§4 of ch-56 describes OpenRLHF's async PPO as three Ray primitives:
`rollout_queue`, `rollout_slots`, `vllm_lock`. This source is where
those primitives are specified and where the 1.6–2.0× throughput
number is attested. Without async rollout, OpenRLHF's Ray-pool design
would be a simple wrapper around synchronous training; with it, it
becomes a genuine architectural improvement.

---

## The core insight, attested

Source §Core Insight:

> Synchronous RL training leaves the GPU rollout engine idle during
> optimizer steps (and vice versa). Decoupling rollout and training
> into two asynchronous processes connected by a queue recovers most
> of that idle time — at the cost of a bounded off-policyness of ≤1
> generation-cycle, which importance-sampling correction handles
> cleanly.

The tradeoff is explicit: **bounded staleness in exchange for
throughput**, and the IS correction ([[excerpts/openrlhf-ppo]] three
modes) is what keeps the bias bounded.

---

## OpenRLHF async primitives, attested

Source §Technical Details:

> OpenRLHF async primitives (from `openrlhf/trainer/ppo_trainer_async.py`):
> - `rollout_queue`: `ray.util.queue.Queue`, capacity 1–2.
> - `rollout_slots`: companion queue carrying `global_step` tokens
>   (backpressure).
> - `vllm_lock`: `ray` asyncio.Lock to serialize weight-broadcast vs
>   generate.
> - Partial-rollout flag: `strategy.args.train.partial_rollout_enable`.

Ch-56 §4's actor-pool description is a direct map of these primitives
onto the physical cluster layout in `figures/openrlhf-ray.html`.

---

## The staleness bound

Source §Technical Details:

> Staleness bound: `k = queue_depth + partial_rollout_depth`;
> typical k=1–2.

A rollout generated at global step `t` is consumed by the trainer at
step `t + k`. The policy has changed during those k steps, so rollout
and current policy differ. IS correction corrects for this
(V-trace-style; mathematically identical to IMPALA's clip).

---

## The 1.6–2.0x number

Source §Key Figures/Tables to Study:

> OpenRLHF paper Figure 5: sync vs async throughput — 1.9× at 7B,
> 1.6× at 70B.

The 70B number is lower because the optimizer step at 70B is longer
relative to generation, so the hidden idle time is smaller. At 7B,
rollouts are cheap and training is the bottleneck; the gain is larger.
The qualitative trend (async matters more at smaller scale) is a
counter-intuitive lesson that the source makes concrete.

---

## The failure signature

Source §Technical Details:

> Failure signature: `vllm_kl` divergence > 0.1 plus PPO clipfrac
> pegged at 1 indicates the async staleness + sampler-mismatch has
> exceeded what IS correction can handle.

Ch-56 §7 uses the same signature. The fix in the source:

> Drop the rollout entirely if the sequence IS weight exceeds a hard
> cap.

`seq-mask-tis` in OpenRLHF ([[excerpts/openrlhf-ppo]]) is how this is
actually implemented.

---

## Ancestry — IMPALA V-trace

Source §Connections:

> Direct descendant of IMPALA V-trace — the IS-weight clip is
> mathematically the same object.

This is load-bearing. If you have read [[async-rollout]] you know
that V-trace clips `c_t = min(c̄, π/μ)` for the trace and `ρ_t =
min(ρ̄, π/μ)` for the advantage — the exact same clipping shows up in
OpenRLHF's `tis` mode. Async rollout is not a framework hack; it is a
direct implementation of IMPALA's correctness result applied to LLM RL.

---

## Connections

- [[excerpts/openrlhf-ppo]] — the IS-correction branches exist because
  of async rollout.
- [[excerpts/entropy-logging-patterns]] — `vllm_kl` is the OpenRLHF
  metric that diagnoses async failure.
- Host chapter: [[ch-56]] §4.
- Forward to [[ch-57]] (TRL) — TRL has no async rollout (as of 2026);
  this is one of the "outgrown TRL" signals.
