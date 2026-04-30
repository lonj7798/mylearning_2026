---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/async-rollout.md
source_url: https://arxiv.org/abs/2409.19256
created_at: "2026-04-23"
---

# Excerpt: Async rollout for LLM RL — the design behind `vllm_async_server`

**Source library:** `wiki/raw-data/llm-training/papers/async-rollout.md`
**Artifact:** HybridFlow / verl async architecture §4; bounded-k staleness + IS correction; IMPALA V-trace lineage.

---

## Why this excerpt exists in ch-55

Ch-55 §4.2 quotes the async server code. This file is the *why* — the architectural rationale for why `vllm_async_server.py` looks the way it does, why priority + paused-state are load-bearing, and why the SPMD path (PR #4411) had to go.

---

## Core insight from the source

> Synchronous RL training leaves the GPU rollout engine idle during optimizer steps (and vice versa). Decoupling rollout and training into two asynchronous processes connected by a queue recovers most of that idle time — at the cost of a bounded off-policyness of ≤1 generation-cycle, which importance-sampling correction handles cleanly.

Guideline: *"For any RL run >30 min, use async rollout. The staleness is bounded by the queue depth (usually 1–2); IS correction + vLLM-trainer logprob-difference monitoring (`vllm_kl`) keeps it safe. Expect 1.6–2.0× throughput vs sync."*

---

## verl-specific primitives (from the source)

From §Technical Details / verl async primitives:

- `engine.generate(...)` is async-for-iterated per-request.
- `priority` parameter reorders in-flight requests.
- Pause state (≈ line 628) blocks new generates and waits for in-flight to drain before a weight broadcast.

These three primitives map 1:1 onto the code quoted in ch-55 §4.2 (and `excerpts/verl-rollout.md`). The pattern is identical to OpenRLHF's Ray-based version — `rollout_queue` + `vllm_lock` + partial-rollout flag — but threaded through vLLM's internal async scheduler instead of a Ray queue.

---

## Bounded staleness + IS correction

From §Technical Details:

- Staleness bound: `k = queue_depth + partial_rollout_depth`; typical k = 1–2.
- Per-token IS weight `exp(logπ_train − logπ_rollout)`, clipped to `[low, high]`.
- Sequence-level mask when the IS weight exits `[low, high]` for any token.
- Drop the rollout entirely if the sequence IS weight exceeds a hard cap.
- Failure signature: `vllm_kl` divergence > 0.1 plus PPO clipfrac pegged at 1 indicates async staleness + sampler mismatch has exceeded what IS correction can handle.

This is the `rollout_is_weights` argument in `compute_policy_loss_vanilla` (see `excerpts/verl-ppo-loss.md`). Async without IS correction is correct only when vLLM and trainer run identical dtype and sampler — which in practice they never do (bf16 vLLM, fp32 master weights).

---

## Why sync SPMD was retired (context for PR #4411)

Sync SPMD colocated vLLM with the trainer, so:
- No per-request priority → no partial-rollout RL → every training step waits on the longest rollout.
- No pause-state gate → weight broadcasts race in-flight decoding → mid-sequence mixed-weight outputs.
- Throughput ceiling ≈ 2× HFRollout; async continuous-batching sits at 5–8×.

All three are structural, not implementation bugs. The async rewrite (PR #4411) was the only way forward.

---

## IMPALA / V-trace lineage

The IS-weight clip on per-token `exp(logπ_train − logπ_rollout)` is mathematically the V-trace weight from IMPALA (Espeholt 2018). Async LLM-RL is Ape-X / IMPALA with a Transformer actor, vLLM replacing the Atari env, and a learned reward model or verifier in place of environment reward.

This lineage is why the async correctness story is textbook: the clip thresholds and truncated-IS semantics were worked out in 2018; LLM-RL reuses them verbatim.

---

## Throughput numbers (from the source)

- OpenRLHF paper Figure 5: sync vs async throughput — **1.9× at 7B, 1.6× at 70B**.
- verl blog "Continuous batching RL": timeline of partial rollout + weight broadcast showing the trainer never waiting.

Both frameworks report the same pattern: 7B benefits more in percentage terms because optimizer step is a larger fraction of wall-clock; 70B still benefits but rollout already dominates more completely.

---

## Connections

- [[verl-rollout]] — the code excerpt this paper motivates.
- [[verl-ppo-loss]] — `rollout_is_weights` implements V-trace-style per-token IS on the async staleness.
- [[entropy-logging-patterns]] — `vllm_kl` (OpenRLHF) / `rollout_kl` (verl extension) monitor async drift.
- [[ppo]] — sync on-policy baseline; async is the generalization.
