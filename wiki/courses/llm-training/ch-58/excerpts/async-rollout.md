---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/async-rollout.md
source_url: https://arxiv.org/abs/2405.11143
created_at: "2026-04-23"
---

# Excerpt: async rollout — the throughput evidence for §4 and §6

**Source library:** `wiki/raw-data/llm-training/papers/async-rollout.md`
**Artifacts:** Hu 2024 (OpenRLHF, arXiv:2405.11143), HybridFlow / verl (arXiv:2409.19256), IMPALA ancestry (Espeholt 2018).

---

## Why this source defines ch-58 §4 performance envelope + §6 graduation criteria

Ch-58 §4 is the only chapter section that makes *quantitative* claims ("1.6–2.0× throughput"). Every number here is attested by this source.

## The core insight behind §4

Source:

> Synchronous RL training leaves the GPU rollout engine idle during optimizer steps (and vice versa). Decoupling rollout and training into two asynchronous processes connected by a queue recovers most of that idle time — at the cost of a bounded off-policyness of ≤1 generation-cycle, which importance-sampling correction handles cleanly.

Ch-58 §4 opens with "Rollout idle during optimizer step; optimizer idle during rollout. For 7B on 8×H100, rollout dominates ≥70% wall time." That 70% figure is the gap async recovers.

## The attested throughput numbers (§4 quantitative claim)

Source:

> OpenRLHF paper Figure 5: sync vs async throughput — 1.9× at 7B, 1.6× at 70B.

> Practical deployments report 1.6×–2× throughput improvements over sync on 8×H100 setups for ~7B models.

Ch-58 §4 "OpenRLHF async. ... Hu 2024 Figure 5: 1.9× throughput at 7B, 1.6× at 70B." is this citation verbatim. No independent benchmark is invented; §8 "not a benchmark chapter" disclaims further numbers.

## The staleness bound — §6 graduation criterion

Source:

> **Staleness bound:** `k = queue_depth + partial_rollout_depth`; typical k=1–2.

Ch-58 §4 "staleness bounded at `k = queue_depth + partial_rollout_depth`, typically 1–2" is this verbatim. It's the precondition for IS correction being sufficient — if k blows up past 2, TIS/iCEPO masks won't save the run.

## The three-level scheduling divergence — §4's key insight

Source on OpenRLHF:

> `rollout_queue`: `ray.util.queue.Queue`, capacity 1–2.
> `rollout_slots`: companion queue carrying `global_step` tokens (backpressure).
> `vllm_lock`: `ray` asyncio.Lock to serialize weight-broadcast vs generate.

Source on verl:

> `engine.generate(...)` is async-for-iterated per-request.
> `priority` parameter reorders in-flight requests.
> Pause state (≈ line 628) blocks new generates and waits for in-flight to drain before a weight broadcast.

This is the §4 "queue-level vs engine-level" scheduling claim. OpenRLHF schedules *batches* at the Ray queue level; verl schedules *requests* at the vLLM engine level. Ch-58 §5 Q4 routes ≥128-GPU runs to verl *because* request-level scheduling is the only thing that handles stragglers at that scale.

## The failure signature (a §7 negative-case reminder)

Source:

> **Failure signature:** `vllm_kl` divergence > 0.1 plus PPO clipfrac pegged at 1 indicates the async staleness + sampler-mismatch has exceeded what IS correction can handle.

Ch-58 does not elevate this to a body-text line (the §3 crib sheet alludes to it), but a learner running async PPO who sees these two signals simultaneously should recognize "k is too big; drop queue_depth" as the correct diagnosis.

## The "partial rollout" design choice

Source:

> Partial rollout (continue a response after weight update) is a small but growing line — enables "continuous batching RL" where rollouts are never fully discarded and the trainer never waits.

Ch-58 §4 "verl's 'continuous batching RL' pattern: newly-weighted requests jump ahead of stragglers; in-flight generations finish under old weights then resume" is this attestation. It's the mechanism behind matrix row 14's verl cell.

## IMPALA ancestry — why IS correction is mandatory, not optional

Source:

> The async + IS correction pattern is used by every 2025 production RL stack (verl, OpenRLHF, TRL's forthcoming async trainer, NeMo-Aligner).
> Direct descendant of IMPALA V-trace — the IS-weight clip is mathematically the same object.

Ch-58 §6 "Graduate from TRL to OpenRLHF when ... you hit the vLLM-IS-correction wall: PPO destabilizes within ~50 steps on long completions" is the failure condition. The V-trace lineage explains why truncated-IS is the correct mathematical fix — not a hack.

## What ch-58 inherits verbatim

- 1.6×–2.0× throughput figure for §4 (Hu 2024 Figure 5).
- `k = queue_depth + partial_rollout_depth ≤ 2` staleness formula for §4 and §6.
- OpenRLHF `rollout_queue` / `rollout_slots` / `vllm_lock` primitive names.
- verl `priority` + pause-state (≈ line 628) for matrix row 14 citation.
- The "queue-level vs engine-level" scheduling framing that distinguishes §5 Q3 (OpenRLHF) from §5 Q4 (verl).

## Connections

- **[[verl-rollout]]** — the code-level view of verl's async implementation.
- **[[openrlhf-ppo]]** — the IS-correction modes async rollout relies on.
- **[[entropy-logging-patterns]]** — the `vllm_kl` diagnostic that flags async-exceeding-correction.
- **IMPALA (Espeholt 2018, arXiv:1802.01561)** — ancestral architecture; attested in the source's Connections block.
- **ch-55 verl** / **ch-56 OpenRLHF** — the framework chapters whose async internals this synthesis chapter references.
