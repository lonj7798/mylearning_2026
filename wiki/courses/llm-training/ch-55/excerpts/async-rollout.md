---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/async-rollout.md
source_url: https://arxiv.org/abs/2409.19256
created_at: "2026-04-23"
updated_at: "2026-09-17"
---

# Excerpt: asynchronous rollout — the design behind verl's async server

**Canonical extract:** `wiki/raw-data/llm-training/papers/async-rollout.md`. This file was rewritten in the 2026-09 revision; the earlier version carried throughput ratios and staleness thresholds with no source.

---

## The problem

In a synchronous RL step the generation engine is idle while the optimizer runs, and the optimizer is idle while the longest rollout finishes. Decoupling generation from optimization recovers that time at the cost of training on samples drawn by a slightly older policy.

## What the decoupling requires

1. **A separate process for generation**, so the two can run at once. verl's placement modes make this explicit: HYBRID shares GPUs with weight sync, STANDALONE uses separate GPUs and is off-policy (`replica.py` L54–66).
2. **A safe weight-update point.** verl closes the submission gate, aborts in-flight requests, and clears caches before weights change (`vllm_async_server.py` L932–1003).
3. **A correction for off-policyness.** The policy that produced a token and the policy being updated are no longer identical, so the loss can be multiplied by an importance weight `exp(log π_train − log π_rollout)`, token-level or sequence-level (`core_algos.py` L1364–1365; `config/algorithm.py` L85–89). It is off by default (`rollout_is: null`).
4. **A staleness bound.** verl's fully-async recipe exposes `staleness_threshold` and `partial_rollout`, which interrupts generation before a parameter sync and resumes the interrupted samples afterwards (`fully_async_policy/README.md` L55–58, L222–226).

## Sources of the mismatch

verl's configuration comments name precision mismatch between engines, for example vLLM BF16 against FSDP FP32, and stale rollout checkpoints (`rollout_correction.yaml` L70–72). Precision is one cause: differing kernels and implementations produce a log-probability mismatch even at equal dtype, which is why the correction is written as a general importance weight.

## The one measured comparison in the artifact

**Result (single study):** 128 H20 GPUs, Qwen2.5-Math-7B, DAPO, 400 steps: 1d 16h 48m colocated against 17h 22m fully async at a 64:64 rollout:train split, a 2.35× ratio. AIME-2024 acc mean@1 last 0.2958 colocated against 0.3094 async; max 0.3573 against 0.3521. Colocated `gen` is 177.85 s of a 356.30 s step (`fully_async_policy/README.md` L333–369). Staleness was ablated at 0, 0.1, 0.3, 0.5 (L380–396).

## Corrections to the previous excerpt version

1. "Expect 1.6–2.0× throughput vs sync"; "async continuous-batching sits at 5–8× HFRollout"; "OpenRLHF Figure 5: 1.9× at 7B, 1.6× at 70B" — removed; not supported by the artifacts read here. The one verified ratio is 2.35× in the configuration above.
2. "Staleness bound k = queue_depth + partial_rollout_depth; typical k = 1–2" — removed; verl exposes `staleness_threshold` as a fraction, ablated at 0/0.1/0.3/0.5.
3. "Pause state (≈ line 628) blocks new generates and waits for in-flight to drain" → in-flight requests are aborted, not drained (L932–1003).
4. "Sync SPMD had no per-request priority and no pause gate, so the async rewrite (PR #4411) was the only way forward" — removed as reconstruction; the artifact records only that the SPMD mode was retired in PR #4411 and that `mode: async` is the default.
5. "`vllm_kl` divergence > 0.1 plus clipfrac pegged at 1" as a failure threshold — removed; no source gives a threshold.

## Connections

- [[verl-rollout]] — the code this design describes.
- [[verl-ppo-loss]] — `rollout_is_weights` in the policy loss.
- [[entropy-logging-patterns]] — the drift metrics each framework exposes.
