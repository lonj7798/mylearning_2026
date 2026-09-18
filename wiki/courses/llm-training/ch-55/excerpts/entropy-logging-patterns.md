---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
updated_at: "2026-09-17"
---

# Excerpt: entropy and KL logging — verl's row

**Canonical extract:** `wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md` (verified 2026-09-14; verl read at 1a3b7e2 with current-main status noted). This file was rewritten in the 2026-09 revision to match that card.

---

## What verl logs

| Metric | Definition | Locus |
|---|---|---|
| `actor/ppo_kl` | masked mean of `old_log_prob − log_prob` (k1, old vs current), log-ratio clamped to [−20, 20] | core_algos.py L1329–1333, L1366 |
| `actor/entropy` | categorical entropy from logits, `logsumexp(z) − Σ softmax(z)·z`, aggregated with `loss_agg_mode` | ray_trainer.py L1170, L1433–1446; torch_functional.py L224–238 |
| `actor/entropy_loss` | present when entropy is requested; `policy_loss -= entropy_coeff * entropy_loss` | losses.py L123–130 |
| `actor/pg_clipfrac` | fraction of tokens on a clipped branch | core_algos.py L1353 |
| `actor/pg_clipfrac_lower` | fraction of tokens with A < 0 whose loss exceeded `c·|A|` | core_algos.py L1357–1359 |
| `actor/reward_kl_penalty`, `…_coeff` | in-reward KL, when `algorithm.use_kl_in_reward` | ray_trainer.py L76–115, L1488–1492 |
| `kl_loss`, `kl_coef` | in-loss KL, when `actor.use_kl_loss` | losses.py L132–143 |
| `rollout_corr/kl`, `rollout_corr/k3_kl` | rollout vs old log-prob, when rollout correction is configured | ray_trainer.py L1500–1509 |

## Estimators

With `δ = log π_θ − log π_b` for a sampled token: k1 = δ; k2 = δ²/2; k3 = exp(−δ) − 1 + δ; verl also offers `abs = |δ|`. Input clamped to ±20, k3 output to ±10 (core_algos.py L2177–2183, L2239–2245). verl's own comment states that k1 and k3 have the expected value of the KL but not its expected gradient, that k2 gives the right gradient, and that a `+` suffix uses k2 in the backward pass (L2145–2151, L2201–2213).

Worked values: δ = +0.1 gives k1 = 0.1 and k3 = 0.00484; δ = −0.1 gives k1 = −0.1 and k3 = 0.00517. A k1 mean near zero can therefore hide two-sided drift; k3 is non-negative per token.

## Cross-framework placement (from the card's table)

| Concern | verl | OpenRLHF | TRL PPO | TRL GRPO |
|---|---|---|---|---|
| KL to reference, placement | reward or loss; both off by default | reward (default) or loss | reward only | loss only, if β ≠ 0 (default 0.0) |
| Estimators offered | k1, abs, k2, k3, `+` variants | k1, k2, k3 | k1, k3 | k3 |
| Entropy in loss by default | no (`entropy_coeff: 0`) | no (`entropy_coef` None) | no | no |

## Adaptive controller

`β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon)` (core_algos.py L153–174). verl's default controller is `type: fixed` with `kl_coef: 0.001`, `horizon: 10000`, `target_kl: 0.1` (ppo_trainer.yaml).

## Corrections to the previous excerpt version

1. "'Entropy' almost never means the true categorical entropy" → every logged entropy in these three frameworks is categorical entropy from logits except TRL PPO's `objective/entropy`.
2. "K1 (biased, noisy)" / "K3 … modern default" → the estimator properties are as quoted from the code comments above; verl's KL to the reference is off by default in both placements.
3. "verl `workers/actor/*`" → that directory no longer exists after the engine migration (commit 044bbba, 2026-04-20); the metrics are in `ray_trainer.py` and `losses.py`.
4. "iCEPO" → the OpenRLHF IS-correction type is `icepop`; `vllm_kl` exists only inside the IS-correction branch.
5. Removed as unsupported: "entropy collapse is the single most-common RL-for-LLM failure mode"; the "entropy falls ≥30% in <100 steps" threshold; "PPO will destabilize unless IS correction is enabled".

## Connections

- [[john-schulman-kl-tricks]] — the k1/k2/k3 derivations all three code bases cite.
- [[entropy-mechanism-llm-rl]] — the entropy-collapse study behind `clip_cov` and `kl_cov`.
- [[verl-ppo-loss]] — where these metrics are produced.
