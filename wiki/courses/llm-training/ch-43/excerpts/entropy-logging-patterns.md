---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md
source_url: "verl@1a3b7e2; OpenRLHF@64c1cc4; TRL@a08e713 (April 2026 commits)"
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the card verified on 2026-09-14)"
---

# Excerpt: entropy and KL as implemented in verl, OpenRLHF and TRL

## Estimator forms as implemented
For a sampled token with `δ = log π(a|s) − log π_b(a|s)` (π = current policy, π_b = the comparison policy,
either the reference model or the old/rollout policy): `k1 = δ`; `k2 = δ²/2`; `k3 = exp(−δ) − 1 + δ`; verl
also offers `abs = |δ|`. verl clamps `−δ` to [−20, 20] and `k3` to [−10, 10]; OpenRLHF clamps every estimator
output to [−10, 10]; TRL applies no clamp.

## Code comments on gradients (verl `core_algos.py` L2145-2151)
verl states that k1 and k3 have the expected value of the KL but not its expected gradient, that k2 gives the
right gradient, and that a `+` suffix (for example `k3+`) uses k2 in the backward pass. OpenRLHF's CLI prints
a recommendation of k2 or k3 for KL as a loss and k1 for KL in the reward (`cli/train_ppo_ray.py` L675-680).

## Defaults at the pinned commits
| Setting | verl | OpenRLHF | TRL PPO (experimental) | TRL GRPO |
|---|---|---|---|---|
| KL to reference | off (`use_kl_in_reward: False`, `use_kl_loss: false`) | in the reward, `init_coef` 0.01 | in the reward, `kl_coef` 0.05 | in the loss, β = 0.0 (reference model not loaded) |
| Estimator | `kl_penalty: kl` (k1) in reward; `kl_loss_type: low_var_kl` (k3) in loss; coefficients 0.001 | `estimator` k1 | `kl_estimator` "k1" | k3 |
| Entropy in loss | `entropy_coeff: 0`, `calculate_entropy: false` | `entropy_coef` None (entropy not computed) | none | none |
| Entropy logged | `actor/entropy` (categorical) | `entropy_loss` only when `entropy_coef` is set | `policy/entropy_avg` (categorical) and `objective/entropy` (sum of −log p of sampled tokens) | `entropy` (categorical) |
| Entropy-based loss control | `clip_cov`, `kl_cov` (adapted from the PRIME-RL Entropy-Mechanism-of-RL repository) | none in the checked files | none | `top_entropy_quantile` (default 1.0, no masking) |
| Old-vs-current KL diagnostic | `actor/ppo_kl` (k1, clamped log-ratio) | `ppo_kl` | `policy/approxkl_avg` (k2) | `clip_ratio/*` |

- Adaptive controller (verl `core_algos.py` L153-174):
  `β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon)`. Measured KL above the target raises `β`.
  verl's default controller is `fixed` with `kl_coef: 0.001`, `horizon: 10000`, `target_kl: 0.1`. OpenRLHF
  uses an adaptive controller only when `algo.kl.target` is set; the code carries a comment saying the
  adaptive controller is incompatible at that call site.
- TRL GRPO's `top_entropy_quantile` help text cites "Beyond the 80/20 Rule" and states "The paper recommends a
  value of `0.2`" ([[high-entropy-minority-tokens]]).

## Diagnostics split by advantage sign
verl logs `actor/pg_clipfrac_lower`, which counts negative-advantage tokens where the dual-clip bound applies;
TRL GRPO logs `clip_ratio/low_mean` (ratio below `1 − ε` with a negative advantage) separately from
`clip_ratio/high_mean`; OpenRLHF applies its `dual_clip` bound only to negative advantages but logs one
combined `ppo_clip_ratio`.

## Removed in the 2026-09 revision
"All three default to k3", the claim that entropy coefficients default to 1e-3 in verl presets, the collapse
signature "entropy falls ≥30% in <100 steps with `ppo_kl` > 0.1 and clipfrac → 1", and the attribution of
`top_entropy_quantile` to Cui et al. or DAPO. The code reports no measured effect of any of these settings.
