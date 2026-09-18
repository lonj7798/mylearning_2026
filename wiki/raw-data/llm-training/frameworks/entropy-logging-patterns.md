<!-- scope: how verl, OpenRLHF, and TRL compute, apply, and log entropy and KL during RL training, read from code at pinned April 2026 commits
     deps: [[entropy-mechanism-llm-rl]], [[kl-control-rlhf]]
     see-also: [[john-schulman-kl-tricks]], [[verl-ppo-loss]], [[openrlhf-ppo]], [[trl-ppo]], [[trl-grpo]], [[openrlhf-entropy-debugging]]
-->

# Entropy and KL logging in verl, OpenRLHF, and TRL (code at pinned commits, April 2026)
- **Core Insight:** At the pinned commits, none of the three frameworks adds an entropy term to the loss by default (verl `entropy_coeff: 0`, OpenRLHF `entropy_coef` None, TRL PPO and GRPO have no entropy term), and every logged entropy is categorical entropy from logits except TRL PPO's `objective/entropy`, which sums the negative log-probabilities of the sampled tokens.
- **Guideline:** When comparing KL curves across these frameworks, first check the estimator, the two policies compared (reference vs. old or rollout policy), and the aggregation, because the defaults differ: verl has KL to the reference off by default, OpenRLHF and TRL PPO put k1 into the reward (coefficients 0.01 and 0.05), and TRL GRPO puts k3 into the loss with default β = 0.0.
- **Authors:** verl project (github.com/verl-project); OpenRLHF project (github.com/OpenRLHF); Hugging Face TRL maintainers (github.com/huggingface)
- **Year:** 2026 (verl commit 1a3b7e2, 2026-04-21; OpenRLHF commit 64c1cc4, 2026-04-19; TRL commit a08e713, 2026-04-21)
- **URL:** https://github.com/verl-project/verl/tree/1a3b7e2ab71fb44a542011c2927809a7ca9da3d7 ; https://github.com/OpenRLHF/OpenRLHF/tree/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da ; https://github.com/huggingface/trl/tree/a08e7139f933b770177fc2abc0b43118e26b260b
- **Source type:** released config/code (comparison of three repositories)
- **Relevant topics:** entropy logging, KL estimators k1/k2/k3, KL in reward vs. KL in loss, KL controllers, rollout–training mismatch metrics, entropy-based token masking

## Summary
This card compares code, not one publication. For each framework it records which entropy quantity is computed, whether entropy can enter the loss, which KL estimator is used, which two policies the KL compares, whether KL to the reference enters the reward or the loss, and the logged metric names. File paths below omit the top-level package directory (`verl/`, `openrlhf/`, `trl/`). Status at current main (checked 2026-09-14): TRL deleted `trl/experimental/ppo/` in commit 700b845 ("Remove PPOTrainer (#7020)", 2026-09-04); OpenRLHF at b117b2b computes `ppo_kl` from `raw_policy_log_ratio` and returns a fifth value `is_filter_ratio` (loss.py L185, L258–259); verl at 753aed3 keeps the same `kl_penalty` names (core_algos.py L2188–2239). Default values are in the Recipe ledger below.

## Key Contributions
- Separates the two KL quantities logged under similar names: old or rollout policy vs. current policy (`actor/ppo_kl`, `ppo_kl`, `policy/approxkl_avg`), and reference vs. current policy (`actor/reward_kl_penalty`, `kl_loss`, `kl`, `objective/kl`).
- Records the KL estimator formulas and clamps as implemented.
- Records where entropy is computed and which coefficient adds it to the loss.
- Records the rollout–training mismatch diagnostics each framework exposes.

## Key Figures/Tables to Study
- verl `kl_penalty`, `kl_penalty_forward` (trainer/ppo/core_algos.py L2126–2190) and `ppo_loss` (workers/utils/losses.py L58–145).
- OpenRLHF `PolicyLoss.forward` (models/loss.py L114–182) and `training_step` (trainer/ray/ppo_actor.py L251–346).
- TRL PPO metrics block (experimental/ppo/ppo_trainer.py L883–904) and GRPO `_compute_loss` (trainer/grpo_trainer.py L2418–2601).

## Technical Details
**Estimators.** For a sampled token a after prefix s, let δ = log π(a|s) − log π_b(a|s). Implemented forms: k1 = δ; k2 = δ²/2; k3 = exp(−δ) − 1 + δ; verl also has abs = |δ|. Here π is the current policy and π_b is the comparison policy (the reference model, or the old or rollout policy). verl clamps −δ to [−20, 20] and k3 to [−10, 10] (core_algos.py L2177–2183). OpenRLHF clamps every estimator output to [−10, 10] (models/utils.py L62–79). TRL applies no clamp (ppo_trainer.py L775–776; grpo_trainer.py L2495–2497).
Code comments on estimators: TRL calls k1 "straightforward, unbiased" and k3 "lower variance, unbiased, appears to be a strictly better estimator" (ppo_trainer.py L414–415). verl states that k1 and k3 have the expected value of KL but not its expected gradient, that k2 gives the right gradient, and that a `+` suffix (e.g. `k3+`) uses k2 in the backward pass (core_algos.py L2145–2151). OpenRLHF prints a recommendation of k2 or k3 for KL as a loss and k1 for KL in the reward (cli/train_ppo_ray.py L675–680).

**verl (1a3b7e2).**
- `actor/ppo_kl` = masked mean of `old_log_prob − log_prob` (k1, old vs. current), with the log-ratio clamped to [−20, 20] (core_algos.py L1329–1333, L1366).
- `actor/entropy`: categorical entropy from the old-log-prob recomputation, aggregated with `loss_agg_mode` (trainer/ppo/ray_trainer.py L1170, L1433–1446). `entropy_from_logits` = logsumexp(z) − Σ softmax(z)·z (utils/torch_functional.py L224–238).
- Entropy in loss: `policy_loss -= entropy_coeff * entropy_loss`, logged as `actor/entropy_loss`, when entropy is requested (losses.py L123–130); it is requested when `calculate_entropy` is true or `entropy_coeff != 0` (ray_trainer.py L1200–1202).
- KL to reference in the reward (`algorithm.use_kl_in_reward`): token rewards become `token_level_scores − β·kl_penalty(old_log_probs, ref_log_prob)`; logs `actor/reward_kl_penalty` and `actor/reward_kl_penalty_coeff` (ray_trainer.py L76–115, L1488–1492).
- KL to reference in the loss (`actor.use_kl_loss`, config comment "True for GRPO"): `policy_loss += kl_loss * kl_loss_coef`, logs `kl_loss` and `kl_coef` (losses.py L132–143; trainer/config/actor/actor.yaml L94–95).
- Adaptive controller: β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon) (core_algos.py L153–174).
- Entropy-related loss modes `clip_cov` and `kl_cov`, "Adapted from" the PRIME-RL Entropy-Mechanism-of-RL repository (core_algos.py L1735–1750, L1840–1854).
- Rollout–training mismatch: `rollout_is_weights` multiply the per-token policy loss (core_algos.py L1357–1358); `rollout_corr/kl` (mean of rollout − old log-prob) and `rollout_corr/k3_kl` are logged when rollout correction is configured and rollout log-probs are present (trainer/ppo/ray_trainer.py L1500–1509; trainer/ppo/rollout_corr_helper.py L874–887, L946–959).

**OpenRLHF (64c1cc4).**
- `ppo_kl` = masked mean of `−log_ratio` (models/loss.py L181). `log_ratio` is `log_probs − old_log_probs` (L123), but with vLLM IS correction enabled and `policy_loss_type="ppo"` it is reassigned to `old_log_probs − rollout_log_probs` first (L152–154), so `ppo_kl` then compares rollout and old policies.
- `vllm_kl` = masked mean of `rollout_log_probs − old_log_probs`, computed only inside the IS-correction branch and `None` otherwise (L151–173). IS types are `tis`, `icepop`, and `seq-mask-tis`; the comment cites "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (L150).
- KL to reference in the reward (default): `compute_approx_kl` runs when a reference model exists and `algo.kl.use_loss` is false (trainer/ppo_utils/experience_maker.py L205–213); `compute_reward` adds `−kl_coef·kl` per token and places the scalar reward at the last response token (models/utils.py L82–110; experience_maker.py L276–282).
- KL to reference in the loss: with `--algo.kl.use_loss`, `loss = actor_loss + kl_loss * kl_ctl` (trainer/ray/ppo_actor.py L296–314).
- Controller: `AdaptiveKLController` if `algo.kl.target` is set, otherwise `FixedKLController` (trainer/ppo_trainer.py L171–176). The update call sits under the comment "TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here." (L250–253).
- Entropy: computed only when `actor.entropy_coef` is not None, as logsumexp(z) − Σ softmax(z)·z on float32 logits (models/actor.py L227–234; models/utils.py L164–168); subtracted from the loss when the coefficient is non-zero and logged as `entropy_loss` (ppo_actor.py L318–322, L344–346).

**TRL experimental PPO (a08e713).**
- KL to reference in the reward: `non_score_reward = -args.kl_coef * kl`, with k1 or k3 (experimental/ppo/ppo_trainer.py L775–777); a fixed coefficient and no controller.
- `objective/kl` = per-sequence sum of that KL, averaged over sequences (L884, L891).
- `objective/entropy` = `(-logprobs).sum(1).mean()` over sampled response tokens (L885, L892); padding positions hold `INVALID_LOGPROB = 1.0` (L80, L767) and enter the sum.
- `policy/entropy_avg` = mean per-token categorical entropy of temperature-scaled training logits, under `torch.no_grad()` (L824, L857–869, L903).
- `policy/approxkl_avg` = mean of 0.5·(new − old log-prob)², where old log-probs come from the generation logits of the same policy (k2, old vs. current) (L692–705, L843, L859, L898). The loss is `pg_loss + args.vf_coef * vf_loss` (L849).

**TRL GRPO (a08e713).**
- KL to reference in the loss: `per_token_loss + self.beta * per_token_kl` with k3, only when β ≠ 0 (trainer/grpo_trainer.py L2492–2500, L2544–2545); logged as `kl` (L2580–2582). With β = 0.0 the reference model is not loaded (grpo_config.py L589–596).
- `entropy`: per-token categorical entropy of temperature-scaled logits under `torch.no_grad()`, logged every step (grpo_trainer.py L1094, L1100–1101, L2584–2585; trainer/utils.py L484–514).
- `top_entropy_quantile` < 1.0 keeps only tokens at or above that entropy quantile in the policy loss; the help text cites "Beyond the 80/20 Rule" and states "The paper recommends a value of `0.2`" (grpo_config.py L774–783; grpo_trainer.py L988–1024, L2444–2447).
- With vLLM and `vllm_importance_sampling_correction`, the per-token loss is multiplied by `importance_sampling_ratio` = exp(old − vLLM sampling log-prob), summed per sequence in the default `sequence_mask` mode (L2044–2055, L2541–2542); min, mean, and max are logged as `sampling/importance_sampling_ratio/*` (L2241–2249).

| Concern | verl | OpenRLHF | TRL PPO | TRL GRPO |
|---|---|---|---|---|
| Entropy logged | `actor/entropy`; `actor/entropy_loss` if requested | `entropy_loss` if `entropy_coef` set | `objective/entropy` (sampled-token sum); `policy/entropy_avg` | `entropy` |
| KL to reference, placement | reward or loss; both off by default | reward (default) or loss | reward only | loss only, if β ≠ 0 |
| Estimators offered | k1, abs, k2, k3, `+` variants | k1, k2, k3 | k1, k3 | k3 |
| Old/rollout vs. current | `actor/ppo_kl`, `rollout_corr/kl` | `ppo_kl`, `vllm_kl` | `policy/approxkl_avg` | `clip_ratio/*`; `sampling/importance_sampling_ratio/*` (old vs. vLLM) |
| Entropy-based loss control | `clip_cov`, `kl_cov` | none in checked files | none | `top_entropy_quantile` |

## Recipe ledger
Framework defaults at the pinned commits. Size is "any" and Stage is RL for every row; no repository reports an ablation for these values.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| verl default config | any | RL | KL to reference in reward; estimator; controller; coefficient | `use_kl_in_reward: False`; `kl_penalty: kl` (k1); `type: fixed`; `kl_coef: 0.001` (`horizon: 10000`, `target_kl: 0.1`) | verl@1a3b7e2 trainer/config/ppo_trainer.yaml L77–98 | verified 2026-09-14 | no ablation reported |
| verl default config | any | RL | KL to reference in loss; coefficient; estimator | `use_kl_loss: false`; `kl_loss_coef: 0.001`; `kl_loss_type: low_var_kl` (k3) | verl@1a3b7e2 trainer/config/actor/actor.yaml L95, L105, L108 | verified 2026-09-14 | no ablation reported |
| verl default config | any | RL | entropy coefficient; entropy computation; PPO clip ratio | `entropy_coeff: 0`; `calculate_entropy: false`; `clip_ratio: 0.2` | verl@1a3b7e2 actor.yaml L89, L92, L36 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_ppo_ray.py` CLI default | any | RL | KL coefficient; target (adaptive if set); horizon | `algo.kl.init_coef` 0.01; `algo.kl.target` None (fixed controller); `algo.kl.horizon` 10000 | OpenRLHF@64c1cc4 cli/train_ppo_ray.py L417–419 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_ppo_ray.py` CLI default | any | RL | KL estimator; KL as loss | `algo.kl.estimator` k1; `algo.kl.use_loss` False (KL in reward) | OpenRLHF@64c1cc4 cli/train_ppo_ray.py L421–429, L485 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_ppo_ray.py` CLI default | any | RL | entropy coefficient | `actor.entropy_coef` None (entropy not computed; 0 logs entropy only) | OpenRLHF@64c1cc4 cli/train_ppo_ray.py L431–436 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_ppo_ray.py` CLI default | any | RL | vLLM IS correction; thresholds; type | off; [0.5, 5.0]; `tis` | OpenRLHF@64c1cc4 cli/train_ppo_ray.py L257–270 | verified 2026-09-14 | no ablation reported |
| TRL `PPOConfig` (experimental) default | any | RL | KL coefficient; estimator | `kl_coef` 0.05; `kl_estimator` "k1" | TRL@a08e713 experimental/ppo/ppo_config.py L240–252 | verified 2026-09-14 | no ablation reported |
| TRL `GRPOConfig` default | any | RL | KL coefficient β; KL bias correction | 0.0 (reference model not loaded); `use_bias_correction_kl` False | TRL@a08e713 trainer/grpo_config.py L589–596, L830–836 | verified 2026-09-14 | no ablation reported |
| TRL `GRPOConfig` default | any | RL | entropy masking; vLLM IS correction; IS mode | `top_entropy_quantile` 1.0 (no masking); `vllm_importance_sampling_correction` True; `sequence_mask` | TRL@a08e713 trainer/grpo_config.py L774–783, L792–803 | verified 2026-09-14 | help text: "The paper recommends a value of `0.2`" for entropy masking (L779–780); no ablation in the repository |

## Findings relevant to negative feedback
The code reports no measurements. It exposes diagnostics split by advantage sign: verl `actor/pg_clipfrac_lower` counts negative-advantage tokens where the dual-clip bound applies (core_algos.py L1350–1352, L1367), and TRL GRPO logs `clip_ratio/low_mean` (ratio < 1 − ε with negative advantage) separately from `clip_ratio/high_mean` (ratio > 1 + ε with positive advantage) (grpo_trainer.py L2589–2601). OpenRLHF applies its `dual_clip` bound only to negative advantages (loss.py L145–147) but logs one combined `ppo_clip_ratio` (ppo_actor.py L291).

## Connections
- [[john-schulman-kl-tricks]] — the blog that all three code bases cite for k1/k2/k3.
- [[entropy-mechanism-llm-rl]] — entropy-collapse paper; verl's `clip_cov` and `kl_cov` code is adapted from the `PRIME-RL/Entropy-Mechanism-of-RL` repository.
- [[kl-control-rlhf]] — KL regularization in RLHF; the verl and OpenRLHF controller docstrings cite arXiv:1909.08593.
- [[verl-ppo-loss]], [[openrlhf-ppo]], [[trl-ppo]], [[trl-grpo]] — per-framework loss cards.
- [[openrlhf-entropy-debugging]] — practitioner notes on entropy debugging across the three stacks.

## Verification
- Checked on 2026-09-14 against the three commits in the URL field (verl and TRL: last commit on 2026-04-21; OpenRLHF: last commit on or before 2026-04-21), plus current main where stated in the Summary.
- Corrections: (1) "'Entropy' ... almost never means the true categorical entropy" → all logged entropies except TRL PPO `objective/entropy` are categorical entropy from logits. (2) "K1 (biased, noisy)"; "K2 (Schulman low-variance)"; "K3 ... modern default" → estimator properties are as quoted from code comments above; the defaults are listed in the Recipe ledger. (3) verl entropy "see `verl/workers/actor/*`" → that directory does not exist at 1a3b7e2 (the last commit touching `dp_actor.py` is 044bbba, "deprecate workers, migrate to engines", 2026-04-20); the metrics are in ray_trainer.py L1445 and losses.py L130. (4) verl KL "reward shaping" only → reward or loss placement, both off by default; the k3 excerpt omitted the clamps. (5) OpenRLHF entropy "per-step mean −logp" → categorical entropy, only when `entropy_coef` is set. (6) OpenRLHF estimator "K1" → k1/k2/k3, default k1; KL can be a loss term. (7) OpenRLHF controller condition "if adaptive" → `if self.args.algo.kl.target`. (8) TRL PPO "uses an AdaptiveKLController" → fixed `kl_coef`. (9) TRL `objective/kl` "K1" → k1 or k3 per `kl_estimator`. (10) TRL PPO entropies "disagree when the policy is high-variance" → they differ by definition and aggregation. (11) TRL GRPO KL "added to the loss" → only when β ≠ 0; default 0.0. (12) `top_entropy_quantile` attributed to Cui 2025 and DAPO → help text cites "Beyond the 80/20 Rule". (13) "iCEPO" → `icepop`; `vllm_kl` exists only with IS correction on. (14) TRL `_compute_loss` "≈ 2578–2620" → starts at L2418; metrics at L2571–2601.
- Removed as unsupported by the source: "Entropy collapse ... is the single most-common RL-for-LLM failure mode"; "entropy falls ≥30% in <100 steps, `ppo_kl` spikes ≥0.1, `clipfrac` pegs to 1"; "PPO will destabilize unless IS correction ... is enabled"; "Cui 2025 ... and DAPO argue for token-level entropy masking or entropy annealing".
- Not reported by the sources: any measured effect of entropy coefficients, KL estimators, or KL placement.
