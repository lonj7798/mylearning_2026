<!-- scope: verl's registered "vanilla" PPO clipped policy loss (asymmetric clip, dual-clip on negative advantages, rollout IS weights), loss aggregation modes, and where entropy and KL terms are added
     deps: [[ppo]]
     see-also: [[verl-grpo]], [[openrlhf-ppo]], [[trl-ppo]], [[dr-grpo]], [[john-schulman-kl-tricks]], [[entropy-logging-patterns]]
-->

# verl `verl/trainer/ppo/core_algos.py`: `compute_policy_loss_vanilla` (PPO clipped policy loss) and `agg_loss`
- **Core Insight:** verl's `vanilla` policy loss is the PPO clipped surrogate with separate lower and upper clip ranges (both default 0.2) plus a dual clip that caps the loss of each negative-advantage token at `c·|A|`, default c = 3.0 (core_algos.py L1336–1361; actor.yaml L36–42, L83).
- **Guideline:** When reproducing DAPO's Clip-Higher, set `clip_ratio_low=0.2` and `clip_ratio_high=0.28` (docs/algo/dapo.md L52–57), because DAPO reports entropy collapse in its initial naive PPO/GRPO runs, where ε = 0.2 is described as the common default, and a gain from 36 to 38 AIME24 avg@32 on Qwen2.5-32B when Clip-Higher is added (arXiv:2503.14476 §3.1, Table 1). Otherwise the default symmetric 0.2 is the value of the PPO paper's best clipped run (arXiv:1707.06347 Table 1).
- **Authors:** verl project; README L3: "initiated by ByteDance Seed team and maintained by the verl community"
- **Year:** 2026 (commit 753aed3, 2026-09-14; `core_algos.py` byte-identical at 61134f9, 2026-09-15)
- **URL:** https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/trainer/ppo/core_algos.py#L1285-L1376
- **Source type:** released config/code
- **Relevant topics:** PPO-clip, asymmetric clipping, Clip-Higher, dual-clip PPO, negative advantages, loss aggregation, KL loss vs KL reward, entropy bonus, training-inference mismatch

## Summary
Policy losses register by name with `@register_policy_loss` (L53–67); the actor looks one up with `get_policy_loss_fn(config.policy_loss.loss_mode)` (verl/workers/utils/losses.py L101–112), default `"vanilla"` (actor.yaml L61). Eleven other names are registered, including `gspo`, `sapo`, `clip_cov`, `kl_cov`, `cispo`, and `bypass_mode` (core_algos.py L1379–2414). The docstring states the function is adapted from TRL `ppo_trainer.py#L1122` (L1298–1299). `ppo_loss` in losses.py adds the entropy and KL terms after this function returns (L122–142). A deprecated wrapper with explicit arguments, `compute_policy_loss`, keeps the same algebra (L1209–1282).

## Key Contributions
- Separate `clip_ratio_low` and `clip_ratio_high`, each falling back to `clip_ratio` when unset (L1320–1322).
- Dual-clip PPO for negative advantages, citing arXiv:1912.09729 (L1323–1334, L1355–1361).
- Optional per-token multiplication by `rollout_is_weights` (L1364–1365).
- Five aggregation modes in `agg_loss` (L1140–1206); the docstring states that aggregation over the global batch makes the loss invariant to FSDP/Megatron parallelism (L1150).

## Key Figures/Tables to Study
- core_algos.py L1336–1375 (loss algebra and metrics); L1140–1206 (`agg_loss`); L2188–2245 (`kl_penalty`).
- verl/workers/utils/losses.py L57–144 (`ppo_loss`: policy, entropy, KL composition).

## Technical Details
Code (core_algos.py L1336–1361, verbatim except the None fallbacks at L1343–1346, which are omitted):
```python
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```
Formula: `L_t = max(−A_t·r_t, −A_t·clip(r_t, 1−ε_low, 1+ε_high))` when `A_t ≥ 0`, and `L_t = min(max(−A_t·r_t, −A_t·clip(r_t, 1−ε_low, 1+ε_high)), −A_t·c)` when `A_t < 0`. `r_t = exp(clamp(log π_θ − log π_old, −20, 20))` is the token probability ratio (L1336–1339). `A_t` is the token advantage. `ε_low`, `ε_high` are `clip_ratio_low`, `clip_ratio_high`. `c` is `clip_ratio_c`, asserted > 1.0 (L1331–1334). `∂L_t/∂log π_θ = −A_t·r_t` on the unclipped branch and 0 on a clipped branch.

Behavior on negative-advantage tokens (derived from the formula; A = −1, ε_low = ε_high = 0.2, c = 3):
1. r = 0.5: `max(0.5, 0.8) = 0.8`; the lower clip is active, so the gradient is 0. A token whose probability is already below `(1−ε_low)·π_old` gets no further push-down from its own loss term in this update.
2. r = 1.0: loss 1.0; gradient magnitude `|A|·r` = 1.0.
3. r = 2.0: `max(2.0, 1.2) = 2.0`, below the cap 3.0; gradient magnitude 2.0. The upper clip does not bound the loss when A < 0.
4. r = 5.0: `max(5.0, 1.2) = 5.0`, capped to 3.0; gradient 0. Without the dual clip the loss would be 5.0 with gradient magnitude 5.0, growing linearly in r.
- Ye et al. give the reason: with off-policy data the ratio can be very large, and "when Â_t < 0, such a large ratio r_t(θ) will introduce a big and unbounded variance" (arXiv:1912.09729 Algorithm Design, Eq. 5); they set ε = 0.2 and c = 3 (Experiments, System Setup). verl's default c = 3.0 equals that value.
- DAPO keeps ε_low at 0.2 "because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space" (§3.1); no separate ε_low ablation is given.
- Softmax derivative (standard; not stated in verl): `∂ log p_y/∂z_j = 1[j=y] − p_j`. A step that lowers `log p_y` raises each other logit in proportion to `p_j`, so the removed mass goes mostly to tokens that are already likely.

Metrics (L1371–1375): `actor/pg_clipfrac` is the masked fraction of tokens where the clipped term is selected (L1353). `actor/pg_clipfrac_lower` is the fraction of response tokens that have A < 0 and a clipped loss above `c·|A|` (L1357–1359). `actor/ppo_kl` is the mean of `log π_old − log π_θ` (L1340), a k1 estimate that is logged and not added to the loss.

Rollout correction: `rollout_is_weights` exist only when `algorithm.rollout_correction` is set, `rollout_log_probs` are in the batch, and bypass mode is off (ray_trainer.py L1648–1658). The YAML default is `rollout_is: null` (config/algorithm/rollout_correction.yaml); the options are `"token"` or `"sequence"` weights (config/algorithm.py L85–89). The config names rollout-vs-training precision mismatch (e.g. vLLM BF16 vs FSDP FP32) and stale rollout checkpoints as the sources of off-policyness (L70–72).

Aggregation (`agg_loss`): `token-mean` divides the masked sum by the global valid-token count (L1170–1175); `token-sum` (L1176–1180); `seq-mean-token-sum` and `seq-mean-token-sum-norm`, the latter divided by `loss_scale_factor`, default the response-mask width (L1181–1193); `seq-mean-token-mean` (L1194–1202).

Entropy and KL (losses.py): `policy_loss -= entropy_coeff * entropy_loss` when entropy is computed (L122–129); `policy_loss += kl_loss * kl_loss_coef` when `use_kl_loss` (L131–142). `low_var_kl`/`k3` is `exp(ref − logp) − (ref − logp) − 1`, input clamped to ±20 and output to ±10 (core_algos.py L2239–2245); a `+` suffix uses the k2 gradient by a straight-through estimator (L2201–2213). The alternative in-reward KL, `algorithm.use_kl_in_reward`, subtracts `β·kld` from token scores with a KL controller (ray_trainer.py L78–117).

## Recipe ledger
verl defaults at commit 753aed3, the DAPO docs value, and the paper values the defaults trace to. Framework defaults are not values from a model run.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| verl default | n/a | RL | `clip_ratio` (ε) | 0.2 | actor/actor.yaml L36 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.2 | actor.yaml L39, L42 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `clip_ratio_c` (dual-clip c) | 3.0 (asserted > 1.0) | actor.yaml L83; core_algos.py L1323–1334 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | log-ratio clamp before `exp` | [−20, 20] | core_algos.py L1338 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `loss_agg_mode`; `loss_scale_factor` | `token-mean`; null | actor.yaml L86, L90 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `entropy_coeff`; `ppo_epochs` | 0; 1 | actor.yaml L93, L119 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `use_kl_loss`; `kl_loss_coef`; `kl_loss_type` | false; 0.001; `low_var_kl` | actor.yaml L103, L113, L116 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `use_kl_in_reward`; `kl_penalty` | False; `kl` | ppo_trainer.yaml L98, L101 | verified 2026-09-14 | no ablation reported |
| verl docs, DAPO recipe | n/a | RL | `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.28 | docs/algo/dapo.md L52–57 | verified 2026-09-14 | none in docs |
| DAPO, Qwen2.5-32B base | 32B | RL | ε_low; ε_high | 0.2; 0.28 | arXiv:2503.14476v2 §4.1 | verified 2026-09-14 | Table 1: adding Clip-Higher to the progressive stack, AIME24 avg@32 36 → 38; one run per row |
| Dual-clip PPO, Honor of Kings 1v1 agent | not an LLM | RL | ε; c | 0.2; 3 | arXiv:1912.09729v3 Experiments, System Setup | verified 2026-09-14 | no ablation of c reported |
| PPO, MuJoCo continuous control | not an LLM | RL | ε (symmetric) | 0.2 | arXiv:1707.06347v2 §6.1 Table 1 | verified 2026-09-14 | avg. normalized score 0.82 vs 0.76 (ε = 0.1) and 0.70 (ε = 0.3); 7 environments × 3 seeds |

## Findings relevant to generality and negative feedback
- verl's DAPO reproduction table reports AIME 2024 accuracy 50% for "DAPO w/o Dynamic Sampling" and 44% for "DAPO w/o Token-level Loss & Dynamic Sampling" on Qwen2.5-32B (docs/algo/dapo.md L37–38). The two runs also differ in hardware (H800 vs H20) and image (vLLM 0.8.3 vs 0.7.4), one run each, AIME only.
- Dr. GRPO lists verl's PPO loss among implementations with a response-length bias from length normalization (arXiv:2503.20783 §3.1, Listing 1, Table 2). verl's `seq-mean-token-sum-norm` mode divides sequence-summed losses by `loss_scale_factor`, which the docs recommend setting to a constant such as the maximum response length (L1189–1193; docs/algo/grpo.md L55–60).

## Connections
- [[ppo]] — Eq. (7) clipped surrogate with symmetric ε (arXiv:1707.06347 §3).
- [[verl-grpo]] — GRPO advantages that this loss consumes.
- [[openrlhf-ppo]] — OpenRLHF `PolicyLoss(nn.Module)` at 64c1cc4 has `clip_eps_low/high` default 0.2 and `dual_clip` default None, so dual clip is off unless set (openrlhf/models/loss.py L75–148); verl applies it on every call with c = 3.0.
- [[trl-ppo]] — source of the adapted code (L1298–1299); TRL removed `trl/experimental/ppo/ppo_trainer.py` in commit 700b845 (2026-09-04, PR #7020).
- [[dr-grpo]] — length bias and the constant-normalizer fix.
- [[john-schulman-kl-tricks]] — k1/k2/k3 estimators cited in `kl_penalty` (L2191).

## Verification
- Checked on 2026-09-14 against: https://github.com/verl-project/verl at 753aed3e1c286ba6825a74342b28669e72c083ea (`verl/trainer/ppo/{core_algos.py, ray_trainer.py}`, `verl/workers/utils/losses.py`, `verl/trainer/config/{actor/actor.yaml, ppo_trainer.yaml, algorithm.py, algorithm/rollout_correction.yaml}`, `docs/algo/{dapo,grpo}.md`); arXiv:1912.09729v3; arXiv:2503.14476v2; arXiv:1707.06347v2; arXiv:2503.20783v2; OpenRLHF 64c1cc4; TRL 700b845.
- Corrections to the previous card version:
  - "≈ lines 1080–1140; deprecated wrapper at 1029–1078" → L1285–1376 and L1209–1282 at 753aed3 (the April commit 1a3b7e2 had L1279–1369).
  - "Asymmetric clipping ... is the DAPO/Open-Reasoner-Zero recipe — clipping more aggressively on the upside" → DAPO's Clip-Higher raises ε_high (0.28 > 0.2), which clips less on the upside (§3.1, §4.1); the Open-Reasoner-Zero attribution is not in the source.
  - "Dual clip ... floors the loss" → it caps the per-token loss at `c·|A|` (equivalently floors the objective at `c·A`), which zeroes the gradient for r > c (L1355–1361).
  - "`rollout_is_weights` is a per-token IS correction" → token-level or sequence-level, off by default (algorithm.py L85–89; rollout_correction.yaml).
  - "No entropy bonus in the loss ... `entropy_loss` registry hook" → `ppo_loss` subtracts `entropy_coeff · entropy_loss` (losses.py L122–129; default coefficient 0, actor.yaml L93); no such registry hook exists.
  - "Loss aggregation: token-mean, seq-mean-token-sum (Dr.GRPO style), seq-mean-token-mean" → five modes; the docs map Dr. GRPO to `seq-mean-token-sum-norm` (grpo.md L59).
  - "verl/OpenRLHF subtract β·KL from rewards (token-level)" → KL is indeed outside this function, but verl supports both an actor KL loss (losses.py L131–142) and an in-reward KL (ray_trainer.py L78–117), and the GRPO docs recommend the loss form (grpo.md L43–45).
- Removed as unsupported by the source: "verl is the production RLHF framework powering ByteDance Seed and many open R1 reproductions"; "the closest implementation to Schulman 2017 PPO-clip in modern LLM-RL frameworks"; "vLLM bf16 ≠ actor fp32 — see the TIS/iCEPO patches" (replaced by the config text); "each materially changes gradients on long-tailed completion length distributions"; "TRL `PPOTrainer` ... uses symmetric `cliprange` only" (TRL deleted the trainer, then at `trl/experimental/ppo/ppo_trainer.py`, in commit 700b845; not re-checked at an older commit).
- Not reported by the source: any verl ablation of `clip_ratio_c`, `clip_ratio_low`, or `clip_ratio_high`; effect of dual clip on pass@k or held-out tasks.
