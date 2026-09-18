<!-- scope: GRPO-family advantage estimators in verl (outcome GRPO, vectorized GRPO, Pass@k GRPO), the trainer/config wiring that selects them, and what they do with all-equal and failed groups
     deps: [[grpo]], [[verl-ppo-loss]]
     see-also: [[dr-grpo]], [[trl-grpo]], [[openrlhf-ppo]], [[rloo-vs-grpo]], [[verl-rollout]], [[entropy-logging-patterns]]
-->

# verl `verl/trainer/ppo/core_algos.py`: `compute_grpo_outcome_advantage` and related GRPO advantage estimators
- **Core Insight:** verl's GRPO estimator sums each response's token rewards into one score, z-scores it within its prompt group as `(r − mean) / (std + 1e-6)`, and copies that scalar to every response token (core_algos.py L304–329); a group whose rewards are all equal gets advantage 0, and a group of size 1 gets its raw reward as the advantage (L315–317).
- **Guideline:** When `algorithm.adv_estimator=grpo`, set `actor_rollout_ref.rollout.n` above 1, because the default is 1 (rollout.yaml L127) and a size-1 group has no baseline (core_algos.py L315–317). When the Dr. GRPO objective is wanted, use the documented combination `norm_adv_by_std_in_grpo=False`, `loss_agg_mode=seq-mean-token-sum-norm`, `use_kl_loss=False` (docs/algo/grpo.md L57–62).
- **Authors:** verl project; README L3: "initiated by ByteDance Seed team and maintained by the verl community"
- **Year:** 2026 (commit 753aed3, 2026-09-14; `core_algos.py` byte-identical at 61134f9, 2026-09-15)
- **URL:** https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/trainer/ppo/core_algos.py#L266-L530
- **Source type:** released config/code
- **Relevant topics:** GRPO, group-relative advantage, critic-free RL, Dr. GRPO, Pass@k training, zero-advantage groups, dynamic sampling

## Summary
verl keeps advantage estimation separate from the policy loss. Estimators register under a name with `@register_adv_est` (L116–134) and are listed in the `AdvantageEstimator` enum (14 names, L88–110). The trainer's `compute_advantage` calls `compute_grpo_outcome_advantage` with the prompt `uid` as group index (ray_trainer.py L235–247). The critic is disabled when `adv_estimator != gae` unless `critic.enable` is set (ppo/utils.py L96–107). The policy update then uses the same registered policy loss as PPO, `vanilla` by default ([[verl-ppo-loss]]).

## Key Contributions
- Outcome GRPO with an std on/off switch; the docstring names the "on" case the original GRPO and the "off" case Dr. GRPO, arXiv:2503.20783 (L295–296).
- A vectorized GRPO (`grpo_vectorized`, L334–358) using `group_mean_std` (verl/utils/groupwise.py L164–222).
- A Pass@k estimator (`grpo_passk`, L471–530) implementing arXiv:2503.19595 (L485).
- GDPO (L361–468): per-reward-dimension GRPO normalization, weighted sum, then batch whitening (L466; arXiv:2601.05242, L403).

## Key Figures/Tables to Study
- core_algos.py L304–331 (outcome GRPO body) and L500–530 (Pass@k body).
- docs/algo/grpo.md L29–62 (GRPO and Dr. GRPO configuration); docs/algo/dapo.md L34–38 and L69–86 (DAPO reproduction runs, group filtering).

## Technical Details
Code (core_algos.py L311–329, verbatim; L304 is `scores = token_level_rewards.sum(dim=-1)`):
```python
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
```
Formula: `A_i = (r_i − μ_g) / (σ_g + ε)`, applied to every token of response i. `r_i` is the summed token reward of response i (L304). `μ_g` and `σ_g` are the mean and `torch.std` (sample standard deviation, divisor G−1) over the G responses sharing a `uid` (L319–321). `ε = 1e-6` (L272). With `norm_adv_by_std_in_grpo=False`, `A_i = r_i − μ_g` (L327–328). The function returns the same tensor as advantages and returns (L331).

Worked example (derived from L319–326; binary reward, G = 4):
1. One correct of four: μ = 0.25, σ = sqrt((0.75² + 3·0.25²)/3) = 0.5. The correct response gets +1.5; each wrong response gets −0.5.
2. Three correct of four: μ = 0.75, σ = 0.5. Each correct response gets +0.5; the wrong response gets −1.5.
3. Zero or four correct: σ = 0, so every advantage is 0/(0 + 1e-6) = 0.
4. Without std scaling, case 1 gives +0.75 / −0.25 and case 2 gives +0.25 / −0.75.

Negative signals (code-derived; verl reports no measurement of these effects):
- Wrong responses in a mixed group receive a negative advantage, which the policy loss applies as a decrease in their token log-probabilities (negative as gradient).
- All-wrong and all-correct groups receive advantage 0 and contribute no policy-gradient term. Their tokens stay in `response_mask` (L329), so under the default `token-mean` aggregation they still count in the token denominator (core_algos.py L1170–1175).
- With `rollout.n=1`, each group has one sample and the advantage equals the raw reward divided by 1 + 1e-6 (L315–317, L326). With a 0/1 reward, failures then get advantage 0 and no response gets a negative advantage.
- `algorithm.filter_groups` (default `enable: False`, ppo_trainer.yaml L80–92) is documented as removing groups whose metric values are all equal, then resampling until the prompt batch is full (docs/algo/dapo.md L84–86). The implementation is linked to the separate verl-recipe repository (dapo.md L7).

Pass@k estimator (L500–530): it raises an error for groups smaller than 2 (L516–519), gives only the highest-reward response the advantage `r_max − r_second_max` (L520–527), and divides by the group `torch.std` when `norm_adv_by_std_in_grpo` is true (L524–526). All advantages are ≥ 0, so no response receives a negative advantage. With a 0/1 reward the advantage is non-zero only when exactly one response is correct; for rewards [1, 0, 0, 0] the correct response gets 1/0.5 = 2.0 (derived). Tang et al. §3.1 state the same sparsity: the advantage is non-zero only for the best generation and zero when the best and second-best rewards are equal.

## Recipe ledger
Framework defaults and documented settings at commit 753aed3. These are not values from a model run unless the row says so.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| verl default | n/a | RL | `algorithm.adv_estimator` | `gae` (docs: set `grpo` for GRPO) | ppo_trainer.yaml L74; docs/algo/grpo.md L39 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | samples per prompt `rollout.n` | 1 (docs: set > 1 for GRPO) | rollout/rollout.yaml L127; grpo.md L29 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `norm_adv_by_std_in_grpo` | True | ppo_trainer.yaml L77; config/algorithm.py L658 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | std denominator ε | 1e-6 | core_algos.py L272 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `actor.use_kl_loss`; `kl_loss_coef`; `kl_loss_type` | false (docs: True for GRPO); 0.001; `low_var_kl` | actor/actor.yaml L103, L113, L116; grpo.md L45–47 | verified 2026-09-14 | no ablation reported |
| verl default | n/a | RL | `actor.loss_agg_mode` | `token-mean` | actor.yaml L86; grpo.md L41 | verified 2026-09-14 | docs: all verl GRPO example scripts use it; no ablation reported |
| verl docs, Dr. GRPO | n/a | RL | loss_agg_mode; use_kl_loss; norm_adv_by_std_in_grpo | `seq-mean-token-sum-norm`; False; False | grpo.md L57–62 | verified 2026-09-14 | no ablation reported in docs |
| DAPO reproduction on verl, Qwen2.5-32B base | 32B | RL | group filtering (dynamic sampling) on vs off | AIME 2024 acc. 52% vs 50% | docs/algo/dapo.md L36–37 (commit 4f80e4, 16x8xH800) | verified 2026-09-14 | one training run per row; W&B record linked |

## Findings relevant to generality and negative feedback
- The verl DAPO reproduction table (dapo.md L36–38) reports 52% (DAPO), 50% (without dynamic sampling), and 44% (without token-level loss and dynamic sampling) AIME 2024 accuracy on Qwen2.5-32B. The table gives one run per setting and no other benchmarks, so it does not measure breadth; the 44% run also used different hardware (H20) and a different image.
- The docs state that the original GRPO sample-level loss (`seq-mean-token-mean`) "may be unstable in long-CoT scenarios" (grpo.md L41). No experiment is given in the docs.

## Connections
- [[grpo]] — DeepSeekMath, arXiv:2402.03300: objective Eq. (3) and the outcome-supervision advantage `(r_i − mean(r))/std(r)` (§4.1.2) that L325–326 implements.
- [[dr-grpo]] — arXiv:2503.20783 §3.1: std normalization up-weights questions with low reward std (too easy or too hard); the 1/|o_i| term causes a response-length bias. In verl the length term is set by `loss_agg_mode`, not by this estimator.
- [[verl-ppo-loss]] — the policy loss that consumes these advantages.
- [[trl-grpo]] — TRL at a04ffd3 (2026-09-14) computes group advantages in `_generate_and_score_completions` (grpo_trainer.py L2793–2813, ε = 1e-4), separate from `_compute_loss` (L3113).
- [[openrlhf-ppo]] — OpenRLHF at 64c1cc4 (2026-04-19) implements `group_norm` (ε = 1e-9) and mean-only `dr_grpo` in `experience_maker.py` L267–270.
- [[rloo-vs-grpo]] — alternative leave-one-out baseline (`rloo` estimator, core_algos.py L587–636).

## Verification
- Checked on 2026-09-14 against: https://github.com/verl-project/verl at 753aed3e1c286ba6825a74342b28669e72c083ea (`verl/trainer/ppo/core_algos.py`, `ray_trainer.py`, `utils.py`; `verl/utils/groupwise.py`; `verl/trainer/config/{ppo_trainer.yaml, actor/actor.yaml, rollout/rollout.yaml, algorithm.py}`; `docs/algo/{grpo,dapo}.md`); arXiv:2402.03300v3; arXiv:2503.20783v2; arXiv:2503.19595v2; TRL a04ffd3; OpenRLHF 64c1cc4.
- Corrections to the previous card version:
  - "Singleton groups (n=1) get mean=0, std=1 — zero advantage, the prompt contributes no gradient" → a size-1 group gets advantage = raw reward/(1 + 1e-6); only all-equal groups of size ≥ 2 get zero advantage (L315–326).
  - Line ranges "≈ 290–365" and "≈ 498–550" (April fetch) → L266–358 and L471–530 at 753aed3; the April commit 1a3b7e2 already had the function at L268.
  - "Per-token broadcast ... is the source of GRPO's well-known length bias" → Dr. GRPO §3.1 attributes the length bias to the 1/|o_i| normalization, which verl sets through `loss_agg_mode`.
  - "Dr.GRPO removes std normalization to fix variance bias on prompts where all rollouts succeed/fail" → Dr. GRPO §3.1: std division up-weights low-std questions; all-succeed or all-fail groups have zero advantage with or without std division.
  - "matches Eq. 18 of Shao et al. 2024" → the advantage is defined in DeepSeekMath §4.1.2; Eq. (18) is the PPO gradient coefficient in App. A.1.
  - "Process rewards require the GRPO-Pass@k or process variants" → the Pass@k estimator also sums token rewards to one outcome score (L501).
  - "(advantages, returns) ... value-loss term dropped (`vf_coef = 0`)", "`use_critic = false`" → no `vf_coef` key exists in ppo_trainer.yaml or actor.yaml; `use_critic` is a trainer attribute set by `need_critic` (ray_trainer.py L348), which returns False when `adv_estimator != gae` and `critic.enable` is unset (utils.py L96–107).
  - "TRL bundles advantage normalization, KL term, and clipped objective into a single `_compute_loss`" → TRL computes advantages in `_generate_and_score_completions`.
  - Repo named "volcengine / Bytedance Seed" → repository is verl-project/verl; README says initiated by ByteDance Seed team, maintained by the verl community.
- Removed as unsupported by the source: "verl's implementation is the canonical open-source reference and is what most R1 reproductions actually run"; "In practice `n ≥ 4`"; "GRPO is the loss DeepSeek used for R1-Zero / R1" (not stated in this source; see [[deepseek-r1]]).
- Not reported by the source: measured effect of zero-advantage groups or of negative advantages on pass@k, calibration, or held-out tasks.
