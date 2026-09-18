<!-- scope: OpenRLHF PPO policy loss (PolicyLoss), KL-to-reference wiring, and vLLM importance-sampling correction, pinned at commit 64c1cc4
     deps: [[ppo]]
     see-also: [[openrlhf-ppo-recipe]], [[verl-ppo-loss]], [[trl-ppo]], [[trl-grpo]], [[openrlhf-dpo]], [[entropy-logging-patterns]], [[async-rollout]]
-->

# OpenRLHF `openrlhf/models/loss.py` `PolicyLoss` and PPO KL wiring (commit 64c1cc4)
- **Core Insight:** At commit 64c1cc4, `PolicyLoss` minimizes −min(r·A, clip(r, 1−ε_low, 1+ε_high)·A) with optional dual-clip and vLLM importance-sampling (IS) weights, and the KL-to-reference term is added to per-token rewards with a fixed coefficient of 0.01 unless `--algo.kl.use_loss` moves it into the loss.
- **Guideline:** When rollouts are sampled by vLLM, set `--algo.advantage.is_correction_enable` (default band [0.5, 5.0]) and log `vllm_kl`, because without it the loss treats the trainer's recomputed `old_log_probs` as the sampling policy; the repository reports no ablation for the band.
- **Authors:** OpenRLHF project (github.com/OpenRLHF); framework paper by Jian Hu, Xibin Wu, Wei Shen, Jason Klein Liu, Zilin Zhu, Weixun Wang et al.
- **Year:** 2026 (pinned commit 64c1cc4, 2026-04-19); framework paper arXiv:2405.11143 v1 2024-05
- **URL:** https://github.com/OpenRLHF/OpenRLHF/blob/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da/openrlhf/models/loss.py
- **Source type:** released config/code
- **Relevant topics:** PPO clipped surrogate, asymmetric clipping, dual-clip PPO, GSPO, KL penalty in reward vs loss, KL controllers, train–inference mismatch, IS correction

## Summary
The artifact is the PPO actor-loss path of OpenRLHF at commit 64c1cc4: `PolicyLoss` in `openrlhf/models/loss.py`, the KL estimators and reward shaping in `openrlhf/models/utils.py`, the KL controllers, the experience maker that builds advantages, the Ray actor that combines the losses, and the `train_ppo_ray.py` CLI defaults. Framework defaults are listed in [[openrlhf-ppo-recipe]]. The repository contains no training results for these settings.

## Key Contributions
- One loss module for token-level PPO (`policy_loss_type="ppo"`) and sequence-level GSPO (`"gspo"`) (loss.py L122–134).
- Separate lower and upper clip bounds, plus optional dual-clip for negative advantages (L136–148).
- Three IS corrections for vLLM-vs-trainer log-prob mismatch: `tis`, `icepop`, `seq-mask-tis` (L150–173).
- KL-to-reference as a per-token reward (default) or as a loss term (`--algo.kl.use_loss`), with k1/k2/k3 estimators (utils.py L48–110; ppo_actor.py L296–314).
- Advantage estimators `gae`, `reinforce`, `rloo`, `reinforce_baseline`, `group_norm`, `dr_grpo` (experience_maker.py L264–309).

## Key Figures/Tables to Study (code loci at 64c1cc4)
- `openrlhf/models/loss.py` L75–182 (`PolicyLoss`), L185–215 (`ValueLoss`, clipped value loss × 0.5).
- `openrlhf/models/utils.py` L48–79 (`compute_approx_kl`), L82–110 (`compute_reward`).
- `openrlhf/trainer/ppo_utils/kl_controller.py` L4–29; `openrlhf/trainer/ppo_trainer.py` L171–176, L250–253.
- `openrlhf/trainer/ppo_utils/experience_maker.py` L200–227 (KL), L264–328 (advantages, whitening).
- `openrlhf/trainer/ray/ppo_actor.py` L75–87 (loss construction), L284–322 (total loss).

## Technical Details
Verbatim excerpt, `loss.py` L136–148 and L175–182:
```python
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
if self.dual_clip is None:
    loss = -torch.min(surr1, surr2)
else:
    clip1 = torch.min(surr1, surr2)
    clip2 = torch.max(clip1, self.dual_clip * advantages)
    loss = -torch.where(advantages < 0, clip2, clip1)
...
loss = (masked_mean(loss, action_mask, dim=None) if self.token_level_loss
        else masked_mean(loss, action_mask, dim=-1).mean())
clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
return loss, clip_ratio, ppo_kl, vllm_kl
```
- **Ratio.** PPO: r = exp(log π_θ − log π_old) per token (L122–124). GSPO: the masked mean of the per-token log ratio over the response, exponentiated and broadcast to all tokens (L125–132); if IS correction is on, the GSPO ratio uses `rollout_log_probs` instead of `old_log_probs` (L127–130). GSPO forces sequence-level aggregation (`token_level_loss=False`, L101–103). The file cites arXiv:2507.18071 for GSPO (L126).
- **Aggregation.** Token level: one masked mean over all tokens in the micro-batch; sequence level: per-sequence mean, then batch mean (L175–179). `ppo_actor.py` does not pass `token_level_loss`, so PPO uses the default `True` (L75–87; loss.py L85).
- **Dual-clip.** Applied only where A < 0; `dual_clip` must be > 1.0; the file cites arXiv:1912.09729 (L105–107, L143–148). Default `None` (train_ppo_ray.py L389).
- **IS correction** (only for `policy_loss_type="ppo"`, L152). `old_log_probs` come from an actor forward pass in the experience maker (experience_maker.py L153–158); vLLM log-probs are requested only when correction is enabled (train_ppo_ray.py L78). The weight is w = exp(old_log_probs − rollout_log_probs), detached (L154–171). `tis`: w clamped to [low, high] (L169–172). `icepop`: w set to 0 outside [low, high] (L155–160). `seq-mask-tis`: sequences whose geometric-mean w lies outside [low, high] are dropped; kept tokens use unclamped w (L161–168). `vllm_kl` = masked mean of (rollout_log_probs − old_log_probs) (L173). The code comment cites the blog "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (L150). Defaults: off, band [0.5, 5.0], type `tis` (train_ppo_ray.py L257–271).
- **`ppo_kl` at this commit.** L154 reassigns `log_ratio` inside the IS branch, so when IS correction is on, `ppo_kl` (L181) equals mean(rollout_log_probs − old_log_probs), the same quantity as `vllm_kl`. With IS correction off it is mean(log π_old − log π_θ).
- **KL in the reward (default).** k1 = log π − log π_ref; k2 = (log-ratio)²/2; k3 = exp(−log-ratio) − 1 + log-ratio; all clamped to [−10, 10] (utils.py L62–79). Per-token reward = −β·KL on every token, plus the scalar reward (clipped to `reward.clip_range`, default (−10, 10)) on the last response token (utils.py L92–108; train_ppo_ray.py L437). Reward-side KL is computed only when `--algo.kl.use_loss` is off (experience_maker.py L205–214).
- **KL in the loss.** With `--algo.kl.use_loss`, the actor computes KL against `base_action_log_probs` and adds `kl_loss * kl_ctl` to the policy loss (ppo_actor.py L296–314). The CLI prints a recommendation of k2/k3 for loss-side KL and k1 for reward-side KL (train_ppo_ray.py L675–680).
- **KL controllers.** `AdaptiveKLController` (docstring cites arXiv:1909.08593): error = clip(KL/target − 1, −0.2, 0.2); β ← β·(1 + error·n_steps/horizon) (kl_controller.py L4–19). It is used only if `--algo.kl.target` is set (default `None`), otherwise `FixedKLController(init_coef)` with `init_coef` 0.01 (ppo_trainer.py L171–176; train_ppo_ray.py L417–419). The update call passes n_steps = rollout batch size × samples per prompt and carries the comment "TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here." (ppo_trainer.py L252–253).
- **Entropy term.** If `--actor.entropy_coef` is set, loss −= coef × masked-mean entropy; 0 logs entropy only (ppo_actor.py L319–322; train_ppo_ray.py L431–436).

## Recipe ledger
Framework defaults (not a trained model) are in [[openrlhf-ppo-recipe]].

## Findings relevant to negative feedback (code behavior; no experiments in the source)
Meaning used: negative as gradient (tokens with A < 0).
- **Sources of A < 0.** Baselines: critic values for `gae` (L284–291), `rloo` leave-one-out mean, `reinforce_baseline`/`dr_grpo` group mean, `group_norm` group mean and std (experience_maker.py L264–270); `gae`, `reinforce` and `reinforce_baseline` advantages are then mean-centered across the batch and divided by the batch std unless `--algo.advantage.no_std_norm` (L315–328). The KL reward −β·KL is negative on tokens where the k1 estimate is positive (utils.py L95).
- **Clipping by sign (derived from L136–141).** For A < 0 the objective is A·max(r, 1−ε_low): the gradient is zero when r < 1−ε_low, and the loss grows linearly without bound as r increases. For A > 0 it is A·min(r, 1+ε_high). `clip_eps_low` therefore acts on negative-advantage tokens and `clip_eps_high` on positive ones.
- **Dual-clip bound (derived from L143–148).** For A < 0 the per-token loss is capped at c·|A| and its gradient is zero when r > c.
- **Worked example.** ε_low = ε_high = 0.2, A = −1. r = 0.7: surr1 = −0.7, surr2 = −0.8, loss = 0.8 (constant in r). r = 1.5: loss = 1.5 (gradient flows). r = 5 with `dual_clip` = 3: clip1 = −5, clip2 = −3, loss = 3 (constant in r).
- **Diagnostics.** `clip_ratio` counts tokens with surr2 < surr1 for both signs together and does not count dual-clip activations (L180). IS weights and masks apply to both signs (L155–172).

## Connections
- [[ppo]] — clipped surrogate objective implemented here.
- [[verl-ppo-loss]] — verl applies the same dual-clip bound by default as `clip_ratio_c` = 3.0 (verl@753aed3 `core_algos.py` L1323–1324); OpenRLHF defaults to off.
- [[trl-ppo]] — TRL experimental PPO also puts −kl_coef·KL into per-token rewards (trl@a08e713 `trl/experimental/ppo/ppo_trainer.py` L777–781).
- [[trl-grpo]] — TRL's GRPO `delta` bound plays the same role as `dual_clip` for A < 0.
- [[kl-control-rlhf]], [[john-schulman-kl-tricks]] — KL-penalty and k1/k2/k3 background.
- [[async-rollout]], [[entropy-logging-patterns]], [[openrlhf-dpo]] — async rollout, metric logging, DPO path of the same repository.

## Verification
- Checked on 2026-09-14 against: github.com/OpenRLHF/OpenRLHF@64c1cc4f30d4c0dab772c8aa1dc11288c425e0da (latest `main` commit on or before 2026-04-21; `loss.py` last changed 2026-03-19 in 6a981c8): `openrlhf/models/loss.py`, `models/utils.py`, `trainer/ppo_trainer.py`, `trainer/ppo_utils/kl_controller.py`, `trainer/ppo_utils/experience_maker.py`, `trainer/ray/ppo_actor.py`, `cli/train_ppo_ray.py`. Also read `loss.py` at current `main` b117b2b.
- Corrections to the previous card version:
  - "`main` branch (fetched 2026-04-21)" → pinned to commit 64c1cc4.
  - "`PolicyLoss` ≈ lines 68–168" → L75–182 (forward L114–182).
  - Excerpt renamed the IS log ratio to `log_ratio_v` and described `ppo_kl` as "train-vs-old" → the code reuses `log_ratio` (L154), so with IS correction on `ppo_kl` equals `vllm_kl` (L173, L181).
  - "Dual-clip floors the per-token loss at `dual_clip * advantages`" → it lower-bounds the objective at c·A for A < 0, i.e. caps the loss at c·|A|; requires c > 1 (L106–107, L146–148).
  - "KL is not in the loss" → true only by default; `--algo.kl.use_loss` adds `kl_loss * kl_ctl` to the actor loss (ppo_actor.py L296–314).
  - "`AdaptiveKLController` … this is the InstructGPT recipe" → docstring cites arXiv:1909.08593; the default is `FixedKLController` (`kl.target` = None), and a TODO marks the adaptive controller incompatible at the update call (ppo_trainer.py L171–176, L252).
  - "TRL adds [KL] as `non_score_reward` for logging only" → TRL experimental PPO adds −kl_coef·KL to rewards (ppo_trainer.py L777–781 at trl@a08e713).
- Removed as unsupported by the source: "most-deployed open Ray-based PPO/DPO framework"; "pioneered the token-level-KL-reward + clipped-loss factorization that verl and many forks adopt"; "bf16 vLLM vs fp32 actor … PPO destabilizes within ~50 steps on long completions"; "GSPO reduces ratio-explosion variance"; "IS correction is essential"; "same convention as DAPO"; "algebraically equivalent to verl `compute_policy_loss_vanilla`".
- Not reported by the source: ablations or recommended values for `eps_clip_low_high`, `dual_clip`, or IS thresholds; any training curves.
- Later changes on `main` (b117b2b, 2026-09-14): IS correction is replaced by `is_correction_level {off, token, seq}`, `mode {mask, clip}`, `gating {ratio, binary_kl, tv}` (loss.py L121–172; commit e53dcf8); `ppo_kl` uses the unmodified policy ratio (L185, L258); the PPO log ratio is clamped to [−20, 20] (L187); `forward` also returns `is_filter_ratio` (L259).
