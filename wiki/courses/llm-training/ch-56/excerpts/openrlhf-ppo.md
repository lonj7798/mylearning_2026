---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md
source_url: https://github.com/OpenRLHF/OpenRLHF/tree/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da
created_at: "2026-04-23"
revised: "2026-09 (generality revision; rewritten to match the verified card)"
---

# Excerpt: OpenRLHF `PolicyLoss`, KL wiring, and advantage handling

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md`
**Version/commit:** `64c1cc4f30d4c0dab772c8aa1dc11288c425e0da` (2026-04-19); loss file also read at `main` b117b2b (2026-09-14)
**Files:** `openrlhf/models/loss.py`, `openrlhf/models/utils.py`, `openrlhf/trainer/ppo_trainer.py`,
`openrlhf/trainer/ppo_utils/{kl_controller,experience_maker}.py`, `openrlhf/trainer/ray/ppo_actor.py`,
`openrlhf/cli/train_ppo_ray.py`

---

## What ch-56 takes from this source

### 1. The clipped surrogate (`loss.py` L136–148, L175–182, verbatim)

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

`PolicyLoss` covers token-level PPO (`policy_loss_type="ppo"`, L122–124) and sequence-level GSPO
(`"gspo"`, L125–134, which forces `token_level_loss=False`). `dual_clip` applies only where A < 0 and must
exceed 1.0 (L105–107, L143–148); its default is `None` (`train_ppo_ray.py` L389).

### 2. Behaviour by advantage sign (derived from L136–148)

For A < 0 the objective is A·max(r, 1 − ε_low): the gradient is zero below 1 − ε_low, and the loss grows
without bound as r increases unless `dual_clip` caps it at c·|A|. For A > 0 it is A·min(r, 1 + ε_high).
Worked example at ε_low = ε_high = 0.2, A = −1: r = 0.7 gives loss 0.8 with zero gradient; r = 1.5 gives
loss 1.5 with gradient; r = 5 with `dual_clip` = 3 gives loss 3 with zero gradient.

### 3. KL placement (`models/utils.py` L48–110; `ray/ppo_actor.py` L296–314)

Default: KL is a per-token reward. `compute_approx_kl` returns k1 = log π − log π_ref, k2 = (log-ratio)²/2,
or k3 = exp(−log-ratio) − 1 + log-ratio, clamped to [−10, 10]. `compute_reward` writes −β·kl_t on every
response token plus the scalar reward, clipped to `--reward.clip_range` (default (−10, 10)), on the last
response token. With `--algo.kl.use_loss`, the experience maker zeroes `kl` and the actor adds
`kl_loss * kl_ctl` to the policy loss instead. The CLI prints a recommendation of k1 for reward placement
and k2/k3 for loss placement (`train_ppo_ray.py` L675–680).

### 4. Controllers (`ppo_utils/kl_controller.py` L4–29; `ppo_trainer.py` L171–176, L252–253)

`AdaptiveKLController` docstring cites arXiv:1909.08593; the rule is
`error = clip(KL/target − 1, −0.2, 0.2)`, then `β ← β·(1 + error·n_steps/horizon)`. It is used **only** when
`--algo.kl.target` is set; the default is `None`, which selects `FixedKLController(init_coef=0.01)`. The
update call carries the comment "TODO: KL controller must be FixedKLController; AdaptiveKLController is
incompatible here."

### 5. Advantages and normalization (`experience_maker.py` L264–270, L315–328)

Estimators: `gae`, `reinforce`, `rloo`, `reinforce_baseline`, `group_norm`, `dr_grpo`. For `gae`,
`reinforce` and `reinforce_baseline`, advantages are mean-centred over the batch and divided by the batch
standard deviation unless `--algo.advantage.no_std_norm` is passed.

### 6. vLLM importance-sampling correction (`loss.py` L150–173)

`w = exp(old_log_probs − rollout_log_probs)`, detached. `tis` clamps w into the band; `icepop` zeroes
tokens outside it; `seq-mask-tis` drops sequences whose geometric-mean w is outside it. `vllm_kl` is the
masked mean of `rollout_log_probs − old_log_probs`. Defaults: off, band [0.5, 5.0], type `tis`
(`train_ppo_ray.py` L257–271). At this commit L154 reassigns `log_ratio` inside the correction branch, so
with IS correction on, `ppo_kl` reports the same quantity as `vllm_kl`.

---

## Limits

The repository contains no training results. Every value above is a default or a code behaviour; the card
records no ablation for `eps_clip_low_high`, `dual_clip`, or the IS thresholds.

---

## Links

[[openrlhf-ppo]] · [[openrlhf-ppo-recipe]] · [[verl-ppo-loss]] · [[async-rollout]] · [[openrlhf-dpo]]
