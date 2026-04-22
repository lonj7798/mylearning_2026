<!-- scope: PPO PolicyLoss + KL handling in OpenRLHF
     deps: [[ppo]]
     see-also: [[verl-ppo-loss]], [[trl-ppo]], [[openrlhf-dpo]], [[entropy-logging-patterns]]
-->

# OpenRLHF — PPO PolicyLoss
- **Framework:** OpenRLHF
- **Repo URL:** https://github.com/OpenRLHF/OpenRLHF
- **Version/commit:** `main` branch (fetched 2026-04-21)
- **Relevant file(s):**
  - `openrlhf/models/loss.py` ≈ lines 68–168 (`PolicyLoss`)
  - `openrlhf/trainer/ppo_trainer.py` (KL controller wiring; `AdaptiveKLController`/`FixedKLController` instantiation around line 172)
- **Core pattern:** Module-style loss object that returns `(loss, clip_ratio, ppo_kl, vllm_kl)`. Supports vanilla PPO clip, GSPO sequence-level ratio, dual-clip, and three vLLM importance-sampling correction modes (`tis`, `seq-mask-tis`, `icepop`). KL is *not* in the loss — it's added to per-token rewards through `kl_ctl` outside.
- **Why it matters:** OpenRLHF is the most-deployed open Ray-based PPO/DPO framework; it pioneered the `(token-level KL reward) + (clipped policy loss)` factorization that verl and many forks adopt.

## Context
OpenRLHF separates concerns sharply: `PolicyLoss` is just the clipped surrogate; the KL-to-reference penalty is folded into per-token rewards by the experience builder (`AdaptiveKLController` updates β each iteration based on observed KL, mirroring InstructGPT). The class also bakes in the rollout-vs-train logprob mismatch correction needed when generation runs in vLLM at bf16 while the actor trains at fp32 — without it, ratios are systematically biased and PPO destabilizes within ~50 steps on long completions.

## Code excerpt
```python
# openrlhf/models/loss.py, lines 68–168 (condensed; PolicyLoss.forward body)
class PolicyLoss(nn.Module):
    def __init__(self, clip_eps_low=0.2, clip_eps_high=0.2, dual_clip=None,
                 token_level_loss=True, policy_loss_type="ppo",
                 enable_vllm_is_correction=False,
                 vllm_is_truncated_threshold=None,
                 vllm_is_correction_type="tis"):
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.dual_clip = dual_clip
        self.token_level_loss = token_level_loss
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

    def forward(self, log_probs, old_log_probs, advantages,
                action_mask=None, rollout_log_probs=None):
        if self.policy_loss_type == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # Sequence-level importance ratio (Qwen GSPO)
            base = rollout_log_probs if self.enable_vllm_is_correction else old_log_probs
            log_ratio = log_probs - base
            ratio = (log_ratio * action_mask).sum(-1) / action_mask.sum(-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
        if self.dual_clip is None:
            loss = -torch.min(surr1, surr2)
        else:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            loss = -torch.where(advantages < 0, clip2, clip1)

        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low, high = self.vllm_is_truncated_threshold
            log_ratio_v = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                vllm_is = torch.exp(log_ratio_v).detach()
                vllm_is = vllm_is * ((vllm_is >= low) & (vllm_is <= high))
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                seq_log_ratio = masked_mean(log_ratio_v, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low) & (seq_is <= high)
                vllm_is = torch.exp(log_ratio_v).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:  # "tis"
                vllm_is = torch.exp(log_ratio_v).clamp(min=low, max=high).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        loss = (masked_mean(loss, action_mask, dim=None) if self.token_level_loss
                else masked_mean(loss, action_mask, dim=-1).mean())
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl
```

## What to notice
- **Asymmetric clipping** is exposed as `clip_eps_low` / `clip_eps_high`, same convention as DAPO and verl.
- **GSPO** (Qwen 2025) reduces ratio-explosion variance by computing one ratio per *sequence* (mean log-ratio over the response) instead of per token; here it's a one-line branch on `policy_loss_type`.
- **Dual-clip** floors the per-token loss at `dual_clip * advantages` for negative-advantage tokens (mirrors verl's `clip_ratio_c`).
- **vLLM IS correction** is essential when rollouts are produced by a different inference engine than the trainer's forward pass — `tis` is the simple Truncated IS, `icepop` masks tokens whose IS weight falls outside `[low, high]`, `seq-mask-tis` masks at the sequence level.
- **Two KL metrics returned:** `ppo_kl` (train-vs-old, monitoring), `vllm_kl` (rollout-vs-train, diagnoses sampler drift). KL-to-reference is *not* here — it's applied via `kl_ctl.value` to the per-token reward in the trainer (`openrlhf/trainer/ppo_trainer.py`).
- **`AdaptiveKLController`** (instantiated around line 172 of `ppo_trainer.py`) adjusts β each iteration to keep observed KL near a target; this is the InstructGPT recipe.

## Comparison to paper / to other frameworks
- **vs Schulman 2017 PPO:** identical clipped objective; OpenRLHF adds asymmetric clip + dual-clip + IS correction.
- **vs verl `compute_policy_loss_vanilla`:** algebraically equivalent (verl's `agg_loss` is more flexible); OpenRLHF wraps state in a module so the IS correction config travels with the loss object.
- **vs TRL `PPOTrainer`:** TRL inlines the loss in the train loop and uses one symmetric `cliprange` plus a separate `kl_ctl`; no GSPO, no IS correction.
- **KL handling:** OpenRLHF and verl both apply token-level KL via reward shaping (so KL becomes part of the GAE/GRPO advantage); TRL adds it as `non_score_reward` for logging only and clips to a controller target.
