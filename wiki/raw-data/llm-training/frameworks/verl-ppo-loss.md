<!-- scope: PPO clipped policy loss in verl
     deps: [[ppo]]
     see-also: [[verl-grpo]], [[openrlhf-ppo]], [[trl-ppo]], [[entropy-logging-patterns]]
-->

# verl — PPO Policy Loss
- **Framework:** verl (volcengine / Bytedance Seed)
- **Repo URL:** https://github.com/verl-project/verl
- **Version/commit:** `main` branch (fetched 2026-04-21)
- **Relevant file(s):** `verl/trainer/ppo/core_algos.py` (≈ lines 1080–1140; deprecated wrapper at 1029–1078; `kl_penalty` and `agg_loss` lower in same file)
- **Core pattern:** A `@register_policy_loss("vanilla")`-decorated function computes the standard PPO clipped surrogate with asymmetric low/high clip ratios, dual-clip on negative-advantage tokens, optional rollout importance-sampling correction, and plug-in loss aggregation across token / sequence axes.
- **Why it matters:** verl is the production RLHF framework powering ByteDance Seed and many open R1 reproductions; its PPO loss is the textbook clipped objective with the small-but-load-bearing extras (asymmetric clipping, dual-clip-PPO, K3-style KL approximation) you actually need at scale.

## Context
verl decouples policy-loss algebra from training-loop plumbing through a registry (`@register_policy_loss(name)` / `@register_adv_est(name)`). That makes it easy to swap PPO vanilla, GRPO, GSPO, etc. The vanilla PPO function below is what runs when `actor.policy_loss = "vanilla"` and is the closest implementation to Schulman 2017 PPO‑clip in modern LLM-RL frameworks. Asymmetric clipping (`clip_ratio_low`, `clip_ratio_high`) is the DAPO/Open‑Reasoner‑Zero recipe — clipping more aggressively on the upside. Dual-clip (Ye et al., 2020) prevents catastrophic gradient explosion when an old-policy ratio explodes on a negative-advantage token.

## Code excerpt
```python
# verl/trainer/ppo/core_algos.py, ~lines 1080–1140
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # Dual-clip PPO (Ye et al. 2020): bound the loss for negative-advantage tokens.
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights  # off-policy IS correction (vLLM mismatch)

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics
```

## What to notice
- **Asymmetric clipping:** `clip_ratio_low` and `clip_ratio_high` separate; DAPO uses ε_low ≈ 0.2 and ε_high ≈ 0.28 to allow more upward exploration on rare positive-advantage tokens.
- **K1 ratio is *not* the KL term in the loss** — `ppo_kl = mean(-Δlogp)` is logged for monitoring only; KL-to-reference is added either through the reward (token-level KL) or through a separate `kl_penalty(logprob, ref_logprob, "k3")` call (low-variance Schulman estimator), never inside this function.
- **Dual clip** only fires when `advantages < 0` and the loss has already exceeded `clip_ratio_c · |advantage|`; it floors the loss to prevent ratio blow-ups on long off-policy rollouts.
- **`rollout_is_weights`** is a per-token IS correction for the train-vs-rollout-engine logprob mismatch (vLLM bf16 ≠ actor fp32) — see the TIS/iCEPO patches.
- **Loss aggregation is parametric:** `"token-mean"` (default), `"seq-mean-token-sum"` (Dr.GRPO style), or `"seq-mean-token-mean"` (length-normalized) — each materially changes gradients on long-tailed completion length distributions.
- **No entropy bonus** in the loss — entropy is logged separately and (optionally) regularized via the `entropy_loss` registry hook.

## Comparison to paper / to other frameworks
- **vs Schulman 2017:** identical clipped surrogate; the paper's ε-symmetric clip becomes asymmetric here, and dual-clip is added.
- **vs OpenRLHF (`PolicyLoss`, `openrlhf/models/loss.py`):** OpenRLHF wraps the same algebra in an `nn.Module`; verl's free-function + registry pattern composes more cleanly with GRPO / GSPO variants.
- **vs TRL `PPOTrainer`:** TRL inlines the PPO loss inside the training loop (see [[trl-ppo]]) and uses *symmetric* `cliprange` only.
- **KL handling:** all three frameworks compute KL outside the policy loss — verl/OpenRLHF subtract β·KL from rewards (token-level), TRL adds it as `non_score_reward` for logging and uses an adaptive controller.
