---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl PPO loss — `compute_policy_loss_vanilla`

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-ppo-loss.md`
**Artifact:** `@register_policy_loss("vanilla")` body at `verl/trainer/ppo/core_algos.py` ~lines 1080–1140. The canonical clipped-surrogate objective in verl, plus three extras (asymmetric clip, dual-clip, rollout IS).

---

## Why this excerpt exists in ch-55

Ch-55 §2 walks this function line by line. The function is 100% of the PPO policy-loss algebra in verl — GRPO reuses it unchanged. If you can trace one token from the `advantages` argument to the returned `pg_loss`, you understand verl's RL engine.

---

## The full quoted body

```python
# verl/trainer/ppo/core_algos.py, ~lines 1080-1140
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode="token-mean", config=None, rollout_is_weights=None,
):
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low  = config.clip_ratio_low  if config.clip_ratio_low  is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c    = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio  = torch.exp(negative_approx_kl)
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
        pg_losses = pg_losses * rollout_is_weights   # off-policy IS correction (vLLM mismatch)

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    return pg_loss, {"actor/pg_clipfrac": pg_clipfrac.detach().item(),
                     "actor/ppo_kl": ppo_kl.detach().item(),
                     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item()}
```

---

## Five things from the source's "What to notice"

1. **Asymmetric clipping.** `clip_ratio_low` and `clip_ratio_high` are independent; DAPO uses ε_low ≈ 0.2 and ε_high ≈ 0.28 to let rare positive-advantage tokens keep moving.
2. **K1 is a monitor.** `ppo_kl = mean(-Δlogp)` is logged; never added to the loss. KL-to-ref is reward-shaped via `kl_penalty(...)` (see excerpts/`entropy-logging-patterns.md`).
3. **Dual-clip.** Only fires when `advantages < 0` and the loss has already exceeded `clip_ratio_c · |A|`; it floors the loss to prevent ratio blow-ups on long off-policy rollouts.
4. **Rollout IS weights.** Per-token correction `exp(logπ_actor − logπ_rollout)` for the bf16-vLLM vs fp32-actor logprob mismatch.
5. **Loss aggregation is parametric.** `"token-mean"` default; `"seq-mean-token-sum"` = Dr.GRPO; `"seq-mean-token-mean"` = length-normalized. Changing this flag materially changes the gradient on long-tailed length distributions — it is where the [[dr-grpo]] fix lives.

---

## How this function is called

From `ppo_trainer.py` inside the K-epoch minibatch loop:

```python
# pseudo-trace of one inner step
old_logp    = actor.forward(rollout)              # frozen snapshot
for epoch in range(K):
    logp    = actor.forward(rollout)              # current weights
    loss, m = compute_policy_loss_vanilla(
        old_log_prob=old_logp, log_prob=logp,
        advantages=adv, response_mask=mask,
        config=actor_config, rollout_is_weights=is_weights,
    )
    loss.backward(); optimizer.step()
```

GRPO reuses this exact function unchanged. Only the `advantages` argument changes entry points — `compute_grpo_outcome_advantage` replaces GAE.

---

## Comparison to [[ppo]]

Schulman's 2017 L^CLIP is `E[min(r·A, clip(r, 1-ε, 1+ε)·A)]` with *symmetric* ε. verl matches this when `clip_ratio_low == clip_ratio_high == clip_ratio`. The two extras (asymmetric clip, dual-clip) are production patches. The `entropy bonus` from the paper's combined `L^CLIP+VF+S` is **absent** — verl logs entropy but doesn't add an entropy term to the loss by default (optional via `entropy_loss` registry).

---

## Comparison to other frameworks

- **OpenRLHF `PolicyLoss` (`openrlhf/models/loss.py`)** — same algebra, wrapped in an `nn.Module`. verl's free-function + registry composes more cleanly with GRPO / GSPO variants.
- **TRL `PPOTrainer`** — PPO loss inlined inside the training loop; uses *symmetric* `cliprange` only.
- **KL handling:** all three frameworks compute KL outside the policy loss. verl/OpenRLHF subtract β·KL from rewards (reward-shaping); TRL adds it as `non_score_reward` for logging and uses an adaptive controller.
