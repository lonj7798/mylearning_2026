<!-- scope: PPO trainer in HuggingFace TRL (experimental)
     deps: [[ppo]], [[rlhf-instructgpt]]
     see-also: [[trl-grpo]], [[verl-ppo-loss]], [[openrlhf-ppo]]
-->

# HuggingFace TRL — PPO Trainer
- **Framework:** HuggingFace TRL
- **Repo URL:** https://github.com/huggingface/trl
- **Version/commit:** `main` branch (fetched 2026-04-21). The actor-critic PPO with value head has moved to `trl/experimental/ppo/ppo_trainer.py` as of TRL 0.16; the new default trainers are GRPO/RLOO (value-free).
- **Relevant file(s):** `trl/experimental/ppo/ppo_trainer.py`
  - inner-loop inline PPO update ≈ lines 820–870
  - KL & entropy metric block ≈ lines 883–907
- **Core pattern:** Classic InstructGPT-shaped PPO: generate → score → GAE + value targets → N epochs of per-microbatch clipped policy update with a value head sharing the base model via `modeling_value_head.AutoModelForCausalLMWithValueHead`. KL controller runs outside the loss and subtracts β·KL from the per-token reward.
- **Why it matters:** This is the canonical Ouyang 2022 PPO recipe implemented in a single file — studying the inline loss math next to the GAE and value-head code is the cleanest way to understand why the post-GPT-RL community pivoted to critic-free algorithms (GRPO/RLOO/REINFORCE++).

## Context
TRL's PPO trainer co-located policy + value head (Ouyang 2022 / Stiennon 2020 pattern). Rewards from the reward model have the per-token KL `non_score_reward = −β·KL(π || π_ref)` added token-wise, then GAE produces advantages/returns for the value loss. The inner two-loop update runs `num_ppo_epochs` over the rollout with microbatches. Value clipping is the Engstrom-et-al.-recommended trick: clip `V_new` to `[V_old ± cliprange_value]` before squaring.

## Code excerpt
```python
# trl/experimental/ppo/ppo_trainer.py, ≈ lines 820–870 (inner PPO update)
output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
logits = output.logits[:, context_length - 1 : -1]
logits /= args.temperature + 1e-7
new_logprobs = selective_log_softmax(logits, mb_responses)
new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)

vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
vpredclipped = torch.clamp(vpred,
                           mb_values - args.cliprange_value,
                           mb_values + args.cliprange_value)
vf_losses1 = torch.square(vpred - mb_return)
vf_losses2 = torch.square(vpredclipped - mb_return)
vf_loss    = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2),
                               ~padding_mask_p1[micro_batch_inds])
vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(),
                          ~padding_mask_p1[micro_batch_inds])

logprobs_diff = new_logprobs - mb_logprobs
ratio        = torch.exp(logprobs_diff)
pg_losses    = -mb_advantage * ratio
pg_losses2   = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
pg_loss      = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask[micro_batch_inds])

loss = pg_loss + args.vf_coef * vf_loss
accelerator.backward(loss); optimizer.step(); optimizer.zero_grad()

with torch.no_grad():
    pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(),
                              ~padding_mask[micro_batch_inds])
    prob_dist = torch.nn.functional.softmax(logits, dim=-1)
    entropy   = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
    approxkl  = 0.5 * (logprobs_diff**2).mean()        # Schulman k2 KL
```

```python
# trl/experimental/ppo/ppo_trainer.py, ≈ lines 883–907 (per-iter metric reduction)
mean_kl = kl.sum(1).mean()
mean_entropy = (-logprobs).sum(1).mean()
mean_non_score_reward = non_score_reward.sum(1).mean()
rlhf_reward = mean_non_score_reward + scores.mean()
metrics["objective/kl"]       = gather(mean_kl).mean().item()
metrics["objective/entropy"]  = gather(mean_entropy).mean().item()
metrics["policy/approxkl_avg"] = gather(approxkl_stats).mean().item()
metrics["policy/clipfrac_avg"] = gather(pg_clipfrac_stats).mean().item()
metrics["loss/policy_avg"]    = gather(pg_loss_stats).mean().item()
metrics["loss/value_avg"]     = gather(vf_loss_stats).mean().item()
metrics["policy/entropy_avg"] = gather(entropy_stats).mean().item()
```

## What to notice
- **Symmetric clipping** — one `args.cliprange` only; no asymmetric DAPO-style split.
- **Value clipping** on `vpred`, not on the squared error — standard PPO2 trick that keeps the value target stable across epochs.
- **Entropy computed from logits** via `logsumexp − Σ p·logp` inside `torch.no_grad()` — it's a metric, not a training loss term (no entropy bonus).
- **Two KL metrics:**
  - `kl = logprobs − ref_logprobs` (K1, biased but cheap) — used in reward shaping.
  - `approxkl = 0.5·(Δlogp)^2` (K2, Schulman) — used in the clip-fraction diagnostic.
- **No KL inside the loss** — KL penalty is added into the *reward* as `non_score_reward` before GAE, so it becomes part of `mb_advantage`.
- **Padding masks are double-layered:** `padding_mask` (tokens that must not contribute to loss) and `padding_mask_p1` (one more token because the value head is one-shifted).
- **`INVALID_LOGPROB`** sentinel (usually -inf) prevents pad tokens from contaminating the ratio.

## Comparison to paper / to other frameworks
- **vs Ouyang 2022 (InstructGPT):** matches precisely; the value-clip and K2 approxkl are PPO2 additions (Engstrom 2020, Schulman blog).
- **vs verl `compute_policy_loss_vanilla`:** verl splits algebra from the loop and supports asymmetric+dual clip; TRL's experimental PPO is simpler and shows the GAE/value head clearly.
- **vs OpenRLHF PPO:** OpenRLHF uses a Ray-distributed actor/critic with the same clipped loss; its KL goes through an `AdaptiveKLController` (InstructGPT style), matching TRL's controller.
- **Deprecation note:** HF's main-line alignment stack has pivoted to critic-free `GRPOTrainer` / `RLOOTrainer`; the actor-critic PPO now lives under `trl/experimental/` but remains the recommended reference for teaching GAE + value-function RLHF.
