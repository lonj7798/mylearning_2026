---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-ppo.md
source_url: https://github.com/huggingface/trl/blob/a08e7139f933b770177fc2abc0b43118e26b260b/trl/experimental/ppo/ppo_trainer.py
revised_at: "2026-09-15"
---

# Excerpt: TRL PPOTrainer: per-token reward, GAE, clipped losses (commit a08e713)

Matches [[trl-ppo]] and [[trl-ppo-recipe]] (verified 2026-09-14). Used in read.md §3, §4, §9. The file was removed from TRL in v1.13.0 (PR #7020).

## Reward shaping and GAE (ppo_trainer.py L775-799, verbatim lines)
```python
logr = ref_logprobs - logprobs
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
actual_start = torch.arange(rewards.size(0), device=rewards.device)
actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[actual_start, actual_end] += scores
...
delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
lastgaelam = delta + args.gamma * args.lam * lastgaelam
...
returns = advantages + values
advantages = masked_whiten(advantages, ~padding_mask)
```

## Losses (L831-849, as described in the card)
- Value loss: 0.5·max((V − R)², (clip(V, V_old ± cliprange_value) − R)²), masked mean.
- Policy loss: max(−A·r, −A·clip(r, 1 − cliprange, 1 + cliprange)), masked mean; one `cliprange` sets both bounds.
- Total: pg_loss + vf_coef·vf_loss; no KL and no entropy term in the loss.

## Defaults (ppo_config.py)
- learning_rate 3e-6; num_ppo_epochs 4; num_mini_batches 1; kl_coef 0.05; kl_estimator "k1"; cliprange 0.2; cliprange_value 0.2; vf_coef 0.1; gamma 1.0; lam 0.95; temperature 0.7; response_length 53; whiten_rewards False; missing_eos_penalty None.
- The value model is separate; the TL;DR example initializes it from `reward_model_path`.
