---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-ppo.md
source_url: https://github.com/huggingface/trl/blob/main/trl/experimental/ppo/ppo_trainer.py
created_at: "2026-04-23"
---

# Excerpt: TRL experimental PPO — the demoted reference implementation

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-ppo.md`
**Artifact:** `trl/experimental/ppo/ppo_trainer.py`. Inner-loop inline PPO update at ~L820–870; metric block at ~L883–907. Uses `AutoModelForCausalLMWithValueHead` — the policy and value head share the base transformer.

---

## Why this source anchors ch-57 §5

Ch-57 §5 shows that TRL's mainline RL trainer *used to* be actor-critic PPO — the canonical InstructGPT ([[hf-rlhf-illustrated]]) recipe — and that in 2024 it was demoted into `experimental/` when HF pivoted to critic-free methods (GRPO, RLOO). The code below is the cleanest open-source statement of the PPO+value-head pattern; studying it next to the GRPO file makes clear what "removing the critic" actually means in code.

---

## The inner-loop PPO update that ch-57 §5 quotes verbatim

Source lines 820–870:

```python
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

logprobs_diff = new_logprobs - mb_logprobs
ratio        = torch.exp(logprobs_diff)
pg_losses    = -mb_advantage * ratio
pg_losses2   = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
pg_loss      = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask[micro_batch_inds])

loss = pg_loss + args.vf_coef * vf_loss
accelerator.backward(loss); optimizer.step(); optimizer.zero_grad()
```

Four things to notice:

1. **Value head and policy share the forward.** `output, vpred_temp = forward(model, ...)` is one call that returns both logits and value. That is the `AutoModelForCausalLMWithValueHead` trick — one transformer, two output heads.
2. **Value clipping** on `vpred` (not on the squared error) — the Engstrom 2020 / Schulman-blog PPO2 refinement that keeps the value target stable across epochs.
3. **Symmetric clip** — a single `args.cliprange`, no asymmetric ε_low/ε_high like DAPO. This is the 2017 PPO paper's original form.
4. **Both losses combined in one backward.** `loss = pg_loss + args.vf_coef * vf_loss` — the policy and value head are updated together, coupling their gradients.

---

## KL is on the reward, not in the loss (ch-57 §5 contrasts this with GRPO)

Source lines 883–907:

```python
mean_kl = kl.sum(1).mean()
mean_entropy = (-logprobs).sum(1).mean()
mean_non_score_reward = non_score_reward.sum(1).mean()
rlhf_reward = mean_non_score_reward + scores.mean()
metrics["objective/kl"]       = gather(mean_kl).mean().item()
metrics["objective/entropy"]  = gather(mean_entropy).mean().item()
metrics["policy/approxkl_avg"] = gather(approxkl_stats).mean().item()
metrics["policy/clipfrac_avg"] = gather(pg_clipfrac_stats).mean().item()
```

Two KL quantities are logged separately:

- `kl = logprobs − ref_logprobs` is the **K1** estimator (biased in sign, low variance). Used upstream to build `non_score_reward = −β · KL` which is *added into the per-token reward before GAE runs*. That is the "KL-on-reward" pattern.
- `approxkl = 0.5·(Δlogp)²` is the **K2** estimator (Schulman), used only for the clipfrac diagnostic.

GRPO ([[trl-grpo]]) switched to K3 (`exp(Δ) − Δ − 1`, always ≥ 0, unbiased) and moved the KL *into the loss* rather than onto the reward. The move has a real numerical effect: KL-on-reward propagates through std normalization when that exists; KL-in-loss does not.

---

## The padding-mask double layer

`padding_mask` masks tokens that should not contribute to the policy loss. `padding_mask_p1` is the same mask shifted by one token, because the value head is one-shifted relative to the logits. Missing this detail used to be a common bug in PPO-RLHF reimplementations.

`INVALID_LOGPROB` is a sentinel (usually `-inf`) that prevents pad tokens from contaminating the ratio `exp(new − old)`.

---

## Why this file was demoted

Ch-57 §5's narrative: the critic is solving a problem (value-function fit over 1024-token completions with a single end-of-sequence reward) that does not need solving. [[rlhf-instructgpt]]-style RL with deterministic transitions and full-trajectory rewards collapses GAE to `R − V(s_0)`, so the critic is really learning `V(prompt)` — which is estimated *for free* by the mean reward of G peer samples (GRPO) or the leave-one-out mean (RLOO). The critic is thus pure overhead:

- ~50% more memory (value head + its gradients).
- One more target-fitting loss to balance against the policy.
- Numerical instability from the value-loss coefficient `vf_coef`.

When DeepSeekMath shipped [[grpo]] and R1 used it, the field voted with its feet. `GRPOTrainer` became the default RLHF trainer in TRL; `PPOTrainer` moved into `experimental/`. The file is still the best teaching reference — ch-57 §5 preserves the exact code snippet precisely because it is where students can see the value loss and the policy loss side by side in one backward.

---

## Attested implementation notes

- The trainer uses TRL's own `selective_log_softmax(logits, responses)` — a custom log-softmax that only computes the row corresponding to the actual token, saving memory vs a full `log_softmax(logits, -1)`. Shared with GRPOTrainer.
- Entropy is logged but not added to the loss (no entropy bonus). This matches [[ppo]]'s `c2 = 0.01` canonical value being effectively zero in the LLM regime.
- `args.num_ppo_epochs` controls inner-loop epochs per rollout. The LLM convention is `μ = 1`; increasing it was the 2023 standard but by 2025 most runs went back to single-epoch because the clip rarely binds at higher μ anyway.

---

## Connections to the rest of the track

- [[ppo]] — the 2017 paper whose L^CLIP+VF+S objective this file instantiates.
- [[trl-grpo]] — the successor trainer that removed the value head.
- [[hf-rlhf-illustrated]] — the three-stage diagram this code implements.
- [[verl-ppo-loss]] — verl's counterpart; splits policy loss and GAE into separate modules.
- [[rlhf-instructgpt]] — the Ouyang 2022 paper this recipe comes from.
