<!-- scope: Online DPO trainer in HuggingFace TRL
     deps: [[dpo]], [[self-play-preference]]
     see-also: [[openrlhf-dpo]], [[trl-grpo]], [[self-rewarding-lm]]
-->

# HuggingFace TRL — Online DPO Trainer
- **Framework:** HuggingFace TRL (experimental)
- **Repo URL:** https://github.com/huggingface/trl
- **Version/commit:** `main` branch, `trl/experimental/online_dpo/online_dpo_trainer.py` (fetched 2026-04-21)
- **Relevant file(s):**
  - `trl/experimental/online_dpo/online_dpo_trainer.py` ≈ lines 1089–1275 (`training_step` body; sampling→judging→DPO loss)
  - `_generate_vllm` / `_generate_vllm_colocate` ≈ lines 585–893 (rollout engines)
- **Core pattern:** On every step, sample *two* completions per prompt from the current policy (vLLM server, co-located vLLM, or HF `.generate`), score both with reward functions or a judge LM, pair the winner as `chosen` and the loser as `rejected`, then apply DPO on those fresh pairs. Nothing is stored offline — the preference pair is produced and consumed in the same step.
- **Why it matters:** Online DPO (Guo 2024 / Dong 2024) closes the gap between DPO and PPO by making the preference distribution *on-policy*; this trainer is the reference implementation used by many self-play / Nash-LM experiments.

## Context
Standard DPO assumes a fixed preference dataset. Online DPO turns that into an iterative loop: at each step, two samples per prompt are drawn from the *current* policy, labeled by a reward model / verifiable reward / LLM judge, and a DPO gradient step is taken. This avoids DPO's core failure mode — optimizing against a stale π_ref that no longer resembles the distribution the policy is in — by continuously refreshing the preference pairs from the actor itself. The file below is HF's canonical implementation; the same pattern underlies SPIN, self-rewarding LM training, and XPO.

## Code excerpt
```python
# trl/experimental/online_dpo/online_dpo_trainer.py, ≈ lines 1186–1230 (training_step body, condensed)
# Score both completions with the configured reward function(s)
rewards = self._calculate_rewards_from_functions(
    prompts=2 * prompts, completions=completions,
    completion_ids_list=completion_ids_list, **reward_kwargs,
)
if self.args.missing_eos_penalty is not None:
    rewards[~contain_eos_token] -= self.args.missing_eos_penalty

# Split rewards into chosen/rejected pairs by argmax per prompt
first_half, second_half = rewards.split(batch_size)
mask = first_half >= second_half
batch_range    = torch.arange(batch_size, device=device)
chosen_indices   = batch_range + (~mask * batch_size)
rejected_indices = batch_range + ( mask * batch_size)

cr_indices       = torch.cat((chosen_indices, rejected_indices), dim=0)
cr_logprobs      = logprobs[cr_indices]          # policy logprobs
cr_ref_logprobs  = ref_logprobs[cr_indices]      # frozen reference logprobs
padding_mask     = ~completion_mask.bool()
cr_padding_mask  = padding_mask[cr_indices]

cr_logprobs_sum     = (cr_logprobs     * ~cr_padding_mask).sum(1)
cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

chosen_logprobs_sum,   rejected_logprobs_sum   = torch.split(cr_logprobs_sum,     batch_size)
chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)

pi_logratios  = chosen_logprobs_sum     - rejected_logprobs_sum
ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum
logits = pi_logratios - ref_logratios

if self.args.loss_type == "sigmoid":
    losses = -F.logsigmoid(self.beta * logits)          # standard DPO
elif self.args.loss_type == "ipo":
    losses = (logits - 1 / (2 * self.beta)) ** 2        # IPO
loss = losses.mean()
```

```python
# trl/experimental/online_dpo/online_dpo_trainer.py, ≈ lines 1244–1270 (logging)
kl = logprobs - ref_logprobs
mean_kl = kl.sum(1).mean()
self.stats["objective/kl"].append(gather(mean_kl).mean().item())
non_score_reward = (-self.beta * kl).sum(1)
chosen_rewards   = self.beta * (chosen_logprobs_sum   - chosen_ref_logprobs_sum)
rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
margin = chosen_rewards - rejected_rewards
self.stats["rewards/accuracies"].append((margin > 0).float().mean().item())
self.stats["beta"].append(self.beta)
mean_entropy = -logprobs.sum(1).mean()
self.stats["objective/entropy"].append(gather(mean_entropy).mean().item())
```

## What to notice
- **Two samples per prompt** — the trainer always repeats each prompt twice for generation (`prompts=2 * prompts`), collects two completions, then picks winner/loser. Equivalent to RLOO with n=2 but fed into a DPO loss rather than a REINFORCE gradient.
- **Judge is pluggable:** `_calculate_rewards_from_functions` accepts reward models, verifiable reward functions, or a `Judge` interface that calls a frozen LLM to pick the preferred completion.
- **Missing-EOS penalty** (`missing_eos_penalty`) subtracts a constant from the reward if the completion didn't end with EOS — prevents truncation gaming.
- **Sequence-level logprobs** (`sum(1)` over completion tokens) — DPO uses full-sequence margins, not per-token.
- **KL tracking** is the canary for online DPO collapse: the policy drifts away from the reference much faster than offline DPO because each step uses fresh on-policy samples. `rewards/accuracies` (margin > 0) is the second critical metric.
- **`loss_type` switch** exposes both `sigmoid` (canonical DPO) and `ipo` (Azar 2023 L2-on-margin); extensions for self-rewarding and Nash-MD live in sibling files (`nash_md`, `xpo`, `self_distillation`).

## Comparison to paper / to other frameworks
- **vs Guo 2024 "Direct Language Model Alignment from Online AI Feedback" (OAIF):** this trainer is the exact pattern — on-policy samples, LLM-judge, DPO gradient.
- **vs OpenRLHF DPO (`openrlhf/trainer/dpo_trainer.py`):** OpenRLHF's DPO is offline (fixed chosen/rejected pairs); see [[openrlhf-dpo]]. No equivalent online trainer in OpenRLHF mainline — typically users run OpenRLHF PPO or switch to TRL for online DPO.
- **vs SPIN (Chen 2024):** SPIN is a special case where the "judge" labels human-written data as chosen and model-generated as rejected. TRL's `online_dpo_trainer.py` with a verifier reward is the closest match.
- **vs offline DPO:** the only difference is where the pair comes from — offline from a dataset, online from `.generate()`. The DPO loss algebra is identical.
