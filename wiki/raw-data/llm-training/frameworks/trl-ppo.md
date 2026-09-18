<!-- scope: HuggingFace TRL actor-critic PPOTrainer for RLHF (trl/experimental/ppo), pinned at commit a08e713 (2026-04-21); deleted from TRL on 2026-09-04
     deps: [[ppo]], [[rlhf-instructgpt]]
     see-also: [[trl-ppo-recipe]], [[trl-grpo]], [[verl-ppo-loss]], [[openrlhf-ppo]], [[john-schulman-kl-tricks]]
-->

# PPO Trainer — TRL `trl/experimental/ppo/ppo_trainer.py`
- **Core Insight:** TRL's `PPOTrainer` subtracts a per-token KL penalty (default `kl_coef` 0.05) from the reward, adds the reward-model score at the end of the response, computes GAE advantages from a separate value model, and minimizes `pg_loss + vf_coef·vf_loss` (default `vf_coef` 0.1) with a symmetric ratio clip (default 0.2) for 4 epochs per rollout batch by default (ppo_trainer.py L774–849; ppo_config.py L232–264).
- **Guideline:** When code must run this trainer, pin a TRL version earlier than v1.13.0 and import it from `trl.experimental.ppo`, because PR #7020 deleted `PPOTrainer`, `PPOConfig`, and `modeling_value_head.py` (v1.13.0 release notes, 2026-09-10); otherwise use a critic-free trainer from `trl/trainer/` (`grpo_trainer.py` and `rloo_trainer.py` exist there at commit a08e713).
- **Authors:** Hugging Face TRL maintainers (no per-file author list; PR #7020 by Quentin Gallouédec removed the file)
- **Year:** 2026 snapshot (commit a08e713, 2026-04-21, VERSION `1.3.0.dev0`); first added 2020-03-28 (v1.13.0 release notes)
- **URL:** https://github.com/huggingface/trl/blob/a08e7139f933b770177fc2abc0b43118e26b260b/trl/experimental/ppo/ppo_trainer.py
- **Source type:** released config/code
- **Relevant topics:** PPO, RLHF, GAE, value model, KL penalty in reward, k1/k3 KL estimators, missing-EOS penalty, entropy logging

## Summary
`PPOTrainer` trains a causal LM policy against a scalar reward model with PPO. Each update (1) samples one response per
query, (2) scores it with the reward model and a separate value model, (3) builds a per-token reward from a KL penalty
plus the final score, (4) computes GAE advantages and returns, and (5) runs `num_ppo_epochs` passes of clipped policy
and clipped value updates over minibatches. The docs state that the implementation follows arXiv:2403.17031, "The N+
Implementation Details of RLHF with PPO" (docs/source/ppo_trainer.md L171). The file moved to `trl.experimental.ppo` in
v0.26.0 (2025-12-09, PR #4482) and was removed on 2026-09-04 (commit 700b845, PR #7020, released in v1.13.0).

## Key Contributions
- A single-file RLHF PPO loop with an explicit rollout phase (L671–801) and an explicit optimization phase (L804–880).
- Two KL estimators selectable by `kl_estimator` ("k1" default, "k3"), validated at L412–417 and applied at L775–777.
- A `PolicyAndValueWrapper` (L283–302, adapted from MOSS-RLHF per the comment at L281) so one `accelerator.prepare` call covers policy and value model.
- A documented metric set (docs L39–55) and a Pythia-1B TL;DR benchmark run (docs L173–202).

## Key Figures/Tables to Study
- ppo_trainer.py L774–801: reward shaping, optional reward whitening, GAE, advantage whitening.
- ppo_trainer.py L822–859: clipped value loss, clipped policy loss, and the logged entropy and `approxkl`.
- docs L35–63: definition of each logged metric and the "EOS trick" recommendation.

## Technical Details
Rollout (`train`, L606 onward):
1. Generation uses `temperature` (default 0.7), `top_k` 0.0, `top_p` 1.0, `max_new_tokens=response_length` (default 53) (L622–628; ppo_config.py L151–172).
2. Reference log-probs come from a copy made by `create_reference_model` or, with PEFT, from the policy with the adapter disabled or `ref_adapter_name` set (L441–446, L579–590, L709–716).
3. The value model is a separate `nn.Module` with a `.score` head read on its backbone's last hidden state (L299–302; utils.py L377–387). `PPOTrainer` does not import `modeling_value_head.AutoModelForCausalLMWithValueHead` (import block L15–80). The example script initializes the value model from `reward_model_path` (examples/scripts/ppo/ppo_tldr.py L100–101).
4. The reward-model score is read at the last non-pad token (utils.py L388–396). If `missing_eos_penalty` is set, completions without EOS get that constant subtracted from the score (L759–761).
5. Padded positions of `logprobs` and `ref_logprobs` are filled with `INVALID_LOGPROB = 1.0` (L80, L766–768), so both KL estimators evaluate to 0 there.

```python
# ppo_trainer.py @a08e713 L775-781 and L794-799 (verbatim lines)
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
- Symbols: `logr` = log π_ref − log π_θ per token; `kl_coef` = β; `scores` = reward-model output; `gamma` = γ (default 1.0); `lam` = λ (default 0.95).
- The score is added at index `sequence_length + 1` when that index exists, else at `sequence_length` (L779–781). `padding_mask_p1` keeps that one extra position for values and the value loss (L769–771, L839).

Optimization (L804–859):
1. Each PPO epoch draws a fresh permutation of the local batch (L805) and iterates minibatches and micro-batches.
2. Policy loss: `max(-A·r, -A·clip(r, 1−cliprange, 1+cliprange))`, masked mean over response tokens (L843–848). One `cliprange` sets both bounds.
3. Value loss: `0.5·max((V−R)², (clip(V, V_old±cliprange_value)−R)²)`, masked mean (L831–839).
4. Total loss: `pg_loss + vf_coef·vf_loss` (L849). The loss has no KL term and no entropy term; the KL penalty enters only through the rewards (L777–781).
5. Logged under `torch.no_grad()`: token entropy from the logits (L858) and `approxkl = 0.5·mean((log π_new − log π_rollout)²)` (L859), which the docs define as the KL between consecutive PPO policies, distinct from `objective/kl` (docs L45).

## Recipe ledger
Framework defaults and the documented TL;DR benchmark run are in [[trl-ppo-recipe]] (moved there to keep this card under 120 lines).

## Findings relevant to generality, negative feedback
- Negative feedback: the only penalty terms are the per-token KL term and the optional constant `missing_eos_penalty` (L759–761, L777). The docs recommend the penalty because it "can help the model learn to generate more coherent completions" (docs L63); no ablation is given.
- Measurement: at this commit `objective/entropy` sums `-logprobs` over padded positions, where the value is `INVALID_LOGPROB`, so each padded token adds −1.0 (L885). Commit 190c39c (PR #6121, 2026-07-27) masked those positions.
- Measurement: the docs define `objective/non_score_reward` as `beta * kl.sum(1)` and `objective/rlhf_reward` as `score - non_score_reward` (docs L42–43). The code logs `non_score_reward = -kl_coef·kl` summed over tokens and `rlhf_reward = non_score_reward + score` (L777, L886–887). Both give score − β·KL; the logged `non_score_reward` has the opposite sign to the docs.
- Generality: the docs evaluate one task, TL;DR summarization, with one model size (docs L173–202).

## Connections
- [[ppo]] — the clipped surrogate objective implemented at L843–848.
- [[rlhf-instructgpt]] — RLHF with a reward model and a KL penalty to the SFT policy; the TRL docs cite the OpenAI lm-human-preferences and summarize-from-feedback code, not InstructGPT (docs L9–10).
- [[john-schulman-kl-tricks]] — source of the k1/k3 estimators named in `kl_estimator` (ppo_config.py L244–252).
- [[kl-control-rlhf]] — KL penalty placed in the reward rather than in the loss.
- [[costa-huang-ppo-details]] — general PPO implementation details; the TRL docs cite the RLHF-specific N+ paper instead (docs L171).
- [[trl-grpo]] — critic-free trainer in `trl/trainer/grpo_trainer.py` at the same commit.
- [[verl-ppo-loss]], [[openrlhf-ppo]] — PPO losses in other frameworks.
- [[entropy-logging-patterns]] — entropy metric conventions across frameworks.

## Verification
- Checked on 2026-09-14 against: the URL above (commit a08e713, 2026-04-21), plus `ppo_config.py`, `trl/experimental/utils.py`, `docs/source/ppo_trainer.md`, `examples/scripts/ppo/ppo.py` and `ppo_tldr.py` at the same commit; commit 700b845 (file deletion); commit 190c39c; TRL release notes v0.26.0, v0.29.0, v1.13.0; PR #5174.
- Corrections to the previous card version:
  - "moved to `trl/experimental/ppo/ppo_trainer.py` as of TRL 0.16" → moved in v0.26.0 (2025-12-09, PR #4482); deleted in v1.13.0 (PR #7020).
  - "value head sharing the base model via `AutoModelForCausalLMWithValueHead`" → separate `value_model` with a `.score` head, wrapped with the policy in `PolicyAndValueWrapper` (L283–302, L494).
  - "KL controller runs outside the loss"; "OpenRLHF's `AdaptiveKLController` matching TRL's controller" → no KL controller exists; `kl_coef` is a fixed config value (ppo_config.py L240–243, L777).
  - "`kl = logprobs − ref_logprobs` (K1, biased)" → estimator is "k1" or "k3" by config; the config docstring calls both unbiased (ppo_config.py L95–99).
  - "`approxkl` (K2) used in the clip-fraction diagnostic" → `approxkl` (the k2 form, which the config reserves "for logging purposes", ppo_config.py L98–99) is logged as `policy/approxkl_avg`; the clip fraction is the separate `pg_clipfrac` (L854–859, L898–899).
  - "`INVALID_LOGPROB` sentinel (usually -inf)" → `INVALID_LOGPROB = 1.0` (L80).
  - "`padding_mask_p1` because the value head is one-shifted" → it keeps the position where the score is added (L769–781); the source gives no further reason.
  - Line ranges "≈820–870" and "≈883–907" → L804–880 (update loop) and L883–910 (metrics).
- Removed as unsupported by the source: "canonical Ouyang 2022 PPO recipe" and "matches [InstructGPT] precisely"; "value clipping is the Engstrom-et-al.-recommended trick"; "cleanest way to understand why the community pivoted to critic-free algorithms"; "new default trainers are GRPO/RLOO"; "remains the recommended reference for teaching"; verl "asymmetric+dual clip" comparison (not in this source).
- Not reported by the source: GPU count, global batch, and TRL version for the TL;DR benchmark run; any ablation of the defaults.
