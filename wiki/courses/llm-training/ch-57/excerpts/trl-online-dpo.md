---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-online-dpo.md
source_url: https://github.com/huggingface/trl/blob/main/trl/experimental/online_dpo/online_dpo_trainer.py
created_at: "2026-04-23"
---

# Excerpt: TRL OnlineDPOTrainer — two-sample-per-prompt, DPO-on-the-fly

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-online-dpo.md`
**Artifact:** `trl/experimental/online_dpo/online_dpo_trainer.py`. `training_step` body at ~L1089–1275 (sampling → judging → DPO loss). Rollout engines (`_generate_vllm`, `_generate_vllm_colocate`) at ~L585–893.

---

## Why this source anchors ch-57 §2

Ch-57 §2 introduces the three active TRL RL trainers: GRPO, DPO, online DPO. This excerpt explains why online DPO exists as a *separate trainer* rather than as a `loss_type` on `DPOTrainer`: the training loop structure is genuinely different — it needs on-policy sampling every step, not pre-computed preference pairs.

---

## The pair-construction block ch-57 §2 quotes

Source lines 1186–1230 (condensed):

```python
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

Three things to notice for ch-57 §2:

1. **`prompts = 2 * prompts`** — every prompt is repeated twice for generation. Two completions come back, both from the *current* policy. This is the "on-policy" move; offline DPO would load a preferred/rejected pair from disk.
2. **Argmax pair selection.** `mask = first_half >= second_half` declares which of the two completions scored higher; that one becomes `chosen`, the other `rejected`. No explicit Bradley-Terry sampling; whichever reward function returns the higher number wins.
3. **Same DPO loss algebra.** Once the pair is built, the loss is identical to offline DPO ([[dpo]] Eq. 7). Online DPO differs *only* in where the pair came from. `loss_type` exposes `sigmoid` (vanilla) and `ipo` (Azar L2-on-margin).

---

## Missing-EOS penalty

```python
if self.args.missing_eos_penalty is not None:
    rewards[~contain_eos_token] -= self.args.missing_eos_penalty
```

If a completion did not end with EOS (i.e., hit `max_new_tokens`), subtract a constant from its reward. This prevents the policy from gaming truncation — a common reward-hacking mode where the policy learns that cutting off mid-sentence sometimes scores higher than finishing. Ch-57 §2 treats this as a small-but-important config knob unique to the online setting.

---

## KL tracking — the canary for online-DPO collapse

Source lines 1244–1270:

```python
kl = logprobs - ref_logprobs
mean_kl = kl.sum(1).mean()
self.stats["objective/kl"].append(gather(mean_kl).mean().item())
non_score_reward = (-self.beta * kl).sum(1)
chosen_rewards   = self.beta * (chosen_logprobs_sum   - chosen_ref_logprobs_sum)
rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
margin = chosen_rewards - rejected_rewards
self.stats["rewards/accuracies"].append((margin > 0).float().mean().item())
```

Online DPO drifts from `π_ref` much faster than offline DPO because each step's pairs are drawn from the *current* policy. The KL-to-reference is the early warning that the policy has left the neighborhood where `π_ref` was a reasonable anchor. `rewards/accuracies` (margin > 0) is the second critical metric — it tells you whether the loss is actively getting the ordering right at all.

---

## Rollout engine choice

Three rollout paths live in the same file:

- `_generate` — plain HF `.generate()` on the training model. Correct, slow.
- `_generate_vllm` — remote vLLM server; requires a separate vLLM process and weight sync.
- `_generate_vllm_colocate` — vLLM in-process on the same GPUs as training.

Ch-57 §3's Accelerate-orchestration story applies directly: every step does two completions × batch-size generations per prompt, and any straggler rank blocks the collective. This is why online DPO is so much slower than offline DPO in practice, even though the loss is identical.

---

## Judge interface — LLM-as-judge drop-in

```python
# Config
OnlineDPOConfig(..., judge="some_model", reward_model=None)
```

If a `judge` is provided instead of a reward model, `_calculate_rewards_from_functions` calls a frozen LLM to pick the preferred completion. This is the pattern behind self-rewarding LM and Guo 2024's OAIF — the "reward" is just the judge's vote. Extensions for Nash-MD and XPO live in sibling files (`trl/experimental/nash_md/`, `trl/experimental/xpo/`).

---

## Why online DPO is not just a `DPOTrainer` `loss_type`

Ch-57 §2 makes this distinction explicit: offline DPO iterates over a dataset of fixed preference pairs; online DPO regenerates pairs every step. The training loop structure is genuinely different — buffer management, generation scheduling, reward-function calls per step — so it earns its own trainer. `loss_type` switches inside `_compute_loss` do not capture that.

The trade-off: online DPO closes the on-policy gap with PPO ([[on-off-policy-rlhf]]) but costs roughly `(1 + 2 × generation_cost)` times the GPU time per effective update. Whether it is worth it depends on how fast your policy drifts from `π_ref`.

---

## Attested implementation notes

- The trainer lives under `trl/experimental/` — no semver promise, may change signatures between releases.
- Sequence-level log-ratio (`sum(1)` over completion tokens) — DPO uses full-sequence margins, not per-token. This matters because length-normalized variants like SimPO require a different aggregation that online DPO does not expose.
- `beta` typical values: 0.1 (DPO-standard), 0.05 for more aggressive online drift, 0.2 for stricter anchoring.

---

## Connections to the rest of the track

- [[dpo]] — the offline ancestor; identical loss algebra.
- [[self-play-preference]] / [[self-rewarding-lm]] — the self-play instantiations this trainer supports.
- [[on-off-policy-rlhf]] — why online matters; ~80% of the PPO-vs-offline-DPO gap is distribution shift.
- [[trl-grpo]] — the other active online trainer in TRL; different objective, same Accelerate orchestration.
- [[openrlhf-dpo]] — OpenRLHF's DPO is offline-only; no equivalent online trainer there.
