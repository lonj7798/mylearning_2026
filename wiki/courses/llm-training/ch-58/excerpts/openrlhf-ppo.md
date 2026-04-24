---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF PolicyLoss — why OpenRLHF owns the §5 "production middle" leaf

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md`
**Artifact:** `openrlhf/models/loss.py` L68–168 (`PolicyLoss`), `openrlhf/trainer/ppo_trainer.py` L~172 (`AdaptiveKLController` wiring).

---

## Why this source defines ch-58 matrix rows 4, 8, 10, 12, 14

Ch-58 §5 routes "standard PPO/GRPO at 8–64 GPUs" to OpenRLHF. Every leaf on that branch is attested here.

## Row 4 — "PPO: PolicyLoss module + asymmetric clip + dual-clip + IS"

Source code:

> ```python
> class PolicyLoss(nn.Module):
>     def __init__(self, clip_eps_low=0.2, clip_eps_high=0.2, dual_clip=None,
>                  token_level_loss=True, policy_loss_type="ppo",
>                  enable_vllm_is_correction=False,
>                  vllm_is_truncated_threshold=None,
>                  vllm_is_correction_type="tis"):
> ```

The `nn.Module` carries the IS-correction configuration *with the loss*. Ch-58 §1 says this is the "state travels with the object" design — contrasted against verl's free-function + registry. A learner reading ch-58 §2 matrix cell "PolicyLoss module + asym + dual + IS" should recognize that the five `__init__` parameters here are exactly those four features.

## Row 10 — "vLLM IS correction: tis / seq-mask-tis / icepop"

Source code:

> ```python
> if self.vllm_is_correction_type == "icepop":
>     vllm_is = torch.exp(log_ratio_v).detach()
>     vllm_is = vllm_is * ((vllm_is >= low) & (vllm_is <= high))
>     loss = vllm_is * loss
> elif self.vllm_is_correction_type == "seq-mask-tis":
>     seq_log_ratio = masked_mean(log_ratio_v, action_mask, dim=-1)
>     seq_is = torch.exp(seq_log_ratio)
>     seq_mask = (seq_is >= low) & (seq_is <= high)
>     vllm_is = torch.exp(log_ratio_v).detach()
>     loss = seq_mask.unsqueeze(-1) * vllm_is * loss
> else:  # "tis"
>     vllm_is = torch.exp(log_ratio_v).clamp(min=low, max=high).detach()
>     loss = vllm_is * loss
> ```

Ch-58 matrix row 10 cell "tis / seq-mask-tis / icepop + vllm_kl" is this exact three-mode switch. §6 graduation criterion "you hit the vLLM-IS-correction wall" routes here.

## Row 12 — "GSPO (sequence-level ratio)"

Source code:

> ```python
> elif self.policy_loss_type == "gspo":
>     base = rollout_log_probs if self.enable_vllm_is_correction else old_log_probs
>     log_ratio = log_probs - base
>     ratio = (log_ratio * action_mask).sum(-1) / action_mask.sum(-1)
>     ratio = ratio.exp().unsqueeze(-1) * action_mask
> ```

Ch-58 matrix row 12 cell "policy_loss_type='gspo' (native)" is this one-line branch. This is one of the reasons the production-middle leaf routes here instead of TRL: the GSPO branch is simpler in OpenRLHF than in TRL's GRPO `_compute_loss` monolith.

## Row 8 — "KL location: reward shaping via AdaptiveKLController"

Source:

> ```python
> # openrlhf/trainer/ppo_trainer.py
> self.kl_ctl = (AdaptiveKLController(init_coef, target, horizon)
>                if adaptive else FixedKLController(init_coef))
> # ... each step, per-token reward is shaped as:  reward_t -= kl_ctl.value * kl_t
> self.kl_ctl.update(status["kl"], rollout.batch_size * n_samples_per_prompt)
> ```

The adaptive controller is the InstructGPT recipe — and ch-58 §6 "Graduate from TRL" criterion explicitly cites this: "TRL has one for PPO but it's minimal; OpenRLHF's is the InstructGPT-faithful implementation". This source is that citation.

## Row 14 — "async / partial rollout: ppo_trainer_async.py + partial_rollout_enable"

Companion source `async-rollout.md` (ch-58 already cites) documents:

> `rollout_queue`: `ray.util.queue.Queue`, capacity 1–2.
> `rollout_slots`: companion queue carrying `global_step` tokens (backpressure).
> `vllm_lock`: `ray` asyncio.Lock to serialize weight-broadcast vs generate.

But this `openrlhf-ppo.md` source is also where `vllm_kl` is introduced:

> Two KL metrics returned: `ppo_kl` (train-vs-old, monitoring), `vllm_kl` (rollout-vs-train, diagnoses sampler drift).

Ch-58 §3 crib sheet row "rollout-vs-train KL" cites `vllm_kl` as OpenRLHF-exclusive — that attestation is here.

## Row 11 — "DAPO asymmetric clip: clip_eps_low / clip_eps_high"

Already covered by the `__init__` signature above. Note that OpenRLHF uses *the same* names as DAPO and verl (`clip_eps_low`, `clip_eps_high`) — this is why ch-58 §1 can claim "algebra is the same". The clamp line is verbatim:

> ```python
> surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
> ```

## What ch-58 inherits verbatim

- `clip_eps_low`, `clip_eps_high`, `dual_clip`, `policy_loss_type`, `enable_vllm_is_correction` as the §2 matrix vocabulary for rows 4, 10–12.
- `AdaptiveKLController` as the name the §6 graduation criterion cites.
- `vllm_kl` as the §3 crib sheet metric name.
- The three IS-correction mode names (tis / seq-mask-tis / icepop) verbatim into row 10.

## Connections

- **[[openrlhf-dpo]]** — the sister loss module for offline DPO; matrix row 6.
- **[[async-rollout]]** — the architectural counterpart; matrix row 14.
- **[[verl-ppo-loss]]** — ch-58 §1's "algebra is the same" uses these two sources as the attested pair.
- **[[entropy-logging-patterns]]** — the cross-framework table where `ppo_kl` and `vllm_kl` cells originate.
