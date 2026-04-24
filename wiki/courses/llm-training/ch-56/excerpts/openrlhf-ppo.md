---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF — PolicyLoss and the loss-object-as-logger idiom

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-ppo.md`
**Files:** `openrlhf/models/loss.py` (lines 68–168), `openrlhf/trainer/ppo_trainer.py` (~line 172)
**Version/commit:** `main` branch (fetched 2026-04-21)

---

## Why this source anchors ch-56

`PolicyLoss` is the architectural fulcrum of OpenRLHF. Every other design
decision — the AdaptiveKLController living *outside* the loss, the
`(loss, clip_ratio, ppo_kl, vllm_kl)` return tuple, the three vLLM IS
correction modes — is visible in this one `nn.Module`. Read it once and
you have read the stylistic difference between OpenRLHF and TRL / verl.

---

## The load-bearing five lines

Of the 100-line `PolicyLoss.forward` body, five lines are doing the real work:

```python
log_ratio = log_probs - old_log_probs
ratio = log_ratio.exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
loss = -torch.min(surr1, surr2)
```

This is Schulman 2017 PPO with asymmetric clip. Everything else (`gspo`
branch, `dual_clip`, three vLLM IS modes, `vllm_kl` metric) is an
optional modifier. The point is pedagogical: **the PPO loss is actually
five lines**; the rest is the cost of being a production framework.

---

## The two KLs, attested

Source lines 42–44 and 76 show the loss returns two distinct KL metrics:

> `ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)`
>
> `vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)`

- `ppo_kl` is K1-style train-vs-old-logprob ([[entropy-logging-patterns]]).
  Diagnoses whether the trust region is still holding.
- `vllm_kl` is rollout-engine-vs-trainer-forward. Diagnoses whether
  bf16 vLLM decoding and fp32 trainer forward pass have drifted enough
  to need IS correction.

No other framework in ch-55..ch-57 returns both from the loss. TRL
computes `objective/kl` on the rollout side; verl exposes a `k1/k2/k3`
switch but in `core_algos.py`, not inside the loss object.

---

## Where KL-to-reference actually lives

Source §Context is explicit:

> KL is *not* in the loss — it's added to per-token rewards through
> `kl_ctl` outside.

The instantiation is around line 172 of `ppo_trainer.py`:

```python
self.kl_ctl = (AdaptiveKLController(init_coef, target, horizon)
               if adaptive else FixedKLController(init_coef))
```

This split — loss handles ratio + clip; trainer handles KL-to-ref via
reward shaping — is the InstructGPT recipe ([[rlhf-instructgpt]]
Eq. 2) taken seriously. [[kl-control-rlhf]] explains why: KL-in-reward
keeps GAE (and for GRPO, the group-relative z-score) well-defined.

---

## The three vLLM IS modes — when each applies

Source "What to notice" enumerates:

- **`tis`** — Truncated IS; clamp per-token ratio into `[low, high]`.
  Safe default; what you want if `vllm_kl` is small and bounded.
- **`seq-mask-tis`** — mask the entire sequence if its mean IS weight
  exits `[low, high]`. Aggressive; drops whole rollouts but keeps
  surviving sequences cleanly on-policy.
- **`icepop`** — zero per-token IS weight outside `[low, high]`. Used
  when token-level drift is heterogeneous (some tokens fine, others
  catastrophic).

The lesson for ch-56 §2.1: the correction type is a function of *how*
your rollouts drift, not just whether they do.

---

## Connections

- [[excerpts/openrlhf-dpo]] — the DPO counterpart; `DPOLoss` has the
  same nn.Module style.
- [[excerpts/entropy-logging-patterns]] — cross-framework KL metric
  comparison table; OpenRLHF's `ppo_kl` + `vllm_kl` slot into the K1
  column.
- [[excerpts/async-rollout]] — the IS modes exist because async
  rollout exists.
- Host chapter: [[ch-56]] §2.
- Forward to [[ch-57]] (TRL) — same loss, inlined into the train loop;
  no nn.Module wrapper.
