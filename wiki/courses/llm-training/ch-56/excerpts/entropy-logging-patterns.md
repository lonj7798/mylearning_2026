---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md
source_url: https://github.com/OpenRLHF/OpenRLHF, https://github.com/verl-project/verl, https://github.com/huggingface/trl
created_at: "2026-04-23"
---

# Excerpt: Cross-framework KL/entropy logging — OpenRLHF slotted into the table

**Source library:** `wiki/raw-data/llm-training/frameworks/entropy-logging-patterns.md`

---

## Why this source anchors ch-56

The only way to understand OpenRLHF's logging choices is in contrast
to verl and TRL. This source is the framework-comparison table that
locates OpenRLHF's `ppo_kl` and `vllm_kl` inside the broader
K1/K2/K3 estimator taxonomy and the reward-shaping-vs-loss-regularizer
debate.

---

## The three KL estimators, attested

Source §What to notice:

> - **K1** (`logπ − logπ_ref`): simple, used by verl reward shaping
>   and TRL `objective/kl`. Biased (not non-negative).
> - **K2** (`0.5·Δlogp²`): variance-reduced, always non-negative.
>   TRL `approxkl_stats`.
> - **K3** (`exp(−Δlogp) + Δlogp − 1`): unbiased, always ≥ 0.
>   verl `kl_penalty="k3"` and TRL GRPO loss term — this is the
>   modern default.

OpenRLHF's `ppo_kl` is **K1** — `masked_mean(-log_ratio.detach(), ...)`.
Its AdaptiveKLController's reward-shaping term is also K1-style. This
is a legacy choice (InstructGPT used K1); the modern recommendation is
K3 because it is non-negative by construction, which simplifies
controller logic.

---

## KL placement — reward vs loss

Source comparison table row "KL-to-ref where":

| Framework | KL placement |
|---|---|
| verl | reward shaping |
| OpenRLHF | reward shaping (`kl_ctl`) |
| TRL PPO | reward shaping (`kl_ctl`) |
| TRL GRPO | loss term `β·K3` |

OpenRLHF sits with verl and TRL-PPO on the **reward-shaping** side.
The argument for this side, per [[kl-control-rlhf]]: KL in reward
keeps per-token advantage estimators well-defined. TRL GRPO dissents
because GRPO has no per-token value and a loss-side KL is easier to
reason about when the advantage is a sequence-level z-score.

---

## vllm_kl — the sampler-drift metric

Source "What to notice" § bullet 4:

> **OpenRLHF adds `vllm_kl`:** mean logprob difference between the
> rollout engine and the training forward pass. If this diverges from
> zero, training-inference mismatch is active and PPO will destabilize
> unless IS correction (TIS/iCEPO) is enabled.

This is unique to OpenRLHF among the three frameworks compared. verl
uses IS weights (same math) but does not expose a named scalar; TRL
does not handle rollout-vs-train mismatch at all (TRL generates on the
trainer-model forward, so there is no mismatch to measure).

---

## Entropy — a metric, never a loss

Source §Key Points:

> Entropy bonuses are not default anywhere. Cui 2025 ("Entropy
> Mechanism of RL for LLMs") and DAPO argue for token-level entropy
> masking or entropy annealing; TRL exposes this via
> `top_entropy_quantile`.

OpenRLHF's `entropy_coef = 0.0` default matches the other two
frameworks. The KL-to-ref term is the only explicit regularizer.
Entropy is logged as a failure-signal proxy, not optimized directly.

---

## The entropy-collapse signature

Source §What to notice:

> Entropy collapse signature is identical across frameworks:
> `entropy` falls ≥30% in <100 steps, `ppo_kl` spikes ≥0.1, `clipfrac`
> pegs to 1. Use whichever framework metric maps to these three.

Ch-56 §7 uses this exact triad: `ppo_kl > 0.1` + `clipfrac = 1` is the
diagnostic; the fix is raising `β_init` or lowering the controller's
`K_beta`.

---

## Connections

- [[excerpts/openrlhf-ppo]] — the loss object that emits `ppo_kl`
  and `vllm_kl`.
- [[excerpts/kl-control-rlhf]] — the theoretical case for
  reward-shaping KL over loss-side KL.
- [[excerpts/openrlhf-entropy-debugging]] — the community-standard
  triage order that uses these metrics.
- Host chapter: [[ch-56]] §2 + §7.
- Forward to [[ch-57]] (TRL) — TRL GRPO's `β·K3` loss term is the
  dissenting row in the table.
