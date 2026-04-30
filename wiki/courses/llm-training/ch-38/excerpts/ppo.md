---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ppo.md
source_url: https://arxiv.org/abs/1707.06347
created_at: "2026-04-23"
---

# Excerpt: PPO — the clipped surrogate ch-38 §2 derives

**Source library:** `wiki/raw-data/llm-training/papers/ppo.md`
**Artifact:** `L^{CLIP}`, `L^{CLIP+VF+S}`, GAE λ, canonical hyperparameters.

---

## Why this source anchors ch-38

[[ppo]] is the algorithmic backbone of every chapter from here through ch-45 (GRPO, RLOO, etc. all preserve the `L^{CLIP}` surrogate). The paper's two load-bearing contributions — the pessimistic min over `r·A` and `clip(r)·A`, and the shared actor-critic objective — propagate directly into the InstructGPT PPO-ptx recipe and the Llama-2 appendix.

---

## The identity that drives everything

From the source:

> `L^CLIP(θ) = E_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]`

Ch-38 §2 derives this by casework on the sign of `Â_t` and whether `r_t` is inside or outside the trust region. The key observation is that the `min` is *pessimistic* — it only lets the surrogate improve when both terms agree. Outside the trust region on the "good" side, the clipped term saturates and its gradient is zero; we refuse to push further.

---

## Canonical hyperparameters quoted in ch-38

From the source table:

> Clip ε 0.2 — Value-loss coef c1 1.0 — Entropy coef c2 0.01 — Epochs K 3–10 — Minibatch 64 — GAE λ 0.95 — Discount γ 0.99 — LR 3e-4 (Adam)

Ch-38 §2 carries `ε=0.2`, `c_1=1.0` into the `L^{CLIP+VF+S}` derivation. In §4 it notes that InstructGPT keeps ε=0.2 but drops the entropy bonus (c_2=0) because the `−β·KL(π‖π_SFT)` term already regularizes.

---

## Token-level adaptation

The source lays out the RLHF specialization explicitly:

> Original PPO works at timestep level; RLHF implementations apply the clipped ratio per token and average (token-level). The KL penalty is usually added to the per-token reward before computing advantages (sometimes called "KL-reward" style). Entropy is typically disabled or set very small because the Bradley-Terry KL penalty already regularizes the policy.

This is exactly the pattern ch-38 §7 finds across [[trl-ppo]], [[openrlhf-ppo]], [[verl-ppo-loss]]: token-level ratio, KL added to reward, no entropy.

---

## What ch-38 keeps, changes, drops from Schulman 2017

Keeps: the clipped surrogate, the `min` pessimism, GAE λ=0.95, value-loss clipping. Changes: γ=1.0 (not 0.99), drop entropy bonus, token-level ratio. Drops: MuJoCo-specific observation normalization.
