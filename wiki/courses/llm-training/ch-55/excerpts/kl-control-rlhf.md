---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-04-23"
---

# Excerpt: KL-control in RLHF — why verl puts KL in the reward

**Source library:** `wiki/raw-data/llm-training/papers/kl-control-rlhf.md`
**Artifact:** the Stiennon/Ouyang/Korbak framework; k1/k2/k3 estimator math; reward-shaping vs in-loss KL.

---

## Why this excerpt exists in ch-55

Ch-55 §3 and §5 both claim that verl applies KL-to-reference as a *reward shaper*, not a loss term. This excerpt is the theoretical grounding: why reward-shaping is the right choice for PPO/GRPO, and why the K3 estimator is the modern default.

---

## Core insight from the source

> Standard RLHF is not "pure RL" but KL-regularized RL — the objective is `E[r(x,y)] − β · KL(π‖π_ref)`, which is mathematically equivalent to variational inference over a target distribution `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`.

So β isn't a "fudge factor" — it's the inverse temperature of a tilted posterior the policy is amortized-sampling from. verl's `beta_kl` plays the same role whether you call it a penalty or a Bayesian prior strength.

---

## The canonical RLHF objective

From §Key Contributions:

> `J(φ) = E_{(x,y)~π_φ}[ r_θ(x,y) − β · log( π_φ(y|x) / π_SFT(y|x) ) ] + γ · E_{x~D_pretrain}[log π_φ(x)]`

The KL is **added to the per-token reward**, then standard PPO is run. β ≈ 0.02 in InstructGPT (in reward-scale units); practitioners tune in [0.01, 0.5]. verl defaults to β ≈ 0.04 — the DeepSeekMath/[[grpo]] value.

---

## The three KL estimators

Schulman's blog, quoted in the source:

- `k1 = log(π/π_ref)` — unbiased but high variance, can be negative.
- `k2 = 0.5 · (log(π/π_ref))^2` — biased (always positive), low variance.
- `k3 = (π_ref/π) − 1 − log(π_ref/π)` — **unbiased AND always ≥ 0**; recommended, used in modern TRL / OpenRLHF / verl.

verl's `kl_penalty(logprob, ref_logprob, mode)` exposes all three; default is `k3`. When `actor/kl_loss` under k3 ever goes negative, the ref forward pass is broken — a useful correctness invariant.

---

## Why reward, not loss

From §Key Contributions / Why reward + not loss:

> adding KL to the reward keeps the PPO advantage estimator well-defined per token; adding KL to the loss breaks the advantage-based policy gradient and empirically trains worse.

This is the load-bearing claim for verl's design. verl subtracts `β · kl_penalty(...)` from `token_level_rewards` *before* `compute_grpo_outcome_advantage` runs, so the GRPO group-baseline z-score sees a KL-adjusted reward. The advantage absorbs the KL signal through the same baseline mechanism as RM reward.

TRL-GRPO deliberately chooses the other path (add K3 to the loss). Ch-55 §3 flags that the two choices give different optimizer dynamics on long-tailed KL distributions — same algebra on the mean, different gradients on the tail.

---

## Korbak reformulation — the Bayesian view

`argmax_π E_π[r] − β · KL(π‖π_ref)` has closed-form optimum `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`.

Interpretation: RLHF is not reward maximization. It is *amortized sampling* from the reward-tilted posterior. Large β means the prior π_ref dominates (model barely moves); small β means the RM signal dominates (model wanders, reward hacks).

---

## Failure modes as β-regime diagnosis

From §Technical Details:

- **β too small** → reward hacking and mode collapse (the policy finds degenerate reward-max outputs that the SFT prior would never have generated).
- **β too large** → policy cannot depart from SFT, ignores RM signal (reward-margin barely moves).

These are the two ends of the ch-46 DPO sweep (β ∈ {0.05, 0.1, 0.3}) — the same KL-budget phenomenology plays out whether the KL is an explicit term (PPO/GRPO) or an implicit one (DPO's `r̂_θ = β · log(π_θ/π_ref)`).

---

## Connections

- [[verl-ppo-loss]] — `ppo_kl` is the K1 *monitor*, not a regularizer; the regularizer is in `kl_penalty(..., "k3")`.
- [[entropy-logging-patterns]] — cross-framework table showing verl/OpenRLHF reward-shape while TRL-GRPO adds to loss.
- [[grpo]] — Eq. 3 has KL in the loss (paper); verl intentionally moves it to the reward.
- [[reward-hacking-taxonomy]] — "β too small" is the upstream cause of most hacking modes.
- Schulman's "Approximating KL Divergence" blog — the k1/k2/k3 derivations.
