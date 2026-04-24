---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/maximum-entropy-rl.md
source_url: https://arxiv.org/abs/1801.01290
created_at: "2026-04-23"
---

# Excerpt: Haarnoja 2018 — SAC and max-entropy RL

**Source library:** `wiki/raw-data/llm-training/papers/maximum-entropy-rl.md`
**Authors:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
**Venues:** ICML 2018 (SAC); arXiv:1812.05905 (SAC v2 with auto α)
**Year:** 2018

---

## Why this source anchors ch-43

SAC is the theoretical ancestor of every entropy knob in LLM-RL. The read chapter's §5 ("Max-ent RL ancestry") uses Haarnoja's objective as the reference point and then explicitly enumerates what LLM-RL kept and what it dropped. This excerpt makes the inheritance precise by writing out SAC's objective, soft-Bellman recursion, and auto-α rule — then mapping each piece to its LLM-RL fate.

---

## The max-entropy objective

Source lines 16–17:

> `J(π) = Σ_t E[r(s_t, a_t) + α · H(π(·|s_t))]`

This is the load-bearing equation. Every policy update maximizes reward *plus* per-step entropy, weighted by a temperature `α`. `α = 0` recovers standard RL; `α → ∞` gives uniform exploration regardless of reward.

The resulting optimal policy has a closed form: `π*(a|s) ∝ exp(Q(s,a) / α)`. This is a Boltzmann distribution at "temperature" α — the same shape that appears in Korbak's Bayesian view of RLHF (`π* ∝ π_ref · exp(r/β)`) but with a uniform prior instead of a reference-policy prior.

---

## The soft Bellman recursion

Source lines 18–20:

> Soft value functions: `V(s) = α · log ∫ exp(Q(s,a)/α) da` — log-sum-exp over actions instead of max.
> Optimal-policy form: `π*(a|s) = exp((Q(s,a) − V(s)) / α)`.

The move from standard RL to max-ent RL replaces the `max` operator in the Bellman update with a temperature-weighted log-sum-exp. In a discrete-action / finite-vocab setting, `V(s) = α · log Σ_a exp(Q(s,a)/α)` — which is exactly `softmax → log-sum-exp`. As `α → 0`, `α · log Σ exp(·/α) → max`; as `α → ∞`, it → mean. So α interpolates between greedy and uniform.

For an LLM, the "action" is the next token and `Q(s, a)` is the implicit quality of choosing that token given the context. The soft-Bellman view says: a *max-ent optimal* LLM policy would be

```
π*(y_t | y_<t, x) ∝ exp( Q(y_<t, y_t | x) / α )
```

i.e. Boltzmann over tokens. This is recognizable — it is the softmax head with α as an inverse-temperature. The `Q/α` gap is what the RLHF objective *tries* to produce via the tilt `r/β + log π_ref`.

---

## Automatic α tuning (SAC v2)

Source lines 23, 26, 37:

> Automatic α tuning (v2): solve `α = argmin_α E[−α · log π(a|s) − α · H̄]`; gradient-descent on α with a target entropy like `H̄ = −dim(A)`.

The crucial innovation of SAC-v2: instead of picking a fixed α, pick a **target entropy** `H̄` and let α float so that `H(π) ≈ H̄` at optimum. The Lagrangian-inspired loss on α is

```
L(α) = E_{a~π(·|s)}[ −α · log π(a|s) − α · H̄ ]
     = α · ( H(π) − H̄ )
```

so `∂L/∂α = H(π) − H̄`: gradient descent raises α when entropy is below target (forcing more exploration) and lowers α when entropy is above target (letting the policy commit).

This is the ancestor of Cui 2025's covariance-based interventions in §1 of the read chapter — same goal (keep H above a floor), different surgical tool (Clip-Cov / KL-Cov on high-covariance tokens rather than symmetric α adjustment).

---

## What LLM-RL inherited

The read chapter's §5 enumerates three things:

1. **The loss form.** A3C/PPO's `+ c_H · H(π)` ([[entropy-regularization-ppo]]) is the small-α, on-policy limit of SAC's objective. If you Taylor-expand the soft-Bellman V around the standard Bellman V in α, the first-order correction is the entropy bonus with coefficient α. Production LLM-RL keeps this term (usually with `c_H = 0` in GRPO; `c_H ∈ [0, 1e-2]` in PPO).

2. **The target-entropy idea.** Cui 2025 does not compute `H̄ = −dim(A)`, but the "keep H above a floor" framing is SAC-v2's with a different enforcement mechanism. The `H < 0.1` nat collapse threshold is a target-entropy floor in disguise.

3. **The Boltzmann-tilt view.** Korbak's `π* ∝ π_ref · exp(r/β)` ([[kl-control-rlhf]]) is structurally identical to SAC's `π* ∝ exp(Q/α)` — just with π_ref playing the role of a non-uniform prior.

---

## What LLM-RL dropped

Source line 40 plus the structural differences:

> Contrast with on-policy PPO: SAC uses off-policy replay + soft Q learning; LLM RL is largely on-policy so max-ent becomes an entropy-bonus approximation.

Concretely:

- **Off-policy replay.** SAC trains from a replay buffer. LLM-RL generates fresh rollouts every batch (on-policy) or uses very short replay windows ([[openrlhf-entropy-debugging]] framework defaults). The trade: sample efficiency for training stability in the huge-context / huge-action-space regime.
- **Soft Q critics.** SAC has twin Q networks trained to match the soft-Bellman target. Modern GRPO / REINFORCE-family LLM-RL has no value head at all — advantages are group-relative z-scores. So the entropy term has no critic to interact with; it is purely an actor-loss regularizer.
- **Auto α.** No production LLM stack runs an online α controller. β is tuned by hand or by an adaptive-KL controller that targets per-batch KL, not per-token entropy.
- **True max-ent semantics.** The full soft-Bellman recursion (entropy inside Q, log-sum-exp instead of max) is not used in any production LLM stack. The `+ c_H · H(π)` actor-loss term is only the small-α limit.

---

## Robustness, briefly

Source line 27:

> Robustness: max-ent policies tolerate environment perturbations better than deterministic optimal policies — an argument that reappears in LLM RL where a collapsed policy is brittle to distribution shift.

This is the theoretical reason practitioners care about entropy at all. A max-ent optimal policy covers multiple good-reward modes; a greedy optimal policy commits to one. Under distribution shift (new prompts, different tokenizer, subtly-different RM), the max-ent policy degrades gracefully while the greedy one breaks abruptly. This is the "RL trains a brittle policy" failure that Cui 2025 documents: once `H < 0.1`, the model has lost the modal diversity that would let it cover unseen prompt shapes at deployment.

---

## Connections

- Read-chapter §5 uses this source's objective, soft-Bellman recursion, and auto-α rule verbatim.
- [[excerpts/entropy-mechanism-llm-rl]] — the modern, LLM-specific version of the same control problem.
- [[excerpts/kl-control-rlhf]] — the Korbak Bayesian view is the RLHF analogue of SAC's `π* ∝ exp(Q/α)`.
- [[entropy-regularization-ppo]] — the on-policy small-α limit that LLM-RL actually runs.
- ch-37 (Policy-Gradient Foundations) — the REINFORCE/AC framing underneath both SAC and PPO.
