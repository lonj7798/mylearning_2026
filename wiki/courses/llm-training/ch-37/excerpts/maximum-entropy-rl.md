---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/maximum-entropy-rl.md
source_url: https://arxiv.org/abs/1801.01290
created_at: "2026-04-23"
---

# Excerpt: Max-Ent RL (SAC) — the principled ancestor of LLM entropy bonuses

**Source library:** `wiki/raw-data/llm-training/papers/maximum-entropy-rl.md`
**Artifact:** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* — Haarnoja, Zhou, Abbeel, Levine, 2018 (SAC v1), and the 2018 follow-up that adds automatic α tuning (SAC v2, arXiv:1812.05905).

---

## Why this source anchors ch-37

Ch-37 §5's position — "the entropy term is usually redundant in RLHF because KL-to-ref already regularises; it becomes a real regulariser when exploration is the bottleneck and KL-to-ref is too weak" — needs a principled account of when and why an entropy bonus helps. SAC is that account. It derives the entropy-augmented objective from first principles, shows the closed-form optimal policy, and introduces automatic α tuning, which is the right mental model for "target-entropy schedule" strategies in reasoning RL.

---

## The max-ent objective

From the source (lines 22, §Key Contributions):

> Max-ent objective: `J(π) = Σ_t E[r_t + α H(π(·|s_t))]` — rewards are valued together with high-entropy (random) behavior.

This is the *augmented* objective. The vanilla RL objective is `J(π) = Σ_t E[r_t]`; the max-ent objective adds a per-step entropy bonus weighted by `α`. The consequence is that the optimal policy is no longer deterministic — it is a Boltzmann distribution whose temperature is `α`.

> Notice: the entropy bonus is *per step*, not per trajectory. For LLM-RL this distinction matters because adding a per-token entropy bonus weights high-uncertainty tokens (typical of mid-sequence reasoning) more than low-uncertainty tokens (punctuation, formatting). A per-sequence entropy bonus would not have this property. Most LLM-RL implementations that add entropy add it per token, following the SAC convention.

---

## The closed-form optimal policy

From the source (line 24, §Key Contributions):

> Optimal-policy form: `π*(a|s) = exp((Q(s,a) − V(s)) / α)`.

Equivalently, `π*(a|s) ∝ exp(Q(s,a) / α)` with `V(s) = α · log ∫ exp(Q(s,a)/α) da` acting as the log-partition. This is exactly the Gibbs / softmax-with-temperature form. Two limits:

- `α → 0`: recovers the deterministic greedy policy `π*(a|s) = 𝟙{a = argmax_a Q(s,a)}`.
- `α → ∞`: recovers the uniform policy.

For intermediate `α`, the policy spreads mass smoothly over actions in proportion to `exp(Q/α)`.

> Notice: this form is identical — not just analogous — to the KL-regularised RLHF target distribution. If you take [[kl-control-rlhf]]'s derivation that the optimal KL-constrained policy is `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`, it is SAC's `π*(a|s) ∝ exp(Q(s,a)/α)` with `Q(s,a)/α = log π_ref(a|s) + r(s,a)/β`. The KL-to-ref regulariser *is* an entropy regulariser in disguise, with `π_ref` acting as the prior measure. This is the formal basis of ch-37 §5's "KL-to-ref already regularises entropy" claim.

---

## Automatic α tuning (SAC v2)

From the source (line 26, §Key Contributions):

> Automatic α tuning (v2): solve `α = argmin_α E[−α · log π(a|s) − α · H̄]`; gradient-descent on α with a target entropy like `H̄ = −dim(A)`.

This is the key practical insight. Instead of fixing `α`, treat it as a variable to be tuned so that the *actual* policy entropy stays near a **target entropy** `H̄`. The loss `L(α) = E[−α (log π(a|s) + H̄)]` gradient-descends on `α` to satisfy `E[−log π(a|s)] ≈ H̄`.

The default target entropy in SAC v2 is `H̄ = −dim(action space)`. For continuous control with a d-dimensional action, this is a calibrated target. For discrete actions with |V| = 128K (LLM), the analog is some fraction of `log |V|` — but the paper does not prescribe the LLM adaptation.

> Notice: auto-α is what separates "principled max-ent regularisation" from "bandaid entropy bonus". A fixed `c2 = 0.01` entropy coefficient is a constant nudge regardless of whether the policy is at H = 0 (collapsed) or H = log|V| (uniform). An auto-α with target entropy `H̄` *tracks the policy's actual entropy* and only pulls when the policy is drifting below target. Ch-37 §5's "bandaid vs real regulariser" distinction is exactly this — fixed-coefficient is the bandaid, target-entropy is the real regulariser.

---

## Robustness argument

From the source (line 27, §Key Contributions):

> Robustness: max-ent policies tolerate environment perturbations better than deterministic optimal policies — an argument that reappears in LLM RL where a collapsed policy is brittle to distribution shift.

The argument is that a Boltzmann-distributed policy has non-trivial mass on sub-optimal actions, so when the environment (or prompt distribution) shifts, the policy has "options" already in support and degrades gracefully. A deterministic optimal policy has zero mass on everything but the best action; a small environment shift can render that action suboptimal with no fallback.

> Notice: for LLMs this translates to "a collapsed (low-entropy, near-deterministic) policy fails badly on OOD prompts". The empirical version of this argument appears in [[entropy-mechanism-llm-rl]] and [[openrlhf-entropy-debugging]] — teams report that entropy-collapsed RL policies produce verbatim templates on novel prompts, while higher-entropy ones generalise. SAC's theoretical argument and the LLM-empirical argument point in the same direction.

---

## Why this matters for LLM-RL specifically

From the source (line 38, §Technical Details):

> Why it matters for LLMs: SAC's auto-α is the direct ancestor of "keep policy entropy above H̄ during GRPO" strategies; modern LLM-RL papers ([[entropy-mechanism-llm-rl]]) are rediscovering that a fixed β on H(π) is inferior to a target-entropy-based adaptive term.

The "rediscovery" framing is accurate — the LLM-RL community is slowly adopting SAC's auto-α machinery after initially using fixed-coefficient entropy bonuses. This matches ch-37 §5's position that entropy-bonus-as-bandaid is the default failure mode; the principled version is target-entropy scheduling.

> Notice: "target-entropy scheduling" in LLM RL is not yet a standardised technique. Different teams use different target definitions — per-token entropy floor, per-prompt entropy floor, per-batch average entropy target, running-EMA of SFT model entropy. The SAC v2 recipe is `H̄ = −dim(A)` by analogy, but the LLM analogue is not worked out. This is an open research area; ch-46 (track capstone) is a natural place to instantiate it.

---

## Soft Bellman equation and the KL-regulariser connection

From the source (lines 35–36, §Technical Details):

> Soft Q update: `L_Q(θ) = E[(Q_θ(s,a) − (r + γ (Q_target(s',a') − α log π(a'|s'))))^2]`.

The `−α log π(a'|s')` term in the target is what makes the Bellman operator "soft": the next-state value is `E_a[Q(s',a') − α log π(a'|s')] = E_a[Q(s',a')] + α H(π(·|s'))`. The soft value function `V(s) = E_a[Q(s,a) − α log π(a|s)]` is the max-ent analog of `V(s) = max_a Q(s,a)`.

> Notice: in the RLHF / PPO world the equivalent is the per-step value target `V_target = r_t + γ V(s_{t+1})`, with the KL-to-ref penalty `−β log(π/π_ref)` playing the role of SAC's `−α log π(a|s)`. Same structure, different labels — which is why GAE-with-KL-in-reward and soft-Q-learning often produce similar training dynamics on similar problems.

---

## What ch-37 keeps from this source

- The max-ent objective `J(π) = Σ_t E[r_t + α H(π(·|s_t))]` as the formal basis for entropy bonuses (§5).
- The closed-form `π*(a|s) ∝ exp(Q(s,a)/α)` and its structural equivalence to the KL-regularised RLHF target distribution (§5 framing).
- The target-entropy / auto-α story as the principled alternative to fixed entropy bonuses (§5).
- The "collapsed policy is brittle to distribution shift" robustness argument (§5, when entropy bonus is a real regulariser).

---

## Connections

- **ch-37 §5** — the entropy-term discussion.
- [[entropy-regularization-ppo]] — PPO's `c2 · H(π)` bonus is a fixed-α special case of SAC.
- [[entropy-mechanism-llm-rl]] — LLM-RL rediscovery of target-entropy.
- [[kl-control-rlhf]] — the KL-regularised RLHF target distribution is the same object as SAC's optimal soft policy.
- [[nathan-lambert-entropy-rl]] — practitioner view of entropy-collapse and why it matters.
- [[excerpts/lilianweng-rlhf]] — companion excerpt; Weng's "entropy bonus usually dropped" claim is a specific position SAC informs.
- [[excerpts/trpo]] — TRPO's KL constraint and SAC's entropy bonus are two faces of the same regularisation principle.
