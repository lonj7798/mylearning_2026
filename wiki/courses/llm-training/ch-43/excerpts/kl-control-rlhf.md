---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-04-23"
---

# Excerpt: KL-Control in RLHF — Jaques / Stiennon / Ouyang / Korbak

**Source library:** `wiki/raw-data/llm-training/papers/kl-control-rlhf.md`
**Primary lineage:** Jaques 2019 (Way Off-Policy) → Stiennon 2020 (Summarization) → Ouyang 2022 (InstructGPT) → Korbak 2022 (Bayesian view)
**Years:** 2019 / 2020 / 2022 / 2022

---

## Why this source anchors ch-43

The read chapter's §4 ("KL-to-reward vs KL-as-loss") rests on this lineage. The four papers answer four different questions:

1. **Jaques 2019** — *Why add a KL penalty at all?* To keep dialog agents fluent.
2. **Stiennon 2020** — *How do you combine it with a learned reward model?* Per-token KL on the reward stream.
3. **Ouyang 2022 (InstructGPT)** — *What is the canonical form?* `J = E[r] − β · KL(π ‖ π_ref) + γ · pretrain-mix-loss`, β ≈ 0.02.
4. **Korbak 2022** — *What is this objective, mathematically?* Exact variational inference against a tilted posterior.

Together they are the reason modern RLHF runs are "KL-regularized RL" and not "reward maximization".

---

## The canonical RLHF objective

Source lines 22–25:

> `J(φ) = E_{(x,y)~π_φ}[ r_θ(x,y) − β · log( π_φ(y|x) / π_SFT(y|x) ) ] + γ · E_{x~D_pretrain}[log π_φ(x)]`
> — KL is added to the **per-token reward**, then standard PPO is run.

Unpacked:

- `r_θ(x, y)` — reward model score at the terminal token of response `y`.
- `β · log(π_φ / π_SFT)` — per-token KL penalty (reverse KL, `π_new ‖ π_ref`).
- `γ · E[log π_φ(x)]` — the PPO-ptx pretraining-mix term that stops the post-training from eroding general capability.

The β coefficient sits at **~0.02** in InstructGPT, though production stacks tune it inside `[0.01, 0.5]` of the reward scale ([[openrlhf-entropy-debugging]]).

---

## The reward-stream trick

Source lines 42–43:

> Per-token reward used in TRL/OpenRLHF:
> `r̂_t = r_t − β · (log π_φ(y_t|y_<t,x) − log π_ref(y_t|y_<t,x))`
> with only the sequence-terminal `r_t` being the RM reward (zero elsewhere), and the KL term active on every token.

This is the operational form: at training time, compute per-token log-prob difference between current policy and reference, subtract β times that from the per-token reward, and feed the shaped reward into the advantage estimator (GAE / simple Monte-Carlo / group-relative). PPO then proceeds exactly as in classical RL — the KL "constraint" is absorbed into the reward.

The corresponding advantage formula is

```
A_t = Σ_{t' >= t} γ^{t'-t} · (r̂_{t'} + γ V(s_{t'+1}) − V(s_{t'}))
```

with the RM reward landing at terminal `T` and the KL term contributing every step. GAE λ < 1 discounts further.

---

## Why reward, not loss (for PPO)

Source line 33:

> Why reward + not loss: adding KL to the reward keeps the PPO advantage estimator well-defined per token; adding KL to the loss breaks the advantage-based policy gradient and empirically trains worse.

The issue: PPO's clipped surrogate `L_CLIP = E[min(r(θ) A, clip(r(θ), 1−ε, 1+ε) A)]` is derived under the assumption that A is an unbiased estimator of the token's advantage under the *shaped* reward. If you add `β · KL` *to the loss*, the advantage is still computed against `r_t`, so the gradient of the KL term does not have the trust-region protection that the PPO clip provides to the policy-gradient term. It empirically under-trains and overshoots β-dependent trajectories.

GRPO's convention ([[entropy-logging-patterns]] TRL GRPO excerpt) inverts this: KL goes *into the loss* as a k3 penalty. The read chapter's §4 explains why GRPO can get away with this — no value baseline, advantages are group z-scores, so the PPO derivation's assumption does not apply.

---

## The Korbak view — Bayesian inference

Source lines 30–31:

> Korbak's reformulation: `argmax_π E_π[r] − β · KL(π ‖ π_ref)` has closed-form optimum `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)`. RLHF is therefore amortized sampling from this tilted posterior — not unconstrained reward maximization.

Derivation in one line: form the Lagrangian `L(π) = E_π[r] − β · KL(π‖π_ref)`, take variational derivative w.r.t. `π(y|x)`, set to zero, enforce normalization. You get a Boltzmann-shaped tilt of the reference distribution.

The `r(x,y)/β` term plays the role of a log-likelihood; `π_ref` plays the role of a prior; `π*` is the posterior. RLHF is then a way to *approximate sampling* from π* using PPO as an amortized sampler.

Two implications:

1. **β sets the tilt strength.** Smaller β = larger tilt = policy moves further from reference. Reward hacking is what happens when β is too small to hold the policy near π_ref's support.
2. **DPO is the same tilt, solved exactly.** DPO's implicit reward `r_θ(x, y) = β · log(π_θ / π_ref)` is exactly the log-likelihood ratio of the tilted posterior. So DPO inherits the same overoptimization laws.

---

## KL estimator choice

Source lines 26–29:

> KL estimators (Schulman's blog): three unbiased estimators for `log(p/q)`-style KL:
> - `k1 = log(π/π_ref)` — unbiased but high variance, can be negative.
> - `k2 = 0.5 · (log(π/π_ref))^2` — biased but low variance.
> - `k3 = (π_ref/π) − 1 − log(π_ref/π)` — unbiased AND always ≥ 0; recommended, used in modern TRL / OpenRLHF.

Defer full derivation to [[excerpts/john-schulman-kl-tricks]]. The essential point for KL-control: *which estimator you choose is independent of whether you route KL through the reward or the loss*. You can do k3-in-reward (modern TRL PPO with adaptive controller) or k1-in-reward (early verl default). The read chapter's §3 gives the tradeoffs; §4 gives the reward-vs-loss plumbing.

---

## The reverse-KL direction and mode-seeking

Source line 27:

> KL direction: the penalty is `KL(π_new ‖ π_ref)` (reverse, mode-seeking) — forces the policy to place mass only where the reference also has mass.

Information-theoretic consequence: `KL(π ‖ π_ref)` heavily penalizes placing mass where `π_ref` has near-zero probability, because the integrand `π · log(π/π_ref)` explodes. It therefore forbids the policy from exploring outside the reference's support — even if exploration would be rewarded.

This is the reason RLHF-tuned models often *sharpen* rather than *explore*. It is also the reason a separate entropy term is still needed: KL-to-reference keeps you inside the reference's support, but does not keep the *within-support* distribution wide.

Forward KL `KL(π_ref ‖ π)` would be mass-covering and mode-averaging, and has been tried in variational RL contexts but is not the RLHF convention.

---

## Adaptive β controllers

Source line 45:

> Adaptive KL: some implementations (InstructGPT early ablations, DeepSpeedChat) adapt β to hit a target KL per batch — multiplicatively raise β when KL exceeds target, lower when below.

OpenRLHF ships this as `AdaptiveKLController` ([[entropy-logging-patterns]] OpenRLHF excerpt). The controller takes a *target KL per batch* (e.g. 6 nats per response on a sequence-level KL) and updates β multiplicatively:

```
β ← β · (1 + K_p · (batch_KL / target_KL − 1))
```

capped in some `[β_min, β_max]`. This is safer than fixed-β for new reward functions because it decouples "how much KL budget am I willing to spend" from "what β achieves that KL".

---

## Connections

- Read-chapter §4 uses this source's `r̂_t = r_t − β · log(π/π_ref)` formula verbatim.
- Read-chapter §5 uses Korbak's `π* ∝ π_ref · exp(r/β)` as the link between RLHF and SAC's `π* ∝ exp(Q/α)`.
- [[excerpts/john-schulman-kl-tricks]] gives the estimator used by this chapter.
- [[excerpts/maximum-entropy-rl]] gives the max-ent ancestor of the tilted-posterior view.
- [[excerpts/entropy-logging-patterns]] shows how verl / OpenRLHF / TRL implement the per-token KL shaping.
- Upstream: ch-38 (PPO and InstructGPT). Downstream: ch-44 (RLVR pipelines that still use KL-to-SFT).
