---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/trpo.md
source_url: https://arxiv.org/abs/1502.05477
created_at: "2026-04-23"
---

# Excerpt: TRPO — the monotonic-improvement bound

**Source library:** `wiki/raw-data/llm-training/papers/trpo.md`
**Artifact:** *Trust Region Policy Optimization* — Schulman, Levine, Moritz, Jordan, Abbeel, 2015. The natural-gradient ancestor of PPO; the paper that turned "policy gradient" into "policy gradient with a guaranteed-improvement guarantee".

---

## Why this source anchors ch-37

Ch-37 introduces the score-function estimator; ch-38 introduces the first two practical instantiations (TRPO, PPO). TRPO's contribution is *conceptual scaffolding* that survives in PPO: the surrogate objective is an importance-sampled version of the vanilla-PG expectation, and the whole point of the algorithm is to bound *how far* `π_θ` is allowed to drift from `π_θ_old` per update. The `β · KL(π_θ || π_ref)` term that appears in every RLHF objective in ch-38 onward is a direct descendant of TRPO's trust-region constraint.

---

## The monotonic-improvement bound

From the source (line 19, §Key Contributions):

> Derives monotonic-improvement lower bound: `η(π_new) ≥ L_π_old(π_new) − C · D_KL^max(π_old, π_new)`.

This is the core theoretical result. `η(π)` is the true expected return; `L_π_old(π)` is a **surrogate** that is tractable to maximise given rollouts from `π_old`; `C` is a constant that depends on the reward scale and the horizon.

> Notice: the bound says *monotonic improvement is not automatic* — it holds only when the KL term is accounted for. Naive policy gradient can and does *decrease* `η` because the surrogate `L_π_old` is only a good approximation locally. The KL-as-trust-region reading is: penalise moves that invalidate the local approximation, and you recover monotonicity. Every RL-trained RLHF chapter after ch-37 is re-using this insight under a different name (PPO's clip, GRPO's clip, DPO's `β`).

The paper replaces the penalty form with an **explicit KL constraint** (still attested at line 20):

> Replaces the penalty form with an explicit KL constraint (trust region).

So two equivalent readings — Lagrangian penalty (PPO / DPO style) or hard constraint (TRPO style) — coexist from the start.

---

## The importance-sampled surrogate

From the source (lines 31–32, §Surrogate objective):

> `L_π_old(π) = η(π_old) + E_{s~ρ_π_old, a~π}[ (π(a|s) / π_old(a|s)) · A_π_old(s,a) ]`
> Identical to PPO's unclipped objective; A is the advantage under π_old.

This is the object ch-37 §1 derives from scratch. Reading it against the vanilla score-function form: `E_{a~π}[X(a)] = E_{a~π_old}[ (π/π_old) · X(a) ]` is just importance sampling. So the PG gradient w.r.t. `θ` is:

```
∇_θ L_π_old(π_θ) = E_{s, a~π_old}[ (π_θ(a|s)/π_old(a|s)) · ∇_θ log π_θ(a|s) · A_π_old(s,a) ]
```

At `θ = θ_old` the ratio is 1 and this collapses back to `E[∇log π · A]`. Ch-37's template equation is the surrogate, with the ratio hiding at `θ = θ_old`.

> Notice: the ratio `π_θ(a|s)/π_old(a|s)` is what lets you do **multiple gradient steps per rollout** without re-collecting data. This is the single biggest efficiency difference between pre-TRPO REINFORCE (one update per sample) and PPO (K epochs per rollout). The ratio is also what PPO's clip clamps — the trust region becomes a first-order trust region on the ratio, not on the KL.

---

## The constraint and δ

From the source (lines 35–36, §Constrained problem):

> `max_θ L_π_old(π_θ)  subject to  E_s[ D_KL(π_old(·|s) || π_θ(·|s)) ] ≤ δ`
> δ typically 0.01–0.05.

The `δ ∈ [0.01, 0.05]` range is attested and is the reason InstructGPT-style RLHF uses `β` in roughly this order of magnitude (β=0.04 in Tulu-3 RLVR, β=0.05 in REINFORCE++'s default, β=0.01–0.05 across most RLHF recipes).

> Notice: TRPO imposes KL symmetrically — `D_KL(π_old || π_θ)`. Modern RLHF imposes KL-to-*reference* — `D_KL(π_θ || π_ref)` — which is a different quantity because `π_ref` is the frozen SFT policy, not the previous RL iterate. The two are easy to confuse. TRPO's δ controls step size; RLHF's β controls distance-from-SFT. Ch-38 formalises the distinction.

---

## The natural-gradient step

From the source (lines 39–41, §Natural gradient step):

> `θ_new = θ_old + sqrt( 2δ / g^T F^{-1} g ) · F^{-1} g`
> where g = ∇L_π_old and F is the Fisher information matrix of π_old. F^{-1}g computed via conjugate gradient on Fisher-vector products — no explicit F.

The machinery (conjugate gradient on Fisher-vector products, line search for constraint satisfaction) is why TRPO is expensive. Ch-37 does not use any of it — the score-function estimator is first-order — but the formula matters for ch-38's argument that PPO is "TRPO at 10× less compute".

> Notice: Fisher-vector products require *two* backward passes per CG iteration. On a 7B-parameter LLM this is prohibitively expensive, and high-dim param spaces are hard to tune CG on. This is the concrete, attested reason ([[trpo]] lines 58–60) that nobody runs TRPO on modern LLMs.

---

## Why PPO replaced it

From the source (lines 58–61, §Why PPO replaced it):

> TRPO requires Fisher-vector products → 2nd backward pass per CG step → expensive.
> Conjugate gradient is hard to tune in high-dim LLM parameter space.
> PPO's clip is a cheaper first-order surrogate that empirically matches monotonicity.
> Almost no modern RLHF uses TRPO directly.

This is the transition ch-38 opens with. TRPO is the theory that PPO operationalises cheaply. The monotonic-improvement bound does not carry through PPO's first-order clip in a strict sense — there is no formal proof that PPO preserves monotonicity — but the empirical behaviour is close enough that the field moved on.

> Notice: "empirically matches monotonicity" is a hedge. PPO *fails to be monotonic* in well-documented regimes (K too large, clip too loose, reward scale unstable). [[costa-huang-ppo-details]]'s 37-trick list is in large part the set of implementation choices that keep PPO approximately monotonic in practice. Chapter 38's treatment of PPO reproduces those choices explicitly.

---

## Attested hyperparameters

From the source (lines 47–55, §Hyperparameters):

| Knob | Value |
|------|-------|
| KL target δ | 0.01 |
| CG iterations | 10 |
| CG damping | 0.1 |
| Line-search fraction | 0.5 |
| Line-search max steps | 10 |
| γ | 0.99 |
| GAE λ | 0.95–0.97 |

`γ = 0.99` and `λ = 0.95` are the values ch-37 §3 cites as the RL defaults that get specialised down to `γ = 1.0, λ = 0.95` for LLMs (undiscounted because rewards concentrate at EOS). The TRPO paper is the first place both numbers appear together as a default combination.

---

## What ch-37 keeps from this source

- The surrogate-objective identity `L_π_old(π) = E[(π/π_old) · A]`, which ch-37's template equation is a reduction of.
- The KL trust-region framing, which ch-37 §5 cites when arguing that KL-to-ref already regularises entropy (so entropy bonus is usually redundant).
- The δ ∈ [0.01, 0.05] range, which ch-37 (and ch-38) uses to rationalise RLHF's β choices.
- The "PPO is TRPO at 10× less compute" framing, which sets up ch-38's opening.

---

## Connections

- **ch-37 §1 / §5** — surrogate objective and KL regularisation.
- **ch-38** — PPO, InstructGPT, Llama-2's RLHF — all first-order successors.
- [[ppo]] — direct successor paper.
- [[rlhf-instructgpt]] — applies the same β-KL penalty at the per-token level.
- [[vanilla-pg]] — the unconstrained predecessor; ch-37's §1 derivation.
- [[excerpts/vanilla-pg]] — companion excerpt with the score-function identity and baseline proof.
