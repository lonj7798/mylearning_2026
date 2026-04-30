---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2203.02155 (InstructGPT), https://arxiv.org/abs/2205.11275 (Korbak)
created_at: "2026-04-23"
---

# Excerpt: KL-control — the theory behind OpenRLHF's AdaptiveKLController

**Source library:** `wiki/raw-data/llm-training/papers/kl-control-rlhf.md`
**Authors (lineage):** Jaques 2019, Stiennon 2020, Ouyang 2022 (InstructGPT), Korbak 2022

---

## Why this source anchors ch-56

OpenRLHF's decision to put KL-to-reference in the *reward* and not in
the *loss* is not arbitrary — it inherits from InstructGPT and is
validated by Korbak's Bayesian-inference reinterpretation. Without
this source, ch-56 §2.2's "why KL-into-reward, not KL-into-loss"
is just an assertion; with it, the choice is mathematically forced.

---

## The RLHF objective, attested

Source §Key Contributions:

> RLHF objective (Stiennon / InstructGPT form):
> `J(φ) = E[ r_θ(x,y) − β · log( π_φ(y|x) / π_SFT(y|x) ) ]
>   + γ · E_{x~D_pretrain}[log π_φ(x)]`
> — KL is added to the **per-token reward**, then standard PPO is run.

OpenRLHF's `ppo_trainer.py` implements exactly this: for each rollout
token t,

```
reward_t -= kl_ctl.value * (log pi(y_t|...) - log pi_ref(y_t|...))
```

before GAE runs. This is why `PolicyLoss` sees `advantages` that
already encode the KL cost and does not need its own KL term.

---

## Why reward, not loss — the argument

Source §Key Contributions:

> Why reward + not loss: adding KL to the reward keeps the PPO
> advantage estimator well-defined per token; adding KL to the loss
> breaks the advantage-based policy gradient and empirically trains
> worse.

The empirical tail is what matters: InstructGPT-era ablations
established that loss-side KL converges to a different and worse
optimum than reward-side KL for PPO. GRPO (TRL implementation)
changes this calculus because GRPO has no per-token value function,
but PPO with a learned critic requires reward-side KL.

---

## Korbak's reinterpretation

Source §Key Contributions:

> Korbak's reformulation: `argmax_π E_π[r] − β · KL(π‖π_ref)` has
> closed-form optimum `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`. RLHF is
> therefore amortized sampling from this tilted posterior — not
> unconstrained reward maximization.

This gives ch-56 its philosophical frame: OpenRLHF is not solving
`max E[r]`; it is solving `max E[r] - β KL(π‖π_ref)`, and the
AdaptiveKLController's job is to keep β at the value that hits a
target KL budget per batch. DPO's implicit reward (ch-56 §3) has the
same form — `β log(π/π_ref)` — which is why β plays the same role in
PPO and DPO.

---

## The K1/K2/K3 estimators

Source §Key Contributions:

> KL estimators (Schulman's blog): three unbiased estimators for
> `log(p/q)`-style KL:
> - k1 = log(π/π_ref) — unbiased but high variance, can be negative.
> - k2 = 0.5 · (log(π/π_ref))^2 — biased but low variance.
> - k3 = (π_ref/π) − 1 − log(π_ref/π) — unbiased AND always ≥ 0;
>   **recommended**, used in modern TRL/OpenRLHF.

OpenRLHF's `AdaptiveKLController` uses K1 internally (the source uses
K1 in its reward-shaping path); the source flags K3 as the modern
recommendation. This is one of the places where ch-56 §2 notes
OpenRLHF is *slightly behind* the frontier — a production job that
cares will swap to K3.

---

## Adaptive β — the update rule

Source §Technical Details:

> Adaptive KL: some implementations (InstructGPT early ablations,
> DeepSpeedChat) adapt β to hit a target KL per batch —
> multiplicatively raise β when KL exceeds target, lower when below.

OpenRLHF's `AdaptiveKLController.update(current, n_steps)` is exactly
this rule, implemented as:

```
beta_new = beta_old * (1 + K_beta * clamp((KL_obs - KL_target) / KL_target, -0.2, 0.2))
```

The `K_beta` tuning knob and the `clamp` are the two practical
details that distinguish a stable run from an oscillating one.

---

## Failure modes, attested

Source §Technical Details:

> Failure modes: β too small → reward hacking and mode collapse;
> β too large → policy cannot depart from SFT, ignores RM signal.

Ch-56 §7's "entropy collapse" failure is the first half of this; the
second half is the quiet failure where a training run looks stable
but the evaluation numbers never move — β is holding the policy
against π_SFT.

---

## Connections

- [[excerpts/openrlhf-ppo]] — the loss that does *not* contain the
  KL term, because this source argues it shouldn't.
- [[excerpts/entropy-logging-patterns]] — the K1/K2/K3 table.
- [[excerpts/openrlhf-entropy-debugging]] — practitioner β ranges.
- Host chapter: [[ch-56]] §2.1 + §2.2.
- Backward to [[ch-38]] (DPO) — β plays the same role in DPO's
  implicit reward.
