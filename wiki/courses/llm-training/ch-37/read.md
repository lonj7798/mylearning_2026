<!-- chapter: ch-37
     track: rl
     kind: content
     title: Policy-Gradient Foundations
     deps: [ch-36]
     sources: [[vanilla-pg]], [[trpo]], [[ppo]], [[rloo]], [[reinforce-plus-plus]],
              [[lilianweng-rlhf]], [[nathan-lambert-rl-overview]], [[costa-huang-ppo-details]],
              [[maximum-entropy-rl]]
     figures: figures/pg-variance.html
     opens: rl-track (ch-37..ch-46)
-->

# Chapter 37 — Policy-Gradient Foundations

> **Core insight.** RL for LLMs is not a different beast from supervised learning — it is supervised learning with a *chosen* per-sample weight. The policy-gradient theorem reduces to a single identity: `∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A(s,a)]`. Every modern LLM-RL algorithm — PPO, TRPO, RLOO, GRPO, REINFORCE++ — is a different choice of (a) what `A(s,a)` is, (b) how you estimate it from samples, and (c) what regulariser you tack on to keep `π_θ` near a reference. The algorithm family is a second-order concern; the first-order concern is variance in the `A` estimator, because variance is what decides whether a batch of gradients moves the model or cancels itself out.
>
> **Guideline.** Before picking an RL algorithm, know (1) which form of the score-function estimator you are using (per-timestep vs per-sequence, causal vs full-trajectory), (2) which baseline subtracts the action-independent floor (none / moving average / learned V / leave-one-out / group-mean / global-batch), and (3) whether the regulariser is a KL-to-reference penalty *inside the reward* (RLHF canon) or a separate KL constraint / entropy bonus *in the loss* (classical RL, GRPO). Defaulting to PPO because "PPO works" is how you spend a quarter debugging an advantage estimator whose variance scales with sequence length.

---

## Why this chapter exists

This is the opening chapter of the RL track. The SFT track (ch-30 … ch-36) produced a frozen `checkpoint-final` that is about to become `π_ref` — the reference policy every RL method in the next ten chapters regularises against. Before introducing TRPO, PPO, DPO, GRPO, or RLVR, we need the one object they all share: the score-function estimator of a policy gradient. Nathan Lambert's framing ([[nathan-lambert-rl-overview]]) puts it cleanly — "the field has converged on a small set of algorithmic templates (PPO, DPO, GRPO)" — so this chapter is the template. Everything that follows is a specialisation.

Four deliverables by the end of the chapter: (1) a derivation of `∇J(θ) = E[∇log π · A]` that you can reproduce on paper; (2) a proof that subtracting any baseline `b(s)` is unbiased and reduces variance; (3) an understanding of *why* LM-RL is structurally simpler than robotics RL (and why that simplicity kills half of PPO's machinery, per [[rloo]]); (4) a working mental model of entropy regularisation — what it fixes, what it papers over.

---

## §1 The policy gradient theorem — derivation

Let `π_θ(a|s)` be a stochastic policy and `J(θ) = E_{τ ∼ π_θ}[R(τ)]` the expected return over trajectories `τ = (s_0, a_0, r_0, s_1, a_1, r_1, …)`. We want `∇_θ J(θ)`. The probability of a trajectory under `π_θ` factorises:

```
p_θ(τ) = ρ_0(s_0) · Π_t π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t)
```

Only the middle factor depends on `θ`. The **log-prob trick** (score-function identity) is:

```
∇_θ p_θ(τ) = p_θ(τ) · ∇_θ log p_θ(τ) = p_θ(τ) · Σ_t ∇_θ log π_θ(a_t | s_t)
```

The initial-state and transition terms have no `θ`, so they contribute zero gradient. Pushing `∇_θ` through the expectation:

```
∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ
         = ∫ ∇_θ p_θ(τ) · R(τ) dτ
         = ∫ p_θ(τ) · [Σ_t ∇_θ log π_θ(a_t | s_t)] · R(τ) dτ
         = E_{τ∼π_θ}[ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τ) ]
```

This is the **score-function (REINFORCE) estimator**, first written down in this connectionist form by Williams 1992 ([[vanilla-pg]]). The estimator is *unbiased*: `E[∇̂J] = ∇J` with one sampled trajectory. It is also *high-variance*: the variance of a sum-of-log-prob-gradients-times-one-scalar-return grows with trajectory length and with the scale of `R`.

### Causal form (return-to-go)

The future cannot influence the past. `R(τ) = Σ_{t'} r_{t'}` can be split: for `t' < t` the reward is independent of `a_t`, so those terms contribute zero in expectation. This yields the **causal** form used in every modern implementation ([[vanilla-pg]] §Technical Details):

```
∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]   where   G_t = Σ_{t'≥t} γ^{t'−t} r_{t'}
```

`G_t` is the **return-to-go** from step `t`. Dropping the past-reward terms strictly reduces variance without introducing bias.

---

## §2 Baselines and the variance-reduction proof

For any function `b(s)` that does not depend on the action:

```
E_{a∼π_θ(·|s)}[ ∇_θ log π_θ(a|s) · b(s) ]
    = b(s) · Σ_a π_θ(a|s) · ∇_θ log π_θ(a|s)
    = b(s) · Σ_a ∇_θ π_θ(a|s)
    = b(s) · ∇_θ Σ_a π_θ(a|s)
    = b(s) · ∇_θ 1
    = 0
```

So the **baseline-augmented** estimator is still unbiased ([[vanilla-pg]] Theorem 1):

```
∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]
```

**Why it reduces variance.** Let `X = ∇log π · G` and `X' = ∇log π · (G − b)`. Both have the same mean. The variance of a coordinate of `X'` is:

```
Var(X'_i) = Var(X_i) − 2 · Cov(X_i, ∇log π_i · b) + Var(∇log π_i · b)
```

The minimum-variance constant baseline per coordinate is `b* = E[G · (∇log π_i)^2] / E[(∇log π_i)^2]`, the `∇log π`-weighted mean return. In practice any `b(s)` close to `E[G | s]` cuts variance dramatically — which is why the *value function* `V^π(s) = E_π[G | s]` is the canonical baseline, and why the "advantage" `A(s,a) = Q^π(s,a) − V^π(s)` is what everyone puts in the estimator instead of the raw return.

### The menagerie of baselines

| Method | Baseline | Where it lives | Source |
|---|---|---|---|
| Raw REINFORCE | 0 | — | [[vanilla-pg]] |
| REINFORCE w/ moving avg | `b̄ = (1/S) Σ_s R_s` | scalar EMA | [[vanilla-pg]] |
| Actor-critic / A2C | `V_φ(s)` | learned head | §3 below |
| PPO | `V_φ(s)` + GAE | learned head + λ knob | [[ppo]] |
| RLOO | `(1/(k−1)) Σ_{j≠i} R(y_j, x)` | leave-one-out across rollouts | [[rloo]] |
| GRPO | `(r_i − mean(r)) / std(r)` over group G | group z-score | [[reinforce-plus-plus]] |
| REINFORCE++ | `(G − mean_B(G)) / std_B(G)` over full batch | global z-score | [[reinforce-plus-plus]] |

All are unbiased because all are action-independent given the conditioning state (prompt `x` for LLMs). They differ only in *variance* and in *how much sampling / memory / compute* they cost. [[rloo]]'s contribution was precisely the observation that, once you have `k ≥ 2` rollouts per prompt, the leave-one-out baseline is statistically better than a learned `V_φ(s)` *and* removes the value network from the system entirely — ~50% memory footprint of PPO with strictly higher win-rate on TL;DR and HH-RLHF at matched KL.

### LLM-specific form — one advantage per sequence

For a sequence `y = (y_1, …, y_T)` sampled autoregressively from `π_θ(·|x)` with terminal reward `R(x, y)` ([[vanilla-pg]] §LLM-specific form):

```
∇_θ J = E_{y∼π_θ(·|x)}[ R(x, y) · Σ_t ∇_θ log π_θ(y_t | x, y_<t) ]
```

The sum `Σ_t ∇_θ log π_θ(y_t | x, y_<t) = ∇_θ log π_θ(y | x)` is exactly the SFT cross-entropy gradient with sign flipped, weighted by `R(x, y)`. **That is the whole algorithm**: run generation, compute a per-sequence weight, do a weighted SFT step. [[rloo]] is this expression plus a leave-one-out baseline, nothing more.

---

## §3 Actor-critic: trading bias for variance

Monte-Carlo `G_t` is unbiased but high-variance (it sums `T − t` noisy rewards). A **bootstrapped** estimator substitutes a learned value function:

```
Â_t^{(1)} = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)     (TD residual; biased if V_φ ≠ V^π, low variance)
Â_t^{(∞)} = G_t − V_φ(s_t)                       (MC advantage; unbiased, high variance)
```

[[ppo]] §Technical Details gives the **GAE** interpolation that runs the full spectrum:

```
δ_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)
Â_t^{GAE(λ)} = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + …
```

`λ = 1` recovers Monte-Carlo (unbiased, high variance). `λ = 0` recovers 1-step TD (biased by `V_φ`'s error, low variance). The RLHF default is `λ = 0.95, γ = 1.0` ([[ppo]] canonical hparams, [[lilianweng-rlhf]] RLHF defaults). `γ = 1.0` — undiscounted — because, per [[lilianweng-rlhf]], "rewards concentrate at EOS" in LLM RL: there is exactly one non-zero per-step reward (the RM score at the end-of-sequence token), so any `γ < 1` just throws signal away.

**The bias-variance tradeoff is quantitative.** Assume `V_φ` has mean-squared error `ε^2` against `V^π`. Then the bias of `Â_t^{(1)}` is `O(ε)` per step; over a T-step sum it compounds to `O(Tε)` in the worst case. The variance of the MC estimator scales with `Σ_{t'≥t} Var(r_{t'})` — for LLMs with a single terminal reward this is `Var(R)` regardless of T, so the MC estimator's variance is *bounded*. This is a crucial LLM-specific observation: unlike robotics where long horizons force you to bootstrap, LLM terminal-reward structure makes the unbiased MC estimator competitive — which is exactly why [[rloo]] argues you can drop `V_φ` entirely.

---

## §4 Why LM-RL is special

Classical RL papers (TRPO, PPO, SAC) were written for robotics and games. Four structural properties of LLM post-training break the assumptions those papers optimised for. [[rloo]] §Key Contributions is the clearest enumeration; the list below summarises and extends it with the [[reinforce-plus-plus]] and [[lilianweng-rlhf]] refinements.

**(a) Deterministic dynamics.** Given the prefix `(x, y_<t)` and the KV cache, there is no stochastic environment transition — the only randomness is the sampling of `y_t` from `π_θ(·|x, y_<t)`. Bellman-equation stochasticity vanishes. The variance-reduction machinery built to handle environment stochasticity (target networks, double Q, clipped double-Q) is *irrelevant* for LMs.

**(b) Full-trajectory rewards.** Almost every LLM RL reward is terminal: one scalar `R(x, y)` after `y_T = EOS`. Per-step rewards exist only when you add a shaped KL term or a process-reward model ([[prm800k]], [[math-shepherd]]). For terminal-only rewards, `G_t = R(x, y)` for every `t`, so the causal per-step gradient collapses to `R(x, y) · ∇_θ log π_θ(y | x)` — the LLM-specific form from §2.

**(c) Very long episodes, very short batches.** A rollout might be 2K–32K tokens ([[ppo]] MuJoCo: T=2048 per actor vs. 32K tokens in a modern RLHF run). Per-token advantages are highly correlated along a sequence — they share the same terminal reward. [[reinforce-plus-plus]] argues this correlation is why global batch normalisation outperforms group-local normalisation: variance in per-prompt group-means is itself high-variance when k is small.

**(d) Discrete actions over a 100K-way vocabulary.** Policy entropy, clip thresholds, and KL divergences are all computed over a categorical with `|V| ≈ 128K`. Exact KL `Σ_y π(y|x) log(π(y|x)/π_ref(y|x))` is tractable per token and is the default for token-level KL ([[lilianweng-rlhf]] KL penalty implementation, [[reinforce-plus-plus]] k1 estimator).

Concretely, what these four properties kill from the PPO recipe: the value network becomes optional ([[rloo]] removes it, [[reinforce-plus-plus]] removes it, GRPO removes it). GAE becomes redundant when rewards are terminal (`λ=1` is free of bias). K>1 epochs per rollout are often counterproductive — PPO-clip's trust region gets violated fast when each rollout is a 4K-token sequence; [[rloo]] uses K=1. What survives: the **score-function estimator**, the **baseline**, and the **KL-to-reference regulariser**. Those three objects are the minimum viable LLM-RL method, and the next ten chapters are arguments about how to build each one.

---

## §5 The entropy term — regulariser or bandaid

Classical RL adds an **entropy bonus** to the loss: `+ c2 · H(π_θ(·|s))`. The stated purpose is to keep the policy from collapsing onto a single action before it has explored enough. [[ppo]] uses `c2 = 0.01` on MuJoCo; CleanRL's Atari config uses the same. [[maximum-entropy-rl]] derives the deep reason — the optimal max-ent policy is `π*(a|s) ∝ exp(Q(s,a)/α)`, a soft Boltzmann whose temperature `α` is the entropy coefficient. The soft-Q value function is `V(s) = α · log Σ_a exp(Q(s,a)/α)`, and auto-α tuning (SAC-v2) adjusts `α` to hit a *target entropy* `H̄`.

**In LLM-RL, the entropy-bonus story is different.** [[lilianweng-rlhf]] states it bluntly: "entropy bonus is often dropped in RLHF because KL regularization to reference policy already regularizes the policy toward a stochastic distribution." The canonical InstructGPT-style objective is:

```
L_RLHF(θ) = −E_{y∼π_θ(·|x)}[ R(x, y) ] + β · KL( π_θ(·|x) || π_ref(·|x) )
```

The KL term is simultaneously a trust-region constraint and an entropy lower-bound: `KL(π_θ || π_ref) = E_π_θ[log π_θ − log π_ref] = −H(π_θ) − E_π_θ[log π_ref]`. So penalising KL-to-reference directly bounds `−H(π_θ)` from below (given `π_ref`'s support) — the policy cannot collapse entropy without paying KL cost. An explicit `+ c2 · H(π)` on top is mostly redundant.

**When is an entropy bonus a bandaid?** Two symptoms, both documented. (1) *Entropy collapse* — per-token entropy drives toward zero mid-training, the policy degenerates to greedy, and RL stalls. Causes include a too-small β (KL is not actually constraining), a mis-scaled reward (one high-reward mode swallows the mass), or on-policy staleness. Adding an entropy bonus *masks* this instead of fixing the reward / KL config. (2) *Reasoning RL (R1-style) distribution collapse* — [[nathan-lambert-rl-overview]] notes that length-normalisation artefacts in GRPO can cause shorter responses to be preferred monotonically, contracting the policy. Again the fix is the loss — [[dr-grpo]] corrects the normalisation — not an entropy patch.

**When is it a real regulariser?** When exploration is genuinely the bottleneck and KL-to-reference is too weak (`π_ref` itself is low-entropy on the relevant prompts). [[maximum-entropy-rl]]'s auto-α against a target entropy is the principled solution: set `H̄` to a fraction of the SFT model's per-token entropy, let `α` adjust. Most production RLHF stacks do neither — they leave `c2 = 0` and trust KL-to-ref — and that is the right default for ch-38 onward. The entropy term is a tool to reach for when you *observe* entropy collapse, not a default to ship.

---

## §6 What this template lets us predict about the rest of the track

The template `∇J = E[∇log π · A] + regulariser` collapses the next ten chapters into a table of knobs.

| Chapter | Algorithm | `A` estimator | Regulariser | Distinctive component |
|---|---|---|---|---|
| ch-38 | TRPO / PPO / InstructGPT | GAE-λ with `V_φ` | per-token KL-to-ref `β` | clipped ratio / trust region |
| ch-39 | DPO / IPO / KTO / SimPO / ORPO | closed-form; no explicit `A` | implicit via `β` | offline, no rollouts |
| ch-40 | GRPO / Dr. GRPO | group-mean `(r_i − μ_G)/σ_G` | per-token KL (k3) | no value net; group rollouts |
| ch-41 | RLOO / REINFORCE++ | leave-one-out or global-batch z-score | KL as shaped reward | critic-free |
| ch-42 | RLVR (verifier rewards) | same as GRPO/PPO | KL-to-ref | deterministic `r(x,y)=v(x,y)` |
| ch-43 | Process-reward RL | per-step `r_t` from PRM | KL-to-ref | per-step credit assignment |
| ch-44 | Iterative / online RLHF | flywheel over rounds | KL-to-ref (fresh) | loop structure |
| ch-45 | Self-play / RLAIF | RM or model-judge reward | KL-to-ref | signal source |
| ch-46 | Track capstone | — | — | — |

Every row is the same equation. The algorithmic literature looks bigger than it is because nobody writes the equation down twice — they write a new paper around the knob they changed.

---

## Companion visualization

**[figures/pg-variance.html](figures/pg-variance.html)** — interactive gradient-variance simulator. Pick a baseline type (none / constant / value-function / leave-one-out) and watch the per-iteration variance curve of the estimator on a small simulated RLHF-like problem (k rollouts per prompt, terminal reward). The curves make concrete why [[rloo]]'s leave-one-out beats a moving-average baseline at small k, and why global-batch normalisation ([[reinforce-plus-plus]]) beats group-local normalisation when k is 1–2. Use it *before* reading ch-38 — the variance argument is what drives every algorithm choice that follows.

---

## Connections

- **ch-36 (SFT capstone)** — the `checkpoint-final` from the packed SFT run is `π_ref` for every RL chapter in this track. Every regularisation term in this chapter's template is a distance-to-that-checkpoint penalty.
- **ch-38 (KL-Controlled RLHF)** — TRPO's monotonic-improvement bound ([[trpo]]) and PPO's clipped surrogate ([[ppo]]) are the first two specialisations of the template. InstructGPT is PPO + per-token KL-to-ref.
- **ch-39 (Offline preference)** — DPO replaces the sample-based `∇J` with a closed form under Bradley-Terry preferences; no `π_θ` rollouts at train time.
- **ch-40 / ch-41 (GRPO / RLOO / REINFORCE++)** — three critic-free specialisations, each picking a different baseline.
- **ch-42 (RLVR)** — replaces the learned reward model with a deterministic verifier; the policy-gradient structure is unchanged.
- **ch-47..ch-53 (Eval, Reward, Judge)** — the reward signal `R(x, y)` in this chapter is exactly what those chapters are about constructing.

## Further reading

- [[vanilla-pg]] — Williams 1992. The policy-gradient theorem, the baseline-invariance theorem, and the eligibility/score function. The source this chapter's §1 and §2 derive.
- [[trpo]] — Schulman 2015. Monotonic-improvement bound + KL trust region. First step from "score function" to "trust-region policy optimisation".
- [[ppo]] — Schulman 2017. Clipped surrogate, combined actor-critic loss, GAE. Canonical hparams.
- [[rloo]] — Ahmadian 2024. LLM-RL does not need the value network, GAE, clip, or K>1 epochs; leave-one-out baseline beats PPO at ~50% memory.
- [[reinforce-plus-plus]] — Hu 2025. Global batch advantage normalisation; the `k=1` critic-free recipe.
- [[lilianweng-rlhf]] — RLHF tutorial; `r_total = r(x,y)·1[y=EOS] − β·log(π/π_ref)`, reward whitening, and why entropy bonus is usually dropped.
- [[nathan-lambert-rl-overview]] — algorithm-to-reward-signal framing; reward-signal source is the first-order choice.
- [[costa-huang-ppo-details]] — the 37-trick implementation reference; what makes the paper reproduce.
- [[maximum-entropy-rl]] — SAC's `π*(a|s) ∝ exp(Q/α)`, auto-α tuning, target entropy `H̄`.
