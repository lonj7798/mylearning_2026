<!-- scope: Trust Region Policy Optimization — natural-gradient ancestor of PPO with monotonic improvement bound
     deps: [[vanilla-pg]]
     see-also: [[ppo]], [[rlhf-instructgpt]]
-->

# Trust Region Policy Optimization (TRPO)
- **Core Insight:** Monotonic policy improvement is guaranteed if the KL between successive policies is bounded by δ — constrain each update to stay inside a KL trust region and solve with a natural-gradient (Fisher-vector-product) step.
- **Guideline:** TRPO gives the theoretical monotonic-improvement ceiling; in practice use PPO's first-order clipped surrogate, which approximates the trust region at ~10× less compute.
- **Authors:** John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel
- **Year:** 2015
- **URL:** https://arxiv.org/abs/1502.05477
- **Relevant topics:** natural policy gradient, trust region, KL constraint, monotonic improvement

## Abstract
We propose an iterative procedure for optimizing policies with guaranteed monotonic improvement. Starting from a theoretical bound on policy improvement as a function of KL divergence between consecutive policies, we derive a practical algorithm, Trust Region Policy Optimization (TRPO). The algorithm solves a constrained optimization problem using the natural gradient at each step and achieves strong results on simulated robotics (swim, hop, walk) and Atari.

## Key Contributions
- Derives monotonic-improvement lower bound: `η(π_new) ≥ L_π_old(π_new) − C · D_KL^max(π_old, π_new)`.
- Replaces the penalty form with an explicit KL constraint (trust region).
- Uses conjugate-gradient to compute the natural-gradient step — avoids forming the full Fisher matrix.
- Line-search backtracking to ensure both constraint satisfaction and surrogate improvement.
- Direct predecessor of PPO (same surrogate, looser constraint enforcement).

## Key Figures/Tables to Study
- **Figure 1:** Natural vs vanilla gradient on a toy 2D policy — natural-gradient points at the KL-minimizing direction.
- **Table 2:** MuJoCo returns — TRPO vs vanilla PG, reward-shaping ablations.
- **Section 5 (Algorithm 1).**

## Technical Details

### Surrogate objective
`L_π_old(π) = η(π_old) + E_{s~ρ_π_old, a~π}[ (π(a|s) / π_old(a|s)) · A_π_old(s,a) ]`
Identical to PPO's unclipped objective; A is the advantage under π_old.

### Constrained problem
`max_θ L_π_old(π_θ)  subject to  E_s[ D_KL(π_old(·|s) || π_θ(·|s)) ] ≤ δ`
δ typically 0.01–0.05.

### Natural gradient step
`θ_new = θ_old + sqrt( 2δ / g^T F^{-1} g ) · F^{-1} g`
where g = ∇L_π_old and F is the Fisher information matrix of π_old. F^{-1}g computed via conjugate gradient on Fisher-vector products — no explicit F.

### Line search
Starting from the full natural-gradient step, geometrically shrink until both (a) KL ≤ δ and (b) surrogate actually improved on the sampled batch.

### Hyperparameters
| Knob | Value |
|------|-------|
| KL target δ | 0.01 |
| CG iterations | 10 |
| CG damping | 0.1 |
| Line-search fraction | 0.5 |
| Line-search max steps | 10 |
| γ | 0.99 |
| GAE λ | 0.95–0.97 |

### Why PPO replaced it
- TRPO requires Fisher-vector products → 2nd backward pass per CG step → expensive.
- Conjugate gradient is hard to tune in high-dim LLM parameter space.
- PPO's clip is a cheaper first-order surrogate that empirically matches monotonicity.
- Almost no modern RLHF uses TRPO directly.

## Connections
- First-order successor, default for RLHF: [[ppo]].
- Vanilla baseline: [[vanilla-pg]].
- Natural-gradient family (earlier): Kakade 2001 (Natural Policy Gradient).
