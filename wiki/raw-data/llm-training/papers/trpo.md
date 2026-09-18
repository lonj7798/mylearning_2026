<!-- scope: Trust Region Policy Optimization — KL-constrained policy update solved with conjugate gradient, the predecessor of PPO
     deps: [[vanilla-pg]]
     see-also: [[ppo]], [[rlhf-instructgpt]]
-->

# Trust Region Policy Optimization
- **Core Insight:** A policy update that maximizes the importance-sampled surrogate objective subject to an average-KL constraint of δ = 0.01 gave monotonic improvement on simulated locomotion and on seven Atari games, with the same δ used for every experiment (§8, Table 2, Table 3).
- **Guideline:** When a policy update must be bounded but the parameter count is small enough that conjugate-gradient Fisher-vector products are affordable, use the KL-constrained TRPO step, because the paper reports it solved swimmer, hopper and walker while a fixed-penalty natural-gradient baseline did not produce forward-moving hopping or walking gaits (§8.1, Figure 4). Otherwise use the first-order clipped surrogate of [[ppo]].
- **Authors:** John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel
- **Year:** 2015 (arXiv v1 2015-02; arXiv v5 2017-04; ICML 2015)
- **URL:** https://arxiv.org/abs/1502.05477
- **Source type:** paper
- **Relevant topics:** natural policy gradient, trust region, KL constraint, monotonic improvement, conjugate gradient

## Abstract
The paper describes an iterative procedure for optimizing policies with guaranteed monotonic improvement. A theoretically justified procedure is derived first, then several approximations are made to it to obtain a practical algorithm, Trust Region Policy Optimization (TRPO). The algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Experiments cover simulated robotic swimming, hopping and walking gaits, and playing Atari games from screen images. The authors report that despite the approximations that deviate from the theory, TRPO tends to give monotonic improvement with little hyperparameter tuning (Abstract).

## Key Contributions
- Proves a policy-improvement lower bound in terms of the maximum KL divergence between successive policies, extending Kakade & Langford (2002) (§3, Eq. 9).
- Replaces the theoretical KL penalty with an explicit KL constraint, because the penalty coefficient given by the theory produces very small steps (§4).
- Replaces the maximum-KL constraint with an average-KL constraint, which the cart-pole comparison indicates has a similar effect (§8.1).
- Computes the search direction by conjugate gradient on Fisher-vector products, so the Fisher information matrix is never formed (App. C, C.1).
- Defines two estimators of the objective: single path (whole trajectories) and vine (branching rollouts from sampled states) (§5, Figure 1).

## Key Figures/Tables to Study
- **Figure 1:** illustration of the single-path and vine sampling procedures.
- **Figure 4:** learning curves on cart-pole, swimmer, hopper and walker, comparing single-path TRPO, vine TRPO, CEM, CMA, natural gradient, empirical FIM and max-KL, averaged over five runs (§8.1).
- **Table 1:** Atari scores for TRPO single path and vine against random, human, Deep Q-Learning and UCC-I on seven games.
- **Table 2 and Table 3 (App. E):** the experiment parameters for continuous control and for Atari.

## Technical Details

### Improvement bound
`η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃)`, with `C = 4εγ / (1 − γ)²` and `ε = max_{s,a} |A_π(s,a)|` (Eq. 8, Eq. 9).
`η` is the expected discounted return, `L_π` the importance-sampled surrogate, `D_KL^max` the maximum over states of the KL between the two action distributions, `γ` the discount.

### Constrained problem actually solved
`maximize_θ L_θold(θ) subject to D̄_KL^{ρ_θold}(θ_old, θ) ≤ δ` — the average KL over the state distribution, not the maximum, is used in practice (§4, Eq. 12).

### Step length
The search direction `s ≈ A⁻¹ g` is obtained by conjugate gradient, where `A` is the Fisher information matrix (the quadratic approximation of the KL) and `g` the gradient of the surrogate. The maximal step length is `β = sqrt(2δ / (sᵀ A s))`, derived from `δ ≈ ½ β² sᵀ A s` (App. C).

### Line search
The line search is run on `L_θold(θ) − X[D_KL(θ_old, θ) ≤ δ]`, where `X[·]` is 0 when its argument is true and +∞ otherwise. Starting from the maximal `β`, `β` is shrunk exponentially until the objective improves. The authors report that without the line search the algorithm occasionally takes large steps that cause a catastrophic degradation of performance (App. C).

### Cost
Each conjugate-gradient iteration costs one Fisher-vector product, so `k` CG iterations cost `k` such products per gradient step (App. C.1). The paper does not state a cost ratio against any first-order method.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TRPO locomotion policy | 364 / 4,806 / 8,206 params (swimmer / hopper / walker) | RL | KL step size δ | 0.01 | arXiv:1502.05477v5 App. E Table 2 | verified 2026-09-18 | §8.1: "We used δ = 0.01 for all experiments"; no δ sweep reported |
| TRPO locomotion policy | same | RL | Discount γ | 0.99 | App. E Table 2 | verified 2026-09-18 | no ablation reported |
| TRPO locomotion policy | same | RL | Policy iterations | 200 | App. E Table 2 | verified 2026-09-18 | no ablation reported |
| TRPO locomotion policy | same | RL | Simulation steps per iteration | 50K (swimmer), 1M (hopper), 1M (walker) | App. E Table 2 | verified 2026-09-18 | no ablation reported |
| TRPO locomotion policy | same | RL | Hidden layer size | 30 / 50 / 50 | App. E Table 2 | verified 2026-09-18 | no ablation reported |
| TRPO Atari policy | 33,500 params | RL | KL step size δ | 0.01 | App. E Table 3 | verified 2026-09-18 | no ablation reported |
| TRPO Atari policy | 33,500 params | RL | Policy iterations | 500 | App. E Table 3 | verified 2026-09-18 | no ablation reported |
| TRPO Atari policy | 33,500 params | RL | Simulation steps per iteration | 400K (vine), 100K (single path) | App. E Table 3 | verified 2026-09-18 | no ablation reported |
| TRPO Atari policy | 33,500 params | RL | Compute | about 30 hours on 16 cores for 500 iterations | §8.2, App. E Table 3 | verified 2026-09-18 | not an ablation |

Not reported by the paper: conjugate-gradient iteration count, CG damping, line-search shrink factor and maximum backtracks, GAE λ (GAE is a later paper and is not used here), any learning rate.

## Findings relevant to generality
- The authors state that TRPO learned all of the locomotion gaits with general-purpose policies and simple reward functions, using minimal prior knowledge, in contrast to prior locomotion methods that rely on hand-architected policy classes (§8.1). This is a claim about task generality of one algorithm, not about transfer between tasks. **Result (single study)**.
- The same architecture and hyperparameters were used across all seven Atari games, but each was run once per task; the caption states that performance varies substantially between random initializations and that error statistics could not be obtained (Table 1). Quantitative comparison against the baselines in that table is therefore single-seed.
- TRPO single path and vine do not dominate the baselines on Atari: UCC-I scores higher than both TRPO variants on all seven games in Table 1.

## Connections
- [[ppo]] — first-order successor; the surrogate objective is the same, the constraint is replaced by clipping.
- [[vanilla-pg]] — the unconstrained policy-gradient baseline.
- [[rlhf-instructgpt]] — the KL-constrained formulation carried into language-model RLHF.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1502.05477 (arXiv v5, 20 Apr 2017)
- Corrections to the previous card version:
  - "δ typically 0.01–0.05" → "δ = 0.01 was used for all experiments" (§8.1; App. E Tables 2 and 3).
  - "Figure 1: Natural vs vanilla gradient on a toy 2D policy" → Figure 1 illustrates the single-path and vine sampling procedures; there is no toy 2D gradient figure.
  - "Table 2: MuJoCo returns — TRPO vs vanilla PG, reward-shaping ablations" → Table 2 lists experiment parameters for the continuous-control tasks; the locomotion comparison is Figure 4 and the Atari comparison is Table 1. No reward-shaping ablation is reported.
  - "Section 5 (Algorithm 1)" → Algorithm 1 appears in §3; §5 covers the single-path and vine estimators.
  - "Michael I. Jordan" → the paper's author line reads "Michael Jordan".
  - "the natural-gradient step θ_new = θ_old + sqrt(2δ / gᵀF⁻¹g) · F⁻¹g" → the paper derives `β = sqrt(2δ / (sᵀAs))` for the approximate direction `s ≈ A⁻¹g`, and the applied step is the line-searched, not the full, `βs` (App. C).
- Removed as unsupported by the source: the hyperparameter rows "CG iterations 10", "CG damping 0.1", "line-search fraction 0.5", "line-search max steps 10", "GAE λ 0.95–0.97"; the claim that PPO approximates the trust region "at ~10× less compute"; the claim that PPO's clip "empirically matches monotonicity"; the claim that "almost no modern RLHF uses TRPO directly" (no source stated).
- Not reported by the source: wall-clock or sample-efficiency comparison against first-order methods; any language-model experiment.
