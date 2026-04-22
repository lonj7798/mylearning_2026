<!-- scope: Vanilla policy gradient (REINFORCE) — foundational theorem and estimator
     deps: []
     see-also: [[ppo]], [[trpo]], [[rloo]], [[grpo]]
-->

# Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE)
- **Core Insight:** The gradient of expected return equals the expectation of the score function times the return — `∇J(θ) = E[∇log π_θ(a|s) · R]` — turning RL into on-policy stochastic gradient ascent.
- **Guideline:** Always subtract a baseline b(s) to reduce variance (e.g., learned V(s), leave-one-out mean, or group mean); never omit it in high-dimensional action spaces.
- **Authors:** Ronald J. Williams
- **Year:** 1992
- **URL:** https://link.springer.com/article/10.1007/BF00992696
- **Relevant topics:** policy gradient theorem, score function estimator, Monte Carlo RL, baseline

## Abstract
Introduces the REINFORCE family of algorithms: statistical gradient-following algorithms for connectionist networks with stochastic units. Shows that the gradient of expected reinforcement w.r.t. network weights can be written as an expectation of the eligibility ∂/∂θ log π_θ(a) times the received reinforcement. Provides the foundation for modern policy-gradient RL.

## Key Contributions
- **Policy gradient theorem** for stochastic policies — derives `∇_θ J(θ) = E_{a~π_θ}[ ∇_θ log π_θ(a) · R ]`.
- **Eligibility trace / score function**: `∇_θ log π_θ(a|s)` depends only on the chosen action's log-prob.
- Proves that adding a state-dependent baseline b(s) is unbiased and reduces variance.
- Algorithmic template for every modern policy-gradient method (PPO, TRPO, RLOO, GRPO).

## Key Figures/Tables to Study
- **Section 2:** Derivation of the score function gradient.
- **Theorem 1 (baseline invariance).**
- Modern pedagogical re-presentation: Sutton & Barto Ch. 13.

## Technical Details

### Policy gradient theorem
For episodic return R = Σ_t r_t:
`∇_θ J(θ) = E_{τ~π_θ}[ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τ) ]`

Per-step variant (causal form):
`∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ]`
where `G_t = Σ_{t'≥t} γ^{t'−t} r_{t'}` is the return-to-go.

### Baseline
For any function b(s) independent of the action:
`∇_θ J = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · ( G_t − b(s_t) ) ]`
Still unbiased (because `E_a[∇ log π · b(s)] = 0`), lower variance.

### REINFORCE update
`θ ← θ + α · (R − b) · ∇_θ log π_θ(a | s)`

### Common baseline choices
| Method | Baseline |
|--------|----------|
| Original REINFORCE | Running average of returns |
| Actor-Critic | Learned V_φ(s) |
| RLOO (LLM RL) | Mean of other k−1 rollouts for the same prompt |
| GRPO (LLM RL) | Mean over G group samples, std-normalized |
| PPO | V_φ(s) with GAE advantage |

### LLM-specific form
For a sequence y = y_1,…,y_T given prompt x and terminal reward R(x,y):
`∇_θ J = E[ R(x,y) · Σ_t ∇_θ log π_θ(y_t | x, y_<t) ]`
One advantage per sequence; KL penalty typically added as a token-level shaped reward.

### Variance issues
- Raw REINFORCE variance scales with |y| (sum of T log-prob gradients).
- Mitigations: baseline, importance clipping (PPO), multiple samples per prompt (RLOO/GRPO), advantage normalization.

## Connections
- Modern trust-region: [[trpo]].
- First-order trust region: [[ppo]].
- Leave-one-out LLM variant: [[rloo]].
- Group-baseline LLM variant: [[grpo]].
- Bias-corrected: [[dr-grpo]].
- Variance-reduction survey / 2025 variant: [[reinforce-plus-plus]].
