<!-- scope: Proximal Policy Optimization — clipped surrogate objective for on-policy RL
     deps: [[trpo]], [[vanilla-pg]]
     see-also: [[rlhf-instructgpt]], [[grpo]], [[rloo]]
-->

# Proximal Policy Optimization Algorithms
- **Core Insight:** A clipped ratio objective gives trust-region-like stability without second-order optimization, enabling K epochs of minibatch SGD on the same rollout.
- **Guideline:** Default to PPO-clip for on-policy RL; set clip ε≈0.2, K=3–10 epochs, GAE λ=0.95, add a small entropy bonus; increase K with caution — too many epochs push the policy outside the trust region.
- **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **Year:** 2017
- **URL:** https://arxiv.org/abs/1707.06347
- **Relevant topics:** on-policy RL, policy gradient, trust region, clipping, actor-critic, GAE, RLHF policy optimizer

## Abstract
Proposes a new family of policy-gradient methods for RL that alternate between sampling data through interaction with the environment and optimizing a "surrogate" objective with stochastic gradient ascent. Whereas standard policy-gradient methods perform one gradient update per sample, PPO enables multiple epochs of minibatch updates via a clipped probability ratio that acts as a pessimistic (lower-bound) surrogate of the policy performance. PPO has the benefits of trust-region methods (TRPO) but is simpler, more general, and has better sample complexity.

## Key Contributions
- **Clipped surrogate** L^CLIP — a first-order objective that prevents destructively large policy updates without computing a Fisher-vector product.
- **Multiple epochs per rollout** — the same batch is reused for K epochs of minibatch SGD.
- **Shared actor-critic loss** — combines policy, value, and entropy losses into a single objective L^CLIP+VF+S.
- **GAE-based advantages** — generalized advantage estimation with bias-variance knob λ.
- Strong empirical performance on continuous-control (MuJoCo) and Atari benchmarks.
- Became the default RL optimizer for RLHF (InstructGPT, Claude, early LLaMA).

## Key Figures/Tables to Study
- **Figure 1:** A single timestep of the clipped vs unclipped surrogate as a function of the probability ratio r — visual intuition for why clipping removes the incentive to go outside [1-ε, 1+ε].
- **Figure 3 / Table 3:** MuJoCo learning curves vs A2C / TRPO / vanilla PG — PPO dominates.
- **Section 5 (Algorithm 1):** Actor-critic pseudo-code — canonical reference for implementations.

## Technical Details

### Probability ratio
`r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)`
Equals 1 at the start of each epoch; deviates as θ updates.

### Clipped surrogate objective
`L^CLIP(θ) = E_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]`
Take the element-wise minimum of the unclipped and clipped terms → pessimistic lower bound on the true policy improvement.

### Combined objective
`L^CLIP+VF+S(θ) = E_t[ L^CLIP(θ) − c1 · L^VF(θ) + c2 · S[π_θ](s_t) ]`
- `L^VF = (V_θ(s_t) − V_target)^2` — MSE on returns.
- `S[π_θ] = −Σ_a π_θ(a|s) log π_θ(a|s)` — entropy bonus.

### Generalized advantage estimation
`δ_t = r_t + γ V(s_{t+1}) − V(s_t)` (TD error)
`Â_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + …`
λ trades bias for variance; λ=1 recovers Monte Carlo, λ=0 recovers TD(0).

### Canonical hyperparameters (MuJoCo)
| Knob | Value |
|------|-------|
| Clip ε | 0.2 |
| Value-loss coef c1 | 1.0 |
| Entropy coef c2 | 0.01 |
| Epochs per rollout K | 3–10 |
| Minibatch size | 64 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| Learning rate | 3e-4 (Adam) |
| Horizon T per actor | 2048 |

### Token-level vs sequence-level in LLM RL
Original PPO works at timestep level; RLHF implementations apply the clipped ratio per token and average (token-level). The KL penalty is usually added to the per-token reward before computing advantages (sometimes called "KL-reward" style). Entropy is typically disabled or set very small because the Bradley-Terry KL penalty already regularizes the policy.

## Connections
- Trust-region ancestry: [[trpo]] — same motivation, but second-order.
- LLM instantiation: [[rlhf-instructgpt]] — applies PPO-ptx with a KL term on every token.
- Grouped baseline variant: [[grpo]] — drops the value network and normalizes by group statistics.
- Simpler alternative that beats PPO for LLMs: [[rloo]].
- Modern variance-reduction extension: [[reinforce-plus-plus]].
- 37 implementation details blog: [[costa-huang-ppo-details]].
