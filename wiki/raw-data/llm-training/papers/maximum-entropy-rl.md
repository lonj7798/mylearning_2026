<!-- scope: maximum-entropy RL objective (SAC) — ancestor of LLM entropy bonuses
     deps: [[entropy-regularization-ppo]]
     see-also: [[entropy-mechanism-llm-rl]], [[kl-control-rlhf]]
-->

# Soft Actor-Critic and Maximum Entropy RL
- **Core Insight:** Augment the RL objective with a per-step entropy bonus `α · H(π(·|s))` — the optimal policy is `π*(a|s) ∝ exp(Q_soft(s,a)/α)`, which both explores better and is provably more robust than the greedy solution.
- **Guideline:** Tune the entropy coefficient `α` automatically against a target entropy `H̄`; the LLM-era analogue is a target-entropy schedule (keep per-token entropy above a floor via a learned or annealed coefficient), not a fixed entropy bonus.
- **Authors:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
- **Year:** 2018
- **URL:** https://arxiv.org/abs/1801.01290 (SAC); https://arxiv.org/abs/1812.05905 (SAC v2 with auto α)
- **Relevant topics:** maximum-entropy RL, soft Q-learning, entropy bonus α, exploration, temperature auto-tuning

## Abstract
SAC introduces an off-policy actor-critic algorithm that maximizes expected reward plus expected entropy:
  `J(π) = Σ_t E[r(s_t, a_t) + α · H(π(·|s_t))]`.
The soft Bellman operator becomes
  `Q(s,a) ← r(s,a) + γ E_{s'}[V(s')]`, with `V(s) = E_{a~π}[Q(s,a) − α log π(a|s)]`,
and the optimal stochastic policy is the Boltzmann distribution `π*(a|s) ∝ exp(Q(s,a)/α)`. The follow-up paper adds automatic temperature tuning: keep `α` such that `E[−log π(a|s)] ≈ H̄`, the target entropy. SAC dramatically improves sample efficiency and stability across continuous-control benchmarks (HalfCheetah, Humanoid, Ant) and has become the default max-ent RL reference.

## Key Contributions
- **Max-ent objective:** `J(π) = Σ_t E[r_t + α H(π(·|s_t))]` — rewards are valued together with high-entropy (random) behavior.
- **Soft value functions:** `V(s) = α · log ∫ exp(Q(s,a)/α) da` — log-sum-exp over actions instead of max.
- **Optimal-policy form:** `π*(a|s) = exp((Q(s,a) − V(s)) / α)`.
- **Twin Q networks + target networks:** practical stabilization — two critics trained, min used in targets.
- **Automatic α tuning (v2):** solve `α = argmin_α E[−α · log π(a|s) − α · H̄]`; gradient-descent on α with a target entropy like `H̄ = −dim(A)`.
- **Robustness:** max-ent policies tolerate environment perturbations better than deterministic optimal policies — an argument that reappears in LLM RL where a collapsed policy is brittle to distribution shift.

## Key Figures/Tables to Study
- **Fig. 1** (toy bimodal Q landscape) — shows why the Boltzmann policy keeps both modes while greedy collapses to one.
- **Fig. 3** (HalfCheetah / Humanoid learning curves) — SAC outperforms DDPG, TD3, PPO on sample efficiency.
- **SAC-v2 Fig. 1** (auto α curve) — α adapts per environment.

## Technical Details
- **Soft Q update:** `L_Q(θ) = E[(Q_θ(s,a) − (r + γ (Q_target(s',a') − α log π(a'|s'))))^2]`.
- **Policy update (reparam trick):** minimize `E_{a = f(ε; s)}[α log π(a|s) − Q(s,a)]`.
- **α tuning:** `L(α) = E[−α (log π(a|s) + H̄)]`; target entropy `H̄ = −dim(action space)` is the conventional default.
- **Why it matters for LLMs:** SAC's auto-α is the direct ancestor of "keep policy entropy above H̄ during GRPO" strategies; modern LLM-RL papers (**[[entropy-mechanism-llm-rl]]**) are rediscovering that a fixed β on `H(π)` is inferior to a target-entropy-based adaptive term.

## Connections
- Provides the theoretical ancestor for the entropy bonus used in PPO (**[[entropy-regularization-ppo]]**) and for entropy-targeting in LLM-RL (**[[entropy-mechanism-llm-rl]]**).
- The "policy ∝ exp(Q/α)" form is exactly the Korbak-view target distribution for KL-regularized RLHF (**[[kl-control-rlhf]]**) with reward/α playing the role of logits.
- Contrast with on-policy PPO: SAC uses off-policy replay + soft Q learning; LLM RL is largely on-policy so max-ent becomes an entropy-bonus approximation.
