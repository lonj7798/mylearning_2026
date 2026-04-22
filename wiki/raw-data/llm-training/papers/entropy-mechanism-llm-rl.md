<!-- scope: policy-entropy collapse during RL for reasoning LLMs; mechanism + intervention
     deps: [[entropy-regularization-ppo]]
     see-also: [[maximum-entropy-rl]], [[kl-control-rlhf]], [[deepseek-r1]]
-->

# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models
- **Core Insight:** Policy entropy collapses monotonically and predictably early in RL for reasoning LLMs; the downstream task reward traded for entropy follows an exponential law `R = -a·exp(H) + b`, so performance is bounded by how fast entropy dies.
- **Guideline:** Track policy entropy as a first-class metric every step; when it drops into the collapse regime (typically `H < 0.1` on the last-token distribution), intervene by clipping or KL-penalizing the high-advantage + high-probability tokens that drive covariance instead of globally re-warming temperature.
- **Authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, Ning Ding
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.22617
- **Relevant topics:** policy entropy, entropy collapse, RLVR, PPO/GRPO exploration, covariance of logit updates, Clip-Cov, KL-Cov

## Abstract
The paper investigates the sharp collapse of policy entropy observed when applying RL (PPO, GRPO, RLOO, Reinforce++) to reasoning LLMs. Without explicit entropy management, entropy drops rapidly in the first hundreds of updates and reward saturates. The authors derive a closed-form transformation law between entropy `H` and downstream reasoning accuracy `R`, and show a theoretical link between per-token entropy change and the covariance of the action log-prob with its logit update. They propose two minimally invasive interventions, **Clip-Cov** and **KL-Cov**, that target only the tokens driving the collapse.

## Key Contributions
- **Empirical law:** Across >20 models and settings, `R = -a·exp(H) + b` — once H is small, further entropy loss yields diminishing reward; the "performance ceiling" is reached as `H → 0`.
- **Mechanistic theorem:** For softmax policies under policy-gradient updates, the expected change in token entropy is proportional to `-Cov_{a~π}(log π(a|s), A(s,a))` — large advantages on already-high-probability tokens are what burn entropy.
- **Clip-Cov:** Identify the top-k tokens with highest covariance (high probability × high advantage) and detach them from the gradient — lets the rest of the distribution keep updating without letting the sharp spikes dominate.
- **KL-Cov:** Instead of clipping, apply a per-token KL penalty only to those covariance outliers, preserving average gradient flow.
- **Quantitative gain:** On Qwen2.5-7B / Qwen2.5-Math-7B base RL, entropy is kept meaningfully above the collapse floor for the full run, and AIME / MATH accuracy ceiling rises several points over vanilla GRPO without entropy control.

## Key Figures/Tables to Study
- **Fig. 1** (entropy vs step across algorithms): same exponential-decay shape regardless of PPO/GRPO/RLOO — collapse is algorithm-agnostic.
- **Fig. 2** (R vs H fit): the `R = -a·exp(H) + b` curve with each point being a training checkpoint; reward asymptotes as H approaches 0.
- **Fig. 3** (covariance histogram): long-tail of tokens with huge prob×advantage — these are the entropy killers.
- **Table of interventions:** vanilla vs entropy-bonus vs Clip-Cov vs KL-Cov — entropy-bonus is shown to over-correct and hurt accuracy; Clip-Cov / KL-Cov hit a better point on the entropy-performance frontier.

## Technical Details
- **Entropy definition used:** per-step Shannon entropy of the next-token distribution averaged over tokens in rollout: `H(π) = − E_s E_{a~π(·|s)} log π(a|s)`.
- **Predictive law:** `R(step) = −a · exp(H(step)) + b`; fit `a, b` on the first ~10% of training to predict the ceiling.
- **Collapse signal (practical):** token-level `H < 0.1` nats sustained for multiple updates.
- **Vanilla entropy bonus** (A2C-style) — adds `+ β · H(π)` to the loss, where the paper empirically found β in `{1e-4, 1e-3, 1e-2}` either under-corrects or over-corrects; treating all tokens symmetrically hurts high-quality trajectories.
- **Clip-Cov:** rank tokens by `p_t · A_t` per batch, set gradient of the top fraction (e.g. 2%) to zero.
- **KL-Cov:** for those same top-covariance tokens apply `β_KL · KL(π_new‖π_old)` (forward, token-level, k3 approximation).
- **Base RL recipe:** GRPO with group size 8–16, rollout length up to 8k tokens, learning rate ~1e-6, no SFT warm-start for R1-Zero-style comparisons.

## Connections
- The covariance-driven collapse explains empirically why **[[deepseek-r1]]** needs cold-start SFT + long rollouts and rule-based rewards — high-advantage rule signals can burn entropy faster than dense RM signals.
- Complements **[[maximum-entropy-rl]]**: SAC-style temperature tuning is a symmetric fix; this paper argues asymmetric (covariance-targeted) intervention is better for LLMs.
- Directly relevant to **[[entropy-regularization-ppo]]** debate — entropy bonus alone is not enough at LLM scale.
- Interacts with **[[kl-control-rlhf]]**: KL-to-reference and KL-Cov serve different roles (stay near SFT vs keep distribution wide).
