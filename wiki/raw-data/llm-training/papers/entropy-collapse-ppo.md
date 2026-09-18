<!-- scope: large-scale study of >50 implementation choices in PPO-style on-policy RL on five MuJoCo tasks, including entropy regularization (the slug predates this revision; the paper does not study entropy collapse)
     deps: [[ppo]]
     see-also: [[entropy-regularization-ppo]], [[entropy-mechanism-llm-rl]], [[costa-huang-ppo-details]], [[maximum-entropy-rl]]
-->

# What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
- **Core Insight:** Across more than 250,000 agents trained on five MuJoCo tasks, none of the tested policy regularizers, including an entropy penalty (coefficients 10^-5 to 3×10^-3) and an entropy constraint, helped significantly except on HalfCheetah, while scaling the last policy layer's initial weights down alone raised Humanoid performance by 66% (§3.8, App. K.1, §3.2 Fig. 24).
- **Guideline:** When training PPO for continuous control with the last policy layer initialized with 100× smaller weights (as in the base configuration of the regularization experiment), an entropy bonus is not expected to improve performance (§3.8, App. C); tune the initial action standard deviation (0.5 was best on 4 of 5 tasks), the discount factor (start at 0.99), and the PPO clipping threshold (start at 0.25) first (§3.2, §3.6, §3.1).
- **Authors:** Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, et al. (12 authors; Google Research, Brain Team)
- **Year:** 2020 (arXiv v1 2020-06; no venue listed on the arXiv page)
- **URL:** https://arxiv.org/abs/2006.05990
- **Source type:** paper
- **Relevant topics:** PPO implementation choices, policy losses, entropy regularization, KL regularization, advantage normalization, policy initialization, hyperparameter study design, continuous control

## Abstract
State-of-the-art on-policy RL implementations contain many low- and high-level design decisions that are rarely discussed in papers, which makes it hard to attribute progress. The authors implement more than 50 such choices in one configurable on-policy agent, train over 250,000 agents in five continuous-control environments of different complexity, and derive practical recommendations for on-policy training.

## Key Contributions
- A unified on-policy agent built on the SEED RL code base that exposes each choice as a numbered option (C1-C68); with OpenAI Baselines PPO settings it reproduces the performance reported in the PPO paper (§2, App. B).
- A design for studying interacting choices: eight thematic groups (policy losses, architecture, normalization and clipping, advantage estimation, training setup, timesteps, optimizers, regularization); within a group, choices and the Adam learning rate are sampled uniformly at random, and all other choices stay at a base configuration close to PPOv2 with 256 parallel environments (§2, §3).
- Two analyses per choice: the 95th percentile of performance conditioned on each value, with a binomial confidence interval (an estimate of the result of random search over about 20 configurations), and the distribution of values among the top 5% of configurations (§2).
- Recommendations per group; the authors name the effect of the initial action distribution as the most surprising finding (§1, §3.1-3.7).

## Key Figures/Tables to Study
- **Fig. 1 / Fig. 5:** policy losses compared, with random and with best-conditioned loss hyperparameters.
- **Fig. 2:** initial action standard deviation (C61). **App. C Table 2:** base configuration.
- **Figs. 76-77:** regularization type (none / penalty / constraint). **Fig. 81:** entropy penalty coefficient (C46). **Fig. 83:** entropy constraint threshold (C40). **Table 10:** performance quantiles of the regularization experiment.
- **Fig. 35:** per-minibatch advantage normalization (C67). **Fig. 65:** learning-rate decay (C31).

## Technical Details
- **Environments:** Hopper-v1, Walker2d-v1, HalfCheetah-v1, Ant-v1, Humanoid-v1 from OpenAI Gym, MuJoCo 2.0 (§2).
- **Scoring:** 3 seeds per configuration; 1M environment steps (Hopper, HalfCheetah, Walker2d) or 2M (Ant, Humanoid); evaluation every 100k steps on 100 episodes with the stochastic policy; seed score = mean of evaluations (proportional to area under the learning curve); configuration score = median over seeds (§2).
- **PPO loss as implemented:** L = -min(ρ·Â, clip(ρ, 1/(1+ε), 1+ε)·Â), ρ = π(a|s)/μ(a|s), μ = behavioral policy; the lower bound 1/(1+ε) replaces 1-ε of the PPO paper (App. B.3, footnote 19).
- **Policy losses compared (C14):** PG, V-trace, PPO, AWR, V-MPO, and RPA, L_RPA = -log π(a_t|s_t)·[A_t > 0] (App. B.3). PPO performed better than the other losses on 4 of 5 environments (§3.1, Fig. 1).
- **Regularization modes (App. B.6):** penalty α·R with fixed α; or soft constraint R < ε enforced by a Lagrange multiplier α = exp(c·p), c = 10, p clipped to [log(10^-6)/10, log(10^6)/10]. Regularizers: entropy H(π(·|s)), KL(μ‖π), KL(π‖μ), KL(N(0,1)‖π), decoupled KL(μ‖π). One regularizer per run.
- **Regularization sweep (App. K.1):** 4000 sampled configurations per environment; entropy penalty coefficient in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}; entropy constraint threshold in {0.0, -5.0, -10.0, -15.0}; Adam LR in {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}; 3 seeds each.
- **Regularization result (§3.8):** no regularizer helped significantly except on HalfCheetah, where all constraints helped, the entropy constraint most. The gain was largely independent of the threshold, which the authors attribute to the initial penalty strength before α adapts. Their conjectured reasons (Interpretation): the PPO loss already enforces a trust region, and the policy initialization already gives enough exploration.
- **Action distribution (App. B.8):** Gaussian with a clip or tanh transform; with tanh, the log tanh′(x) term matters only for entropy regularization and bounds the entropy.
- **Other measured effects:** per-minibatch advantage normalization "seems not to affect the performance too much" (§3.3, Fig. 35); linear LR decay to 0 improved 4 of 5 tasks, with small gains except +15% on Ant (§3.7, Fig. 65); observation normalization is needed on all tasks except Hopper (§3.3); PPO-style value clipping and Huber value loss hurt (§3.4); ε = 0.2 and 0.3 work on all tasks, 0.1 or 0.5 are better on some (§3.1).

## Recipe ledger
All rows: unified PPO agent in the base configuration of App. C Table 2 (MLP policy and value, shared, 2 layers × 64 tanh units), five MuJoCo tasks; locus arXiv:2006.05990v1.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Unified PPO agent (base config) | MLP 2×64 | RL | Policy loss; clip ε | PPO; 0.2 | App. C Table 2 | verified 2026-09-14 | §3.1 Fig. 1: PPO best on 4 of 5 tasks; recommendation: start at ε = 0.25 |
| Unified PPO agent (base config) | MLP 2×64 | RL | Parallel envs; transitions per iteration; epochs; minibatch size | 256; 2048; 10; 64 | App. C Table 2 | verified 2026-09-14 | §3.5 Fig. 54: multiple passes needed; Fig. 57: larger batch did not hurt sample complexity in the tested range; Fig. 52: transitions per iteration matter |
| Unified PPO agent (base config) | MLP 2×64 | RL | Optimizer; LR; β1; Adam ε; LR decay (terminal fraction) | Adam; 3e-4; 0.9; 1e-7; 0.0 (linear decay to 0) | App. C Table 2; App. B.5 | verified 2026-09-14 | §3.7 Figs. 65, 69: 3e-4 performs well on all tasks; decay helps 4 of 5 |
| Unified PPO agent (base config) | MLP 2×64 | RL | Advantage estimator; GAE λ; discount γ | GAE; 0.95; 0.99 | App. C Table 2 | verified 2026-09-14 | §3.4: λ = 0.9 recommended; §3.6 Fig. 60: γ = 0.99 reasonable on all tasks |
| Unified PPO agent (base config) | MLP 2×64 | RL | Regularization type | None | App. C Table 2 | verified 2026-09-14 | §3.8 Figs. 76-77: no significant gain except HalfCheetah |
| Unified PPO agent (regularization sweep) | MLP 2×64 | RL | Entropy penalty coefficient (C46) | {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3} | App. K.1 | verified 2026-09-14 | Fig. 81; §3.8 reports no significant gain |
| Unified PPO agent (base config) | MLP 2×64 | RL | Per-minibatch advantage normalization; gradient-norm clip | False; 0.5 | App. C Table 2 | verified 2026-09-14 | §3.3 Fig. 35 (little effect), Fig. 34 (small gain from clipping) |
| Unified PPO agent (base config) | MLP 2×64 | RL | Initial action std; last policy layer scaling | 1.0; 0.01 | App. C Table 2 | verified 2026-09-14 | §3.2 Fig. 2: 0.5 best except Hopper; recommendation: 100× smaller last-layer weights |
| Unified PPO agent | MLP 2×64 | RL | Compute (hardware, hours) | not reported | checked §2, §3.5, App. C, App. K | not reported | — |

## Findings relevant to generality, negative feedback
- **Negative feedback:** RPA trains only on actions with positive advantage and V-MPO uses only the top half of advantages in each batch (App. B.3). PPO, which uses advantages of both signs, outperformed all other losses on 4 of 5 tasks, and still outperformed them on Humanoid and Ant when each loss used its best hyperparameters (§3.1, Figs. 1, 5). The paper does not attribute this difference to the use of negative advantages.
- **Generality of the recommendations:** several optima differ by task: policy MLP width (Fig. 18), γ ("should be tuned per environment", §3.6), clipping ε (§3.1), initial std (Hopper prefers higher values, Fig. 2). Ant and Humanoid were not in the original PPO paper, so the base hyperparameters were not tuned for them (§3.1, footnote 7). The authors state that the study does not claim one policy loss is better in general (§3.1).
- **Scope:** continuous control only; no discrete-action or language-model experiments, and the word "collapse" does not occur in the paper (full-text search).

## Connections
- [[ppo]] — base algorithm; the base configuration is close to the OpenAI Baselines PPOv2 defaults, scaled up to 256 parallel environments (§2).
- [[costa-huang-ppo-details]] — catalogue of PPO implementation details; this paper measures many of them in controlled sweeps.
- [[entropy-regularization-ppo]] — A3C paper where the entropy regularizer tested here was used in actor-critic training.
- [[entropy-mechanism-llm-rl]] — LLM RLVR study that also reports entropy loss did not improve accuracy, in a different setting.
- [[maximum-entropy-rl]] — SAC, cited here [21] as an off-policy method for continuous control.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2006.05990 (arXiv v1, the only version).
- Corrections to the previous card version:
  - Title "What Matters in On-Policy RL (Andrychowicz 2020) + Entropy Collapse in LLM PPO" → exact paper title; the old card joined the paper with LLM observations that have no source.
  - "~250,000 MuJoCo trials across a factorial grid of 50 hyperparameters" → over 250,000 agents, more than 50 choices, random sampling within eight groups, not a factorial grid (Abstract, §2).
  - "Ranking: policy loss, advantage normalization, LR, epochs, clipping; entropy coefficient mid-tier; advantage normalization matters more than the entropy bonus" → no global ranking is given; per-minibatch advantage normalization had little effect (§3.3); regularizers gave no significant gain except HalfCheetah (§3.8).
  - "c_H ∈ [0, 0.005] is a reasonable default; c_H swept over [0, 1e-4, 1e-3, 1e-2, 1e-1]" → sweep {1e-5 … 3e-3} (App. K.1); no entropy default recommended; base configuration uses no regularization (Table 2).
  - "ε = 0.2 near optimal; smaller ε reduces entropy-collapse risk" → 0.2 and 0.3 work on all tasks, recommendation 0.25 (§3.1); no entropy statement.
  - "linear decay beats constant; constant LR drives entropy to zero" → decay helps 4 of 5 tasks, +15% on Ant (§3.7); no entropy statement.
  - "Fig. 3 effect-size ranking; Fig. 7 entropy × advantage-normalization heatmap" → Fig. 3 = training curves (App. D); Fig. 7 = policy-loss analysis (C14); entropy coefficient = Fig. 81.
  - "exact analytical entropy per categorical distribution" → Gaussian action distributions (App. B.8).
- Removed as unsupported by the source: the whole "Collapse in LLM PPO" section (entropy 2-3 nats → <0.1 nats in a few hundred updates, conditions, "sudden inflection" diagnostic, "lower the advantage normalization std floor"); "entropy collapse is repeatedly the mechanism behind failures"; "largest-scale ablation to date"; "report entropy curves in any on-policy RL paper"; per-position entropy and "fraction of tokens with H < 0.1" logging advice; "the qualitative ranking survives for LLMs"; the "Cui 2025 Fig. 1 reproduction" pointer.
- Not reported by the source: compute budget; any LLM, discrete-action, or entropy-trajectory result.
