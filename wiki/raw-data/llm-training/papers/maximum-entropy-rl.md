<!-- scope: Soft Actor-Critic (Haarnoja et al., ICML 2018): maximum-entropy RL objective, soft policy iteration, off-policy stochastic actor-critic for continuous control
     deps: [[entropy-regularization-ppo]]
     see-also: [[entropy-mechanism-llm-rl]], [[kl-control-rlhf]], [[ppo]]
-->

# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
- **Core Insight:** Maximizing expected reward plus policy entropy with an off-policy stochastic actor-critic lets SAC match the baselines (DDPG, PPO, SQL, TD3) on the easier continuous-control tasks and outperform them on the harder ones, including the 21-action-dimension Humanoid (rllab), over 5 seeds per method (§5.1, Figure 1).
- **Guideline:** When a maximum-entropy objective is used with a fixed temperature, tune the reward scale (the inverse temperature) per task, because in SAC a small scale gave a near-uniform policy that failed to exploit the reward and a large scale gave a near-deterministic policy that stalled in poor local minima (§5.2, Figure 3b); the authors report reward scale as the only hyperparameter they tuned per environment (§5.2, App. D Table 2).
- **Authors:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
- **Year:** 2018 (arXiv v1 2018-01; ICML 2018)
- **URL:** https://arxiv.org/abs/1801.01290
- **Source type:** paper
- **Relevant topics:** maximum-entropy RL, soft policy iteration, soft Q-function, temperature, reward scale, off-policy actor-critic, exploration, stochastic policies

## Abstract
Model-free deep RL methods have high sample complexity and brittle convergence that needs careful hyperparameter tuning. The paper proposes soft actor-critic (SAC), an off-policy actor-critic algorithm in the maximum-entropy RL framework, where the actor maximizes expected reward and entropy at the same time. Earlier deep maximum-entropy methods were Q-learning methods. SAC combines off-policy updates with a stable stochastic actor-critic formulation. It reports state-of-the-art results on continuous-control benchmarks against prior on-policy and off-policy methods, and similar performance across random seeds.

## Key Contributions
- Soft policy iteration for the tabular case with convergence results: soft policy evaluation (Lemma 1), soft policy improvement (Lemma 2), and convergence to the optimal policy within a policy class Π (Theorem 1) (§4.1, App. B).
- A practical algorithm with a soft state-value network, two soft Q-networks, a Gaussian policy, and a target value network, trained from a replay buffer (§4.2, Algorithm 1).
- A reparameterized policy gradient that extends DDPG-style gradients to any tractable stochastic policy (§4.2, Eq. 11–13).
- Comparison with DDPG, PPO, SQL and TD3 on six benchmarks (§5.1), plus Trust-PCL and two SAC variants (App. E).
- Ablations of stochastic vs deterministic policy, evaluation mode, reward scale, and target smoothing τ (§5.2).

## Key Figures/Tables to Study
- **Figure 1:** returns on Hopper-v1, Walker2d-v1 (x-axis 1M steps), HalfCheetah-v1, Ant-v1 (3M), Humanoid-v1, Humanoid rllab (10M); 5 seeds, one evaluation rollout every 1000 environment steps; line = mean, band = min/max (§5.1).
- **Figure 2:** 5 seeds of SAC vs a deterministic variant on Humanoid (rllab) (§5.2).
- **Figure 3:** Ant-v1 sensitivity: (a) deterministic vs stochastic evaluation, (b) reward scale 1/3/10/30/100, (c) τ = 0.0001/0.001/0.01/0.1 (§5.2).
- **App. D Tables 1–2:** shared hyperparameters and per-environment reward scale.

## Technical Details
- **Objective (Eq. 1):** `J(π) = Σ_{t=0..T} E_{(s_t,a_t)~ρ_π} [ r(s_t,a_t) + α·H(π(·|s_t)) ]`.
  `ρ_π` is the state-action marginal of the trajectory distribution under policy `π`; `r` is the bounded reward; `H` is entropy; `α` is the temperature. The standard objective is recovered as `α → 0`. The paper then drops `α` and absorbs it into the reward by scaling the reward by `α⁻¹` (§3.2). The discounted form is Eq. 14 (App. A).
- **Soft Bellman backup (Eq. 2–3):** `T^π Q(s_t,a_t) = r(s_t,a_t) + γ·E_{s_{t+1}~p}[V(s_{t+1})]`, with `V(s_t) = E_{a_t~π}[Q(s_t,a_t) − log π(a_t|s_t)]`. `γ` is the discount factor and `p` the transition density.
- **Policy improvement (Eq. 4):** `π_new = argmin_{π'∈Π} D_KL( π'(·|s_t) ‖ exp(Q^{π_old}(s_t,·)) / Z^{π_old}(s_t) )`. `Z` is the partition function; it does not affect the gradient. The proofs assume a finite action set `|A| < ∞` (§4.1).
- **Value loss (Eq. 5–6):** `J_V(ψ) = E_{s_t~D}[ ½ (V_ψ(s_t) − E_{a_t~π_φ}[Q_θ(s_t,a_t) − log π_φ(a_t|s_t)])² ]`. `D` is the replay buffer; actions in the estimator are sampled from the current policy, not the buffer.
- **Q loss (Eq. 7–8):** `J_Q(θ) = E_{(s_t,a_t)~D}[ ½ (Q_θ(s_t,a_t) − Q̂(s_t,a_t))² ]`, `Q̂ = r + γ·E_{s_{t+1}~p}[V_ψ̄(s_{t+1})]`. `ψ̄` is an exponential moving average of `ψ`.
- **Policy loss (Eq. 11–12):** `a_t = f_φ(ε_t; s_t)` with noise `ε_t ~ N`; `J_π(φ) = E_{s_t~D, ε_t~N}[ log π_φ(f_φ(ε_t;s_t)|s_t) − Q_θ(s_t, f_φ(ε_t;s_t)) ]`.
- **Two Q-functions:** trained independently; their minimum is used in the value gradient (Eq. 6) and the policy gradient (Eq. 13). One Q-function can learn the 21-dimensional Humanoid, but two sped up training on harder tasks (§4.2).
- **Action bounds:** a tanh squashing of a Gaussian sample, with `log π(a|s) = log μ(u|s) − Σ_i log(1 − tanh²(u_i))` (App. C, Eq. 21).
- **Evaluation:** SAC is evaluated with the mean action; in Figure 3a this gives higher return than stochastic evaluation (§5, §5.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SAC (Figure 1 runs, all 6 tasks) | 2 hidden layers × 256 units, all networks | RL | optimizer; learning rate | Adam; 3·10⁻⁴ | arXiv:1801.01290v2 App. D Table 1 | verified 2026-09-14 | no ablation reported |
| SAC (Figure 1 runs) | same | RL | discount γ; replay buffer size | 0.99; 10⁶ | App. D Table 1 | verified 2026-09-14 | no ablation reported |
| SAC (Figure 1 runs) | same | RL | samples per minibatch; nonlinearity | 256; ReLU | App. D Table 1 | verified 2026-09-14 | no ablation reported |
| SAC (Figure 1 runs) | same | RL | target smoothing τ; target update interval; gradient steps per env step | 0.005; 1; 1 | App. D Table 1; §5.2 | verified 2026-09-14 | §5.2 Figure 3c (Ant-v1): large τ unstable, small τ slower; same τ used on all tasks |
| SAC (hard target update variant) | same | RL | τ; target update interval; gradient steps | 1; 1000; 4 (1 on humanoids) | App. D Table 1 | verified 2026-09-14 | App. E Figure 4: comparable to SAC except Humanoid (rllab) |
| SAC | same | RL | reward scale per task | Hopper-v1 5, Walker2d-v1 5, HalfCheetah-v1 5, Ant-v1 5, Humanoid-v1 20, Humanoid (rllab) 10 | App. D Table 2 | verified 2026-09-14 | §5.2 Figure 3b: sweep 1–100 on Ant-v1 only |
| SAC | same | RL | seeds; evaluation frequency | 5 per algorithm; 1 rollout per 1000 env steps | §5.1 | verified 2026-09-14 | no ablation reported |
| SAC | same | RL | automatic temperature; target entropy | not reported (α is absorbed into reward scale) | checked §3–5, App. A–E | not reported | n/a |

## Findings relevant to generality
The setting is continuous control, not language models.
- The authors state that the objective lets the policy capture multiple modes of near-optimal behavior and give equal probability mass to equally attractive actions (§3.2). This is stated as a property, not measured.
- A deterministic variant of SAC showed higher return variability across 5 seeds on Humanoid (rllab) than SAC; the paper gives the curves but no numeric spread statistic (§5.2, Figure 2).
- A near-deterministic policy caused by a large reward scale reached poor local minima, which the authors attribute to lack of exploration (§5.2, Figure 3b).
- Robustness of maximum-entropy policies to model and estimation errors is cited from Ziebart (2010) and is not tested in this paper (§1).

## Connections
- Follow-up artifact (no card in this library): "Soft Actor-Critic Algorithms and Applications", Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan et al., https://arxiv.org/abs/1812.05905 (v1 2018-12). It drops the separate value network (§4.2 footnote 1), constrains average entropy `E[−log π_t(a_t|s_t)] ≥ H̄` (§5, Eq. 11), learns α with `J(α) = E_{a_t~π_t}[−α log π_t(a_t|s_t) − α H̄]` (Eq. 18), and uses target entropy `−dim(A)`, e.g. −6 for HalfCheetah-v1 (App. D Table 1). Automatic α should be cited to that paper.
- [[entropy-regularization-ppo]]: SAC §2 notes that earlier actor-critic methods (TRPO, PPO, A3C) use entropy as a regularizer instead of maximizing it inside the objective.
- [[ppo]]: on-policy baseline in Figure 1; §5.1 attributes its slower learning to the large batch sizes PPO needs on high-dimensional tasks.
- [[kl-control-rlhf]]: (Interpretation, not in SAC) the KL-regularized RLHF optimum `π_ref·exp(r/β)/Z` has the same exponential form as the Eq. 4 target, with a reference policy in place of a uniform base.
- [[entropy-mechanism-llm-rl]]: studies entropy collapse in RL for reasoning language models; SAC does not study language models.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/1801.01290 (arXiv v2, 2018-08-08); Connections pointer checked against https://arxiv.org/abs/1812.05905 (arXiv v2, 2019-01-29).
- Corrections to the previous card version:
  - The card mixed two artifacts (authors and first URL of 1801.01290; automatic α and "SAC-v2 Fig. 1" from 1812.05905) → card now describes 1801.01290; automatic α moved to a Connections pointer with loci.
  - Title "Soft Actor-Critic and Maximum Entropy RL" → exact published title.
  - "provably more robust than the greedy solution" → the paper proves convergence of soft policy iteration (Theorem 1); robustness is cited from Ziebart (2010), not proven or tested (§1).
  - Soft Q target `r + γ(Q_target(s',a') − α log π(a'|s'))` → in this paper the target is `r + γ·E[V_ψ̄(s')]` with a target value network (Eq. 7–8).
  - "two critics trained, min used in targets" → the minimum is used in the value gradient and policy gradient (Eq. 6, Eq. 13; §4.2).
  - "Fig. 1 (toy bimodal Q landscape)" → Figure 1 is training curves on six benchmarks; the paper has no toy bimodal figure.
  - "Fig. 3 (HalfCheetah / Humanoid learning curves)" → Figure 3 is Ant-v1 hyperparameter sensitivity.
  - "dramatically improves sample efficiency and stability across HalfCheetah, Humanoid, Ant" → comparable on easier tasks, better on harder tasks (§5.1).
  - "Automatic α tuning (v2)" and "L(α) = E[−α(log π + H̄)], H̄ = −dim(action space)" → not in 1801.01290; this paper fixes the temperature and tunes reward scale per task (§3.2, §5.2, Table 2); the α loss and target entropy are in 1812.05905 Eq. 18 and App. D Table 1.
- Removed as unsupported by the source: "V(s) = α·log ∫ exp(Q(s,a)/α) da" (this paper defines V by Eq. 3); "has become the default max-ent RL reference"; "modern LLM-RL papers are rediscovering that a fixed β on H(π) is inferior to a target-entropy-based adaptive term"; the LLM target-entropy-schedule guideline; "SAC-v2 Fig. 1 (auto α curve)".
- Not reported by the source: automatic temperature adjustment, target entropy, wall-clock time or hardware.
