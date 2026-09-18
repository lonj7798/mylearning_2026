<!-- scope: Proximal Policy Optimization (Schulman et al. 2017) — clipped surrogate objective, adaptive KL penalty baseline, actor-critic PPO with truncated GAE; MuJoCo, Roboschool, Atari experiments
     deps: [[trpo]], [[vanilla-pg]]
     see-also: [[rlhf-instructgpt]], [[grpo]], [[rloo]], [[costa-huang-ppo-details]]
-->

# Proximal Policy Optimization Algorithms
- **Core Insight:** Taking the minimum of the unclipped and ratio-clipped surrogate objectives lets a policy-gradient method run several epochs of minibatch updates on each batch of samples using only first-order optimization; on 7 MuJoCo tasks, clipping with ε = 0.2 gave an average normalized score of 0.82, against −0.39 with no clipping or penalty and at most 0.74 with a KL penalty (Table 1).
- **Guideline:** When an on-policy actor-critic method reuses each batch for several epochs of minibatch SGD, use the clipped surrogate with ε = 0.2, because it had the highest average normalized score among clipping (ε = 0.1, 0.2, 0.3), adaptive-KL, and fixed-KL variants on 7 MuJoCo tasks with 3 seeds each (§6.1, Table 1). For the Atari setting the paper instead uses ε = 0.1·α with α annealed linearly from 1 to 0 (Table 5).
- **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
- **Year:** 2017 (arXiv v1 2017-07; v2 2017-08)
- **URL:** https://arxiv.org/abs/1707.06347
- **Source type:** paper
- **Relevant topics:** on-policy RL, policy gradient, trust region, clipped surrogate, adaptive KL penalty, actor-critic, generalized advantage estimation

## Abstract
The paper proposes a family of policy-gradient methods that alternate between sampling data by interacting with the environment and optimizing a "surrogate" objective with stochastic gradient ascent. Standard policy-gradient methods perform one gradient update per data sample; the proposed objective allows multiple epochs of minibatch updates. The methods, called proximal policy optimization (PPO), keep some benefits of trust region policy optimization (TRPO) while being simpler to implement, more general, and empirically more sample-efficient. On simulated robotic locomotion and Atari, PPO outperforms other online policy-gradient methods and balances sample complexity, simplicity, and wall-time.

## Key Contributions
- Clipped surrogate objective L^CLIP, a pessimistic bound on the unclipped objective (§3, Eq. 7).
- Adaptive KL-penalty coefficient as an alternative, reported to perform worse than clipping (§4).
- Actor-critic PPO algorithm with a combined policy, value, and entropy objective and truncated GAE (§5, Eq. 9-12, Alg. 1).
- Surrogate-objective comparison on MuJoCo (§6.1) and comparisons against TRPO, CEM, vanilla PG with adaptive step size, A2C, A2C with trust region (§6.2), and A2C and ACER on 49 Atari games (§6.4).

## Key Figures/Tables to Study
- **Fig. 1:** one term of L^CLIP as a function of the ratio r for Â > 0 and Â < 0.
- **Fig. 2:** surrogate objectives along the update direction on Hopper-v1; L^CLIP peaks where KL from the old policy is about 0.02.
- **Table 1:** normalized MuJoCo scores for each surrogate variant and hyperparameter.
- **Alg. 1:** PPO, actor-critic style.
- **Tables 3-5 (App. A):** hyperparameters for MuJoCo, Roboschool, Atari.

## Technical Details
- **Probability ratio:** r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t), so r(θ_old) = 1 (§3). π_θ is the policy, θ_old the parameters before the update, s_t and a_t the state and action at timestep t.
- **Conservative policy iteration surrogate:** L^CPI(θ) = Ê_t[r_t(θ)·Â_t] (Eq. 6). Ê_t is the empirical average over a batch; Â_t is an advantage estimate. Unconstrained maximization of L^CPI leads to excessively large updates (§3).
- **Clipped surrogate:** L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1 − ε, 1 + ε)Â_t)] (Eq. 7), with ε a hyperparameter, "say, ε = 0.2" (§3). L^CLIP equals L^CPI to first order around θ_old (§3). Clipping in log space was tried and was no better (§6.1).
- **TRPO reference point:** TRPO maximizes Ê_t[r_t Â_t] subject to Ê_t[KL[π_θold, π_θ]] ≤ δ (Eq. 3-4). A fixed penalty coefficient β instead of the constraint is hard to choose across and within problems (§2.2).
- **Adaptive KL penalty:** optimize L^KLPEN = Ê_t[r_t Â_t − β·KL[π_θold, π_θ]] (Eq. 8); after each update compute d = Ê_t[KL]; if d < d_targ/1.5 then β ← β/2; if d > 1.5·d_targ then β ← 2β. The constants 1.5 and 2 are heuristic, and the initial β is reported as unimportant (§4).
- **Combined objective:** L^{CLIP+VF+S}_t(θ) = Ê_t[L^CLIP_t(θ) − c_1·L^VF_t(θ) + c_2·S[π_θ](s_t)] (Eq. 9). L^VF_t = (V_θ(s_t) − V_t^targ)² is the value loss, S is an entropy bonus, c_1 and c_2 are coefficients. The value term is needed when policy and value function share parameters (§5).
- **Truncated GAE:** Â_t = δ_t + (γλ)δ_{t+1} + … + (γλ)^{T−t+1}δ_{T−1}, with δ_t = r_t + γV(s_{t+1}) − V(s_t) (Eq. 11-12). Here r_t is the reward, γ the discount, λ the GAE parameter, T the segment length. With λ = 1 it reduces to the finite-horizon estimator of Eq. 10 (§5). GAE itself is from Schulman et al. 2015a.
- **Algorithm 1:** each iteration, N parallel actors each run π_θold for T timesteps and compute advantages; the surrogate on the N·T timesteps is optimized for K epochs with minibatch size M ≤ N·T, usually with Adam; then θ_old ← θ (§5).
- **Continuous control results:** PPO outperforms the compared methods on almost all of the 7 MuJoCo environments at 1M timesteps (§6.2, Fig. 3).
- **Atari results (Table 2), games won out of 49, 3 trials:** average reward over all training: A2C 1, ACER 18, PPO 30, tie 0; average reward over the last 100 episodes: A2C 1, ACER 28, PPO 19, tie 1 (§6.4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| PPO-Clip, MuJoCo 1M-timestep benchmark (7 Gym "-v1" tasks) | MLP, 2 hidden layers of 64 units, tanh; policy and value not shared | RL | horizon T; epochs K; minibatch | 2048; 10; 64 | arXiv:1707.06347v2 App. A Table 3; §6.1 | verified 2026-09-14 | no ablation reported (only ε, β, d_targ were searched, §6.1) |
| same | same | RL | optimizer, step size; γ; GAE λ | Adam, 3 × 10^-4; 0.99; 0.95 | Table 3 | verified 2026-09-14 | no ablation reported |
| same | same | RL | clip ε | 0.2 | §6.1 Table 1; §6.2 | verified 2026-09-14 | Table 1: 0.82 vs 0.76 (ε = 0.1) and 0.70 (ε = 0.3); 7 tasks × 3 seeds |
| same | same | RL | value coefficient c_1; entropy coefficient c_2 | not used: no parameter sharing, no entropy bonus | §6.1 | verified 2026-09-14 | not applicable |
| same | same | RL | timesteps per task; seeds; number of actors N | 1M; 3; N not reported | §6.1; checked Table 3, §5 | verified 2026-09-14 (N: not reported) | not applicable |
| KL-penalty baselines, same benchmark | same | RL | d_targ (adaptive); β (fixed); initial β | 0.003, 0.01, 0.03; 0.3, 1, 3, 10; 1 | §6.1 Table 1 | verified 2026-09-14 | best adaptive 0.74 (d_targ = 0.01); best fixed 0.72 (β = 3) |
| PPO, Roboschool humanoid tasks | not reported | RL | horizon T; epochs; minibatch; actors | 512; 15; 4096; 32 (locomotion), 128 (flagrun) | App. A Table 4 | verified 2026-09-14 | no ablation reported |
| same | not reported | RL | Adam step size; γ; λ; log std of action distribution | adjusted from a target KL (value not given); 0.99; 0.95; LinearAnneal(−0.7, −1.6) | Table 4 | verified 2026-09-14 | no ablation reported |
| same | not reported | RL | clip ε; training length | not reported; learning curves run to 50M (Humanoid) and 100M (Flagrun, FlagrunHarder) timesteps | Table 4; Fig. 4 axes | not reported (ε); verified 2026-09-14 (length) | not applicable |
| PPO-Clip, Atari (49 games) | network of Mnih et al. 2016; size not given | RL | horizon T; actors; epochs; minibatch | 128; 8; 3; 32 × 8 | App. A Table 5 | verified 2026-09-14 | no ablation reported |
| same | same | RL | Adam step size; clip ε; annealing | 2.5 × 10^-4 × α; 0.1 × α; α linear from 1 to 0 over training | Table 5 | verified 2026-09-14 | no ablation reported |
| same | same | RL | c_1; c_2; γ; λ | 1; 0.01; 0.99; 0.95 | Table 5 | verified 2026-09-14 | no ablation reported |
| same | same | RL | training length; seeds | 40M frames (10M timesteps); 3 | Table 6 caption; §6.4 | verified 2026-09-14 | not applicable |

## Findings relevant to generality and negative feedback
- **Negative advantages (negative as gradient):** the ratio is clipped at 1 + ε when Â > 0 and at 1 − ε when Â < 0 (§3, Fig. 1). Because of the minimum, a change in the ratio is ignored only when it would improve the objective and is kept when it makes the objective worse (§3). For an action with Â < 0, lowering its probability below (1 − ε)·π_θold gives no further objective gain, while raising its probability is penalized without a clip (follows from Eq. 7).
- **Hyperparameter breadth:** each surrogate setting is scored by one normalized score averaged over 7 tasks × 3 seeds rather than tuned per task (§6.1), and one Atari hyperparameter table is used for the 49-game comparison (§6.4, Table 5). The paper contains no language-model experiments.

## Connections
- [[trpo]]: the KL-constrained surrogate (Eq. 3-4) that PPO replaces with a first-order clipped objective.
- [[vanilla-pg]]: the likelihood-ratio policy gradient estimator behind L^PG (Eq. 1-2).
- [[rlhf-instructgpt]]: uses PPO for the RL stage of language-model fine-tuning.
- [[grpo]]: DeepSeekMath's critic-free PPO variant with group-based advantages.
- [[rloo]]: REINFORCE with a leave-one-out baseline, evaluated against PPO for LLM RLHF.
- [[reinforce-plus-plus]]: critic-free method that keeps the PPO clipped ratio.
- [[costa-huang-ppo-details]]: implementation details of PPO that the paper does not state.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/1707.06347 (v2, 2017-08-28; v1 2017-07-20)
- Corrections to the previous card version:
  - "Canonical hyperparameters (MuJoCo): value-loss coef c1 1.0, entropy coef c2 0.01" → MuJoCo runs share no parameters and use no entropy bonus (§6.1); c_1 = 1 and c_2 = 0.01 are the Atari values (Table 5).
  - "Epochs per rollout K 3-10" → MuJoCo 10 (Table 3), Roboschool 15 (Table 4), Atari 3 (Table 5).
  - "Figure 3 / Table 3: MuJoCo learning curves vs A2C / TRPO / vanilla PG — PPO dominates" → Fig. 3 holds the learning curves and PPO "outperforms the previous methods on almost all" environments (§6.2); Table 3 lists hyperparameters.
  - "L^VF — MSE on returns" → squared error to V_t^targ (Eq. 9).
  - "GAE-based advantages" as a contribution → the paper uses a truncated version of GAE from prior work (§5, Eq. 11).
- Removed as unsupported by the source: "Became the default RL optimizer for RLHF (InstructGPT, Claude, early LLaMA)"; "λ = 0 recovers TD(0), λ = 1 recovers Monte Carlo" (the paper states only that λ = 1 gives Eq. 10); "increase K with caution — too many epochs push the policy outside the trust region"; "add a small entropy bonus" as a default; the whole "Token-level vs sequence-level in LLM RL" section (per-token clipping, KL-in-reward, and "Bradley-Terry KL penalty" are not in this paper).
- Not reported by the source: number of actors for MuJoCo; clip ε for Roboschool; network size for Roboschool and Atari; any language-model setting.
