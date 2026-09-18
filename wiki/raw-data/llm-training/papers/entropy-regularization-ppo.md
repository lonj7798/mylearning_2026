<!-- scope: A3C paper (Mnih et al. 2016): asynchronous actor-learners and the entropy-regularization term in the actor-critic policy gradient that PPO later cites for its entropy bonus
     see-also: [[ppo]], [[entropy-collapse-ppo]], [[entropy-mechanism-llm-rl]], [[maximum-entropy-rl]]
-->

# Asynchronous Methods for Deep Reinforcement Learning
- **Core Insight:** A3C trains actor-critic agents with 16 CPU threads and no replay memory and adds β∇H(π) to the policy gradient (β = 0.01 on Atari and TORCS) to discourage "premature convergence to suboptimal deterministic policies"; the LSTM agent reached a mean human-normalized score of 623.0% on 57 Atari games after 4 days on CPU (§4, Supp. §8, Table 1).
- **Guideline:** When an actor-critic policy converges to a near-deterministic policy before it finds high-return behavior, adding an entropy term to the policy objective is the intervention this paper used, because the authors report that it "improved exploration" (§4); the paper reports no ablation of β, so its values (0.01 for discrete actions, 10^-4 for continuous actions) are untested outside its own setups (Supp. §8-9).
- **Authors:** Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, et al. (8 authors; Google DeepMind, MILA)
- **Year:** 2016 (arXiv v1 2016-02, v2 2016-06; ICML 2016)
- **URL:** https://arxiv.org/abs/1602.01783
- **Source type:** paper
- **Relevant topics:** entropy regularization, advantage actor-critic, asynchronous actor-learners, n-step returns, shared RMSProp, exploration, Atari, MuJoCo

## Abstract
The paper proposes a lightweight deep RL framework that runs several actor-learners in parallel on CPU threads of one machine and applies their gradients asynchronously to shared parameters. It gives asynchronous variants of one-step Q-learning, one-step Sarsa, n-step Q-learning, and advantage actor-critic, and shows that parallel actor-learners stabilize training for all four. The best method, asynchronous advantage actor-critic (A3C), exceeds the previous state of the art on Atari while training for half the time on a single multi-core CPU, and also learns continuous motor control tasks and navigation of random 3D mazes from visual input.

## Key Contributions
- Parallel actor-learners on one machine with Hogwild!-style updates replace experience replay; different exploration policies per thread add diversity (§4).
- Four asynchronous algorithms, including A3C with n-step advantage estimates and a shared convolutional network with softmax policy and linear value outputs (§4).
- Entropy regularization of the A3C policy gradient, credited to Williams & Peng (1991) (§4).
- Shared RMSProp (statistics shared across threads) found more robust than per-thread RMSProp and momentum SGD (§4, Supp. §7, Fig. S5).
- Results on 57 Atari games, TORCS, MuJoCo, and Labyrinth (§5).

## Key Figures/Tables to Study
- **§4, A3C paragraph:** gradient of the objective with the entropy term.
- **Supp. §8 and §9:** β = 0.01 (Atari, TORCS) and 10^-4 (MuJoCo differential-entropy cost).
- **Table 1:** mean and median human-normalized scores on 57 Atari games. **Table 2:** speedup vs number of threads.
- **Fig. 2:** A3C final scores for 50 learning rates and initializations on five games.
- **Algorithm S3:** A3C pseudocode per thread.

## Technical Details
- **Advantage estimate (§4):** A(s_t, a_t; θ, θ_v) = Σ_{i=0}^{k-1} γ^i r_{t+i} + γ^k V(s_{t+k}; θ_v) - V(s_t; θ_v). γ = discount, r = reward, V = value network with parameters θ_v, k ≤ t_max (steps since the last update).
- **Policy gradient with entropy regularization (§4):** ∇_θ' log π(a_t|s_t; θ')·(R_t - V(s_t; θ_v)) + β·∇_θ' H(π(s_t; θ')). θ' = thread-specific policy parameters, R_t = n-step return, H = entropy of the action distribution (for a discrete distribution, -Σ_a π(a|s) log π(a|s)), β = strength of the entropy term. The paper gives this gradient, not a loss function.
- **Stated purpose (§4):** "adding the entropy of the policy π to the objective function improved exploration by discouraging premature convergence to suboptimal deterministic policies"; Williams & Peng (1991) found it "particularly helpful on tasks requiring hierarchical behavior".
- **Discrete actions (Supp. §8):** β = 0.01 for all Atari and TORCS experiments.
- **Continuous actions (Supp. §9):** Gaussian policy with mean from a linear layer and variance σ² from a SoftPlus; an entropy cost on the differential entropy of the normal distribution, written -½(log(2πσ²) + 1), with a constant multiplier of 10^-4 on all MuJoCo tasks; policy and value networks share no parameters.
- **Pseudocode gap:** Algorithm S3 accumulates dθ ← dθ + ∇_θ' log π(a_i|s_i; θ')(R - V(s_i; θ'_v)) and does not write the entropy term, although §4 and Supp. §8 state it was used.
- **Setup for the Atari-subset (Figs. 1, 3, 4, Table 2) and TORCS experiments (Supp. §8):** 16 actor-learner threads on one machine, no GPU; update every 5 actions (t_max = 5); shared RMSProp with decay α = 0.99; γ = 0.99; action repeat 4 on Atari; network conv 16 filters 8×8 stride 4, conv 32 filters 4×4 stride 2, fully connected 256, ReLU. 50 experiments per game with learning rate sampled from LogUniform(10^-4, 10^-2) and annealed to 0.
- **57-game protocol (§5.1):** learning rate and gradient-norm clipping searched on six games (Beamrider, Breakout, Pong, Q*bert, Seaquest, Space Invaders), then fixed for all 57; the feedforward agent uses the architecture of Mnih et al. (2015), and the LSTM agent adds 256 LSTM cells after the last hidden layer.
- **Table 1 (mean / median human-normalized):** A3C FF, 1 day on CPU: 344.1% / 68.2%. A3C FF, 4 days: 496.8% / 116.6%. A3C LSTM, 4 days: 623.0% / 112.6%. Prioritized DQN, 8 days on GPU: 463.6% / 127.6%.
- **Scaling (Table 2):** A3C speedup with 16 threads is 12.5×, averaged over seven Atari games.
- **Robustness (Fig. 2):** on each of five games a range of learning rates gives good scores for all random initializations.

## Recipe ledger
Locus for all rows: arXiv:1602.01783v2. No row has an ablation of the entropy weight.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| A3C (all Atari and TORCS experiments) | Atari subset and TORCS: conv16-conv32-FC256; 57 games: Mnih et al. (2015) network, LSTM variant +256 cells | RL | Entropy regularization weight β | 0.01 | Supp. §8 | verified 2026-09-14 | no ablation reported |
| A3C (MuJoCo, continuous actions) | 200 ReLU units + 128 LSTM cells (state input) | RL | Differential-entropy cost multiplier | 10^-4 | Supp. §9 | verified 2026-09-14 | no ablation reported |
| A3C (Atari subset, TORCS) | conv16-conv32-FC256 | RL | Threads; update interval t_max | 16; 5 | Supp. §8 | verified 2026-09-14 | Table 2: speedup 12.5× at 16 threads |
| A3C (Atari subset, TORCS) | conv16-conv32-FC256 | RL | Optimizer; RMSProp decay α; discount γ | shared RMSProp; 0.99; 0.99 | Supp. §8 | verified 2026-09-14 | Supp. §7 Fig. S5: shared RMSProp most robust of three optimizers |
| A3C (Atari subset, TORCS) | conv16-conv32-FC256 | RL | Learning rate | LogUniform(10^-4, 10^-2), annealed to 0; 50 runs per game | Supp. §8 | verified 2026-09-14 | Fig. 2: good scores over a range of learning rates |
| A3C FF / A3C LSTM (57 Atari games) | Mnih et al. (2015) network (+256 LSTM cells) | RL | Hyperparameter selection; compute | LR and gradient-norm clipping tuned on 6 games, fixed for 57; 16 CPU cores, 1 or 4 days | §5.1; Table 1 | verified 2026-09-14 | — |
| A3C FF / A3C LSTM (57 Atari games) | Mnih et al. (2015) network (+256 LSTM cells) | RL | Gradient-norm clipping threshold | not reported | checked §4, §5.1, Supp. §7-9 | not reported | — |

## Findings relevant to generality
- Hyperparameters were selected on 6 games and applied unchanged to 57 games (§5.1). On those 57 games, A3C LSTM has the highest mean (623.0%) but a lower median (112.6%) than Prioritized DQN (127.6%) (Table 1).
- The authors report that giving each thread a different exploration policy "helps improve robustness" (§4).

## Connections
- [[ppo]] — PPO Eq. (9) adds "an entropy bonus to ensure sufficient exploration, as suggested in past work [Wil92; Mni+16]" with coefficient c2; c2 = 0.01 in PPO's Atari runs (PPO Table 5), no entropy bonus in its MuJoCo comparison (PPO §6.1), and its "A2C" baseline is "a synchronous version of A3C" (PPO §6.2).
- [[entropy-collapse-ppo]] — Andrychowicz et al. (2020) test entropy penalties and constraints in PPO on MuJoCo and find no significant gain except on HalfCheetah.
- [[entropy-mechanism-llm-rl]] — in LLM RLVR, an entropy loss with coefficient 0.01 caused entropy explosion and 0.005 did not outperform baselines.
- [[maximum-entropy-rl]] — SAC, which places entropy inside the RL objective instead of using it as a regularizer.
- [[openrlhf-entropy-debugging]], [[entropy-logging-patterns]] — how LLM RL frameworks expose and log entropy.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/1602.01783 (arXiv v2); PPO statements in Connections against https://arxiv.org/abs/1707.06347 (arXiv v2).
- Corrections to the previous card version:
  - The card combined two artifacts (A3C, arXiv:1602.01783, and PPO, arXiv:1707.06347) under the title "Entropy Regularization in A2C/PPO". It now describes A3C, the paper that uses the term "entropy regularization" named in the slug and title (§4). PPO has its own card, [[ppo]]; its entropy-bonus facts appear under Connections with loci.
  - "A3C / A2C (2016)" → the A3C paper does not define A2C; PPO §6.2 describes A2C as a synchronous version of A3C.
  - "L_actor = -E[log π·A] - β·H(π)" as the A3C loss → A3C states the gradient ∇log π·(R_t - V) + β∇H (§4).
  - "c_H ∈ [0, 0.01] typical on Atari and continuous control"; "ε = 0.2, c_v ≈ 0.5, c_H ≈ 0.01 as Atari defaults" → A3C: β = 0.01 (Atari, TORCS), 10^-4 (MuJoCo) (Supp. §8-9). PPO Atari: c1 = 1, c2 = 0.01, clip 0.1×α (PPO Table 5); PPO MuJoCo: no entropy bonus (PPO §6.1).
  - "A3C Fig. 4 / ablation table: entropy bonus vs without" → A3C has no entropy ablation; Fig. 4 compares training speed across numbers of actor-learners.
  - "PPO appendix lists the default entropy coefficient per environment" → only PPO Table 5 (Atari) gives c2.
  - "Andrychowicz 2020: c_H is second-tier, interacting with LR and advantage normalization" → that paper reports no significant gain from regularizers except HalfCheetah and gives no such interaction claim ([[entropy-collapse-ppo]]).
- Removed as unsupported by the source: "survives essentially unchanged in modern LLM RL loops" and "preserved in every LLM-RL framework (TRL, OpenRLHF, verl, Reinforce++)"; the TRL `entropy_coef` statement; "GRPO omits the bonus, which is one reason entropy collapse became a named problem"; "the bonus is the small-α limit of SAC"; "c_H > 0 consistently helps exploration and reduces seed variance on Atari and MuJoCo"; "LLM RL coefficient range 0.0 to 1e-3"; "advantage normalization rescales the effective entropy coefficient"; the LLM tail-entropy failure-mode paragraph.
- Not reported by the source: an ablation of β; gradient-norm clipping threshold; any language-model experiment.
