<!-- scope: off-policy REINFORCE without importance sampling (AsymRE); how the baseline V sets the weight on positive vs negative rewards, a phase transition at the behavior policy's mean reward, and LLM runs with delayed behavior-policy updates
     deps: [[grpo]], [[reinforce-plus-plus]]
     see-also: [[topr-tapered-off-policy-reinforce]], [[negative-sample-reinforcement]], [[async-rollout]], [[lazy-likelihood-displacement-grpo]], [[dpo-positive]]
-->

# Asymmetric REINFORCE for off-Policy Reinforcement Learning: Balancing positive and negative rewards
- **Core Insight:** In off-policy REINFORCE with advantage r − V, the tabular limit policy improves on the behavior policy µ and keeps a wide support while V < V^µ (µ's expected reward), and its support "suddenly shrinks" to generically one element once V ≥ V^µ (Thm. 4.2); with Llama-3.1-8B-Instruct on MATH and behavior-policy refresh every 250 steps, all 7 runs with δV = 0 collapsed, and the authors report greater training stability for the 7 runs with δV = −0.1 (§5.2, Fig. 5 left).
- **Guideline:** When RL rollouts come from a stale policy and no importance-sampling correction is applied, set the baseline slightly below the per-prompt mean reward (the paper suggests δV ≈ −0.1 with ±1 rewards, §5.3), because baselines at or above that mean produced train and test collapse for Llama 8B and Qwen 3B (Figs. 3-4), with the same conclusion reported for Llama 3B and NuminaMath (App. C); a very small baseline slows convergence (§4).
- **Authors:** Charles Arnal, Gaëtan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, Remi Munos (FAIR at Meta; NYU Courant Institute and CDS)
- **Year:** 2025 (arXiv v1 2025-06; v2 2025-11)
- **URL:** https://arxiv.org/abs/2506.20520
- **Source type:** paper
- **Relevant topics:** off-policy RL, REINFORCE baseline, negative rewards as gradient, asynchronous and delayed-update RL, entropy and diversity collapse, GRPO comparison

## Abstract
Off-policy RL is simpler to implement and more data-efficient than on-policy RL for LLMs but often performs worse. The paper studies algorithms between off-policy RL and supervised fine-tuning through a simple off-policy REINFORCE with advantage A = r − V, where V is a tunable baseline. A lower V emphasizes high-reward samples; a higher V penalizes low-reward samples more. The theoretical analysis shows a policy improvement guarantee when V lower-bounds the expected reward. The analysis indicates that on-policy updates can use both positive and negative signals, while off-policy updates benefit from focusing more on positive rewards. Experiments use a stochastic bandit and LLM fine-tuning on reasoning tasks.

## Key Contributions
- Defines Asymmetric REINFORCE (AsymRE): gradient ascent on J(π) = E_{y∼µ}[log π(y)(r(y) − V)] with no importance-sampling correction (§4, Def. 4.1, Eq. 1).
- Theorem 4.2: tabular softmax limit policy for V < V^µ, V = V^µ, and V > V^µ; supports are nested and shrink as V grows (§4).
- Theorem 4.3: repeated AsymRE with V < V^µ increases expected reward at each iteration, concentrates mass exponentially fast, and reaches the optimal reward if and only if V < V_{0,µ} (§4).
- Bandit validation of the phase transition and of policy iteration (§5.1).
- LLM validation on Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, and Qwen2.5-3B-Instruct on MATH and a NuminaMath subset (§5.2, App. C).

## Key Figures/Tables to Study
- **Fig. 1b:** support of the bandit limit policy vs V; drops to a single arm when V crosses V^µ ≈ 0.54.
- **Fig. 2:** 40 policy-improvement iterations of 500 steps each for several V.
- **Figs. 3-4:** train and test accuracy on MATH for δV ∈ {−0.5, −0.3, −0.2, −0.1, 0, 0.1, 0.2, 0.3} (Llama 8B, Qwen 3B; 3 seeds).
- **Fig. 5:** left, 7 runs each at δV = 0 and δV = −0.1; right, AsymRE vs GRPO test accuracy.
- **Figs. 7-8:** policy entropy vs V (bandit, Llama 8B).

## Technical Details
- **Baseline role:** on-policy, V only changes variance and does not change expected dynamics or the limit; off-policy, V changes both the dynamics and the limit policy (§4). With r ∈ {0, 1}, V = 0 equals supervised learning on successes only and V = 1 equals learning from failures only (§4, Intuition).
- **Limit policy, V < V^µ:** π*(y) = (µ(y)(r(y) − V) − τ_{µ,V})_+ / (V^µ − V), where τ_{µ,V} is set by Σ_y (µ(y)(r(y) − V) − τ_{µ,V})_+ = V^µ − V and x_+ = max(x, 0) (Thm. 4.2). Here µ is the behavior policy, r the reward, V^µ = E_{y∼µ}[r(y)].
- **V = V^µ:** support = argmax_y µ(y)(r(y) − V), with probability ratios inherited from the initial policy π_0. **V > V^µ:** the limit can be any element of a set that depends on π_0 (Thm. 4.2; the gradient flow is a replicator equation, App. A.1.4).
- **Speed trade-off:** more steps per iteration means more off-policy training and a need for a smaller V; a very small V gives slower convergence (§4). If V ≥ V^µ, the reward of the limit can be below µ's and later iterations cannot recover (§4).
- **Context-corrected LLM loss:** E_{x∼D, y∼µ(·|x)}[log π(y|x)(r(y, x) − V^{µ(·|x)} − δV)], so the critical value is δV = 0 for every prompt (Eq. 4). Implemented with V̂ = mean reward of G = 8 samples per prompt (§5.2).
- **Bandit (§5.1):** 100 arms, rewards uniform in [0, 1], µ = softmax of logits y/10, learning rate 1, π_0 = µ. V^µ ≈ 0.5405; best expected reward across V ≈ 0.89 vs max_y r(y) = 0.999. In policy iteration, V = 0.525 and 0.54 converge to a worse limit than V ≤ 0.5 (§5.1, Fig. 2).
- **LLM results:** for δV < 0, training accuracy rises with δV and test accuracy follows "less strongly"; at δV ≥ 0 train and test accuracy "suffer a catastrophic collapse", earlier for larger δV; at δV = 0 collapse occurs at the end of the run (§5.2, Figs. 3-4). Entropy falls faster for larger V (App. C, Fig. 8).
- **GRPO comparison:** AsymRE with δV = −0.1 trained faster and reached higher test accuracy than GRPO on MATH (Fig. 5 right) and NuminaMath (Fig. 13) with N = 250; the authors write that "further testing would be needed" (§5.2). No numeric values are given in the text.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 8B | RL | optimizer; peak LR | AdamW; 6 × 10^-8 | arXiv:2506.20520v2 App. B | verified 2026-09-14 | no ablation reported |
| Llama-3.1-8B-Instruct | 8B | RL | samples per prompt; trajectories per gradient step; max trajectory length | 8; 128; 2048 tokens | App. B | verified 2026-09-14 (stated in the Llama-3.1-8B paragraph only) | no ablation reported |
| Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct | 8B, 3B, 3B | RL | samples per prompt G; reward | 8; +1 correct, −1 otherwise | §5.2 | verified 2026-09-14 | no ablation reported |
| same three models | 8B, 3B, 3B | RL | behavior-policy update interval N | 250 gradient steps | §5.2; Figs. 3-5, 9-13 | verified 2026-09-14 | App. B says the interval was varied; only N = 250 is shown |
| same three models | 8B, 3B, 3B | RL | baseline offset δV | tested −0.5 to 0.3; suggested ≈ −0.1 | Figs. 3-4; §5.3 | verified 2026-09-14 | Fig. 5 left: 7/7 runs at δV = 0 collapse; "greater training stability" for 7 runs at −0.1 (Llama 8B, MATH) |
| same three models | 8B, 3B, 3B | RL, eval-gate | sampling | train: temperature 1.0, top-p 1.0; eval: temperature 0.1, top-p 0.95 | App. B | verified 2026-09-14 | no ablation reported |
| same three models | 8B, 3B, 3B | RL | data | MATH (12.5k problems); NuminaMath subsets of 142k (train) and 2k (eval) | §5.2; App. C | verified 2026-09-14 | not applicable |
| Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct | 3B | RL | LR; batch; max length; total steps; KL; GPU count; GRPO baseline settings | not reported (checked §5.2, App. B-C, figure captions) | — | not reported | none |

## Findings relevant to generality, negative feedback
- **Negative as gradient (type 4):** δV moves the weight between pushing up successes and pushing down failures; the paper's stated reason for favoring positives off-policy is that failures of another policy are less informative because the current policy "might not be likely to produce" them (§1, §4). Result (single study) for the phase transition; Interpretation for the reason.
- **Diversity and test accuracy:** the authors expect support collapse to be "detrimental to a language model" through "severe overfitting of the training set, and consequently of poor test results" (§4, Interpretation); in the LLM runs test accuracy collapsed together with training accuracy at δV ≥ 0 (§5.2).
- **Measurement limits:** test sets are MATH and a 2k NuminaMath subset from the training distributions; no out-of-domain benchmark, pass@k, or importance-sampling or KL-regularized variant is reported (§5.2, §6 Limitations).
- **Asynchronous training:** generation workers and trainers run on separate GPUs, and the trainer refreshes the workers' weights every N gradient steps (App. B).

## Connections
- [[grpo]]: the on-policy group-baseline method compared against in Figs. 5 and 13.
- [[reinforce-plus-plus]], [[rloo-vs-grpo]]: other REINFORCE-style baselines for LLM RL.
- [[topr-tapered-off-policy-reinforce]]: concurrent asymmetric importance-sampling variant cited in §2.
- [[negative-sample-reinforcement]]: cited on-policy study of positive vs negative samples (§2).
- [[on-off-policy-rlhf]]: cited for the preference for on-policy LLM training (§3).
- [[async-rollout]], [[areal-async-rl]], [[rollout-training-mismatch-tis]]: systems that create the stale-policy setting studied here.
- [[entropy-collapse-ppo]], [[entropy-mechanism-llm-rl]]: entropy decline, which Fig. 8 ties to larger V.
- [[lazy-likelihood-displacement-grpo]], [[dpo-positive]]: other analyses of when negative gradients reduce the likelihood of good outputs.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2506.20520 (v2, 2025-11-28; v1 2025-06-25).
- Audit claims not found in the source: "this negative term is not tempered by the current policy's own probabilities" is not stated; the paper's reason is that another policy's failures are less informative (§1, §4). The audit's "all runs with δV = 0 collapse" is scoped in the source to 7 Llama-3.1-8B-Instruct runs on MATH (Fig. 5 left), not to all three models.
- Not reported by the source: hyperparameters for the 3B models, number of GPUs, results for update intervals other than N = 250, numeric GRPO comparison values.
