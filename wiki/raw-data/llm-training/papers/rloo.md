<!-- scope: REINFORCE Leave-One-Out — k-sample REINFORCE with per-prompt baseline, beats PPO/DPO on LLM RLHF
     deps: [[vanilla-pg]], [[ppo]]
     see-also: [[grpo]], [[reinforce-plus-plus]], [[rloo-vs-grpo]]
-->

# Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs (RLOO)
- **Core Insight:** For LLM RLHF the "environment" is deterministic per-token given the KV-cache and each prompt generates full episodes — almost every PPO bell and whistle (critic, GAE, clip, many epochs) is overhead, so k-sample REINFORCE with a leave-one-out baseline beats PPO at a fraction of the cost.
- **Guideline:** For online RLHF with a trained RM, sample k∈{2,4} rollouts per prompt, compute the leave-one-out baseline, and take a single REINFORCE step — no value head, no GAE, no epoch loop.
- **Authors:** Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, Sara Hooker
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.14740
- **Relevant topics:** variance reduction, REINFORCE, online RLHF, critic-free, LLM alignment

## Abstract
PPO has been positioned as the canonical RL algorithm for RLHF. We revisit the alignment of LLM RLHF with classical RL assumptions and show that many PPO components — value networks, GAE, multiple PPO epochs, clipping — are unnecessary or counterproductive in the LLM setting. We derive a REINFORCE-style estimator with a leave-one-out (RLOO) baseline that is simpler, uses less memory, and outperforms both PPO and "RL-free" methods such as DPO and RAFT across TL;DR and HH-RLHF benchmarks.

## Key Contributions
- Analyzes which PPO assumptions break in LLM RLHF (deterministic transitions, full-trajectory rewards, long episodes).
- **RLOO estimator** — k-sample REINFORCE where each sample's baseline is the mean reward of the other k−1 samples.
- Empirically beats PPO on TL;DR summarization and HH-RLHF by 5–20% win rate at comparable KL.
- Removes the value network and the critic loss → ~50% memory footprint of PPO.
- Provides a "one-line" algorithm that non-RL practitioners can implement.

## Key Figures/Tables to Study
- **Figure 3:** TL;DR win rate vs KL Pareto frontier — RLOO dominates PPO and DPO at every KL.
- **Figure 5:** k=2 vs k=4 vs k=8 — diminishing returns beyond k=4.
- **Section 3 / Equation 6:** The RLOO gradient estimator.

## Technical Details

### k-sample REINFORCE setup
For prompt x, sample k responses y_1,…,y_k ~ π_θ(·|x). RM gives R(x, y_i).

### RLOO gradient estimator
`∇_θ J ≈ (1/k) Σ_{i=1..k} [ R(y_i, x) − (1/(k−1)) Σ_{j≠i} R(y_j, x) ] · ∇_θ log π_θ(y_i | x)`
- Baseline `b_i = (1/(k−1)) Σ_{j≠i} R(y_j, x)` is unbiased (independent of y_i given x).
- Reduces variance relative to a moving-average baseline `b_MA = (1/S) Σ_s R(x^s, y^s)`.

### KL regularization
Applied as a per-token shaped reward, identical to InstructGPT:
`R̃(x, y) = R(x, y) − β · KL(π_θ(·|x) || π_ref(·|x))`
No explicit KL term inside the loss, no entropy bonus.

### What is removed vs PPO
| Component | PPO | RLOO |
|-----------|-----|------|
| Value network | required | removed |
| GAE | yes | no (full-sequence reward) |
| Clip ε | yes | no |
| Epochs per rollout K | 4 | 1 |
| Baseline | learned V | leave-one-out across k |

### Hyperparameters (paper)
| Knob | Value |
|------|-------|
| k (rollouts per prompt) | 2 or 4 (main: k=4) |
| KL coef β | 0.05 (tuned per task on Pareto curve) |
| Learning rate | 1e-6 to 3e-6 (AdamW) |
| Batch size (prompts) | 32–64 |
| Sampling T | 1.0 |
| Max new tokens | 53 (TL;DR), 256 (HH) |
| π_ref | SFT checkpoint, frozen |

### Relationship to GRPO
GRPO's advantage `(r_i − mean(r))/std(r)` over a group of G is equivalent (up to scaling) to RLOO's leave-one-out when G is large; GRPO additionally clips the ratio, normalizes by std, and computes KL in the loss.

## Connections
- Direct ancestor: [[vanilla-pg]].
- Over-engineered baseline: [[ppo]] (paper argues PPO overcounts).
- Group-normalized variant: [[grpo]].
- Variance-reduction successor: [[reinforce-plus-plus]].
- Systematic comparison: [[rloo-vs-grpo]].
- Framework implementations: [[trl-ppo]] (RLOO in newer TRL), [[openrlhf-ppo]].
