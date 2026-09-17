---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2504.20073 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2504.20073
created_at: "2026-09-15"
---

# Excerpt: RAGEN — Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning (StarPO / StarPO-S)

- **Authors:** Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, et al.
  (last authors Li Fei-Fei, Lijuan Wang, Yejin Choi, Manling Li)
- **Year:** 2025 (arXiv v1 2025-04-24; v2 2025-05)
- **Source type:** paper
- **Used in:** [[read]] §5, Negative samples, Common mistakes, Generalization lens

## Abstract (verbatim extract)
"We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level
agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on four
stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo
Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with
trajectory filtering, critic incorporation, and gradient stabilization. Second, we find the shaping of RL
rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling.
Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through
multi-turn RL and they may show shallow strategies or hallucinated thoughts."

## Echo Trap (§4.1, Fig. 4)
- Named failure mode: the model "repeatedly reuses memorized reasoning paths when trained on self-generated
  trajectories" (§4.1).
- Three monitored signals: "Reward standard deviation is an early indicator of convergence"; "Gradient norm
  spikes indicate irreversible collapse"; "Entropy should follow a stable decay trend during effective learning"
  (Fig. 4 caption and §4.1).

## StarPO-S (§4.2, App. D)
1. **Uncertainty-based trajectory filtering:** "retain only the top p% highly-uncertain prompts at each training
   step", ranked by reward standard deviation, with default `p = 25%`. Reported effect: "retaining 75% of
   rollouts extends stability in FrozenLake from 100 to 140 steps, while 50% avoids collapse entirely" (§4.2,
   Fig. 5). Groups whose rollouts all succeed or all fail carry zero reward variance and are dropped first.
2. **Critic incorporation:** a PPO critic is used in the stabilized variant.
3. **Gradient stabilization:** "KL Term Removal and Clip-Higher (Asymmetric Clipping)" with `ε_high = 0.28` and
   `ε_low = 0.2` (App. D).

## Setup (§3.1-§3.2)
Four stylized environments: Bandit, Sokoban, Frozen Lake, WebShop. Rollout configuration: "P=8 prompts, with
N=16 rollouts per prompt, up to 5 turns and 10 actions".

## Reasoning decay (§4.4, Table 4)
"Models produce hallucinated reasoning, revealing a mismatch between thoughts and environment states ...
reasoning gradually decays during training if the reward signal focuses only on final outcomes." Table 4 shows
reasoning traces shrinking over training across tasks.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2504.20073 (abstract page) and the arXiv HTML rendering
  (v2) for §3.1, §3.2, §4.1, §4.2, §4.4 and App. D.
- **Not confirmed and therefore not cited in the chapter:** specific StarPO vs StarPO-S success-rate numbers for
  Sokoban and Bandit. The v2 HTML presents those comparisons as curves (Fig. 6) without numbers in the text.
- Not extracted here: the StarPO objective equation, the PPO/GRPO instantiation details, and the rollout-shaping
  ablations of finding 2.
