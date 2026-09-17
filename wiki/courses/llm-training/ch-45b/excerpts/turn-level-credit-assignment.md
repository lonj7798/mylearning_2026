---
chapter: ch-45b
course: llm-training
phase: read
excerpt_of: arXiv:2505.11821 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2505.11821
created_at: "2026-09-15"
---

# Excerpt: Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment

- **Authors:** Quan Wei, Siliang Zeng, Chenliang Li, Zhongruo Wang, William Brown, Oana Frunza, Wei Deng,
  Anderson Schneider, Yuriy Nevmyvaka, Yang Katie Zhao, Alfredo Garcia, Mingyi Hong
- **Year:** 2025 (arXiv v1 2025-05-17; later versions through 2026). The slug keeps the earlier title's wording
  ("turn-level credit assignment"); the current title is the one printed above.
- **Source type:** paper
- **Used in:** [[read]] §4, Check your understanding

## Abstract (verbatim)
"Reinforcement Learning (RL) approaches have been wildly used to enhance the reasoning capabilities of Large
Language Model (LLM) agents in long-horizon, multi-turn scenarios. Such interactions can be formalized as
turn-level Markov decision processes (MDPs), where intermediate rewards are often available. However, most prior
work relies on sparse trajectory-level rewards, resulting in poor credit assignment, while dense turn-level
rewards remain underexplored. In this paper, we investigate how to effectively leverage dense turn-level reward
structures in RL algorithms, specifically Group Relative Policy Optimization (GRPO) and Proximal Policy
Optimization (PPO), to enable fine-grained credit assignment. We categorize reward structures into three types
based on their granularity: (1) terminal reward; (2) delayed reward; (3) per-turn reward, each corresponding to a
distinct turn-level MDP formulation, and derive GRPO and PPO algorithms tailored to each case, respectively.
Experiments on multi-turn search and game agent tasks show that, for both GRPO and PPO, dense per-turn reward
structures consistently outperform sparse terminal and delayed reward structures in terms of training dynamics
and numerical results. Furthermore, on search tasks, PPO with dense per-turn rewards achieves greater training
stability and faster convergence, and delivers the highest answer correctness among all state-of-the-art methods
across diverse question-answering datasets."

## What the chapter uses
- The three-way taxonomy of reward granularity — terminal, delayed, per-turn — as the vocabulary for comparing
  outcome-only recipes against step-level ones.
- The claim that per-turn rewards beat terminal and delayed rewards for both GRPO and PPO on multi-turn search
  and game tasks, and that PPO with per-turn rewards is the more stable of the two on search.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2505.11821 (abstract page).
- Not extracted here: the turn-level advantage derivations, model sizes, benchmark names, and every numeric
  result. No number from this source is cited in the chapter.
