---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2501.17161
created_at: "2026-09-17"
---

# Excerpt: "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training"

**Artifact.** Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans,
Quoc V. Le, Sergey Levine, Yi Ma. arXiv:2501.17161 (v1 2025-01; v2 2025-05-26). Read on 2026-09-17.
No library card existed when ch-07 was revised.

## Setting

- Two tasks with separable rule and visual variation: **GeneralPoints**, an arithmetic card game where the
  goal is to reach 24 using each card exactly once, and **V-IRL**, a real-world navigation environment
  (§4.1–4.2). Each has a language-only (`-L`) and a vision-language (`-VL`) version.
- Backbone: **Llama-3.2-Vision-11B** (§5). RL uses a multi-turn formulation with PPO as the backbone
  algorithm (§3).
- Out-of-distribution is defined by changing a **rule** (for example whether `J, Q, K` count as 11, 12, 13
  or all as 10) or a **visual attribute** (card colour; landmark appearance), holding the task fixed
  (§4.1–4.2).

## Results quoted in ch-07

- Rule variation, OOD success rate, initial → final (§5.1):

  | Task | RL | SFT |
  |---|---|---|
  | GP-L | 11.5% → 15.0% (+3.5) | 11.5% → 3.4% (−8.1) |
  | V-IRL-L | 80.8% → 91.8% (+11.0) | 80.8% → 1.3% (−79.5) |
  | GP-VL | 11.2% → 14.2% (+3.0) | 11.2% → 5.6% (−5.6) |
  | V-IRL-VL | 35.7% → 45.0% (+9.3) | 35.7% → 2.5% (−33.2) |

- Visual variation (§5.2): RL +17.6% (23.6% → 41.2%) and +61.1% (16.7% → 77.8%); SFT −9.9%
  (23.6% → 13.7%) and −5.6% (16.7% → 11.1%).
- Both methods raise in-distribution performance; the separation appears only on the OOD variants (§5.1,
  Fig. 5).
- **SFT is still needed before RL in this setup**: the base Llama-3.2-Vision-11B fails to follow the task
  instructions, so RL from the base model does not train (§5.4, App. D.3). The authors state that because
  this depends on the backbone, their results do not show that SFT is unnecessary for downstream RL in
  general (§5.4).
- **RL does not repair an overfitted checkpoint**: starting RL from an extremely overfitted SFT checkpoint
  leaves overall success rate below 1% (App. D.3). The authors list the underfit/overfit boundary as an
  unresolved question (§6).
