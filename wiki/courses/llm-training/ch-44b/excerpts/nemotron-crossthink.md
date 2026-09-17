---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2504.13941v3 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2504.13941
created_at: "2026-09-15"
---

# Excerpt: Nemotron-CrossThink: Scaling Self-Learning beyond Math Reasoning

- **Authors:** Syeda Nahida Akter, Shrimai Prabhumoye, Matvei Novikov, Seungju Han, Ying Lin, Evelina Bakhturina, et al. (NVIDIA, CMU, Boston University)
- **Year:** 2025 (arXiv v1 2025-04; v3 2026-03-15)
- **Source type:** paper
- **Used in:** [[read]] §3, Generalization lens

## Setup
GRPO in veRL on Qwen2.5-7B and Qwen2.5-32B; 287.4K released multi-domain examples; general-purpose reasoning (GPR) data from MMLU, Natural Reasoning, and synthesized MCQ and open-ended items, plus math data (NuminaMath and synthesized math). Reward is an accuracy reward (1 if the response equals the ground truth, else 0) combined with a format reward (App. A). Single-source runs train for 250 steps and are evaluated at the final checkpoint (§5).

## Blends (§5, Table 3; Qwen2.5-7B; B_gpr↑ is a 2:1 ratio of GPR to math data)
| Blend | MMLU | MMLU-Pro | GPQA-D | AGIEval | SuperGPQA | MATH-500 | AMC 23 | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (M) | 74.20 | 45.00 | 31.82 | 48.59 | 25.36 | 48.30 | 40.00 | 44.75 |
| Open-Reasoner-Zero | 73.20 | 48.90 | 29.30 | 63.49 | 27.60 | 81.40 | 62.50 | 55.20 |
| B_gpr↑ (2:1 GPR:math) | 74.94 | 57.82 | 38.58 | 63.71 | 29.16 | 77.60 | 65.00 | 58.12 |
| B_only_mr (math only) | 74.24 | 54.26 | 38.58 | 61.39 | 27.69 | 78.60 | 70.00 | 57.82 |
| B_only_gpr (GPR only) | 72.77 | 52.06 | 37.06 | 56.56 | 27.44 | 72.20 | 55.00 | 53.30 |
The best blend is 13.36 points above the untrained model M and 0.30 points above the math-only blend; the math-only blend is higher on MATH-500 and AMC 23 and lower on MMLU-Pro and AGIEval. GPR-only data alone "underperforms in math tasks and trails 4.2% on average across non-math reasoning tasks" relative to the mixed blend (§5).

## Response length (§6, Figure 3, Table 12)
Across tasks, correct responses from B_gpr↑ use "on average 28% fewer tokens" than from B_only_mr; on MMLU the mean correct-response length is 229 tokens for B_gpr↑ and 351 for B_only_mr. B_gpr↑ raises its mean length by 62% on math tasks relative to GPR tasks (622 vs 385 tokens), against 14% for B_only_mr (731 vs 639) and 12% for Open-Reasoner-Zero.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2504.13941 (v3, 2026-03-15): Abstract, §1, §2, §4, §5, §6, Tables 2–3, 12, App. A.
- Not reported: GRPO hyperparameters in the body text of the cited sections (learning rate, rollouts per prompt, steps for the blend runs are partly in the appendix); seeds.
