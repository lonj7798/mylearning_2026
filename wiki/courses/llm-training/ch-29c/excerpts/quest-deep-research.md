---
chapter: ch-29c
course: llm-training
phase: read
excerpt_of: "QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks, arXiv:2605.24218v1 (2026-05-22)"
source_url: https://arxiv.org/abs/2605.24218
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug quest-deep-research). Values read from the v1 PDF on 2026-09-15."
---

# Excerpt: QUEST — rubric-tree tasks for objective and open-ended research

**Authors:** Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, et al. (The Ohio State University; Amazon AGI SF Lab).

## Why not single-answer tasks (§2)
> "training on such data does not generalize to the broad range of tasks that deep research agents may encounter" and "single-answer queries typically induce a binary correctness reward for reinforcement learning, which limits effective credit assignment."

## Rubric tree (§2.1)
- Root = overall score aggregated from children; leaves are directly verifiable criteria (factual correctness, source attribution) with binary scores; internal nodes aggregate their children. Based on Mind2Web 2, but constructed synthetically.

## Generation pipeline (§2.2)
- G_syn = Claude Sonnet 4.5 unless stated.
- Objective tasks: trending keywords from Google Trends → G_syn browses the web and derives verifiable constraints → rubric tree → iterative refinement; trees that still cannot be resolved into "a consistent, reliably evaluable structure" are discarded (judged by Claude Sonnet 4.5) → G_syn writes the question → GPT-5 writes an executable Python evaluation script.
- Open-ended tasks: root children fixed to four criteria from DeepResearch Bench (instruction following, comprehensiveness, readability, insight); task-specific child nodes and weights from G_syn, weights averaged over three generations. Score = J(r_cand) / (J(r_cand) + J(r_ref)), J ∈ [0, 1], with a G_syn reference report; > 0.5 means the candidate beat the reference.
- Manual check (App. A.2): of 50 sampled objective tasks, 2 scripts were non-executable and 6 contained rubric-related errors.
- Retention (App. A.1 Table 4): objective 17,000 generated → 8,737 after rubric refinement → 6,230 after rubric-tree verification → 5,934 after removing erroneous Python scripts; open-ended 3,000 → 2,856 (valid format) → 2,498 (qualified) → 2,227 (reference report score). 5,934 = 5,070 SFT + 864 RL; 2,227 = 1,958 SFT + 269 RL (Table 1).

## Trajectories (§4.1, App. A.4-A.5)
- Teacher G_traj = Tongyi DeepResearch; GPT-5.2 polishes open-ended reports. Keep a trajectory if its score exceeds ε: ε = 1 for objective tasks, 0.475 for open-ended tasks. Failed objective tasks get a reflection-based retry with the evaluation result as a hint.
- 5 rollouts per task; 3 additional retries per failed trajectory; maximum 100 tool calls; context threshold 16K.
- > "Tasks that cannot be successfully completed by G_traj in any rollout are reserved for reinforcement learning, resulting in 864 objective RL tasks."

## Reward (§4.4)
- Open-ended pairwise score mapped to s_rubric: > 0.5 → 1.0; [0.475, 0.5) → 0.75; [0.45, 0.475) → 0.5; [0.425, 0.45) → 0.25; < 0.425 → 0.
- s_fact = supported / (supported + unsupported) citations, labeled by GPT-5-mini.
- Eq. 1: R = 0.75 · s_rubric + 0.25 · min(s_fact, s_rubric). Eq. 2: group-normalized advantage; GRPO "excluding the KL penalty".
- Pointwise 0/0.5/1 scoring reached about 1 in about 50% of cases, and win/tie/lose against a stronger teacher collapsed to "lose" (§7.4). DPO on report pairs gave no improvement (§7.3).

## Data (Table 1)
| Stage | Type | Tasks | Trajectories | Sessions |
|---|---|---|---|---|
| MT | Context summarization | 309,346 | — | — |
| MT | Relevant information extraction | 1,052,663 | — | — |
| SFT | Objective | 5,070 | 19,435 | 39,861 |
| SFT | Open-ended | 1,958 | 4,485 | 11,903 |
| RL | Objective | 864 | — | — |
| RL | Open-ended | 269 | — | — |
- Context summarization data exclude benchmark evaluations "To avoid data leakage" (App. A.3).

## Results (Table 3; §6.2-6.3)
- QUEST-35B (Qwen3.5-35B-A3B): BC 45.5 (64.6 with discard-all), BC-Plus 61.0 (69.5), M2W2 30.7, WideSearch 60.6, HLE 37.2, GAIA 80.8, DRB 48.2, LRB 68.2.
- At 30B scale, Tongyi-DR (single-answer synthetic data) is best on BrowseComp 43.4, HLE 32.9, GAIA 70.9; OpenResearcher is best on BrowseComp-Plus 54.8, "a fully offline benchmark that closely matches its data synthesis recipe"; QUEST-30B is best on 4 of 8. Authors: "the capabilities exhibited by a deep research agent are shaped by its data synthesis recipe".
- Stage ablation: SFT degrades open-ended benchmarks vs the vanilla model and hurts BC-Plus because "heavy SFT makes the model more likely to call disallowed tools due to overfitting to the training-time tool-use pattern"; RL improves open-ended tasks but "slightly sacrifices performance on HLE and GAIA" (§6.3).
