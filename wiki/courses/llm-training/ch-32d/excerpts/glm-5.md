---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: GLM-5 Team (Zhipu AI and Tsinghua University), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763v2 (2026-02-24)
source_url: https://arxiv.org/abs/2602.15763
created_at: "2026-09-15"
source_type: official technical report
---

# Excerpt: GLM-5 pre-training and mid-training stages, base-model evaluation

No library card for this source exists at the time of writing (planned slug `glm-5`). This chapter-local extract covers only the passages used by [[read]] §1, §2, §9, §10, and the Recipe table.

## §1 Introduction ("Methods" paragraph)

- "Our Base Model training began with a massive 27 trillion token corpus, prioritizing code and reasoning early on. We then employed a distinct Mid-training phase to progressively extend context length from 4K to 200K, focusing specifically on long-context agentic data to ensure stability in complex workflows."
- Post-training is described as SFT, then Reasoning RL, Agentic RL, and General RL, with on-policy cross-stage distillation "to prevent catastrophic forgetting".
- Model size: 744B total parameters; training token budget 28.5T tokens (§1; Table 10: 744B total, 40B activated).

## §2 Pre-Training

- "Similar to GLM-4.5, the base model of GLM-5 goes through two stages: pre-training for general language and coding capacity, and mid-training for agentic and long-context capacity. We extend the training token budget for all the training stages of GLM-5, totaling 28.5 trillion tokens for the base model."

## §2.3 Mid-Training

- "We progressively extend the context window across three stages: 32K (1T tokens), 128K (500B tokens), and 200K (50B tokens). Compared to the 128K maximum in GLM-4.5, the additional 200K stage substantially improves the model's ability to process ultra-long documents and complex multi-file codebases. Long documents and synthetic agent trajectories are up-sampled at the later stages accordingly."
- Software engineering data: "We retain the paradigm of concatenating repo-level code files, commit diffs, GitHub issues, pull requests, and relevant source files into unified training sequences." Relaxed repository-level filtering gives "approximately 10 million issue–PR pairs"; "After filtering, the issue–PR portion of the dataset comprises approximately 160B unique tokens."
- Long-context data: natural data (books, papers, documents with PPL, deduplication, and length filtering) plus synthetic data built with interleaved packing of similar texts; "At the 200K stage, we additionally incorporated a small proportion of MRCR-like data". "Empirically, we find that increasing data diversity progressively enhances the model's long-context performance; notably, a subsequent 200K mid-training stage, building upon the initial 128K phase, further bolstered the model's performance even within the 128K context window." No numbers are printed for this statement.

## App. A Hyper-Parameters

- "we follow the setting of GLM-4.5, including the Muon optimizer, cosine decay, and batch size warmup. The learning rate goes through a warmup stage from 0 to 2e-4, and a decaying stage to 4e-5 until the end of the pre-training stage. In the mid-training stage, the learning rate decreases linearly from 4e-5 to 1e-5."

## App. B.1, Table 11 (base models; EM unless noted)

| Benchmark | GLM-4.5-Base | GLM-5-Base |
|---|---|---|
| SimpleQA | 30.0 | 36.0 |
| BBH | 86.2 | 87.4 |
| MMLU | 86.1 | 88.3 |
| HellaSwag | 87.1 | 88.1 |
| TriviaQA | 80.0 | 80.9 |
| EvalPlus (Pass@1) | 78.1 | 87.0 |
| LiveCodeBench-Base (Pass@1) | 28.1 | 34.4 |
| GSM8K | 79.4 | 68.8 |
| MATH | 61.0 | 56.4 |
| C-Eval | 86.9 | 88.8 |

The report does not state which checkpoint (before or after mid-training) Table 11 evaluates, and it prints no evaluation of the same model before and after mid-training.

## §3.1 SFT (used only for the negatives contrast)

- "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions." This is an SFT-stage statement, not a mid-training statement.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2602.15763 (v2, 2026-02-24), §1, §2.2-§2.3, §3.1, App. A, App. B.1.
- Not reported by the source: mixture percentages per mid-training stage; the share of synthetic agent trajectories in any stage; how the agent trajectories were generated for mid-training; the stage in which the 160B issue–PR tokens are used and their epoch count; mid-training batch size; any ablation of agentic mid-training data.
