---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2410.14872v2 (How to Evaluate Reward Models for RLHF)
source_url: https://arxiv.org/abs/2410.14872
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: Preference Proxy Evaluations (PPE) and the downstream RLHF test

Used by [[read]] Core insight, §4.3, the negatives section, and the Recipe. Authors: Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, et al. (UC Berkeley). arXiv v1 2024-10; checked against v2 (2024-10-22) on 2026-09-15.

## Human-preference set (§4)
- 16,038 Chatbot Arena pairs between 20 models; 4,583 instruction-following, 5,195 hard, 2,564 math prompts (overlapping); over 121 languages; 6,120 voters (§4.1).
- Metrics: pairwise accuracy (ties excluded), Spearman and Kendall correlation of the induced model ranking, row-wise Pearson of win-rate matrices, separability, confidence agreement, Brier score (§4.2).

## Correctness set (§5)
- MMLU-Pro, MATH, GPQA, MBPP-Plus, IFEval; 500 prompts each (or all if fewer); 32 samples per prompt from each of Llama-3-8B-Instruct, Gemma-2-9b-it, Claude-3-Haiku, GPT-4o-mini; labels from the benchmark's verifier (§5.1).
- Prompts where all samples are right or all wrong are removed, and so are prompts with fewer than 10% or more than 90% correct samples; 128 responses sampled per model (§5.1).
- Best-of-K curve: `E_{S_K}[g(argmax_{s∈S_K} R̂(s))]` for K = 1..32, where `g` is the verifier and `R̂` the RM (§5.2.1).

## Downstream experiment (§6)
1. Nine general-purpose RMs chosen from RewardBench (§6.1, footnote 4).
2. 8,000 prompts (7,000 from Arena, 500 MMLU-Pro, 500 MATH, disjoint from PPE), 16 samples each from Llama-3.1-8B-Instruct at a temperature drawn from a triangular distribution (a = 0.0, b = 1.0, c = 1.3) (§6.1).
3. Each RM picks chosen = highest score and rejected = a uniformly sampled rank (seeded identically across RMs) (§6.1).
4. DPO on Llama-3.1-8B-Instruct: τ = 0.1, constant schedule, global batch 64, max length 8192, TRL DPOTrainer, AdamW (0.9, 0.999); learning rate printed as "2.00 × 10−0.6" (App. A.3).
5. 13-model blind Arena cohort, 12,190 votes, September 10-16, 2024; BT Arena scores (§6.2).

## Table 3 (post-DPO Arena score, 95% CI)
| Model | Score | CI |
|---|---|---|
| Meta-Llama-3.1-70B-Instruct (baseline) | 1228 | 1218–1238 |
| Athene-RM-70B | 1216 | 1206–1226 |
| Athene-RM-8B | 1209 | 1199–1219 |
| InternLM2-7B-Reward | 1204 | 1194–1212 |
| Llama-3-OffsetBias-RM-8B | 1200 | 1191–1209 |
| ArmoRM-Llama3-8B-v0.1 | 1189 | 1181–1198 |
| Meta-Llama-3.1-8B-Instruct (baseline) | 1178 | 1168–1187 |
| Skywork-Reward-Llama-3.1-8B | 1176 | 1166–1185 |
| Skywork-Reward-Gemma-2-27B | 1173 | 1163–1182 |
| InternLM2-20B-Reward | 1173 | 1163–1182 |
| Nemotron-4-340B-Reward | 1172 | 1163–1180 |
| Meta-Llama-3-8B-Instruct (baseline) | 1152 | 1143–1162 |

## Findings (§7)
- Accuracy on the human-preference set is the best predictor of post-DPO Arena score; Spearman and Kendall correlations have nearly zero correlation with it.
- Math is the most predictive single correctness domain; ROC AUC correlates across benchmarks.
- Lower-quantile aggregation over categories increases correlation in nearly every metric; accuracy peaks at 0.80 (Fig. 5). Authors: "Any domain weakness in a reward model can be exploited by the LLM during training."
- Among top models, RewardBench score is negatively correlated with downstream performance (§2.2, Fig. 4).
- Overall evaluations reach 77% Pearson correlation with downstream performance (§9).

## Limits (§8)
Benchmark leakage is possible; nine RMs; one base model; DPO (offline) rather than PPO, so over-optimization effects under online RL are not tested.
