---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2410.14872
created_at: "2026-09-15"
---

# Excerpt: How to Evaluate Reward Models for RLHF (Preference Proxy Evaluations, PPE)

**Authors:** Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. González, Ion Stoica (UC Berkeley)
**Year:** 2024 (arXiv v1 2024-10; v2 2024-10-22)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v2 PDF text.

## Why ch-49 uses it

It is the only source in this chapter that closes the loop: train policies with different reward models, deploy them to real users, and then ask which offline reward-model metric predicted the outcome.

## Construction (Abstract, §3, Table 1)

- A human preference proxy set of 16,038 labeled pairwise comparisons sampled from Chatbot Arena battles, plus a verifiable-correctness preference set built from standard benchmarks with ground truth.
- 12 metrics across 12 domains. Totals in Table 1: 18,593 prompts and 113,836 responses, against RewardBench's 2,985 prompts and 5,970 responses.
- Correctness metrics include accuracy, ROC AUC, Best-of-K, loss, max score, and end score (§4, App. Tables 23–24).

## End-to-end experiment (§6)

1. For each of nine selected reward models, build 8,000 DPO rows from responses sampled from Llama-3.1-8B-Instruct, with the sample seed fixed across reward models so the candidate pool is identical.
2. Train Llama-3.1-8B-Instruct with DPO on each dataset, producing one policy per reward model.
3. Deploy the 13-model cohort (the nine DPO models plus Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, and Llama-3-8B-Instruct) to Chatbot Arena for blind voting; temperature 0.2 for all except Llama-3-8B-Instruct at 0.7.
4. Collect 12,190 human votes over six days (2024-09-10 to 2024-09-16), about 2,032 battles per model and 190 per unique model pair. Arena scores are computed with the Bradley–Terry model.

## Findings (§7)

1. **Per-pair accuracy on the human preference set is the best predictor** of the post-DPO Arena score. Row-wise Pearson correlation, confidence agreement, and separability have some predictive power but do not exceed accuracy.
2. **Spearman and Kendall rank correlations have nearly zero correlation** with the final Arena score. The authors' explanation: accuracy is measured per preference pair, a more granular scale, while rank correlations aggregate reward-model signal into higher-order preferences.
3. **Low-quantile aggregation predicts better than the mean.** Rescaling each category to mean 0, SD 1 and varying the aggregation quantile, correlation with downstream Arena score rises as the quantile falls; accuracy peaks at 0.80 Pearson correlation at low-quantile aggregation (Fig. 5). Their reading: a reward model must be robust on every input distribution, because any weak domain can be exploited during training.
4. On correctness metrics, the mean across domains correlates well under all metrics, math is the most predictive single domain, and ROC AUC correlates more highly than accuracy across benchmarks.
5. **RewardBench:** the paper reports that, as reward models have improved, there is now a negative correlation between RewardBench evaluation scores on top models and downstream RLHF performance (§1; Fig. 4 shows the Pearson correlation of RewardBench rankings against post-DPO Arena rankings).
6. Under style-controlled Arena scores the human-preference correlations drop slightly, while the correctness-preference correlations are unchanged (§7, App. A.5).

## Limits stated by the paper

Benchmark leakage is acknowledged as possible, with the mitigation that both datasets can be refreshed (§8.1). End-to-end testing covers nine reward models only, and uses DPO rather than PPO, which the authors note may interact differently with over-optimization (§8.2).

## Connections

[[rewardbench]] and [[rm-bench]] (the offline benchmarks it measures against downstream outcomes), [[chatbot-arena]] (both the prompt source and the deployment platform), [[arena-hard-benchbuilder]] (source of the separability and confidence-agreement metrics reused here).
