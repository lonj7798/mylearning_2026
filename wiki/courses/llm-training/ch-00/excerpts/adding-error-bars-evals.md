---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:2411.00640v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2411.00640
created_at: "2026-09-15"
---

# Excerpt: Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations

**Paper:** Evan Miller (Anthropic). arXiv v1 2024-11-01. Source type: paper. Tables 1, 2, 3, and 5 use fictional models and numbers; Table 4 uses Anthropic models.

## Framework (§2.1, Eq. 1-3)
- Eval questions are treated as drawn from an unseen super-population. With n independent questions, scores s_i, and mean s̄: SE = sqrt(Var(s)/n) = sqrt( (1/(n−1)) Σ_i (s_i − s̄)² / n ). For 0/1 scores: SE = sqrt(s̄(1 − s̄)/n). CI_95 = s̄ ± 1.96·SE.
- Using the 0/1 formula for fractional scores (for example F1) gives standard errors that tend to be too wide (§2.1).

## Clustered questions (§2.2, Eq. 4, Table 4)
- Questions drawn in groups (passages with several questions; one question in many languages) need clustered standard errors.
- Anthropic models: clustered vs naive SE — DROP 1.34 vs 0.44 (ratio 3.05); RACE-H 0.51% vs 0.46% (1.10); MGSM 1.62% vs 0.86% (1.88).

## Variance reduction (§3)
- Var(μ̂) = (Var(x) + E[σ_i²])/n: variance across questions plus mean within-question sampling variance.
- Resampling each question K times reduces the second term to E[σ_i²]/K (§3.1); scoring next-token probabilities removes it for non-chain-of-thought multiple-choice evals (§3.2).
- Do not lower sampling temperature to reduce variance: it can move variance into the question term or bias the estimate (§3.3).

## Comparing models (§4, Eq. 5-7)
- Unpaired: SE_{A−B} = sqrt(SE_A² + SE_B²); CI = μ̂_{A−B} ± 1.96·SE_{A−B}; z = μ̂_{A−B}/SE_{A−B}.
- Paired on the same questions: SE from per-question differences, equivalently sqrt(SE_A² + SE_B² − 2·SE_A·SE_B·Corr(s_A, s_B)). Positive correlation between models' per-question scores makes the paired SE smaller.
- Recommendations: report n and the SE with each score; report pairwise differences, pairwise SEs, and correlations; use power analysis to check whether an eval can detect the hypothesized difference (§1, §5).

## Verification
- Read on 2026-09-15 against arXiv:2411.00640v1 PDF text (§1-4).
