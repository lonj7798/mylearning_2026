---
chapter: ch-32f
course: llm-training
phase: read
excerpt_of: primary source arXiv:2411.00640v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2411.00640
created_at: "2026-09-15"
---

# Excerpt: Adding Error Bars to Evals (Miller), §4 comparing two models

**Paper:** Evan Miller (Anthropic), "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations". arXiv v1 2024-11-01. Source type: paper. Table 5 uses fictional models and numbers.

## Unpaired analysis (§4.1, Eqs. 5-6)

```
μ̂_{A−B} = μ̂_A − μ̂_B
SE_{A−B} = sqrt(SE_A² + SE_B²)
CI_{A−B,95%} = μ̂_{A−B} ± 1.96 × SE_{A−B}          (5)
z_{A−B} = μ̂_{A−B} / SE_{A−B}                       (6)
```

## Paired analysis (§4.2, Eq. 7)

With s_{A−B,i} = s_{A,i} − s_{B,i} the score difference on question i and s̄_{A−B} its mean over n questions:

```
SE_{A−B,paired} = sqrt( Var(s_{A−B}) / n ) = sqrt( (1/(n−1)) Σ_i (s_{A−B,i} − s̄_{A−B})² / n )     (7)
Var(μ̂_{A−B,paired}) = Var(μ̂_{A−B,unpaired}) − 2 Cov(x_A, x_B)/n
SE_{A−B,paired} = sqrt( SE_A² + SE_B² − 2 SE_A SE_B Corr(s_A, s_B) )
```

"We can thus reduce the variance with paired differences as long as the conditional means of the model scores are correlated; that is to say, if the two models have some amount of agreement on which questions are 'easy' and which questions are 'hard'." The author recommends "using the paired version of the standard error estimate wherever practicable" and reporting pairwise differences, pairwise standard errors, and score correlations (§4.2). A clustered version (Eq. 8) applies when questions are drawn in related groups.

## Used in

ch-32f §5 (paired short-context regression gate). Copied from the ch-36 excerpt; Eq. 7 re-checked against arXiv:2411.00640v1 §4.2 on 2026-09-15.
