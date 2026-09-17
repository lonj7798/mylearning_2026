---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/adding-error-bars-evals.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2411.00640
created_at: "2026-09-15"
---

# Excerpt: Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations

**Author:** Evan Miller (Anthropic)
**Version read:** arXiv:2411.00640v1 (1 Nov 2024).
**Status:** no library card existed for this slug on 2026-09-15; the equations below were checked against the v1 PDF text. ch-53 applies them in a lab.

## Single model (§2.1)
- Eq. 1: SE_CLT = sqrt( Var(s)/n ) = sqrt( (1/(n−1)) Σ_i (s_i − s̄)² / n ), where s_i is the score on question i and n the number of questions.
- Eq. 2: for s_i ∈ {0, 1}, SE_Bernoulli = sqrt( s̄(1 − s̄)/n ).
- Eq. 3: CI_95% = s̄ ± 1.96 × SE_CLT.
- On bootstrapping: "we regard bootstrapping as unnecessary unless a complicated sampling scheme or estimator is being used."

## Clustered questions (§2.2)
When questions come in related groups (for example several questions about one passage), Eq. 4 adds within-cluster covariance terms. Table 4 caption: "clustered standard errors can be over 3X larger than naive standard errors."

## Two models (§4)
- Unpaired: SE_{A−B} = sqrt(SE_A² + SE_B²); CI_{A−B,95%} = μ̂_{A−B} ± 1.96 × SE_{A−B} (Eq. 5); z_{A−B} = μ̂_{A−B} / SE_{A−B} (Eq. 6).
- Paired, when both models answer the same questions (Eq. 7): with s_{A−B,i} = s_{A,i} − s_{B,i},
  SE_{A−B,paired} = sqrt( (1/(n−1)) Σ_i (s_{A−B,i} − s̄_{A−B})² / n ).
- Variance relation: Var(μ̂_{A−B,paired}) = Var(μ̂_{A−B,unpaired}) − 2 Cov(x_A, x_B)/n. "We can thus reduce the variance with paired differences as long as the conditional means of the model scores are correlated; that is to say, if the two models have some amount of agreement on which questions are 'easy' and which questions are 'hard'."
- Table 5 suggests reporting, per eval, the difference with its standard error, the 95% confidence interval, and the correlation between the two models' question scores.

## Note on the worked variance example in §4.2
The paper's illustration (uniform scores, correlation 0.5) states a reduction "from 1/6 to 1/9". With Var(s_A) = Var(s_B) = 1/12 and Cov = 1/24, the variance relation above gives 1/6 − 2/24 = 1/12. ch-30a does not reuse this example and uses its own worked example instead.

## How ch-30a uses it
§1 (paired confidence interval for a forgetting delta), §6 checklist, figure `figures/paired-forgetting-delta.html`.
