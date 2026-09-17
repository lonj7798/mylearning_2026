---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: "primary source arXiv:2411.00640v1, 2024-11-01 (planned library card adding-error-bars-evals; not present on 2026-09-15)"
source_url: https://arxiv.org/abs/2411.00640
created_at: "2026-09-15"
note: "Read from the cached v1 text on 2026-09-15. Tables 1, 2, 3 and 5 of the paper use fictional models and numbers; Table 4 uses 'non-fictional numbers' from Anthropic models."
---

# Excerpt: Adding Error Bars to Evals — A Statistical Approach to Language Model Evaluations (Miller 2024)

**Author:** Evan Miller (Anthropic). Source type: paper.

## Standard error of one score (§2.1, Eqs. 1-3)

With per-question scores s₁ … s_n and mean s̄:

```
SE_C.L.T.   = sqrt( Var(s)/n ) = sqrt( (1/(n−1)) Σ_i (s_i − s̄)² / n )      (1)
SE_Bernoulli = sqrt( s̄(1 − s̄)/n )                                          (2)   [only when s_i ∈ {0,1}]
CI_95%       = s̄ ± 1.96 × SE_C.L.T.                                         (3)
```

On bootstrapping: "the Central Limit Theorem is applicable to any evals having scores with finite variance and a
large number of questions, and so we regard bootstrapping as unnecessary unless a complicated sampling scheme or
estimator is being used" (§2.1). On the Llama 3 report: it "incorrectly estimates all of its standard errors using
SE_Bernoulli, even when s_i takes fractional values, such as with an F1 score. In these cases, SE_Bernoulli will
tend to be conservative (too wide) compared to SE_C.L.T." (§2.1). The Inspect framework computes SE_C.L.T. in its
`stderr()` metric (§2.1).

## Clustered questions (§2.2, Eq. 4, Table 4)

Applies when questions arrive in related groups: several questions about one passage (DROP, QuAC, RACE, SQuAD) or
the same question in many languages (MGSM).

```
SE_clustered = [ SE²_C.L.T. + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_{i,c} − s̄)(s_{j,c} − s̄) ]^(1/2)     (4)
```

s_{i,c} is the score of question i inside cluster c; clusters are assumed independent. The adjustment is "a kind of
'sliding scale'" between perfectly correlated clusters (each cluster is one observation) and uncorrelated ones
(Eq. 4 reduces to Eq. 1).

Measured on Anthropic models (Table 4, non-fictional): DROP SE_clustered 1.34 vs SE_C.L.T. 0.44, ratio 3.05;
RACE-H 0.51% vs 0.46%, ratio 1.10; MGSM 1.62% vs 0.86%, ratio 1.88. Table 3 reports DROP with 9,622 questions in
588 clusters, RACE-H 3,498 in 1,045, MGSM 2,500 in 250, and recommends printing the cluster count next to the
question count.

## Repeated generations per question (§3.1, §3.3)

Var(µ̂) = Var(s)/n = (Var(x) + E[σ²_i])/n, with x the per-question conditional mean (a property of the question
super-population) and σ²_i the conditional variance of question i under sampling. Answering each question K times
gives Var(s_i) = σ²_i/K. In the worked example with binary scores and uniformly distributed difficulty,
Var(x) = 1/12 and E[σ²_i] = 1/6, so

```
Var(µ̂ | K) = Var(µ̂ | K = 1) × (1 + 2/K)/3
```

K = 2 removes 1/3 of the variance, K = 4 one half, K = 6 five ninths; the limit as K → ∞ is 2/3 of the K = 1
variance. Pooling all K·n answers as if independent is inconsistent (§3.1).

Temperature: "It may be tempting to reduce the 'sampling temperature' of the model in order to reduce (or
eliminate) the conditional variance. However, we advise against this practice, unless the purpose is to study the
model at the new temperature" (§3.3). Two single-token worked cases: with x_{T=1} ~ U[0,1], Var(x_{T=1}) = 1/12 but
Var(x_{T=0}) = 1/4; with x_{T=1} ~ U[1/3, 1], E[x] moves from 2/3 to 3/4 and Var(x) from 1/27 to 3/16.
Recommendation: use next-token probabilities when available; otherwise pick K with E[σ²_i]/K ≪ Var(x). "In neither
case should the sampling temperature be adjusted for the sake of reducing variance in the scores."

## Comparing two models (§4, Eqs. 5-8)

```
unpaired:  SE_{A−B} = sqrt( SE²_A + SE²_B )                                        (5)
           z_{A−B}  = µ̂_{A−B} / SE_{A−B}                                          (6)
paired:    SE_{A−B,paired} = sqrt( Var(s_{A−B})/n )                                (7)
                           = sqrt( SE²_A + SE²_B − 2 SE_A SE_B Corr(s_A, s_B) )
           Var(µ̂_{A−B,paired}) = Var(µ̂_{A−B,unpaired}) − 2 Cov(x_A, x_B)/n
clustered paired: Eq. 8, computed directly on the per-question differences s_{A−B,i,c}
```

With uniform continuous scores and Corr(x_A, x_B) = 0.5, pairing lowers the estimator variance by 1/3 (1/6 → 1/9).
Recommendation: "using the paired version of the standard error estimate wherever practicable", and reporting
pairwise differences, pairwise standard errors, and score correlations. In the paper's running example, applying
pairing and clustering reverses the informal reading of the results table: the difference is significant on MATH
and indistinguishable from noise on HumanEval and MGSM (§4.2).

## Power analysis (§5, Eqs. 9-10)

With ω² = Var(x_A) + Var(x_B) − 2Cov(x_A, x_B), σ²_A = E[σ²_{A,i}], σ²_B = E[σ²_{B,i}], significance level α,
power 1 − β, minimum detectable effect δ, and K_A, K_B generations per question:

```
n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²                (9)
δ = (z_{α/2} + z_β) sqrt( (ω² + σ²_A/K_A + σ²_B/K_B) / n )          (10)
```

Worked examples: with σ²_A = σ²_B = 0, ω² = 1/9, δ = 0.03, α = 0.05, β = 0.20,
n = (z_0.025 + z_0.20)² (1/9)/(0.03)² ≈ 969 independent questions, which the author reads as a case for new evals
containing at least 1,000 questions. With σ²_A = σ²_B = 1/6, ω² = 1/9, Corr(x_A, x_B) = 0.5, n = 198, α = 0.05,
β = 0.20, raising K_A = K_B from 1 to 10 moves the minimum detectable effect from 13.2% to 7.5%.

## Used in

ch-51 §2 (standard errors, clustered standard errors), §3 (paired comparison), §4 (power and eval sizing),
Recipe, Common mistakes.
