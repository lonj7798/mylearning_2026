---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (arXiv:2411.00640v1, 2024-11-01)"
source_url: https://arxiv.org/abs/2411.00640
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug adding-error-bars-evals). Values read from the v1 PDF on 2026-09-15. ch-36/excerpts/adding-error-bars-evals.md covers §4 of the same paper."
---

# Excerpt: Adding Error Bars to Evals (Miller), §2-§5

**Author:** Evan Miller (Anthropic). Source type: paper. Tables 1, 2, 3 and 5 use fictional models and numbers; Table 4 uses "non-fictional numbers" from Anthropic models.

## Standard error of one score (§2.1, Eqs. 1-3)

With question scores s₁ … s_n and mean s̄:

```
SE_C.L.T. = sqrt( Var(s)/n ) = sqrt( (1/(n−1)) Σ_i (s_i − s̄)² / n )      (1)
SE_Bernoulli = sqrt( s̄(1 − s̄)/n )      (2, only when s_i ∈ {0,1})
CI_95% = s̄ ± 1.96 × SE_C.L.T.           (3)
```

On bootstrapping: "we regard bootstrapping as unnecessary unless a complicated sampling scheme or estimator is being used" (§2.1). The Llama 3 report is cited as using SE_Bernoulli even for fractional scores such as F1, which makes its intervals too wide there (§2.1).

## Clustered questions (§2.2, Eq. 4, Table 4)

When questions arrive in related groups (one passage, one problem instantiated in many languages), independence fails:

```
SE_clustered = [ SE²_C.L.T. + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_{i,c} − s̄)(s_{j,c} − s̄) ]^(1/2)     (4)
```

Measured ratios on Anthropic models (Table 4): DROP 1.34 vs 0.44 (3.05×), RACE-H 0.51% vs 0.46% (1.10×), MGSM 1.62% vs 0.86% (1.88×).

## Variance reduction by resampling (§3.1)

Var(µ̂) = Var(s)/n = (Var(x) + E[σ²_i])/n, where x is the per-question conditional mean (a property of the question super-population) and σ²_i the conditional variance of question i. Answering each question K times gives Var(s_i) = σ²_i/K. In the worked example with binary scores and uniformly distributed difficulty, Var(x) = 1/12 and E[σ²_i] = 1/6, so

```
Var(µ̂ | K) = Var(µ̂ | K = 1) × (1 + 2/K)/3
```

K = 2 reduces total variance by 1/3, K = 4 by 1/2, K = 6 by 5/9; the limit as K grows is 2/3. Pooling all K·n answers as if independent is inconsistent (§3.1).

## Do not lower the temperature (§3.3)

"It may be tempting to reduce the 'sampling temperature' of the model in order to reduce (or eliminate) the conditional variance. However, we advise against this practice, unless the purpose is to study the model at the new temperature." Two worked single-token examples: with x_{T=1} ~ U[0,1], Var(x_{T=1}) = 1/12 but Var(x_{T=0}) = 1/4; with x_{T=1} ~ U[1/3, 1], E[x] moves from 2/3 to 3/4 and Var(x) from 1/27 to 3/16.

## Paired comparison of two models (§4.2, Eqs. 7-8)

```
SE_{A−B,paired} = sqrt( Var(s_{A−B})/n )                                    (7)
                = sqrt( SE²_A + SE²_B − 2 SE_A SE_B Corr(s_A, s_B) )
Var(µ̂_{A−B,paired}) = Var(µ̂_{A−B,unpaired}) − 2 Cov(x_A, x_B)/n
```

With uniform continuous scores and correlation 0.5, pairing reduces estimator variance by 1/3 (from 1/6 to 1/9). Equation 8 gives the clustered version computed directly on the differences. Recommendation: "using the paired version of the standard error estimate wherever practicable", and reporting pairwise differences, standard errors, and score correlations in technical reports (§4.2).

## Power analysis (§5, Eqs. 9-10)

With ω² = Var(x_A) + Var(x_B) − 2Cov(x_A, x_B), σ²_A = E[σ²_{A,i}], σ²_B = E[σ²_{B,i}], significance level α, power 1 − β, minimum detectable effect δ, and K_A, K_B answers sampled per question:

```
n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²                (9)
δ = (z_{α/2} + z_β) sqrt( (ω² + σ²_A/K_A + σ²_B/K_B) / n )          (10)
```

Worked examples: with σ²_A = σ²_B = 0, ω² = 1/9, δ = 0.03, α = 0.05, β = 0.20, n ≈ 969 independent questions, which the author reads as a case for new evals containing at least 1,000 questions. With σ²_A = σ²_B = 1/6, ω² = 1/9, Corr(x_A, x_B) = 0.5, n = 198, α = 0.05 and β = 0.20, raising K_A = K_B from 1 to 10 moves the minimum detectable effect from 13.2% to 7.5%.

## Used in

ch-46a §7 (paired intervals, resampling budget, and the power calculation for the generality gate) and the Common-mistakes table.
