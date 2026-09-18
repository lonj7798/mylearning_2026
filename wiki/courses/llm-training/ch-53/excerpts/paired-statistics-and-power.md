---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/adding-error-bars-evals.md
source_url: https://arxiv.org/abs/2411.00640
primary_text_checked: arXiv:2411.00640v1 (Miller, Anthropic, 2024-11), 2026-09-17
---

# Excerpt: the five statistical rules ch-53 implements

Used by ch-53 §5 and §9. The paper's own recommendation list (§1):

1. Compute standard errors of the mean using the Central Limit Theorem.
2. When questions are drawn in related groups, compute clustered standard errors.
3. Reduce variance by resampling answers and by analyzing next-token probabilities.
4. When two models are compared, conduct inference on the question-level paired differences rather
   than on the population-level summary statistics.
5. Use power analysis to decide whether an eval, or a random subsample of it, can test the hypothesis
   of interest.

## Framework (§2.1)

Each question score decomposes as `s_i = x_i + ε_i`, where `x_i` is the conditional mean of question i
(the mean over repeated sampling from the model) and `ε_i` is a zero-mean random component with
conditional variance `σ_i² = Var(ε_i)`. Questions are treated as drawn from an unseen super-population,
which is what makes a confidence interval meaningful for a fixed benchmark.

## Clustered standard error (§2.2, Eq. 4)

```
SE_clustered = [ SE_CLT² + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_{i,c} − s̄)(s_{j,c} − s̄) ]^(1/2)
```

`s_{i,c}` is the ith question score in cluster c, `s̄` the overall mean, `n` the number of questions,
and the triple sum runs over clusters and the cross terms inside each cluster. The estimator
interpolates between "each cluster is one observation" (perfect within-cluster correlation) and the
ordinary standard error (no correlation).

Measured on Anthropic models (Table 4, real numbers): DROP 1.34 clustered vs 0.44 naive (ratio 3.05),
MGSM 1.62% vs 0.86% (1.88), RACE-H 0.51% vs 0.46% (1.10). Examples of clustered evals given by the
paper: DROP, QuAC, RACE, SQuAD (several questions per passage) and MGSM (one question translated into
many languages).

## Paired difference (§4.2, Eq. 7)

```
SE_{A−B,paired} = sqrt( Var(s_{A−B}) / n ) = [ (1/(n−1)) Σ_i (s_{A−B,i} − s̄_{A−B})² / n ]^(1/2)
```

with `s_{A−B,i} = s_{A,i} − s_{B,i}`. The variance identity:

```
Var(µ̂_paired) = Var(µ̂_unpaired) − 2 Cov(x_A, x_B) / n
```

so pairing reduces variance whenever the two models agree about which questions are hard. Worked
illustration in the paper: with continuous scores uniform on [0,1] for both models and correlation
0.5, `Var(s_A) = Var(s_B) = 1/12` and `Cov(x_A, x_B) = 1/24`, so pairing reduces estimator variance by
1/3 (from 1/6 to 1/9). An equivalent form from single-model standard errors:

```
SE_{A−B,paired} = sqrt( SE_A² + SE_B² − 2 SE_A SE_B Corr(s_A, s_B) )
```

The clustered paired version is Eq. 8, computed directly on the per-question differences.

The paper's suggested reporting format (Table 5) carries, per eval: the difference, its standard
error, the 95% interval, and the score correlation between the two models.

## Variance reduction by resampling (§3.1) and by next-token probabilities (§3.2)

With K samples per question, in the paper's uniform-difficulty example
`Var(µ̂ | K) = Var(µ̂ | K=1) × (1 + 2/K)/3`: K = 2 reduces total variance by 1/3, K = 4 by 1/2, K = 6
by 5/9, with an upper limit of 2/3. Pooling all K·N answers as if independent gives an inconsistent
standard error. For non-chain-of-thought multiple-choice evals, scoring the next-token probability of
the correct option removes the conditional variance entirely and reaches the same 2/3 limit.

## Temperature is not a variance knob (§3.3)

Reducing sampling temperature can move variance from the conditional component (which the two methods
above can reduce) into the variance of the conditional means (which they cannot), and can shift the
expected score. Paper's examples: a single-token true/false eval with `x_{T=1} ~ U[0,1]` has
`Var(x_{T=1}) = 1/12`, while at T = 0 the distribution becomes Bernoulli(1/2) with variance 1/4; with
`x_{T=1} ~ U[1/3, 1]`, moving to T = 0 changes the expected score from 2/3 to 3/4 and the variance from
1/27 to 3/16.

## Power and sample size (§5, Eq. 9 and Eq. 10)

```
n = (z_{α/2} + z_β)² (ω² + σ_A²/K_A + σ_B²/K_B) / δ²
δ = (z_{α/2} + z_β) sqrt( (ω² + σ_A²/K_A + σ_B²/K_B) / n )
```

`n` = independent questions, `α` = Type I error rate, `β` = Type II error rate (power = 1 − β),
`δ` = minimum detectable effect, `ω² = Var(x_A) + Var(x_B) − 2 Cov(x_A, x_B)`, `σ_A² = E[σ_{A,i}²]` and
`σ_B² = E[σ_{B,i}²]` the expected conditional variances, `K_A, K_B` the samples per question.

Worked example from the paper: `σ_A² = σ_B² = 0`, `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20` gives
`n = (1.96 + 0.84)² (1/9) / 0.03² ≈ 969`. The author concludes that new evals should contain at least
1,000 questions to have good signaling ability.

Also recommended for reporting (Tables 2, 3): the number of questions in each eval, the standard error
of each estimate, and the cluster count when clustered errors are used.

Related: [[adding-error-bars-evals]], [[signal-and-noise-eval]], [[read]].
