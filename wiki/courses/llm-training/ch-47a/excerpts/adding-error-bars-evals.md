---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (arXiv:2411.00640v1, 2024-11-01)"
source_url: https://arxiv.org/abs/2411.00640
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug adding-error-bars-evals). Values and formulas read from the v1 PDF on 2026-09-15."
---

# Excerpt: paired comparison and power analysis for evaluations

**Author:** Evan Miller (Anthropic).

## Framing (§1)

Evaluation questions are treated as a sample from an unseen super-population, so a benchmark score is an estimate with a standard error, and a difference between two models is a statistical comparison rather than an ordering.

Five recommendations (§1):
1. Compute standard errors of the mean using the Central Limit Theorem.
2. Use clustered standard errors when questions come in related groups.
3. Reduce variance by resampling answers, or by reading next-token probabilities where the eval permits it.
4. Analyze question-level **paired differences** rather than population-level summary statistics.
5. Use power analysis to decide whether an eval can detect the effect of interest.

Clustered standard errors on two real evals with Anthropic models are up to about 3× the naive ones (Table 4).

## Paired analysis (§4.1, §4.2, Eqs. 5–7)

Unpaired: `SE_{A−B} = sqrt(SE_A² + SE_B²)`.

Paired, with `s_{A−B,i} = s_{A,i} − s_{B,i}` the per-question score difference and `s̄_{A−B}` their mean:

```
SE_{A−B,paired} = sqrt( (1/(n−1)) Σ_i (s_{A−B,i} − s̄_{A−B})² / n )
```

- `n` = number of questions; `s_{A,i}` = score of model A on question i.
- `CI_{95%} = s̄_{A−B} ± 1.96 SE_{A−B}` and `z = s̄_{A−B} / SE_{A−B}`.
- `Var(µ̂_{A−B,paired}) = Var(µ̂_{A−B,unpaired}) − 2 Cov(x_A, x_B)/n`, so pairing helps whenever the two models agree about which questions are hard. Worked example in §4.2: uniform scores on [0,1] with correlation 0.5 give a one-third relative variance reduction.
- Table 5 shows the suggested reporting format: difference, standard error, 95% interval, and the score correlation, per eval.

## Power analysis (§5, Eqs. 9–10)

```
n = (z_{α/2} + z_β)² (ω² + σ_A²/K_A + σ_B²/K_B) / δ²
δ = (z_{α/2} + z_β) sqrt( (ω² + σ_A²/K_A + σ_B²/K_B) / n )
```

- `ω² = Var(x_A) + Var(x_B) − 2 Cov(x_A, x_B)`, the variance of the per-question conditional means of the difference; `σ_A²`, `σ_B²` the mean conditional variances of the two models' answers; `K_A`, `K_B` the number of samples drawn per question; `δ` the minimum detectable effect; `α` the false-positive rate; `1 − β` the power.
- Worked example (§5): with `σ_A² = σ_B² = 0`, `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20`, the requirement is `n ≈ 969`. The author writes that new evals should hold at least 1,000 questions to have good signalling ability.

## Sampling temperature (§3.3)

The paper advises against lowering sampling temperature to reduce variance: in its examples, moving from T = 1 to T = 0 tripled the minimum variance in one case (1/12 → 1/4) and shifted the expected score in another. Temperature should be changed only when the intention is to study the model at that temperature.

## Used in

ch-47a §7 (audit protocol: paired comparisons and minimum detectable effect), Recipe, Common mistakes.
