---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.10229v1 (planned library card papers/benchmark-variance-quantified.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2406.10229
created_at: "2026-09-15"
---

# Excerpt: Quantifying Variance in Evaluation Benchmarks

**Paper:** Lovish Madaan, Aaditya K. Singh, Rylan Schaeffer, Andrew Poulton, Sanmi Koyejo, Pontus Stenetorp, Sharan Narang, Dieuwke Hupkes (Meta GenAI; UCL; Stanford; Cohere). arXiv v1 2024-06-14 (preprint). Source type: paper.

## Setup (§2.1-2.2)

- Ten Llama-2-7B-architecture models trained from scratch, identical except for the initialization seed (same data mixture and data loading), 210B tokens each, 21 checkpoints per run: 210 "seed models". Also 41 Llama-1/2-architecture models on the same mixture and 32 public models; over 280 models in total.
- 13 benchmarks: AGIEval, ARC-C, BIG-Bench Hard, COPA, GSM8k, HellaSwag, HumanEval, MATH, MMLU, Natural Questions, PIQA, SIQA, TriviaQA.
- The seed variance is pretraining-seed variance. The paper does not measure fine-tuning seeds.

## Definitions (§3.1)

- Seed variance E(S, M): standard deviation of metric S across the 10 seed models, averaged over the 21 checkpoints.
- 95% CI: bootstrapped per checkpoint with the `bootstrapped` library; the analytic form for discrete metrics is CI_analytic(M) = 1.96 · sqrt(S_M(1 − S_M)/N), where S_M is the score and N the number of test instances. "Bootstrapped and Analytic CIs converge when the number of bootstrap samples is large."
- The Table 1 value is "an average of 210 (one for each model) confidence interval sizes" (§3.2).

## Table 1 (7B seed models; size, chance, mean µ, seed std E, 95% CI, monotonicity discrete/continuous)

| Benchmark | Size | Chance | µ | E | 95% CI | mon_disc / mon_cont |
|---|---|---|---|---|---|---|
| ARC-C | 1165 | 25 | 39.71 | 0.80 | 2.74 | 0.88 / 0.91 |
| GSM8k | 1319 | 0 | 4.10 | 0.41 | 0.87 | 0.74 / 0.30 |
| HellaSwag | 10042 | 25 | 70.08 | 0.21 | 0.93 | 0.99 / 0.99 |
| HumanEval | 164 | 0 | 11.89 | 1.11 | 3.98 | 0.79 / 0.98 |
| MMLU | 14042 | 25 | 25.86 | 0.57 | 0.72 | 0.09 / 0.15 |
| MMLU-Cloze | 14042 | 25 | 37.47 | 0.22 | 0.79 | 0.95 / 0.96 |
| TriviaQA | 11313 | 0 | 42.69 | 0.45 | 0.83 | 0.99 / – |
| COPA | 100 | 50 | 78.80 | 2.15 | 8.30 | 0.56 / 0.90 |

Statements (§3.2): "Benchmarks with few test examples (like COPA and HumanEval) exhibit high variance (both seed variance and 95% CIs). Generally, the 7B seed variance is well below the 95% CI for the same benchmark, though the ratio of the two is quite variable." Continuous metrics have higher signal-to-noise ratio than discrete metrics for all benchmarks in Table 2.

## Check performed for ch-36 (derived)

With the analytic formula, ARC-C at S = 0.3971 and N = 1165 gives 1.96 · sqrt(0.3971 × 0.6029 / 1165) = 0.0281, and HellaSwag at 0.7008 and 10042 gives 0.0090. The printed values 2.74 and 0.93 (in points) are close to these, so the Table 1 "95% CI" column is read as the half-width 1.96·SE in accuracy points.

## MMLU formulation (§3.3)

Standard MMLU stays near chance for the 7B seed models after 210B tokens (monotonicity 0.09); MMLU-Cloze has lower seed variance and monotonicity 0.95. The authors recommend cloze formulations for small-model ablations.

## Used in

ch-36 §6.1 (item-level intervals) and §6.3 (seed variance), Generalization lens (c).
