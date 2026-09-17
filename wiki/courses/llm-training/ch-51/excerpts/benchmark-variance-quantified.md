---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: "primary source arXiv:2406.10229v1 (planned library card papers/benchmark-variance-quantified.md; not present on 2026-09-15)"
source_url: https://arxiv.org/abs/2406.10229
created_at: "2026-09-15"
note: "Read from the cached v1 text on 2026-09-15. Numbers below are transcribed from Tables 1-4 and §3-§5 of that text."
---

# Excerpt: Quantifying Variance in Evaluation Benchmarks (Madaan et al. 2024)

**Paper:** Lovish Madaan, Aaditya K. Singh, Rylan Schaeffer, Andrew Poulton, Sanmi Koyejo, Pontus Stenetorp,
Sharan Narang, Dieuwke Hupkes (Meta GenAI; UCL; Stanford; Cohere). arXiv v1 2024-06-14. Source type: paper.

## Setup (§2)

- 10 Llama-2-7B-architecture models trained from scratch, identical except the initialization seed (same data
  mixture, deterministic data ordering), 210B tokens each, 21 checkpoints per run: 210 "seed models".
  Additional 41 Llama-1/2-architecture models and 32 public models; about 70 models are used for the item
  analysis of §4.
- 13 benchmarks: AGIEval, ARC-C, BIG-Bench Hard, COPA, GSM8k, HellaSwag, HumanEval, MATH, MMLU,
  Natural Questions, PIQA, SIQA, TriviaQA.
- The variance measured is **pretraining-seed** variance. Fine-tuning seeds are not measured.

## Definitions (§3.1)

- Seed variance E(S, M): standard deviation of the metric across the 10 seed models, averaged over the 21
  checkpoint timesteps.
- 95% CI: bootstrapped per checkpoint with the `bootstrapped` library; the analytic form for discrete metrics is
  `CI_analytic(M) = 1.96 · sqrt(S_M(1 − S_M)/N)`, with S_M the score and N the number of test instances.
  "Empirically, we observe that, for the distributions we consider, bootstrapped and Analytic CIs converge when
  the number of bootstrap samples is large." The Table 1 CI column is an average over 210 per-model intervals
  (§3.2), and footnote 4 states the intervals are 1.96 times the standard error.
- Monotonicity (mon_disc / mon_cont): Kendall rank correlation between the score sequence over checkpoints and a
  monotonically increasing (discrete) or decreasing (continuous) array.

## Table 1 — 7B seed models (size, chance, mean µ, seed std E, 95% CI, mon_disc / mon_cont)

| Benchmark | Size | Chance | µ | E | 95% CI | mon_disc / mon_cont |
|---|---|---|---|---|---|---|
| AGIEval | 2546 | 20 | 23.44 | 0.77 | 1.63 | 0.37 / 0.29 |
| ARC-C | 1165 | 25 | 39.71 | 0.80 | 2.74 | 0.88 / 0.91 |
| BIG-Bench (Hard) | 6511 | 0 | 29.10 | 0.87 | 1.07 | 0.77 / – |
| COPA | 100 | 50 | 78.80 | 2.15 | 8.30 | 0.56 / 0.90 |
| GSM8k | 1319 | 0 | 4.10 | 0.41 | 0.87 | 0.74 / 0.30 |
| HellaSwag | 10042 | 25 | 70.08 | 0.21 | 0.93 | 0.99 / 0.99 |
| HumanEval | 164 | 0 | 11.89 | 1.11 | 3.98 | 0.79 / 0.98 |
| MATH | 5000 | 0 | 1.52 | 0.23 | 0.28 | 0.52 / – |
| MMLU | 14042 | 25 | 25.86 | 0.57 | 0.72 | 0.09 / 0.15 |
| MMLU-Cloze | 14042 | 25 | 37.47 | 0.22 | 0.79 | 0.95 / 0.96 |
| Natural Questions | 3610 | 0 | 16.43 | 0.60 | 1.04 | 0.91 / – |
| PIQA | 1838 | 50 | 76.93 | 0.41 | 1.99 | 0.87 / 0.93 |
| SIQA | 1954 | 33 | 46.69 | 0.55 | 2.21 | 0.66 / 0.81 |
| TriviaQA | 11313 | 0 | 42.69 | 0.45 | 0.83 | 0.99 / – |

Statement in §3.2: "Benchmarks with few test examples (like COPA and HumanEval) exhibit high variance (both
seed variance and 95% CIs). Generally, the 7B seed variance is well below the 95% CI for the same benchmark,
though the ratio of the two is quite variable."

## Discrete vs continuous metrics (Table 2, SNR)

Signal-to-noise ratio computed over the final checkpoints of the 10 seeds. Selected rows
(discrete µ, discrete std, discrete SNR → continuous SNR): ARC-C 39.71, 0.87, 45.89 → 381.64;
GSM8k 4.10, 0.52, 7.88 → 15.24; HellaSwag 70.08, 0.12, 608.23 → 1921.15; HumanEval 11.89, 1.75, 6.79 → 124.08;
MMLU 25.86, 0.49, 52.45 → 347.57. Continuous SNR exceeds discrete SNR for all benchmarks in Table 2, and
mon_cont > mon_disc for all benchmarks in Table 1.

## MMLU formulation (§3.3)

Standard MMLU stays near chance (25.86 with chance 25) for the 7B seed models after 210B tokens, monotonicity
0.09. MMLU-Cloze reaches 37.47 with seed std 0.22 and monotonicity 0.95. On fully trained large models the
standard format scores higher (78.7% vs 60.6% MMLU-Cloze for Llama 3 70B) and the two formats correlate at
Pearson 0.92 over the 70 models of §2.1. A Llama-2-13B-like model trained from scratch shows a jump at about
800B tokens, after which standard MMLU overtakes cloze. Recommendation: use cloze formulations for pretraining
ablations.

## Item analysis and item response theory (§4, §5, Tables 3-4)

- Item discrimination = correlation between per-item scores and overall model scores. Pruning low-discrimination
  items gave "modest improvements" in standard error and monotonicity, with a drift in the estimated accuracy;
  the authors "would not suggest the use of item analysis-based methods for understanding variance in language
  model evaluations".
- Applying the 100-item IRT subsets of Polo et al. 2024 (tinyBenchmarks) to the same models, seed variance E
  increases: ARC-C 0.80 → 1.80 (IRT) / 1.86 (IRT++); GSM8k 0.41 → 1.16 / 1.49; HellaSwag 0.21 → 2.06 / 2.42
  (Table 3). Monotonicity falls: ARC-C mon_disc 0.88 → 0.64 / 0.63; HellaSwag 0.99 → 0.84 / 0.80 (Table 4).
  Mean estimates also drift (ARC-C overestimated by 7% under IRT). Conclusion: "the tiny-benchmarks method may
  have limited use during pretraining ablations as it makes model comparisons more likely to be confounded by
  randomness from the initialisation and data ordering seed."

## Used in

ch-51 §1.2 (measured run-to-run noise), §1.4 (MMLU vs MMLU-Cloze), §7 (metric choice and the IRT subset result),
Recipe, Generalization lens (c).
