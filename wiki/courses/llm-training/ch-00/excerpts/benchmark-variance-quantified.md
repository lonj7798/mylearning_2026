---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.10229v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2406.10229
created_at: "2026-09-15"
---

# Excerpt: Quantifying Variance in Evaluation Benchmarks

**Paper:** Madaan, Singh, Schaeffer, Poulton, Koyejo, Stenetorp, et al. (Meta GenAI; UCL; Stanford; Cohere). arXiv v1 2024-06-14 (preprint). Source type: paper.

## Setup (§2)
- Ten Llama-2-7B-architecture models trained from scratch on the same data mixture and data order, differing only in initialization seed; 210B tokens each, 21 checkpoints per run ("seed models"). Plus 41 Llama-1/2-architecture models on the same mixture and 32 public models; over 280 models including checkpoints; 13 benchmarks.

## Definitions (§3.1)
- Seed variance E(S, M): standard deviation of metric S across the seed models, averaged over the 21 checkpoints.
- 95% CI: bootstrapped per model; analytic form 1.96·sqrt(S_M(1 − S_M)/N) with N test instances.
- Monotonicity: Kendall rank correlation between a run's scores over checkpoints and a monotonic sequence.

## Table 1 (7B seed models; size, chance, mean, seed std, 95% CI, monotonicity discrete/continuous)
- ARC-C: 1165, 25, 39.71, 0.80, 2.74, 0.88/0.91.
- COPA: 100, 50, 78.80, 2.15, 8.30, 0.56/0.90.
- HellaSwag: 10042, 25, 70.08, 0.21, 0.93, 0.99/0.99.
- HumanEval: 164, 0, 11.89, 1.11, 3.98, 0.79/0.98.
- MMLU: 14042, 25, 25.86, 0.57, 0.72, 0.09/0.15.
- MMLU-Cloze: 14042, 25, 37.47, 0.22, 0.79, 0.95/0.96.
- The seed standard deviation is generally below the 95% CI; continuous metrics have higher signal-to-noise ratio than discrete ones for all benchmarks (Table 2).

## MMLU formulation (§3.3, Fig. 2)
- Standard MMLU stays at chance for the 7B seed models after 210B tokens; MMLU-Cloze scores higher, with lower seed variance and monotonicity 0.95 instead of 0.09.
- Llama 3 70B: 78.7% standard vs 60.6% cloze; standard and cloze correlate at Pearson 0.92 over 70 models.
- A Llama-2-13B-like model trained from scratch shows a jump near 800B tokens, after which standard MMLU exceeds cloze.
- Recommendation: use cloze formulations for pretraining ablations.

## Item analysis (§4, Table 3)
- Pruning low-discrimination items gives modest standard-error and monotonicity improvements but shifts estimated accuracy; IRT-selected 100-item subsets raise seed standard deviation (ARC-C 0.80 full vs 1.80 IRT).

## Verification
- Read on 2026-09-15 against arXiv:2406.10229v1 PDF text (§1-4, Tables 1-3).
