<!-- scope: signal (spread across models), noise (checkpoint-to-checkpoint variability), and their ratio as predictors of benchmark reliability for small-scale decisions and scaling-law prediction; three interventions (subtask filtering, checkpoint averaging, bits-per-byte)
     deps: [[datadecide]], [[task-scaling-model-ladders]]
     see-also: [[olmes]], [[benchmark-variance-quantified]], [[adding-error-bars-evals]], [[olmo-2]]
-->

# Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation
- **Core Insight:** Across the OLMES tasks, a benchmark's signal-to-noise ratio (SNR) at small scale correlates with the decision accuracy of predicting 1B DataDecide rankings from 60M-750M models (R = 0.791, R² = 0.626), while signal or noise alone does not (§4.1, Fig. 2); switching the metric to bits-per-byte raises the 30-task average SNR from 10.0 to 31.5 and decision accuracy from 77.0% to 83.7% (Fig. 6).
- **Guideline:** When selecting benchmarks or metrics for small-scale development decisions, prefer those with high SNR at the model scale in use, and compute scores as bits-per-byte or as an average over the final checkpoints, because in this study bits-per-byte improved decision accuracy for 90.0% of benchmarks and checkpoint averaging raised 30-task decision accuracy by 2.4 points (§5.2-5.3, Table 1). When a multi-task average has low SNR, the authors recommend comparing models on individual high-SNR tasks (App. B.3).
- **Authors:** David Heineman, Valentin Hofmann, Ian Magnusson, Yuling Gu, Noah A. Smith, Hannaneh Hajishirzi, et al. (Allen Institute for AI; University of Washington)
- **Year:** 2025 (arXiv v1 2025-08; preprint)
- **URL:** https://arxiv.org/abs/2508.13144
- **Source type:** paper
- **Relevant topics:** evaluation noise, benchmark selection, signal-to-noise ratio, decision accuracy, scaling-law prediction error, bits-per-byte, checkpoint averaging, benchmark saturation

## Abstract
Model development relies on decisions made with small experiments evaluated on multi-task suites. The paper defines two benchmark properties: signal, the ability to separate better models from worse ones, and noise, the sensitivity of a score to random variability between training steps. Benchmarks with a better signal-to-noise ratio give more reliable decisions at small scale, and benchmarks with less noise have lower scaling-law prediction error. Three interventions follow: using a metric with better signal and noise (for example perplexity instead of accuracy), filtering noisy subtasks to raise an aggregate SNR, and averaging the outputs of a model's intermediate checkpoints. The authors recommend that benchmark creators and users aim for high signal and low noise. The study uses 30 benchmarks and open-weight models from 60M to 32B parameters and releases 900K evaluation results covering 200M instances.

## Key Contributions
- Definitions of signal (relative dispersion), noise (relative standard deviation over the final n checkpoints), and SNR for LM evaluation (§3, Eq. 2).
- Evidence that SNR predicts decision accuracy and that target-model noise tracks scaling-law prediction error (§4, Fig. 2-3).
- Three interventions tested on both settings: SNR-based subtask filtering, checkpoint averaging, and bits-per-byte (§5, Fig. 4, Table 1, Fig. 6).
- SNR tables at OLMo 2 compute scales (1.5B-4T to 32B-6T) that show benchmark saturation and late usefulness (App. B.3, Table 4).
- A released dataset of evaluations over DataDecide, ladder, OLMo 2, seed/data-order, and 73 external models (App. A.5).

## Key Figures/Tables to Study
- **Fig. 1:** 25 DataDecide training curves on HellaSwag (low noise, low signal), ARC Challenge (high noise, high signal), MMLU (low noise, high signal).
- **Fig. 2:** signal, noise, and SNR against decision accuracy.
- **Fig. 3:** noise at the 13B target against scaling-law error across 30 tasks.
- **Fig. 4:** SNR-sorted versus random subtask subsets of MMLU and AutoBencher.
- **Table 1 and Fig. 6:** checkpoint averaging and bits-per-byte versus primary metric.
- **Table 4 (App. B.3):** SNR per benchmark at OLMo 2 scales.

## Technical Details
- **Noise:** Rel. Std.(m) = sqrt((1/(n−1)) Σ_i (m_i − m̄)²) / m̄, where m_i is the score of checkpoint i among the final n checkpoints of one run and m̄ their mean (§3.1).
- **Signal:** Rel. Dispersion(M) = max_{j,k} |m_j − m_k| / m̄, over final checkpoints of a population of models trained with similar compute (§3.2). Among the 20 spread measures considered, relative dispersion had the highest SNR-to-decision-accuracy R² (0.5687) versus relative std (0.5657) and Gini coefficient (0.0944) (App. A.4, Table 3).
- **SNR:** Rel. Dispersion(final checkpoint) / Rel. Std.(final n checkpoints) (Eq. 2).
- **Why final-n noise:** the final 30 checkpoints of 1B models on ARC Challenge span 1.7% accuracy (§3.1). For 10 seed runs and 10 data-order runs of a 1B-5xC model, seed noise, data-order noise, and whole-run total variation correlate with final-n relative std at R² 0.82, 0.86, and 0.95 (§3.1, Fig. 7).
- **Decision accuracy:** share of model pairs ranked the same at small and large scale (Eq. 1); equals (Kendall's τ + 1)/2 when ties are excluded (App. A.2). Setup: DataDecide 60M-750M models predict 1B rankings; noise uses the final 5 checkpoints averaged over the small models (§4.1).
- **Scaling-law setup:** two-step fit (loss from N and D, then a sigmoid from task loss to metric) on ladder models 190M-3.2B, predicting OLMo 2 13B; target noise from the final 30 checkpoints of the 13B run, spaced 1000 steps (§4.2; App. A.1). Noise versus error across 30 tasks: R = 0.653, R² = 0.426; MBPP+, SocialIQA, MMLU, and TriviaQA have similar error (about 2-3%) but different noise (§4.2).
- **Subtask filtering:** the top 16 MMLU subtasks and top 6 AutoBencher subtasks by SNR have higher SNR than the full sets, with +2.6% (MMLU) and +5% (AutoBencher) decision accuracy (§5.1, Fig. 4). Of 20 MMLU subtasks with label errors in at least 5% of instances, 10 are among the 20 lowest-SNR subtasks (§5.1).
- **Checkpoint averaging (Table 1, bits-per-byte scores):** 30-task decision accuracy (60M-5xC to 1B-5xC) 68.9% (final checkpoint) → 71.3% (average of final checkpoints for both small and target models); improved in all but two tasks. 13B prediction error 1.03 → 0.86 absolute %, improved for 20 of 30 tasks (§5.2). For early stopping, an exponential moving average gives higher decision accuracy than a single checkpoint at almost any training step (Fig. 5).
- **Bits-per-byte (BPB):** negative log-likelihood of the correct answer divided by its UTF-8 byte count (§5.3). SNR rises from 1.2 to 7.0 on GSM8K and from 2.0 to 41.8 on MBPP; decision accuracy (150M to 1B) improves for 90.0% of benchmarks and 13B prediction error drops for 73.3% (§5.3). Examples (primary → BPB decision accuracy): Minerva MATH 51.0 → 90.0, HellaSwag 74.3 → 95.3, MMLU 89.0 → 92.0 (Fig. 6).
- **Choice of n:** under a normality assumption, n = 9 checkpoints give a 95% probability that the sample std lies within 1σ of the true noise; on OLMo 2 7B, n = 5 meets ±1σ for almost all benchmarks and n = 20 meets ±0.2σ for 34 of 39 benchmarks (App. A.3.2, Table 2).
- **Benchmark size:** gains flatten after about 1K instances for some benchmarks; AutoBencher (33K instances) has the highest checkpoint noise, and a 300-instance subset of ARC Easy has lower noise than the full AutoBencher; a 1,000-question ARC Easy subset has higher decision accuracy than MMLU with 90% fewer instances (App. B.2, Fig. 9).
- **Large-scale SNR:** signal uses open-weight base models within ±10% of estimated FLOPs (at least 8 per size). SNR drops from 1.5B-4T to 32B-6T for ARC Easy (7.89 → 5.10) and SocialIQA (8.73 → 1.95); Minerva MATH 500 rises from 0.91 (1.5B-4T) to 4.45 (7B-4T) (App. B.3, Table 4).
- **Model count:** the abstract says 375 models; §1 and App. A.5.1 say 465. App. A.5.1 lists 25 ladder models with 7B-4T and 13B-5T targets, 225 DataDecide models, 20 seed/data-order models, 120 OLMo 2 final checkpoints, and 73 external base models.
- **Cost:** 94K H100 hours for evaluation; 23K GPU hours to train the 20 seed/data-order models on 2×8 H100 per run (App. A.5.1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DataDecide small models (decision-accuracy setup) | 60M-750M | eval-gate | noise window | final 5 checkpoints, averaged over small models | arXiv:2508.13144v1 §4.1; App. A.3.2 | verified 2026-09-14 | Table 2: n = 5 within ±1σ for almost all benchmarks |
| OLMo 2 13B (scaling-law target) | 13B | eval-gate | noise window | final 30 checkpoints, 1000 steps apart | §4.2, footnote 4 | verified 2026-09-14 | "adequate trade-off between sample size and compute cost" (footnote 4) |
| Early-stopping comparison, DataDecide 1B-5xC | 1B | eval-gate | smoothing | EMA with N = 2, 5, 20 versus single checkpoint | Fig. 5 legend | verified 2026-09-14 | Fig. 5 |
| Seed and data-order runs | 1B-5xC | pretrain-stable | data; runs; compute | OLMoE mix; 10 seeds + 10 data orders; 23K GPU hours, 2×8 H100 per run | App. A.5.1 | verified 2026-09-14 | not applicable |
| Seed and data-order runs | 1B-5xC | pretrain-stable | tokens, batch, LR, schedule | not reported | checked §2-3, App. A | not reported | none |

## Findings relevant to generality
- **Checkpoint noise can change model rankings.** When one of the final 5 checkpoints of the small and large models is resampled, benchmarks with higher SNR give a sampled decision-accuracy distribution with higher mean and lower variance (App. B.1, Fig. 8). Result (single study).
- **Saturation:** SNR at one scale does not transfer to another; ARC Easy and SocialIQA lose SNR at 32B-6T, and some math benchmarks gain it at 7B (App. B.3). Result (single study).
- **Limits stated by the authors:** only decision accuracy and scaling-law error are studied, and only training-time noise; noise from evaluation configuration is left to future work (§6).

## Connections
- [[datadecide]]: supplies the 25-corpus models and the decision-accuracy metric.
- [[task-scaling-model-ladders]]: ladder models and two-step scaling-law fit (Bhagia et al.).
- [[olmes]]: evaluation standard used where applicable.
- [[olmo-2]]: final checkpoints at 1B-32B used for noise and large-scale SNR.
- [[benchmark-variance-quantified]]: Madaan et al., closest prior SNR measure with seed noise from 10 models (§6).
- [[adding-error-bars-evals]]: cited instance-level statistical noise; this paper measures modeling noise instead (§6).
- [[overtraining-downstream-scaling]]: Gadre et al., multi-task averages for downstream scaling laws (§4.2).
- [[emergent-abilities]]: cited for tasks that stay at chance until large compute (§2).
- [[mmlu-pro]]: cited as benchmark expansion; evaluated among the 30 tasks.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2508.13144 (v1, 2025-08-18).
- Audit claims not found in the source: none. Note: the audit's "375 models" is the abstract figure; the body reports 465 (§1, App. A.5.1).
