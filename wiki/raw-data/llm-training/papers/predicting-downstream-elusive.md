<!-- scope: Schaeffer et al. 2024: why multiple-choice benchmark scores are harder to predict from pre-training compute than per-sample log-likelihoods; the identified mechanism is probability mass on incorrect choices; five model families (Pythia, Cerebras-GPT, OLMo, INCITE, LLM360 Amber) on multiple-choice benchmarks from the LM Evaluation Harness
     deps: [[kaplan-scaling-laws]]
     see-also: [[emergent-abilities-mirage]], [[emergence-loss-perspective]], [[observational-scaling-laws]], [[overtraining-downstream-scaling]], [[same-loss-better-downstream]]
-->

# Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?
- **Core Insight:** On ARC-Challenge, about 90% of benchmark samples have a per-sample score-compute correlation above 0.75 when the score is the log-likelihood of the correct choice, but only 40% stay above 0.75 once that probability is renormalized over the available choices, and Accuracy lowers the correlations further (§4, Fig. 3); the authors trace the loss of predictability to probability mass moving among the incorrect choices (§5).
- **Guideline:** When forecasting a multiple-choice benchmark from smaller models, do not extrapolate Accuracy or Brier Score trends directly; model the probability mass on the incorrect choices as well as on the correct choice, because the metrics that depend on incorrect choices lost score-compute correlation in at least 82.4% (Pearson), 85.6% (Spearman), and 90.4% (Kendall) of tested tuples (§4). The paper does not test whether per-choice forecasts work (§6, App. B).
- **Authors:** Rylan Schaeffer, Hailey Schoelkopf, Brando Miranda, Gabriel Mukobi, Varun Madan, Adam Ibrahim, et al. (9 authors; Stanford Computer Science, EleutherAI, MILA, University of Cambridge)
- **Year:** 2024 (arXiv v1 2024-06; v2 2025-02, marked "Preliminary work. Under review by the International Conference on Machine Learning (ICML)")
- **URL:** https://arxiv.org/abs/2406.04391
- **Source type:** paper
- **Relevant topics:** downstream scaling prediction, multiple-choice evaluation, evaluation metrics, emergent abilities, scaling laws

## Abstract
Pre-training loss scales predictably, but downstream capabilities do not. The paper studies five model families and twelve multiple-choice benchmarks. It shows that downstream metrics are computed from negative log-likelihoods through a sequence of transformations, and that each transformation weakens the statistical relationship between the score and scale. The mechanism is that the metrics compare the correct choice against a small number of specific incorrect choices. Predicting them therefore requires predicting how probability mass on the incorrect choices changes with scale, not only mass on the correct choice. The authors measure how mass on correct and incorrect choices co-varies with compute and suggest that scaling laws for incorrect choices may be achievable.

## Key Contributions
- A decomposition of multiple-choice metrics into four steps: log-likelihood over the vocabulary → probability over the vocabulary → probability over the available choices → Accuracy or Brier Score (§4, Fig. 1).
- A per-sample measurement of score-compute correlation for every (model family, benchmark, metric, correlation metric) combination, summarized by four statistics (§4, Fig. 4).
- Identification of the mechanism: metrics after renormalization depend on mass placed on incorrect choices (§5, Fig. 5).
- Evidence that Brier Score, a continuous metric, does not recover the lost predictability (§2 finding 3, Fig. 3C).
- Preliminary evidence that mass on correct and incorrect choices co-varies with compute (§6, Fig. 6).

## Key Figures/Tables to Study
- **Figure 1** — the transformation sequence from negative log-likelihoods to Accuracy.
- **Figure 3** — complementary CDFs of per-sample correlations for each metric on ARC-Challenge (Spearman).
- **Figure 4** — the four correlation-distribution statistics across benchmarks and model families.
- **Figure 5** — pairs of metrics before and after each transformation (MathQA, MMLU Conceptual Physics).
- **Figure 6** — mass on correct choices versus mass on incorrect choices as compute grows.
- **Appendix G (Figs. 7–9) and H** — per-benchmark results for Pearson, Spearman, and Kendall correlations.

## Technical Details
- **Model families (§3, App. E):**
  - Pythia: 8 models from 70M to 12B trained on the Pile for 300B tokens (§3). App. E uses fully trained checkpoints of 9 sizes, including a later 14M model (parameter scaling), and 8 checkpoints of Pythia-12B at 2M, 64M, 2B, 6B, 20B, 60B, 200B, and 300B tokens (data scaling).
  - Cerebras-GPT: 7 models from 111M to 13B trained on the Pile with about 20 tokens per parameter; one checkpoint per model (§3, App. E).
  - OLMo: §3 describes 7 checkpoints of the 1B model (84B to 3T tokens) and 7 of the 7B model (4B to 2.4T tokens). App. E lists OLMo-7B checkpoints at 4B, 44B, 133B, 442B, 885B, 1.5T, and 2.4T tokens. OLMo 1B checkpoints below 84B tokens were lost by their creators (§3, footnote 1).
  - INCITE: 7B only, because the 3B model has one checkpoint; 6 checkpoints at 240B to 1T tokens of RedPajama-v1 (§3, App. E).
  - LLM360 Amber 7B: 13 checkpoints at 0B to 1.26T tokens (App. E).
- **Benchmarks (§3):** ARC Easy and Challenge, HellaSwag, MathQA, MCTACO, MMLU (each of the 57 subjects analyzed separately), OpenbookQA, PIQA, RACE, SciQ, SIQA, WinoGrande, and XWinoGrad En, with default LM Evaluation Harness settings.
- **Compute (§3):** C ≈ 6ND, where N is the parameter count excluding embeddings and D is the number of training tokens seen. The approximation ignores attention FLOPs, which the authors call negligible when d_model >> n_ctx/12 (§3, footnote 2).
- **Transformation sequence (§4):**
  1. L^Vocab_θ(correct) is the negative log-likelihood of the correct choice for one benchmark sample, computed over the model's vocabulary. It is per sample, not an average over a corpus.
  2. p^Vocab_θ(correct) = exp(−L^Vocab_θ(correct)).
  3. p^Choices_θ(correct) = p^Vocab_θ(correct) / Σ_i p^Vocab_θ(choice_i), after masking continuations that are not available choices.
  4. Accuracy_θ = 1[correct choice = argmax_i p^Choices_θ(choice_i)]; Brier Score_θ = Σ_i (1[choice_i = correct] − p^Choices_θ(choice_i))².
  θ denotes the model parameters, i indexes the available choices, and 1[·] is the indicator function.
- **Correlation measurement (§4, Eq. 1):** for each sample s, C_s is the correlation between compute and score across the models of one family. The distribution over samples is summarized by the empirical complementary CDF Ŝ(c) = (1/S) Σ_s 1{C_s > c}, where S is the number of samples and c is a threshold. The four statistics are the mean, the median, the area under Ŝ, and the negative of the smaller Wasserstein distance to an all-correlations-equal-1 or all-equal-−1 distribution.
- **ARC-Challenge results (§4, Fig. 3, Spearman):** about 90% of samples exceed 0.75 for log-likelihoods, for every model family. The exponential step does not change rank correlations. Renormalizing over choices leaves 40% of samples above 0.75. Brier Score has little to no further effect; Accuracy decreases correlations further. The authors report that other benchmarks show similar patterns (App. H).
- **Ordering test (§4):** Corr(log p^Vocab) > Corr(p^Choices) ≥ Corr(Brier Score) > Corr(Accuracy), checked with strict inequalities, held in at least 82.4% of tuples for Pearson, 85.6% for Spearman, and 90.4% for Kendall.
- **Mechanism example (§5):** on a 4-way question where the correct choice has probability 0.4, spreading the remaining mass uniformly over the incorrect choices gives Accuracy = 1, while concentrating it on one incorrect choice gives Accuracy = 0. With uniform spread each incorrect choice has 0.2 (derived: 0.6 / 3); with concentration one incorrect choice has 0.6.
- **Information loss (Fig. 5 caption):** p^Choices(correct) > 0.5 always gives Accuracy = 1; below 0.5, p^Choices carries little information about Accuracy. Brier Score is less variable than Accuracy given p^Choices, but still variable.
- **Averaging (§6):** mass shifting from one incorrect option to another changes Accuracy only if either option competes with the correct answer, so the effects do not cancel when averaged over a benchmark. The authors state that predictions for Accuracy should be made per sample.
- **Co-variation (§6, Fig. 6):** mass on correct and incorrect choices positively co-varies and typically increases with compute, but for a given mass on the correct choice, mass on incorrect choices can vary by many orders of magnitude. Whether per-sample, per-choice trends can be fit and extrapolated is left open.

## Findings relevant to generality
- **Measuring breadth with benchmarks:** per-sample Accuracy is less correlated with pre-training compute than per-sample log-likelihood of the correct choice, across the benchmarks and model families tested (§4, Fig. 4). A small Accuracy difference between two checkpoints is therefore weaker evidence of a capability change than a log-likelihood difference on the same samples (Interpretation).
- **Scope limit:** the study covers only log-likelihood-based multiple-choice formats; generative evaluations are not tested. The authors cite Lyu et al. (2024), who find that multiple-choice scores often diverge from generative evaluations (App. B, Direction 1).
- **Forecasting limit:** the analyses use whole model families over several orders of magnitude of FLOPs and do not use backtesting (App. B, Direction 2).
- **Contamination:** INCITE is a slight outlier, which the authors speculate is due to benchmark contamination of its pre-training data (§3).

## Connections
- [[kaplan-scaling-laws]] — source of the C ≈ 6ND compute approximation used here (§3).
- [[emergent-abilities]] — cited as prior observation of unpredictable changes in benchmark performance with scale (§1).
- [[emergent-abilities-mirage]] — earlier work by the same first author on metric choice; this paper finds Brier Score insufficient to restore predictability (§2).
- [[emergence-loss-perspective]] — Du et al. (2024), cited for the claim that downstream capabilities become predictable only after pre-training loss falls below a threshold (§1).
- [[overtraining-downstream-scaling]] — Gadre et al. (2024), cited for clearer trends when results are aggregated across many benchmarks (§1, App. A).
- [[observational-scaling-laws]] — Ruan et al. (2024), cited as concurrent work mapping model families to a shared capability space (App. A).
- [[pythia]] — the model suite that provides both parameter-scaling and data-scaling checkpoints (§3, App. E).
- [[same-loss-better-downstream]] — a separate reason downstream scores are not a function of pre-training loss alone; not cited by this paper.
- [[resolving-scaling-discrepancies]] — compute-optimal scaling fit on pre-training loss; its §5.2 names prediction of downstream capabilities as an open problem; not cited by this paper.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2406.04391 (arXiv v2, 2025-02-05; full PDF text including App. A–H).
- Audit claims not found in the source: none.
- Not reported by the source: a forecasting method that recovers Accuracy from small models; results on generative benchmarks; results for instruction-tuned models.
