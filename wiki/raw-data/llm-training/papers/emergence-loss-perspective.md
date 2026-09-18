<!-- scope: pre-training loss (not parameter count or compute) as the predictor of downstream performance, and a loss-threshold definition of emergent abilities
     deps: [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]]
     see-also: [[emergent-abilities]], [[emergent-abilities-mirage]], [[same-loss-better-downstream]], [[pythia]]
-->

# Understanding Emergent Abilities of Language Models from the Loss Perspective
- **Core Insight:** Across more than 30 models (300M-32B) trained on one corpus, tokenizer, and architecture, downstream performance on 12 tasks follows pre-training loss regardless of model or data size, and MMLU, C-Eval, GSM8K, and GSM8K-Chinese stay at random-guess level until the training loss falls to about 2.2 (§2.3, §3.1).
- **Guideline:** When comparing checkpoints that share a corpus and tokenizer, use pre-training loss rather than parameter count or training compute to predict downstream performance, because points from different model sizes fall on one performance-vs-loss curve (Fig. 1, Table 2) while performance-vs-compute points do not (App. E). Do not extrapolate a task that is still at random level from higher-loss checkpoints (§4).
- **Authors:** Zhengxiao Du, Aohan Zeng, Yuxiao Dong, Jie Tang (Zhipu AI; Tsinghua University)
- **Year:** 2024 (arXiv v1 2024-03; NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2403.15796
- **Source type:** paper
- **Relevant topics:** emergent abilities, scaling laws, downstream prediction, evaluation metrics, pre-training loss

## Abstract
Earlier work questioned whether emergent abilities are exclusive to large models, because smaller models trained on more data can score well on such tasks and because discontinuous metrics may create apparent jumps. The authors study emergence as a function of pre-training loss instead of model size or compute. With a fixed data corpus, tokenization, and architecture, Transformer models with the same pre-training loss but different model and data sizes have the same performance on various downstream tasks. On some tasks, performance stays at random-guess level until the loss falls below a threshold, and this holds for both discontinuous and continuous metrics. The authors redefine an emergent ability as one present in models with lower pre-training loss and absent in models with higher loss, which implies it cannot be predicted by extrapolating from higher-loss models.

## Key Contributions
- Pre-trains 1.5B, 6B, and 32B models to 3T, 3T, and 2.5T tokens, plus 28 smaller models (300M-6B, 33B-500B tokens), and evaluates 12 English and Chinese datasets (§2.3, §2.4, Table 1).
- Shows the performance-vs-loss relation is shared across sizes and token counts, and checks it on LLaMA 7B-65B figures and Pythia checkpoints (§2.5, App. F).
- Separates tasks that improve from the start from tasks with a loss threshold near 2.2 (§3.1).
- Tests the metric argument of [[emergent-abilities-mirage]] with CorrectChoiceProb and Brier Score; the threshold remains (§3.2, App. C).
- Gives a loss-based definition and links it to size-based emergence through the loss power law (§4, Eq. 3-5).

## Key Figures/Tables to Study
- Fig. 1 and Table 2: performance vs training loss for intermediate checkpoints of 1.5B/6B/32B, with Spearman and Pearson coefficients.
- Fig. 2 / Fig. 5: final and intermediate checkpoints of the 28 smaller models.
- Fig. 4: MMLU and C-Eval under accuracy, CorrectChoiceProb, and Brier Score.
- Fig. 6 (App. E): the same checkpoints plotted against training compute do not share one curve.
- Fig. 8 (App. G): three BIG-bench tasks (word unscramble, modular arithmetic, IPA transliterate) with a tipping point.

## Technical Details
- Checkpoints of the three large runs are saved about every 43B tokens (§2.3).
- Spearman correlation of performance with loss: TriviaQA -0.996, HellaSwag -0.996, RACE -0.977, WinoGrande -0.978, NLPCC-KBQA -0.984, ClozeT -0.986, C3 -0.988, CLUEWSC -0.947, MMLU -0.804, C-Eval -0.831, GSM8K -0.975, GSM8K-Chinese -0.948 (Table 2).
- Pearson correlation, same order: -0.994, -0.994, -0.963, -0.988, -0.982, -0.985, -0.993, -0.972, -0.903, -0.884, -0.874, -0.829 (Table 2).
- Above a loss of 2.2, MMLU accuracy remains around 25%, the four-option random level; the thresholds of MMLU, C-Eval, GSM8K, and GSM8K-Chinese are all around 2.2 (§3.1).
- The 28 smaller models are close to random on those four tasks with fewer than 500B tokens (§2.4).
- The overall loss, computed on mixed English and Chinese tokens, predicts both English and Chinese tasks (§2.3).
- LLaMA exception: above loss 1.8, LLaMA-65B is below smaller models at the same loss; these outliers are in the first 10% of training tokens, and the authors name exponential smoothing in the original plots as one possible explanation (§2.5).
- Pythia and LLaMA cannot test emergence: the largest Pythia model does not exceed random on MMLU and GSM8K, and LLaMA has no released intermediate checkpoints (App. F).
- Brier Score (Eq. 2) = (1/N) Σ_i Σ_j (y_ij − ŷ_ij)²; N = number of samples, j = answer class, y_ij = ground-truth probability, ŷ_ij = predicted probability. A uniform predictor on four balanced options scores 0.75; the constant prediction (1,0,0,0) scores 1.5 (§3.2, App. C).
- Random-guess Brier baselines for the four BIG-Bench tasks in [[emergent-abilities-mirage]] Fig. 6 (2, 2, 4, 5 options) are 0.25, 0.25, 0.1875, 0.16, and the authors state only the largest model passes them (App. C).
- Eq. 3: normalized performance = f(L) if L < η, else 0; L = pre-training loss, η = threshold, f = monotonically decreasing, random guess = 0 (§4).
- Eq. 4-5: L(N) = L∞ + (N0/N)^αN, so performance is non-zero only when N ≥ N0·(η − L∞)^(−1/αN); N = model size at fixed tokens, L∞ = irreducible loss, N0 and αN = fitted constants (§4).
- Recipe values are in [[emergence-loss-perspective-recipe]].

## Findings relevant to generality
- Breadth measurement: on 8 of the 12 tasks (QA, commonsense NLI, reading comprehension, coreference, in both languages) performance improves with lower loss from the start of training, with |Spearman| between 0.947 and 0.996; on the 4 exam and math word-problem tasks the relation is weaker because of the threshold (Table 2, §3.1). The authors hypothesize that task difficulty explains the split (Interpretation, §3.1).
- Scope limits stated by the authors: loss values are not comparable across tokenizers or corpora; architectures (routed or non-Transformer) and optimizers other than AdamW were not tested; new tipping points at larger scale are not guaranteed; instruction tuning can raise zero-shot MMLU and GSM8K (§7).
- The study measures prompted base-model performance, not fine-tuned performance; the paper cites Tay et al. finding that models with equal pre-training loss can differ after fine-tuning (§5).

## Connections
- [[emergent-abilities]] — the size-based definition this paper restates in terms of loss.
- [[emergent-abilities-mirage]] — the metric-choice explanation tested in §3.2 and App. C.
- [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]] — loss scaling relations used in §4 and the motivation in §1.
- [[same-loss-better-downstream]] — examines cases where equal pre-training loss does not give equal downstream results.
- [[pythia]] — suite whose checkpoints are used in App. F.
- [[training-on-the-test-task]], [[predicting-downstream-elusive]], [[scaling-laws-unreliable-downstream]] — other work on predicting downstream performance from scale.
- [[large-batch-training-noise-scale]] — another result where model size acts mainly through the loss reached.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2403.15796 (arXiv v3, 15 Jan 2025, PDF read in full including App. A-I).
- Audit claims not found in the source: none.
- Not reported by the source: warmup, final learning rate, weight decay, gradient clipping; the unit of "Batch Size" in Tables 4-5 (sequences or tokens).
