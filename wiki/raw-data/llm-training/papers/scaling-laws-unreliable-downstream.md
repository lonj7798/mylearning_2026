<!-- scope: Lourie, Hu, Cho (NYU, Jul 2025; EMNLP Findings 2025) — re-analysis of downstream scaling data from Gadre et al. 2025 and DataDecide showing predictable loss-to-task scaling in 18 of 46 tasks, validation-corpus reversals, and setup-dependent scaling behavior
     deps: [[kaplan-scaling-laws]], [[overtraining-downstream-scaling]]
     see-also: [[datadecide]], [[paloma]], [[olmes]], [[emergent-abilities]], [[emergent-abilities-mirage]], [[observational-scaling-laws]], [[task-scaling-model-ladders]]
-->

# Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check
- **Core Insight:** Re-classifying the 46 downstream tasks of Gadre et al. (2025), only 18 tasks (39%) improve smoothly and predictably with validation loss; the other 28 show inverse, nonmonotonic, noisy, trendless, or breakthrough scaling (abstract, §4, Fig. 1).
- **Guideline:** When a small-scale experiment is used to choose a pretraining corpus or data mix for downstream tasks, first check with plots and regression diagnostics that the task scales predictably in that exact setup (validation corpus, harness, prompt format, number of answer choices), because changing any one of these reversed or removed the scaling trend in the re-analyzed data (§3, §5, §6).
- **Authors:** Nicholas Lourie, Michael Y. Hu, Kyunghyun Cho (New York University; Cho also Prescient Design and CIFAR LMB; first two authors equal contribution)
- **Year:** 2025 (arXiv v1 2025-07; v2 2025-10; EMNLP Findings 2025 per arXiv comment)
- **URL:** https://arxiv.org/abs/2507.00885 ; code https://github.com/nicholaslourie/scale-fails
- **Source type:** paper
- **Relevant topics:** downstream scaling laws, emergence, inverse scaling, validation-loss surrogates, data-mix selection, evaluation-harness sensitivity

## Abstract
Downstream scaling laws try to predict task performance of large models from smaller models. Some work reports linear trends after transforming the metric; other work points to emergence and inverse scaling. The authors run a meta-analysis of existing downstream scaling data and find predictable scaling in a minority of cases, 39% of the time. Changes to the experimental setting that appear benign can change the scaling behavior completely. The authors conclude that the conditions under which scaling laws succeed need to be understood, and that models of the loss-to-task relationship must account for cases that deviate from linear trends.

## Key Contributions
- A six-way qualitative taxonomy of loss-to-task scaling: predictable, inverse, nonmonotonic, noisy, trendless, breakthrough (Fig. 2, App. C).
- Re-classification of the 46 tasks from Gadre et al. (2025): 18 predictable, 28 degenerate (Fig. 1, §4, Figs. 5–10).
- Counterexamples in which the validation corpus exaggerates or reverses which pretraining corpus looks better (§3, Fig. 3, Fig. 12).
- A cross-study comparison on 10 overlapping tasks, using 200 DataDecide checkpoints re-evaluated on C4 validation data, showing qualitative changes in scaling behavior between two controlled setups (§5, Fig. 4, Fig. 11).

## Key Figures/Tables to Study
- Figure 1 (18 of 46 predictable); Figure 2 (taxonomy examples); Figure 3 (C4 vs RedPajama under C4 and Paloma 100 PLs validation, HellaSwag and CoQA); Figure 4 (MMLU, BoolQ, CommonsenseQA across setups); Figures 5–10 (tasks by category, with R² of each fit); Figure 11 (all 10 overlapping tasks).

## Technical Details
**Functional form (§2, App. C)**
- Best case: f(y) = a·g(x) + b, where x is validation cross-entropy loss, y is the downstream metric, f and g are monotonic transformations that depend on the metric, and a, b are fitted constants (§2).
- Gadre et al. form, used for the fits shown in Figures 5–10: y = ε − k·exp{−γx}, with y the error rate, x the validation loss, and ε, k, γ fitted constants; applied to accuracy through accuracy = 1 − error (§2, App. C).
- Fitting: minimize mean squared error with SciPy `differential_evolution`; bounds 0–1 for ε, 0–20 for ln(k) (k searched on a log scale), 0–20 for γ; popsize multiplier 60; R² reported per task (App. C).
- Emergence, inverse, and U-shaped scaling create structural breaks, points where the function that describes one part of the curve does not describe another; without a single global form, extrapolation from small to large models is not possible (§2).

**Category definitions (App. C)**
- Definitions as printed: predictable, "Performance increases, without too much variation around the trend"; inverse, "Performance decreases as loss improves"; nonmonotonic, "Performance switches between increasing and decreasing"; noisy, "Performance increases but varies greatly around the trend"; trendless, "Performance is flat or there is too much noise to discern a trend"; breakthrough, "Performance starts flat, abruptly increases, and then plateaus".
- The definitions are informal by design, so the conclusions do not depend on one scaling-law form (App. C). The per-category counts of the 28 non-predictable tasks are shown only in Figures 6–10; the text gives no breakdown.

**Validation corpus effects (§3, Fig. 3)**
- Setup: Gadre et al. models pretrained on C4 or RedPajama; validation loss on C4 or on Paloma's 100 Programming Languages (100 PLs).
- HellaSwag: with C4 validation both corpora follow the same scaling law; with 100 PLs validation, C4 pretraining appears better even at worse validation loss.
- CoQA: with C4 validation RedPajama reaches better performance sooner; with 100 PLs validation the relationship reverses.
- Stated consequence: data mixes cannot be compared by their validation losses, or by downstream scaling laws built on them, without considering pretraining data, validation data, and task together (§3).

**Setup effects (§5, App. A–B)**
- DataDecide checkpoints evaluated: 20M, 60M, 150M, 300M, 1B parameters; corpora Dolma, C4, DCLM-Baseline, RefinedWeb, FineWeb-Pro; 5,000 to 40,000 steps in intervals of 5,000; 200 released checkpoints in total (§5).
- C4 validation perplexity computed with ai2-olmo at batch size 64 on a mix of A100 and H100 GPUs; inference over the C4 validation set took approximately 10 minutes (App. B).
- MMLU: positive trend in both setups, with different noise and shape. CommonsenseQA: nonmonotonic under Gadre et al., a clean scaling law under Magnusson et al. BoolQ: trendless under Magnusson et al. because those models span too small a validation-loss range (§5, Fig. 4).
- Harness differences: Gadre et al. use LLM Foundry (46 tasks, few-shot, shot count varies by task, own task versions with changed answer-choice counts for CommonsenseQA and SIQA); Magnusson et al. use OLMES (10 tasks, 5 curated examples each); task formulation (multiple-choice vs cloze) and prompts also differ (App. A).
- SIQA is excluded from the analyses because LLM Foundry v0.4.0, used by Gadre et al. for at least some experiments, had incorrect gold labels (fixed in v0.5.0); Figure 11 shows it for completeness (App. A).

## Findings relevant to generality
- Measurement of general capability: for the benchmark suite of one published study, a lower pretraining validation loss predicted downstream gains smoothly for 18 of 46 tasks (Result (single study); Fig. 1). The authors conclude that predictable scaling must be established for the given task before relying on it (§7).
- Choice of the validation corpus changes which pretraining corpus looks better on the same task (§3, Fig. 3).
- Differences in evaluation setup (architecture details, prompts, number of shots, task format, answer-choice count) coincide with CommonsenseQA scaling cleanly in one study and nonmonotonically in the other (§5, Fig. 4, Fig. 11 caption). A benchmark with wrong gold labels (SIQA in LLM Foundry v0.4.0) was excluded (App. A).
- The background cites Magnusson et al. (2025) as finding that comparing models at small scale without extrapolation chose the best data mix as well as or better than scaling laws (§2).
- Limits stated by the authors: only data and checkpoints from two existing studies, which may share unknown biases; the results describe current practice, and better measurement techniques may reduce the problem (Limitations).

## Connections
- [[overtraining-downstream-scaling]] — Gadre et al. (2025), the source of the 46-task data and the exponential loss-to-error form.
- [[datadecide]] — Magnusson et al. (2025), the source of the 200 checkpoints and the OLMES-based setup in §5.
- [[paloma]] — provides the 100 Programming Languages validation set used in §3.
- [[olmes]] — evaluation standard used by DataDecide; one of the two harnesses compared.
- [[emergent-abilities]], [[emergent-abilities-mirage]], [[emergence-loss-perspective]] — emergence results that the paper treats as structural breaks.
- [[observational-scaling-laws]], [[task-scaling-model-ladders]] — surrogate-based downstream prediction methods the paper discusses (§2).
- [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]] — pretraining-loss scaling laws that are reliable for loss but not directly for tasks.
- [[scaling-laws-for-transfer]] — an earlier loss-denominated transfer law that avoids downstream accuracy metrics.
- [[c4]], [[dolma]] — pretraining corpora among those compared.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2507.00885 (arXiv v2, 2025-10-09; v1 2025-07-01), full PDF text including Appendices A–E; per-category figure captions checked on the arXiv HTML v2.
- Audit claims not found in the source: none. Scope note: the 39% figure is the share of the 46 tasks in Gadre et al. (2025), one study's setup, not a pooled rate across all downstream scaling studies (§4, Fig. 1).
- Not reported by the source: task names and counts per non-predictable category in text form, numeric R² values in the text.
