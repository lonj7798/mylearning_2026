---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "Training on the Test Task Confounds Evaluation and Emergence (arXiv:2407.07890v3, 2025-04-21; v1 2024-07-10; ICLR 2025)"
source_url: https://arxiv.org/abs/2407.07890
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug training-on-the-test-task). Values read from the v3 PDF on 2026-09-15. Code and fine-tuned models: github.com/socialfoundations/training-on-the-test-task."
---

# Excerpt: training on the test task, and how to adjust for it

**Authors:** Ricardo Dominguez-Olmedo, Florian E. Dorner, Moritz Hardt (Max Planck Institute for Intelligent Systems, Tübingen AI Center, ETH Zurich).

## Definition (§1)

*Training on the test task* is the use of knowledge about the evaluation task at training time, without using the test data. Named instances: including instruction or question-answering data in pre-training (Qwen, OLMo 1.7, MAP-Neo), reformulating pre-training data to resemble downstream tasks (StableLM 2), and selecting pre-training mixtures by ablations on downstream benchmarks (Gemma, Llama 3). The authors treat it as legitimate and unavoidable, and as a confound that must be adjusted for.

## Method (§2, §2.1)

- 56 base models, 70M to 70B parameters, evaluated on MMLU and GSM8K, with accuracy regressed against pre-training compute.
- Adjustment: fine-tune **every** model on the same task-relevant data before evaluation.
  - MMLU: the auxiliary training set from the HF MMLU repository (training splits of other multiple-choice QA benchmarks), about 100,000 examples and 30M tokens. It is not an i.i.d. split of MMLU.
  - GSM8K: MetaMathQA plus Orca-Math, about 600,000 examples and 200M tokens.
  - Three epochs, standard hyperparameters (App. A.2); fine-tuning compute is small against pre-training compute. A hyperparameter robustness check is in App. A.3.
- Regression model (Eq. 1): `A = α max(0, log C − c_e) + θ N + r + ε`, where `A` is benchmark accuracy, `C` pre-training compute, `c_e` the compute at which the task emerges, `r` random-chance accuracy, `N` an indicator for release after November 2023, and `θ` the average difference attributed to being a newer model. R² > 0.9 for all fits; standard errors clustered by model family.
- The November 2023 cutoff is chosen because technical reports from that point begin describing such practices; App. C.1 shows the finding is robust to moving the cutoff by a few months.

## Results (§2.2, §4.2, §4.3, Figs. 1–3, 7–8, 18)

- Before adjustment, models released after November 2023 outperform older ones at equal pre-training compute by over 7 accuracy points on MMLU and over 19 on GSM8K, statistically significant.
- After adjustment, the estimated `θ` is small and not statistically significant on either benchmark: the scaling trends of older and newer models coincide.
- Older models gain far more from the adjustment fine-tuning than newer models (Fig. 2), which the authors read as newer models already having been exposed to task-relevant data.
- Rankings move under adjustment: GSM8K average shift 7.7 ranks, maximum 21; MMLU average shift 4.8 ranks, maximum 16 (§4.2, Fig. 7, Fig. 18).
- Emergent-capability claims weaken: emergence fades gradually as models train more on the test task, recovering log-linear scaling (§5).
- Appendix D extends the analysis to MMLU-Pro, GPQA, BBH, MuSR and MATH Level 5 (the OpenLLM Leaderboard v2 set), and Appendix F to 36 instruction and chat models, where the authors report the findings generalize.

## Stated relationship to fresh benchmarks (App. D)

The authors note that measures aimed at contamination, including dynamic benchmarks, do not remove the confounding effect of training on the test task, because no test data need be leaked for the effect to occur.

## Used in

ch-47a §1 (taxonomy), §6 (why a fresh test set does not fix everything), §7, Generalization lens.
