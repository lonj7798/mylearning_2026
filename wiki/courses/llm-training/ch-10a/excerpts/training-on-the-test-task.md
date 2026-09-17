---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/training-on-the-test-task.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2407.07890
created_at: "2026-09-15"
---

# Excerpt: Training on the Test Task Confounds Evaluation and Emergence

**Authors:** Ricardo Dominguez-Olmedo, Florian E. Dorner, Moritz Hardt (Max Planck Institute for Intelligent Systems, Tübingen AI Center, ETH Zurich)
**Version read:** arXiv:2407.07890v3 (21 Apr 2025); v1 July 2024.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v3 PDF text.

## Definition (Abstract, §1)
"Training on the test task" groups "practices that utilize knowledge about evaluation tasks at training time". It is distinguished from training on test data, leakage, and contamination and "is not a malpractice". Examples named: instruction-tuning data or QA templates in pretraining (Qwen, OLMo 1.7, MAP Neo, StableLM 2), and pretraining mixtures "selected through ablations on downstream benchmark evaluations" (Gemma, Llama 3) (§1, §2).

## Setup
- 56 base models, 70M to 70B parameters; compute C ≈ 6ND; split by release before or after November 2023 (§2).
- Eq. 1: A = α max(0, log C − c_e) + θ N + r + ε. A accuracy, C pretraining compute, c_e emergence point, N = 1 if trained after November 2023, r chance accuracy, θ average newer-vs-older difference at equal compute; clustered standard errors by model family (§2.2).
- Adjustment (§2.1): fine-tune every model on the same task-relevant data before evaluation. MCQA: MMLU auxiliary training set (≈100,000 examples, 30M tokens; training sets of other MC benchmarks, not an i.i.d. MMLU split). Math: MetaMathQA + Orca-Math (≈600,000 examples, 200M tokens). Three epochs. App. A.2: LR 2e-5 (< 10B params) or 2e-6 (> 10B), cosine with 50 warmup steps to 10% of peak, AdamW (0.9, 0.95, 1e-8), batch 64.

## Results
- Fig. 1: before adjustment θ = 0.073 (MMLU) and 0.191 (GSM8K), significant; after adjustment θ = 0.005 on both, not significant. Text: newer models outperform older ones "on average by over 7 accuracy points in MMLU and 19 accuracy points in GSM8K" (§2.2).
- §3.1, Fig. 3: fine-tuning older models (one epoch) recreates the gap (θ = 0.086 MMLU, 0.223 GSM8K); further fine-tuning of both groups removes it (0.006).
- §3.2, Fig. 4: ARC-Challenge and HellaSwag in cloze form: θ = 0.001 and 0.012 (not significant). Reformulated as MMLU-style multiple choice: θ = 0.120 and 0.114. After adjustment: 0.014 and 0.009. "the adjustment data need not closely resemble the test set, but rather the test task."
- Fig. 5: MMLU in cloze form: θ = 0.008; the authors conclude standard MMLU "conflates knowledge-testing with testing a models' ability to answer multiple choice questions".
- §4: family rankings on MMLU and GSM8K change after adjustment (Fig. 6-7).
- Limitations (§1.1): adjustment needs "generally significant fine-tuning"; task data may be unavailable.

## How ch-10a uses it
§6 (benchmark-targeted selection as training on the test task; detection by cloze-vs-MC and equal fine-tuning), Generalization lens (c).
