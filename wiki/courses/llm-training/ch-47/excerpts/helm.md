---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: primary source arXiv:2211.09110v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2211.09110
created_at: "2026-09-15"
---

# Excerpt: Holistic Evaluation of Language Models (HELM)

**Paper:** Liang, Bommasani, Lee, Tsipras, Soylu, Yasunaga, et al. (Stanford CRFM). arXiv v1 2022-11, read at v2 (2023-10-01); Transactions on Machine Learning Research 08/2023. Source type: paper.

## Coverage (Abstract, §1)
- Taxonomy of scenarios (use cases) and metrics (desiderata), with missing or underrepresented areas stated, e.g., question answering for neglected English dialects and trustworthiness metrics.
- 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) on 16 core scenarios, measured for 98 of 112 (scenario, metric) pairs (87.5%); 7 targeted evaluations on 26 targeted scenarios; 30 models.
- Before HELM, the models had been evaluated on 17.9% of the core scenarios on average; HELM raises this to 96.0%.

## Calibration metric (§4.4, Fig. 17)
- Expected calibration error (ECE) with 10 bins of equal size: the weighted mean over bins of |accuracy − mean predicted probability|. Fig. 17 example with 2 bins: bin 1 accuracy 0.5, mean probability 0.15, error 0.35; bin 2 accuracy 0.75, mean probability 0.85, error 0.1; ECE = (4/8)·0.35 + (4/8)·0.1 = 0.225.
- Selective classification: accuracy on the C fraction of examples with highest model probability.

## Findings used in ch-00 (§1.2 "Empirical findings")
- Finding 3: the relation between accuracy and calibration depends on scenario; on HellaSwag improving accuracy worsens calibration, on OpenBookQA it improves calibration.
- Finding 4: TNLG v2 (530B) on NarrativeQA falls from 72.6% standard accuracy to 38.9% under robustness perturbations.
- Finding 22: all models are sensitive to prompt formatting, choice of in-context examples, and number of in-context examples, across all scenarios and metrics.
- Finding 23: OPT (175B) on HellaSwag scores 79.1% when each answer choice is scored in a separate 0-shot prompt and 30.2% when choices are presented jointly in a 5-shot multiple-choice prompt.
- Finding 24: bits-per-byte on The Pile is a poor predictor of downstream accuracy across model families (some models were trained on The Pile and others not).

## Verification
- Read on 2026-09-15 against arXiv:2211.09110v2 PDF text (Abstract, §1-§1.3, §4.4).
