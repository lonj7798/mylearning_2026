# Data Value in the Age of Scaling

<!-- scope: Data Value in the Age of Scaling; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** Head and tail learning can respond differently to scaling mixed real and synthetic data.
- **Guideline:** Measure tail slices separately when increasing synthetic volume.
- **Authors:** Haohui Wang et al.
- **Year:** 2025
- **URL:** [Primary source](https://arxiv.org/abs/2511.13640)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

The paper analyzes phases of head/tail learning and proposes low-cost data valuation in mixed-data scaling.

## Key Contributions

- Result (single study): Head and tail learning can respond differently to scaling mixed real and synthetic data.
- Interpretation: Measure tail slices separately when increasing synthetic volume.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

Head/tail distinctions and the valuation proposal are abstract-level claims in this extract.

## Limits

No verified dialogue-specific recipe or guarantee that more filtered synthetic data preserves rare behavior.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary abstract verified on 2026-10-06; first release 2025-11-17.

## Full-text follow-up (2026-10-07)

§5.2 three-phase empirical demonstration uses ResNet-18 on CIFAR-100 and corrupted CIFAR-100-C as a controlled synthetic surrogate; Zipf class weighting i^-2, head cutoff 70 of 100 classes,real fraction0.0625. This is not natural LLM dialogue validation. NLP subset-utility valuation experiments are separate evidence.

Checked primary arXiv HTML full text at the cited sections; these facts replace the earlier abstract-only scope for this extract.
