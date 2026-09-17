---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:2306.10062v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2306.10062
created_at: "2026-09-15"
---

# Excerpt: Revealing the structure of language model capabilities

**Paper:** Burnell, Hao, Conway, Hernández-Orallo (University of Cambridge; Alan Turing Institute; New Mexico State University; Universitat Politècnica de València). arXiv v1 2023-06-14 (preprint). Source type: paper.

## Data (§2)
- HELM results for 29 LLMs (Table 1: sizes 0.41B-530B, instruction tuning and RLHF flags) on 27 tasks after removing tasks with missing data or without a clear cognitive basis (§2.1-2.2).

## Results (§3)
- Mean correlation between tasks across models r = 0.56 (median 0.6) (§3.1).
- The Hull method suggests 3 factors; maximum-likelihood exploratory factor analysis with oblimin rotation: variance explained 0.33, 0.31, 0.17 (cumulative 0.82), interpreted as comprehension, language modeling, reasoning (§3.2, Table 2, Fig. 1).
- Fit statistics are poor for the small sample: CFI = 0.70, TLI = 0.61, RMSEA = 0.26 (§3.2). A Bayesian factor analysis gives a matching 3-factor solution; LSAT is not assigned to a factor (§3.3).
- Pearson correlations with 95% intervals (Table 3): log model size 0.70 [0.46, 0.85] comprehension, 0.49 [0.16, 0.72] language modeling, 0.51 [0.19, 0.73] reasoning; instruction tuning 0.23 [−0.13, 0.54], −0.50 [−0.72, −0.17], 0.44 [0.11, 0.69]; training tokens near zero for all three.
- Factor correlations: comprehension-language modeling 0.43, comprehension-reasoning 0.51, language modeling-reasoning 0.22 (Table 3).

## Authors' conclusions and limits (Abstract, §4)
- Capabilities are not monolithic; changes that improve one ability might impair others (Abstract; Interpretation from cross-sectional correlations).
- Results depend on the tasks included; abilities not tested by HELM tasks (for example creative writing) are not represented (§4).

## Verification
- Read on 2026-09-15 against arXiv:2306.10062v1 PDF text (§1-4, Tables 1-3, Fig. 1-2).
