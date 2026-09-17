---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2305.13169 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2305.13169
created_at: "2026-09-15"
revised: 2026-09 (generality revision)
---

# Excerpt: A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity

- **Authors:** Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, et al.
- **Year:** 2023 (arXiv v1 2023-05; read: arXiv v2, 13 Nov 2023)
- **Source type:** paper
- **Checked on:** 2026-09-15 against the arXiv v2 PDF. The planned library card `papers/pretrainers-guide-training-data.md` did not exist when ch-10 was revised; this excerpt holds only what ch-10 cites, with loci.

## Setting (§2)
- 28 decoder-only models of 1.5B parameters ("LM-XL", t5.1.1-XL-like, T5X), plus 20M "LM-Small" for scaling checks (Abstract, §2.4).
- Pretraining data: C4 and the Pile. Their C4 version does not apply the original bad-words filter; both datasets are further deduplicated with the approximate method of Lee et al. (§2.1).
- Evaluation: each pretrained model is fine-tuned per task and scored on the same test data; results are reported relative to a model trained on the unfiltered dataset (§2.3). QA domains come from MRQA and UnifiedQA (§2.3).
- Quality filter: the PaLM/GLaM classifier, score 0 (high quality) to 1 (low quality); documents above thresholds 0.975, 0.95, 0.9, 0.7 removed, plus an inverse filter (§2.2).
- Toxicity filter: Perspective API; documents above 0.95, 0.9, 0.7, 0.5, 0.3 removed, plus an inverse filter that removes the least toxic documents (§2.2).

## Results used in ch-10
- **Toxicity filtering on C4** (§5, Figure 7, change in QA score by domain; percentage of data kept in parentheses):

| Filter | Wiki | Web | Books | Biomed | Academic | Common Sense | Contrast Sets | Average |
|---|---|---|---|---|---|---|---|---|
| Inverse T=0.06 (92%) | 0.4 | −1.4 | 3.8 | 0.7 | 4.9 | 4.1 | 2.7 | 1.7 |
| T=0.9 (95%) | −2.2 | −1.1 | −0.6 | −3.0 | 0.2 | 2.9 | 0.2 | −0.7 |
| T=0.5 (76%) | −4.2 | −2.4 | −0.9 | −3.3 | −1.1 | −0.3 | −0.1 | −2.0 |
| T=0.3 (61%) | −3.8 | −4.4 | −1.4 | −2.5 | −0.3 | −1.3 | −3.5 | −2.7 |

- "Toxicity filtering trades off generalization and toxicity identification ability for reduced risk of toxic generation" (§5 findings box). "The strongest performance on toxicity identification for every dataset comes from the inverse toxicity filter" (§5).
- **Quality filtering on C4** (§5, Figure 6): T=0.975 (91% kept) average +2.5, Books −2.2, Biomed +6.1, Academic +6.4; T=0.7 (46% kept) average +0.7, Books −6.7. Both the quality filter and the inverse quality filter increased toxic generation (Figure 5). The authors write that the benefits of quality filtering "are not predictable from text characteristics" (§1).
- **Domain composition** (§6): removing Common Crawl, Books, or OpenWeb from the Pile degraded average downstream performance most; "Data source heterogeneity is more important than data quality or size" (§6 findings box).

## Stated limits (§8)
- Each pretraining configuration was run once ("single shot experiments").
- Evaluation is in the fine-tuned setting; transfer to zero- or few-shot prompting is not established.
- Two English datasets only; Perspective API may change over time.

## Connections
- [[c4]] — the base corpus for the filter experiments.
- [[dolma]] — reaches a related conclusion on source heterogeneity with Paloma perplexity (§9.2).
- ch-10a covers the classifier quality filter in depth.
