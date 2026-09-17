---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the card verified on 2026-09-14; the earlier version contained a leave-one-stage-out ablation table that the paper does not report, a wrong stage order, and a document-drop PII rule
---

# Excerpt: Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research

Soldaini, Kinney, Bhagia, Schwenk, Atkinson, Authur, et al. (36 authors; AI2 and others), arXiv 2024-01, ACL 2024. Library card: [[dolma]]. Loci refer to arXiv v2.

## Web pipeline (§5)
- 25 Common Crawl snapshots, 2020-05 to 2023-06; web subset 2.28T tokens (§5). Order: CCNet output → URL dedup → document dedup → quality filters → content filters → paragraph dedup (§5.5).
- **CCNet stage** (§5.1): fastText English score ≥ 0.5 (61.7% removed by bytes); within-snapshot paragraph dedup (about 70% of paragraphs); CCNet removes 84.2% overall, 175.1 TB → 27.7 TB.
- **Quality** (§5.2): "we keep all the Gopher rules (Gopher All) and keep a single heuristic from C4 designed to remove paragraphs that do not end in punctuation (C4 NoPunc)". Gopher All tags 15.23% and C4 NoPunc 22.73% of UTF-8 characters. The datasheet describes the 22.73% rule as "Remove documents with more than half of their line not ending in '.', '?', '!', or '"'" (App. N.4).
- **Toxicity** (§5.3): fastText "hate" and "NSFW" classifiers trained on Jigsaw Toxic Comments score sentences; sentences above the threshold are removed. τ = 0.4 removes 5.5–7.3% "but generally yields lower downstream performance" than τ = 0.0004 (29.1–34.9% removed); τ = 0.4 adopted "to ensure we meet our minimum token count requirement".
- **PII** (§5.3): email addresses, IP addresses, phone numbers by regular expression. "For documents with 5 or fewer PII spans, we replace the span with a special token … this affects 0.02% of documents. Otherwise, we remove entire documents … this affects 0.001% of documents." Removal vs replacement "had no effect on model performance". The datasheet lists 0.05% tagged for masking and 0.11% for removal (App. N.4).
- **Deduplication** (§5.4, Bloom filter): exact URL 53.2% of documents; exact document 14.9% of URL-deduplicated documents; exact paragraph 18.7% of paragraphs. URL and document dedup first for efficiency; paragraph dedup last because "paragraph removal risks disrupting content analysis". The order is not ablated.

## Ablations (§4.2, §5)
- Models: 1.2B parameters (OLMo architecture), stopped at 150B tokens; 8 zero-shot tasks: ARC-E, ARC-C, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande (§4.2, Table 2, App. D).
- Quality (Figure 1): "C4 NoPunc on its own outperforms both C4 All as well as Gopher All on both perplexity and downstream tasks. Finally, combining Gopher All + C4 NoPunc offers the best performance."
- Stacking (Figure 3): "positive compounding effect" of quality filters, dedup, and content filters on HellaSwag.
- Results appear as curves; HellaSwag in the main text, other tasks and Paloma subsets in App. O. No numeric ablation table is given.
- Heuristic filters did not change CCNet KenLM bucket proportions (high 21.9%, medium 28.5%, low 49.6%) (§5.2).

## Domain fit (§9.2, Figure 5)
1.2B models on 150B tokens of C4, mC4-en, RedPajama v1, RefinedWeb, the Pile, and Dolma, evaluated on Paloma: the Pile model fits diverse domains well; Dolma and, to a lesser extent, RedPajama give similar coverage; single-source C4, mC4-en, and RefinedWeb give higher average perplexity.

## Limitations
English only ("reinforces the expectation of English being the 'default' language for NLP"); one 1B-scale architecture, which "might result in design decisions that are not relevant at larger model sizes"; Dolma v1.6 as released is not decontaminated (App. N.4).
