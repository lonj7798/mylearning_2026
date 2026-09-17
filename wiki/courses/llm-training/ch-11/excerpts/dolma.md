---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the card verified on 2026-09-14; the earlier version described a regex/NER/LLM "three-tier PII cascade" with precision, recall, and latency numbers that the paper does not report
---

# Excerpt: Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research

Soldaini, Kinney, Bhagia, Schwenk, Atkinson, Authur, et al. (36 authors; AI2 and others), arXiv 2024-01, ACL 2024. Library card: [[dolma]]. Loci refer to arXiv v2.

## PII and secrets (what the paper reports)
- **Web** (§5.3, App. I): regular expressions for email addresses, IP addresses, and phone numbers. "For documents with 5 or fewer PII spans, we replace the span with a special token … this affects 0.02% of documents. Otherwise, we remove entire documents … this affects 0.001% of documents." Removal versus replacement had no measured effect on model performance in the ablations. The datasheet lists 0.05% of data tagged for masking and 0.11% for removal (App. N.4).
- **Code** (§6.3): documents with matches from the detect-secrets tool are removed.
- **Reddit** (§7): documents with PII are removed rather than masked.
- No model-based PII detector, no recall estimate, and no latency numbers are reported.

## Other stages that affect provenance and memorization
- **Deduplication** (§5.4, Bloom filter): exact URL 53.2% of documents; exact document 14.9% of URL-deduplicated documents; exact paragraph 18.7% of paragraphs.
- **Decontamination** (App. L): for OLMo-1B, documents containing a paragraph longer than 13 tokens that appears in Paloma are removed (≤ 0.02% of documents). Dolma v1.6 as released is not decontaminated (App. N.4).
- **Contamination audit** (App. L, Figure 11): WSC, SICK, GLUE AX, SemEval 2014 Task 1, SuperGLUE COPA, and SuperGLUE AXb are 100% contained in Dolma; HumanEval, WiC, e-SNLI, and SNLI over 90%; these datasets are excluded from the paper's evaluations.

## Tokenizer used in the ablations
Dolma data-ablation models (1.2B, 150B tokens) use the GPT-NeoX tokenizer, ALiBi, SwiGLU, context 2048 (App. D.1).

## Toolkit (separate artifact, per the card)
github.com/allenai/dolma docs: `dolma tag` runs taggers, `dolma dedupe` writes duplicate-span attributes under `attributes/`, and `dolma mix` builds a dataset from a JSON or YAML config over compressed JSONL files. The paper itself describes only the "filtering" and "mixing" operations (§4.1).

## How ch-11 uses it
§6 (PII removal as a pipeline stage; masking versus removal), Recipe rows, Common mistakes.
