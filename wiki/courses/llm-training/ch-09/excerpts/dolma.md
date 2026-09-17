---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md (card verified 2026-09-14)
source_url: https://arxiv.org/abs/2402.00159
primary_version: arXiv:2402.00159v2 (2024-06-06; v1 2024-01)
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research

Values and quotations used by ch-09 `read.md`, matching the verified library card. Authors: Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, et al. (36 authors; Allen Institute for AI and others). The paper describes Dolma v1.6.

## Table 1: sources (Llama tokens)
| Source | Tokens | Share of total (derived) |
|---|---|---|
| Common Crawl | 2,479B | 81.0% |
| GitHub | 411B | 13.4% |
| Reddit | 89B | 2.9% |
| Semantic Scholar (peS2o) | 70B | 2.3% |
| Project Gutenberg | 6.0B | 0.20% |
| Wikipedia and Wikibooks | 4.3B | 0.14% |
| Total | 3,059B | 100% |

These are corpus sizes, not training sampling weights; the paper does not report the per-source sampling rates used to train OLMo-1B (card, "Not reported by the source"). The corpus is released under ODC-By and the toolkit under Apache 2.0 (Conclusion).

## Domain fit across corpora (§9.2, Fig. 5; 1.2B models, 150B tokens)
> "(1) The model trained on Pile performs well as it is comprised of many diverse sources, despite its overall smaller scale. (2) Larger multi-source datasets like Dolma and, to a lesser extent, RedPajama v1 yield models with similar coverage of diverse domains to Pile. (3) Finally, training on single-source corpora like C4, mC4 (English-only), and RefinedWeb leads to models with poor fit to diverse domains as indicated by higher average perplexity."

> "Our controlled perplexity analysis reveals the importance of including non-web data from diverse curated sources."

## Mixture ablation (App. M, Table 4, Fig. 12; 1B models)
Token share web / code / reference / books: Naïve 83.5 / 13.8 / 2.5 / 0.2%; Web Only 100 / 0 / 0 / 0%; Reference+ 81.2 / 13.5 / 4.9 / 0.4%; Gopher-like 68.4 / 5.4 / 24.2 / 2.0%.

Figure 12 caption: "All mixture perform similarly on web data (left), while excluding code increases perplexity on code datasets (center). Finally, increasing reference material by upsampling papers and Wikipedia yields lower perplexity on S2ORC (right). Overall, source distribution is linked to downstream capabilities; thus, Dolma users should sample subsets according to their needs."

The card records that Reference+ (4.9% reference) and Gopher-like (24.2%) are nearly identical on S2ORC.

## Code share (App. M, Table 3; 1B, 5 seeds)
Code share 0% / 5% / 15% of a C4 + Stack mixture: bAbI in-context exact match 0.0 / 8.8 / 10.1; WebNLG Rouge-2 16.8 / 19.3 / 22.0; GSM8K fine-tuned 0.0 throughout; GSM8K with program-aided fine-tuning 11.8 / 14.2 / 14.7.

## Contamination (App. L, Fig. 11)
WSC, SICK, GLUE AX, SemEval 2014 Task 1, SuperGLUE COPA, and SuperGLUE AXb are 100% contained in Dolma; HumanEval, WiC, e-SNLI, and SNLI are over 90%; many copies are in the code subset. Dolma v1.6 as released is not decontaminated (App. N.4).

## Stated limits
Ablations use one 1B-scale architecture and zero-shot tasks, so decisions may not hold at 7B-70B (Limitations).

## Correction to earlier ch-09 material
The April 2026 version of ch-09 gave a "Dolma v1.7, ~3T" composition of about 80% web, 8% code, 5% academic, 3% Reddit, 2% books, and 2% Wikipedia. The 3T-token release described in the paper is v1.6, and its Table 1 shares are 81.0% web, 13.4% GitHub, 2.9% Reddit, 2.3% peS2o, 0.20% Gutenberg, and 0.14% Wikipedia and Wikibooks.
