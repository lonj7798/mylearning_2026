---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md (card verified 2026-09-14)
source_url: https://arxiv.org/abs/2402.00159
primary_version: arXiv:2402.00159v2 (2024-06-06; v1 2024-01; ACL 2024)
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the verified card; the earlier version placed language identification after URL dedup, described a `dolma-ngram` T = 1.0 setting and an ablated filter order that the paper does not report, and attributed decisions about semantic dedup and cross-source dedup to the authors
---

# Excerpt: Dolma (deduplication sections)

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, et al. (36 authors; Allen Institute for AI and others). Library card: [[dolma]].

## Web pipeline order (§5.1, §5.5)
CCNet output (fastText English ≥ 0.5; within-snapshot duplicate-paragraph removal of about 70% of paragraphs) → URL dedup → document dedup → quality filters → content filters → paragraph dedup.

## Deduplication with a Bloom filter (§5.4)
- Exact URL dedup removes 53.2% of documents.
- Exact document dedup removes 14.9% of the URL-deduplicated documents.
- Exact paragraph dedup removes 18.7% of paragraphs; the datasheet gives 19.1% of UTF-8 characters (App. N.4).
- URL and document dedup run first to reduce later processing; paragraph dedup runs last because "paragraph removal risks disrupting content analysis" (§5.4).
- The order is justified by efficiency and is not ablated (§5.4; card Verification).

## Interaction with other filters
- Stacking quality filters, dedup, and content filters has a compounding positive effect on HellaSwag at 1.2B parameters and 150B tokens (§5.5, Figure 3).
- Document-level correlations between filters are generally low; Gopher rules correlate negatively with dedup, most in the CCNet "Low" perplexity bucket (−0.36, 43M documents), because they remove random strings that dedup does not catch (App. J, Figure 9).

## Other sources (§6–§8)
- Code: The Stack, already deduplicated with MinHash and LSH by its creators (Allal et al., 2023) (§6).
- Reddit: document-level dedup (§7). Project Gutenberg: dedup by exact title (§8). C4: re-run through the web pipeline except URL dedup (§8).

## Decontamination (App. L)
For OLMo-1B, documents containing a paragraph longer than 13 tokens that appears in Paloma are removed (≤ 0.02% of documents); Dolma v1.6 as released is not decontaminated (App. N.4).
