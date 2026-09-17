---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2502.14854v2 (CLIPPER, COLM 2025), §2-4, Tables 1-2 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2502.14854
created_at: "2026-09-15"
---

# Excerpt: CLIPPER — compress the book, then generate claims

- **Authors:** Chau Minh Pham, Yapei Chang, Mohit Iyyer (University of Maryland; University of Massachusetts Amherst)
- **Year:** 2025 (arXiv v1 2025-02; v2 2025-08 used here; COLM 2025)
- **Source type:** paper
- **Used in:** ch-29a §3.3, §6, Negative samples

## Task and data (§2.1)
Narrative claim verification: true/false minimal pairs; "The model is considered accurate only if it correctly verifies both claims in a
pair". 479 Project Gutenberg books, average 90K tokens, 23 chapters; books over 128K tokens excluded.

## Naive generation (§2.2, Table 1)
Claude-3.5-Sonnet-v1 receives the whole book and writes claim pairs with chain-of-thought. Human annotation of 52 claims from six books:
invalid 11.5%, misattribution 28.9%, explicit references 15.4%, duplication 17.3%, any error 73.1%. Cost about $0.07 per claim.

## CLIPPER (§2.3-2.4)
1. Compression: GPT-4o writes a book summary (about 618 tokens); Claude writes per-chapter outlines (synopsis, 5–7 events, characters);
   average outline 8,745 tokens vs book 90,437 tokens, compression rate 10.0%.
2. Claims: book-level claims from 2–3 events across the outlines of at least 2 chapters; chapter-level claims from one chapter outline plus the summary.
3. Deduplication by Claude; validation by GPT-4o "against the source chapter outlines"; 59.4% removed as duplicates and 2.4% as invalid.
   Authors disagreed with the GPT-4o filter on 1 of 72 reviewed pairs.
- Human annotation of 66 CLIPPER claims: invalid 9.1%, misattribution 4.6%, explicit references 0.0%, duplication 3.0%, any error 16.7%;
  "83.3% of the 66 claims are found to be completely free of errors" vs 26.9% for naive. Cost $0.05 per claim.
- Claude's verification accuracy given outlines is 98.6% vs 40.3% on NoCha given full text (§2.2).

## Fine-tuning (§3.1, Table 2)
Split: 16K train claims (8K pairs), 2K validation, 1K test; test books disjoint from training books. LR 1e-6, batch 16, one epoch;
the target is the chain-of-thought plus the answer, the prompt is book text plus claim.
| Model | CLIPPER-test | NoCha | NarrativeQA | MuSR | ∞Bench QA |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 27.9% | 16.5% | 47.7% | 40.3% | 47.8% |
| Llama-3.1-8B-CLIPPER | 76.0% | 32.2% | 49.0% | 43.6% | 46.5% |
| Qwen2.5-7B-Instruct | 51.0% | 24.1% | 40.3% | 41.2% | 35.3% |
| Qwen2.5-7B-CLIPPER | 73.9% | 32.4% | 46.0% | 45.2% | 42.3% |
| ProLong-512K-8B-CLIPPER | 75.0% | 32.3% | 49.0% | 44.5% | 38.5% |
| ProLong-512K-8B-WritingPrompts (short data) | 63.0% | 24.1% | 31.0% | 45.2% | 35.8% |

- §4.2: "its transferability is not universal across domains"; Llama-CLIPPER and ProLong-CLIPPER decline on ∞Bench QA.
- §4.6: Qwen-CLIPPER fails on 97 false claims vs 37 true claims among 1,000 book-level pairs.
