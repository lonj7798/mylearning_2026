---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
---

# Excerpt: Dolma — the ablation protocol and `attributes/` pattern ch-17 inherits

**Source library:** `wiki/raw-data/llm-training/papers/dolma.md`
**Paper:** Soldaini et al. 2024, "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (ACL 2024).

---

## Why Dolma sets the methodology, not just the stages

[[ccnet]] gave ch-17 its *stages*. [[dolma]] gives ch-17 its *discipline*: run filters as streaming passes, write per-document attributes, defer keep/drop to a cheap final pass, and ablate each stage against a downstream eval rather than by eye.

From the source (lines 29–44):

> **Filter cascade (order matters):**
> 1. URL / document-level deduplication first using Bloom filters — remove exact-URL repeats across CC snapshots.
> 2. Language identification via fastText; keep English (lang score ≥ threshold).
> 3. Quality filters adapted from Gopher/C4 heuristics (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction).
> 4. Content filters: fastText classifiers trained on the Jigsaw Toxic Comments dataset produce `hate` and `NSFW` scores; documents above threshold are dropped.
> 5. PII filtering targets three high-precision categories: email addresses, IP addresses, phone numbers.
> 6. Paragraph-level deduplication last, via Bloom filter exact-match on paragraphs.

Ch-17 collapses Dolma's 6 stages into 4 (no URL-list, no content-toxicity, no PII, no paragraph-dedup) because at 1 GB the first three are mostly noise and paragraph-dedup duplicates MinHash's coverage at lab scale. But the *order* matches: lang-ID early, quality next, dedup last.

The source spells out the rationale for putting dedup last:

> **Why this order:** doing paragraph dedup last is deliberate — earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

This is the non-obvious methodological insight of Dolma §6. If dedup runs first, you compute hashes over a document set 10× the size, and ~90% of that work will be discarded by later stages. More subtly: the "surviving distribution" after quality-filtering has *different* duplicate patterns than the raw crawl. A boilerplate pattern that a language-ID filter removes for being German is now not in the dedup denominator. Ch-17's decision to run exact + MinHash after lang-ID + perplexity is not cost-saving — it is a correctness choice.

---

## The `attributes/` pattern ch-17 uses for every stage

From the source (lines 43–44):

> **Tooling:** the `dolma` CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document `attribute` files (one score per filter) so the final keep/drop decision is a separate, cheap pass.

Every stage in ch-17's pipeline (`lang_id.py`, `perplexity.py`, …) writes an attribute record — not a filtered document. The keep/drop pass reads all attribute files and applies thresholds. The point is: threshold sweeps (what happens if I move the perplexity cut from 33% to 50%?) do not require re-running perplexity scoring. The expensive pass runs once; the cheap pass runs 20 times.

This is not an ornamental pattern. The ablation grid in the lab asks you to produce rows at different filter combinations; if each row re-ran lang-ID from scratch the whole grid would cost 5× what it does. The `attributes/` separation is what makes the grid feasible at a GPU-day budget.

---

## The Dolma ablation-table structure the memo copies

The source flags two tables as key:

> - **Pipeline diagram** showing the six-stage cascade.
> - **Ablation table** — effect of removing each stage on downstream task accuracy.

The ablation table is the template for the memo's per-filter tables. Dolma's format is: rows = filter configurations (cumulative), columns = downstream tasks (HellaSwag, ARC-E, PIQA, SIQA, OBQA, BoolQ, MMLU). The lab drops the bottom three and adds LAMBADA — but the shape is Dolma's.

One detail the lab adopts without naming: Dolma reports average-across-tasks in the ablation table. The memo should *not*. Averaging across tasks hides the LAMBADA-vs-HellaSwag sign disagreement that dedup produces — and that sign disagreement is one of the lab's most valuable observations. Leave the columns un-averaged; let each filter's paragraph discuss per-column signs. That is exactly the ablation move [[karpathy-training-neural-net-recipe]] calls "review the 10 worst validation examples" applied to tasks rather than examples.
