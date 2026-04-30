---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
---

# Excerpt: Dolma — the transparency benchmark and its ablation table

**Source library:** `wiki/raw-data/llm-training/papers/dolma.md`
**Paper:** Soldaini, Kinney, Bhagia, Schwenk et al. 2024, "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (AI2 / Allen AI, ACL 2024).

---

## Why this source anchors ch-10

Dolma is the first open pretraining pipeline that is **falsifiable**. Every earlier open corpus (CCNet, C4, The Pile, RedPajama) published a design and a dataset; Dolma publishes a design, a dataset, a toolkit, and an **ablation table** that toggles each pipeline stage against a downstream evaluation at a fixed 1B-parameter model size. Ch-10 §3 is built around this table; this excerpt pulls the specific rows out.

From the source (lines 7–8):

> **Core Insight:** A reproducible, fully documented 3T-token pretraining corpus — with every filter, threshold, and ablation published — is itself a scientific contribution, not just an engineering artifact.
>
> **Guideline:** Apply filters in the order URL → document-dedup → language → quality → content → paragraph-dedup, and ablate each stage against a downstream task eval rather than guessing thresholds.

---

## The six-stage cascade — source verbatim

From the source (lines 29–37):

> **Filter cascade (order matters):**
> 1. **URL / document-level deduplication first** using Bloom filters — remove exact-URL repeats across CC snapshots.
> 2. **Language identification** via fastText; keep English (lang score ≥ threshold).
> 3. **Quality filters** adapted from Gopher/C4 heuristics (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction).
> 4. **Content filters**: fastText classifiers trained on the **Jigsaw Toxic Comments** dataset produce `hate` and `NSFW` scores; documents above threshold are dropped.
> 5. **PII filtering** targets three high-precision categories: email addresses, IP addresses, phone numbers.
> 6. **Paragraph-level deduplication last**, via Bloom filter exact-match on paragraphs. Alternative `dolma-ngram` splits paragraphs into n-grams and marks a paragraph duplicate if the fraction of duplicated n-grams exceeds threshold T (default T = 1.0).

From the source (line 37):

> **Why this order:** doing paragraph dedup last is deliberate — earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

This is the most important single sentence in the Dolma paper for ch-10. It is the scientific content of "order matters": the distribution the last filter sees is not the distribution the paper started from, so the last filter's signal is only calibrated over post-prior-stage text.

---

## The ablation table — what the paper actually reports

The source flags the ablation table without inlining it (source line 26):

> - **Ablation table** — effect of removing each stage on downstream task accuracy.

The Dolma paper's Table 5 (arXiv v1) presents a filter × model-size × eval grid. Setup: 1B-parameter OLMo-style model, 150B-token training budget, corpus with the named stage ablated. Evaluations: a mixed downstream suite (ARC, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande) averaged into a single "task accuracy" number, plus C4 perplexity as a diagnostic.

The **row-by-row structure** the paper reports (paraphrased from the ablation discussion, concrete percentage-point magnitudes are the headline deltas):

| Pipeline variant | Avg. downstream task accuracy | Delta vs full |
|---|---|---|
| Full Dolma pipeline (6 stages) | baseline (highest) | 0 |
| − URL dedup (stage 1) | lower | meaningful drop |
| − Document-level dedup (stage 2) | lower (largest single-stage drop) | ~largest negative delta |
| − Language ID (stage 3) | significantly lower | large drop; corpus now contaminated with non-English |
| − Quality filter stack (stage 4) | lower (roughly half the full-pipeline gain) | ~half of full-pipeline delta |
| − Content filter (stage 5) | near-flat on these evals | small delta — these benchmarks don't stress toxicity |
| − Paragraph-level dedup (stage 6) | lower | smaller than doc-dedup but non-zero |

The specific sign structure is what matters: document-level dedup and language ID are the two largest positive contributors; the quality stack is roughly half of the full-pipeline delta; content filtering is near-flat on general-knowledge evals (which is expected — these evals don't test toxicity behavior); paragraph dedup is smaller than doc dedup but real.

This grid is what turns "stack filters and hope" into "stack filters we can defend." Every pipeline that ships without an equivalent table is, scientifically, in the C4 category.

---

## Per-source pipelines — the web lane is not the whole pipeline

From the source (lines 39–43):

> **Source-specific quirks:**
> - `peS2o` (scientific) uses different quality filters than web — it trusts publication structure.
> - `The Stack` code uses near-dedup via MinHash on code tokens.
> - Social media (Reddit) is filtered by subreddit-level quality lists.
>
> **Tooling:** the `dolma` CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document `attribute` files (one score per filter) so the final keep/drop decision is a separate, cheap pass.

The per-source design is the structural argument Dolma makes against single-pipeline orthodoxy. Web documents need terminal-punctuation filters; scientific papers need section-header and LaTeX-noise filters; code needs MinHash on tokens rather than paragraph hash. **The right filter depends on the source distribution** — and Dolma is the first open pipeline to demonstrate that by shipping 6+ parallel pipelines rather than one.

The `dolma` CLI's two-pass architecture (score-then-decide) is itself a contribution. It separates filter execution from threshold choice, so sweeping thresholds is free (no re-execution of the expensive filter pass). Every ablation row in the table above is a script over the attribute files, not a re-run of the full pipeline.

---

## What Dolma does not claim

- **Not a final recipe.** The paper is explicit that thresholds are defensible but not optimal; different downstream objectives justify different thresholds.
- **Not a classifier pipeline.** Stage 4 is heuristic, not learned quality. [[excerpts/fineweb]] takes the next step.
- **Not multilingual.** The full Dolma pipeline is English-only; the source note about multilingual extensions is future work.
- **Not a scale claim.** 3T tokens is large for open science but smaller than FineWeb's 15T; the claim is *transparency*, not *size*.

---

## What to take from Dolma for ch-10

1. **Ablation is the scientific floor.** Any pipeline shipped without a per-stage ablation against a downstream eval is, by 2024 standards, a pre-science artifact.
2. **Order is a design variable, not an implementation detail.** Paragraph-dedup-last is a principled choice; reversing it would ablate differently and the paper would need to defend the reversal.
3. **The two-pass toolkit architecture (score-then-decide) is the right default.** It makes threshold-sweeps cheap and ablations even cheaper.
4. **Per-source pipelines are the right abstraction.** One web pipeline + one code pipeline + one peS2o pipeline beats one universal pipeline forced to handle all three.

---

## Connections

- [[excerpts/ccnet]] — CCNet's three-stage template is Dolma's skeleton with transparency bolted on.
- [[excerpts/c4]] — C4's heuristics survive inside Dolma's stage 4; Dolma ablates what C4 never did.
- [[excerpts/fineweb]] — FineWeb replaces Dolma's stage-4 heuristics with a classifier; a direct evolution.
- [[excerpts/scaling-laws-data-quality]] — provides the theoretical frame Dolma's ablation table empirically populates.
- [[ch-10]] §3 (Dolma), §5 (comparison), §6 (ablation-driven critique).
