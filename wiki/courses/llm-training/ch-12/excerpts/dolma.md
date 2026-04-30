---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
---

# Excerpt: Dolma — the Six-Stage Filter Cascade and Why Paragraph Dedup Runs Last

**Source library:** `wiki/raw-data/llm-training/papers/dolma.md`
**Authors:** Soldaini, Kinney, Bhagia, Schwenk, and ~30 others at Allen AI (ACL 2024)

---

## Why this source anchors ch-12

Ch-12 §7 builds its canonical dedup cascade from Dolma's six-stage recipe. The paper is the most transparent documentation published of a full open-source filter-and-dedup pipeline — every stage has a config file, every stage has a published ablation, every stage has a measurable effect on downstream OLMo eval. For ch-12's purposes, Dolma answers the question "in what order should dedup run?"

---

## The filter cascade, quoted

From [[dolma]] §Technical Details:

> **Filter cascade (order matters):**
> 1. URL / document-level deduplication first using Bloom filters - remove exact-URL repeats across CC snapshots.
> 2. Language identification via fastText; keep English (lang score >= threshold).
> 3. Quality filters adapted from Gopher/C4 heuristics (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction).
> 4. Content filters: fastText classifiers trained on the Jigsaw Toxic Comments dataset produce hate and NSFW scores; documents above threshold are dropped.
> 5. PII filtering targets three high-precision categories: email addresses, IP addresses, phone numbers.
> 6. Paragraph-level deduplication last, via Bloom filter exact-match on paragraphs. Alternative dolma-ngram splits paragraphs into n-grams and marks a paragraph duplicate if the fraction of duplicated n-grams exceeds threshold T (default T = 1.0).

Two dedup stages bookend the pipeline: URL-hash first, paragraph-hash last. Everything between them is filtering.

---

## Why URL-hash is first, paragraph-hash is last

From [[dolma]]:

> **Why this order:** doing paragraph dedup last is deliberate - earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

Expanded to ch-12 §7's framing:

- **URL-hash first** is cheap and catches the most common duplication pattern (the same URL re-crawled across 96 CC snapshots). Running it first reduces the input to every subsequent stage by ~30%, which pays for itself in downstream compute.
- **Paragraph-hash last** is expensive and only correct over the filtered distribution. If you run paragraph dedup before quality filtering, you might delete one copy of a paragraph whose sibling copy later fails quality; the corpus ends with zero copies of a paragraph that both would have been filtered individually but together survived. Paragraph-last avoids this.

The broader rule ch-12 extracts: **cheap and coarse first, expensive and fine last.** URL-hash is coarse (doc-level identity), paragraph-hash is fine (span-level identity). Stacking them in cheap-to-expensive order amortizes cost, and the coarse stage has already shrunk the input by the time the expensive stage runs.

---

## dolma-ngram and the T = 1.0 default

From [[dolma]]:

> Alternative dolma-ngram splits paragraphs into n-grams and marks a paragraph duplicate if the fraction of duplicated n-grams exceeds threshold T (default T = 1.0).

`T = 1.0` means a paragraph is flagged as duplicate only if *100%* of its n-grams appear elsewhere — essentially exact paragraph match, just implemented via n-gram Bloom. This is a very conservative setting; lower T values would behave more like a local MinHash. Dolma keeps T = 1.0 because:

- Paragraph-level exact-match is a known-safe operation.
- Lower T risks deleting paragraphs whose n-grams overlap for innocuous reasons (shared vocabulary in short paragraphs, common sentence-starters).
- The expensive near-duplicate work is delegated to upstream document-level MinHash on specific sub-corpora (The Stack uses MinHash; peS2o uses custom methods).

Ch-12 treats dolma-ngram as the paragraph-granularity equivalent of URL-hash: a cheap exact-match stage that catches boilerplate without the risk of false positives that a fuzzy setting would introduce.

---

## Source-specific dedup

From [[dolma]] §Source-specific quirks:

> - peS2o (scientific) uses different quality filters than web - it trusts publication structure.
> - **The Stack code uses near-dedup via MinHash on code tokens.**
> - Social media (Reddit) is filtered by subreddit-level quality lists.

The Stack's MinHash-on-code is the relevant ch-12 point: dedup tooling is **per-source**, not **global**. Code benefits from MinHash because near-duplicate function bodies across repositories are true redundancy; scientific papers benefit from different methods because their redundancy lives in citations and standardized section structures; Reddit benefits from subreddit-level quality filtering before any dedup stage.

Ch-12 §8.2 generalizes this: **running a single dedup algorithm across mixed-source corpora is a category error.** A paper's method section is a near-duplicate of a blog post summarizing the paper; a cross-source dedup run deletes one of them, and the choice is not what the pipeline intended. Dolma's per-source approach is the reference implementation of the fix.

---

## What Dolma does not do (vs. Lee 2021 and FineWeb)

Dolma does not ship:

- **Suffix-array ExactSubstr.** Like FineWeb, the team deemed it cost-prohibitive at 3T tokens. Paragraph-exact-match via Bloom is the approximation.
- **Semantic dedup (SemDeDup / D4).** Dolma's position is that semantic dedup is research-grade, not production-proven for pretraining; the team points to D4's diversity-narrowing risk as the reason for caution.
- **Classifier-based quality filtering.** Dolma defends heuristic filtering (Gopher/C4-style) against FineWeb's classifier-based approach — the two open pipelines disagree on this axis.

The silence on semantic dedup is notable for ch-12 §6: the most transparent 2024 pretraining pipeline looked at SemDeDup, evaluated the tradeoffs, and declined to ship it. The decision is not "SemDeDup is wrong" but "the downstream-eval justification was not strong enough at Dolma's scale and target."

---

## The ablation-driven defense

From [[dolma]] §Key Contributions:

> Ablation-driven defense of the filter order: each stage is shown to improve downstream OLMo eval scores.

This is the methodological stance ch-12 §8 ends on. Dedup aggressiveness is justified by ablation, not by intuition. The cascade that FineWeb runs is defensible *because* the per-snapshot-vs-global ablation exists; the cascade that Dolma runs is defensible *because* each of the six stages has a published effect on OLMo evals.

Ch-12's rule: if you cannot point to an ablation that justifies a dedup decision, you are guessing.

---

## Operational artifacts the Dolma team shipped

From [[dolma]] §Tooling:

> The dolma CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document attribute files (one score per filter) so the final keep/drop decision is a separate, cheap pass.

This design — per-document attribute files for every filter, with the keep/drop decision deferred to a final cheap pass — is what lets Dolma ship ablations. You can change a threshold and re-score without re-running expensive upstream filters. Ch-11 (tokenizer and lineage) treats this as the reference implementation of dataset versioning for the pretraining layer.

---

## Connections

- [[excerpts/deduplicating-training-data]] — the Lee 2021 tools Dolma's cascade operationalizes (with its own choices).
- [[excerpts/minhash-lsh]] — used in Dolma's The Stack sub-pipeline; the primitive behind "MinHash on code tokens."
- [[excerpts/fineweb]] — the sibling 2024 open pipeline; compare and contrast on dedup ordering and classifier-based filtering.
- [[excerpts/d4]] — the semantic-dedup method Dolma deliberately declines to ship.
- [[ch-12]] §7 (production cascade), §8 (source-specific failure modes).
- [[ch-11]] — dataset versioning via per-document attribute files.
