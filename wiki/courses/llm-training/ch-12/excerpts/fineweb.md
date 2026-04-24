---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
---

# Excerpt: FineWeb — Per-Snapshot MinHash and the Dedup-Aggression Surprise

**Source library:** `wiki/raw-data/llm-training/papers/fineweb.md`
**Authors:** Penedo, Kydlicek, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf (2024)

---

## Why this source anchors ch-12

Lee 2021 set the default: dedup aggressively, ship a model that memorizes less and trains faster. FineWeb is the 2024 update that partially overturns the "aggressively" part. The headline finding for ch-12: **global MinHash across 96 Common Crawl dumps hurt downstream accuracy compared to per-snapshot MinHash.** This is the single most important post-Lee 2021 update to how pretraining pipelines dedup, and ch-12 §7 builds the chapter's canonical cascade around it.

---

## The per-snapshot vs global ablation

From [[fineweb]] §Key Figures/Tables to Study:

> **Per-dump MinHash vs global dedup** comparison — HF found per-dump outperforms naive global dedup on downstream tasks (surprising; tied to removing near-identical re-crawls).

The mechanism, expanded:

1. A high-quality page (Wikipedia article, canonical Stack Overflow answer) is present in most of 96 CC snapshots.
2. Across 96 snapshots, HTML extraction is not bit-identical — Trafilatura sees slightly different boilerplate at different crawl times, so the extracted texts differ in a few paragraphs.
3. Their shingle-Jaccard is ~0.95-0.99: clearly "near-duplicate" under any reasonable threshold.
4. Global MinHash deletes 95 of the 96 copies. Per-snapshot MinHash deletes the duplicates *within each snapshot* but retains one copy per snapshot.

Per-snapshot therefore preserves the high-quality page ~96 times over; global preserves it once. The downstream question is whether that 96x factor is "helpful repetition" or "wasteful memorization." FineWeb's ablation says: helpful, by a measurable margin on downstream evals.

Ch-12 §7 reads this as: **dedup aggression is not free. Each level of aggression deletes a population, and the population's value to the model is not monotone in its redundancy.** This overturns the Lee 2021 intuition that "more dedup is more better" and replaces it with a recall/precision tradeoff mediated by downstream evals.

---

## Why the surprise is only half-surprising

Two framings to reconcile FineWeb with Lee 2021:

**Framing A: data-constrained scaling.** The pre-LLM dedup literature lived in a world where training data was cheap and dedup removed "waste." By 2024, data-constrained scaling ([[data-constrained-scaling]]) had shown that 4-epoch training on a smaller corpus can match 1-epoch training on a larger one. In that regime, duplicates that are *high-quality* become *controlled repetition*, not waste. Per-snapshot dedup is a crude but effective way to preserve that repetition.

**Framing B: crawl artifacts vs genuine redundancy.** What looks like near-duplication across snapshots is partly a *crawl-pipeline artifact* (the same source rendered at different times through different extraction configurations). Treating that as "duplicate" is a category error — the content is the same but the rendering noise is not redundant.

Both framings suggest the same prescription: dedup *within* a snapshot, not *across*. FineWeb implements this and ships.

---

## The full FineWeb dedup pipeline

From [[fineweb]] §Technical Details:

> 1. URL filter (blocklist) on Common Crawl WARC files.
> 2. Trafilatura for HTML-to-text extraction (higher-quality than CCNet's extractor).
> 3. fastText language ID -> English only.
> 4. Quality heuristics adapted from Gopher + C4 (symbol ratios, line length, etc.).
> 5. **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.
> 6. PII redaction (email, phone).

Two process details ch-12 §7 highlights.

**No suffix-array ExactSubstr.** FineWeb does not ship Lee 2021's second tool. The team's position is that MinHash plus paragraph-level heuristics is sufficient at their scale, and the suffix-array cost is prohibitive over 15T tokens. This is a reasonable 2024 call; the tradeoff is that small verbatim blocks (see the 61-word sentence) are not surgically removed.

**Dedup position in the cascade.** FineWeb puts MinHash *after* language ID and quality filters but *before* PII redaction. Compare to Dolma, which puts paragraph dedup *last* after content filtering. The argument for Dolma's order: earlier filters change which paragraphs survive, so dedup operates over a cleaner distribution. The argument for FineWeb's order: dedup before redaction avoids the case where two near-duplicates differ only in their PII content (rare but real for some document types).

Both orders are defensible. The invariant: **dedup runs at least once in the pipeline**, and at least one of its runs is at document or shingle granularity.

---

## FineWeb-Edu and the "one classifier beats stacked heuristics" argument

Adjacent to dedup but important for ch-12's framing. From [[fineweb]]:

> **Why classifier > heuristics at scale:** heuristics (C4, CCNet) plateau - adding more heuristic filters doesn't help MMLU. A single LLM-labeled educational-value classifier captures what no regex stack can: the vibe of a textbook vs the vibe of a forum post.

The parallel to SemDeDup ([[excerpts/d4]]) is direct: both use a learned similarity-or-quality function to make keep/drop decisions. The difference:

- FineWeb-Edu classifier: keeps documents scoring >= 3 on an LLM-labeled educational-value rubric.
- SemDeDup: keeps one representative per embedding-space cluster.

They compose: quality-filter first, SemDeDup the filtered pool. Ch-12 §6 warns against stacking aggressions; FineWeb-Edu at threshold 3 plus SemDeDup at tau = 0.80 would narrow coverage more than either alone, and the paper does not currently ship that combination for exactly this reason.

---

## What FineWeb does not answer

Three open questions ch-12 flags:

1. **Is per-snapshot always better than global?** FineWeb tested 96 CC dumps. For 10 dumps the answer might differ. For a single-crawl corpus it is undefined. The rule generalizes only to "dedup at the granularity where crawl-artifact redundancy lives."

2. **Does this interact with domain mixing ([[ch-13]], DoReMi)?** Per-snapshot dedup preserves more copies of high-quality pages; DoReMi-style domain weighting later in the pipeline sees an artificially inflated mass on those pages. The two interactions have not been ablated together publicly.

3. **Does FineWeb-Edu's classifier-first approach generalize to non-English?** The educational-value rubric was LLM-labeled by Llama 3 in English. Applying the same recipe to French or Chinese requires re-labeling; the dedup decisions are downstream of that labeling and may shift accordingly.

---

## Connections

- [[excerpts/deduplicating-training-data]] — Lee 2021's defaults, which FineWeb partially overturns.
- [[excerpts/minhash-lsh]] — the primitive; FineWeb tunes its aggressiveness.
- [[excerpts/d4]] — the semantic-dedup alternative FineWeb does not ship.
- [[excerpts/dolma]] — the other open-pipeline reference; Dolma and FineWeb differ on dedup ordering.
- [[ch-12]] §7 (production cascade), §8.2 (cross-domain collision as related failure mode).
- [[ch-13]] — DoReMi and domain mixing interact with dedup aggression.
