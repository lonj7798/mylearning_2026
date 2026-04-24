---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ccnet.md
source_url: https://arxiv.org/abs/1911.00359
created_at: "2026-04-23"
---

# Excerpt: CCNet — the four-stage template ch-17 reimplements

**Source library:** `wiki/raw-data/llm-training/papers/ccnet.md`
**Paper:** Wenzek et al. 2019, "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data."

---

## Why CCNet is the ch-17 backbone

Ch-17 asks you to build *a* filter pipeline, but specifies the exact four stages: lang-ID → perplexity → exact dedup → MinHash near-dedup. That is CCNet's template. [[dolma]] and [[fineweb]] both descend from it; the chapter makes you reimplement the ancestor before the children.

From the source (lines 22–26):

> Input is raw Common Crawl snapshots. Pipeline stages: text extraction, language identification, deduplication, quality scoring, shard export. Quality scoring is based on language-model perplexity / similarity relative to cleaner reference text. CCNet is multilingual, which matters because later open stacks often copied only the English slice of the recipe.

Two details in that short quote drive the whole lab:

1. **"Language-model perplexity relative to cleaner reference text."** This is the §2 perplexity filter. The reference is Wikipedia, the LM is a 5-gram KenLM. Nothing about this is a black box — it is a count-based n-gram model you can train in an afternoon on a Wikipedia XML dump. That is the point. A classifier-based filter ([[fineweb]] FineWeb-Edu) is genuinely newer machinery; the lab asks you to live inside the pre-classifier world so you feel what it bought and what it left on the table.

2. **"Multilingual … later stacks copied only the English slice."** CCNet runs the pipeline *per language* after fastText lang-ID. The single-language labs copy the English slice and inherit a quiet assumption: that lang-ID errors are rare enough to not distort the perplexity distribution. At small scale (~1 GB) they are not rare. That is why the lab bumps the lang threshold from CCNet's 0.5 to 0.65 — the one piece of the recipe ch-17 tells you to break.

---

## CCNet's bucket-by-perplexity scheme (reused verbatim in §2)

The paper splits each shard into three perplexity buckets: `head` (lowest third), `middle`, `tail`. The original release ships all three; downstream users are expected to pick. Ch-17's §2 says "keep `head + middle`," following [[dolma]]'s choice — but the bucketing itself is CCNet's.

Why buckets rather than a scalar threshold? Because the absolute perplexity number is corpus-dependent (different CC dumps have different word distributions), so a threshold of "PPL < 200" will admit wildly different fractions of different dumps. Percentile-bucketing is self-calibrating.

If ch-17 student asks "why can't I just threshold at PPL < 150?" the CCNet answer is: your 150 is my 90 is someone else's 400. Buckets are portable; thresholds are not.

---

## What CCNet does *not* do (and ch-17 therefore does not do)

- No URL / blocklist stage. [[c4]] did that; [[dolma]] re-added it; CCNet did not. Ch-17 inherits CCNet's decision — not because URL lists are bad, but because on a 1 GB slice they remove almost nothing.
- No content-quality classifier. [[fineweb]] added one; CCNet did not. This is the single biggest thing the lab-internal "oracle" stretch row tests.
- No paragraph-level dedup. [[dolma]] added it after the MinHash pass; CCNet did not. The lab skips it to keep the cascade four stages, not six.

Every missing stage is a deliberate ch-17 boundary. If a student's ablation memo has rows for stages this excerpt says CCNet does not include, they are no longer reimplementing CCNet — they are reimplementing Dolma or FineWeb, and the memo should say so explicitly.
