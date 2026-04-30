---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ccnet.md
source_url: https://arxiv.org/abs/1911.00359
created_at: "2026-04-23"
---

# Excerpt: CCNet — the three-stage template every later pipeline mutates

**Source library:** `wiki/raw-data/llm-training/papers/ccnet.md`
**Paper:** Wenzek, Lachaux, Conneau, Chaudhary, Guzmán, Joulin, Grave 2019, "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data" (FAIR / Facebook AI).

---

## Why this source anchors ch-10

CCNet is the reason the rest of the chapter has a common vocabulary. Every open pipeline from 2019 onward inherits its three-stage shape — **paragraph dedup → language ID → quality scoring** — and argues about step three. Ch-10 cites CCNet five times: as the template in §1, as the ancestor of Dolma and FineWeb in §3–§4, as the source of the "perplexity against Wikipedia" idea, as the multilingual baseline C4 narrowed, and as the head/middle/tail partition convention most downstream users do not realize they are relying on.

From the source (lines 7–8):

> **Core Insight:** Large web crawls become useful pretraining corpora only after language ID, deduplication, and quality filtering against a trusted reference like Wikipedia.
>
> **Guideline:** For Common Crawl preprocessing, run language ID and exact/near dedup first, then score documents by similarity to a clean anchor corpus before building the final shard.

The guideline is the shape. The core insight is the *why* — raw Common Crawl is not training data, it is the raw material from which training data is refined.

---

## The three stages — what each actually does

From the source (lines 22–26):

> - Input is raw Common Crawl snapshots.
> - Pipeline stages: text extraction, language identification, deduplication, quality scoring, shard export.
> - Quality scoring is based on language-model perplexity / similarity relative to cleaner reference text.
> - CCNet is multilingual, which matters because later open stacks often copied only the English slice of the recipe.

Each stage fixes a distinct failure mode. Text extraction decides *what counts as a document* (WET sidesteps HTML parsing but loses structure; later FineWeb's Trafilatura revisits exactly this). Language ID prevents a corpus from being a noisy mix of detected-as-English-but-actually-code-switched documents. Deduplication removes the boilerplate that dominates any real crawl — site chrome, cookie banners, "Please enable JavaScript to view this page." Quality scoring, finally, is what separates CCNet from simple crawl cleanup: rather than dropping documents by rule, it ranks them by similarity to a known-good corpus and lets the downstream user pick a partition.

The crucial ordering decision is **dedup first, then langID**. Dolma reverses this because Dolma's dedup is document-level and expensive. CCNet's paragraph-level hash dedup is cheap enough that running it on the full multilingual crawl is fine. The ordering difference is a direct consequence of the dedup granularity.

---

## Why Wikipedia as the anchor

CCNet's quality signal is 5-gram KenLM perplexity against Wikipedia of the target language. This is the first time in open pretraining literature that "quality" is made quantitative rather than heuristic. The choice of Wikipedia is not neutral but it is defensible: it is the only 2019-era multilingual corpus that is (a) human-cleaned, (b) large enough per-language to train a useful LM, and (c) stylistically aligned with the encyclopedic-explanation register pretraining mostly wants to imitate.

The per-language KenLM is also what makes CCNet genuinely multilingual rather than English-with-bolted-on-language-ID. Each language's perplexity distribution is calibrated against its own Wikipedia, so the "head partition" in Swahili is the Swahili-Wikipedia-like slice of the Swahili crawl, not a Swahili-text-that-resembles-English-Wikipedia slice.

---

## The head/middle/tail convention — what CCNet ships vs what users ship

CCNet the pipeline does not discard the tail. It **partitions** documents by perplexity percentile and ships all three buckets. Ch-10 §1 calls this out explicitly: *"Label, don't drop — CCNet emits all three partitions and lets the downstream user pick."*

Downstream users — RedPajama-V1, Llama-1's training mix, early open releases — picked the head partition as "CCNet-filtered Common Crawl" and threw away the middle and tail. This is a **use-site decision**, not a pipeline decision, and it is where most of CCNet's effective quality-filtering power actually sits. If you adopt CCNet, you are really adopting CCNet-head.

This matters for ch-10's critique framework: the answer to "what threshold does CCNet use?" is "none — it partitions." The answer to "what threshold does CCNet-head use?" is "the top-third perplexity cutoff per language, determined post hoc by the downstream consumer."

---

## The lineage — why the three-stage shape survived

From the source's Connections (lines 29–30):

> - Direct ancestor of [[c4]], [[dolma]], and [[fineweb]].
> - Pairs with [[deduplicating-training-data]] and [[minhash-lsh]] as the practical dedup lineage.

This is the structural claim of ch-10. The shape persists because each stage is orthogonal (no stage's output is strongly conditional on the next), each stage uses cheap signals (hash tables for dedup, tiny classifier for langID, n-gram LM for quality), and the shape scales — ten years and several orders of magnitude of crawl size later, CCNet still runs on a handful of CPU nodes. Later pipelines pay more at each stage (Trafilatura is more expensive than WET; FineWeb-Edu's classifier is more expensive than KenLM) but the stage budget is still three.

---

## What CCNet does not do — and why that matters

- No **content filtering** (toxicity, NSFW). Dolma adds this as stage 5.
- No **PII handling**. Dolma and FineWeb both add this.
- No **per-source pipelines**. The single-lane web-only design is what Dolma generalizes.
- No **ablation table**. The paper defends the pipeline by showing downstream model improvements but does not toggle individual stages against a fixed eval at a fixed model size.

Each of these absences is a design gap that a later pipeline fills. Read the CCNet → Dolma → FineWeb arc as filling them one at a time: Dolma adds content filtering, PII, per-source pipelines, and ablations; FineWeb adds the classifier-as-quality-signal and the per-dump MinHash finding.

---

## What to take from CCNet for ch-10

1. **Three stages, orthogonal, cheap.** The skeleton is structural-dedup-quality in some order; everything else is parameterization.
2. **Perplexity against an anchor is the first quantitative quality signal.** The classifier era of FineWeb-Edu is a descendant of this move, not a replacement for the shape.
3. **Partition, don't drop.** CCNet's "head/middle/tail" is a better design than a single threshold — it defers the threshold decision to the consumer. Dolma's two-pass (score-then-decide) architecture generalizes exactly this idea.
4. **Multilingual by default is cheap if you build it in from stage 1.** Later English-only pipelines cannot retrofit multilinguality without redoing langID and quality calibration.

---

## Connections

- [[excerpts/c4]] — CCNet's heuristic cousin; drops multilingual, adopts rules.
- [[excerpts/dolma]] — generalizes CCNet's shape with per-source pipelines + ablations.
- [[excerpts/fineweb]] — replaces CCNet's KenLM perplexity with a classifier.
- [[excerpts/scaling-laws-data-quality]] — theoretical frame for why quality-signal choice matters at scale.
- [[ch-10]] §1 (template), §5 (comparison table), §6 (checklist).
