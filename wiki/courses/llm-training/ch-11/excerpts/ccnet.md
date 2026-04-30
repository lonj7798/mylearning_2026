---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ccnet.md
source_url: https://arxiv.org/abs/1911.00359
created_at: "2026-04-23"
---

# Excerpt: CCNet — hash-based dedup and the content-addressed lineage ancestor

**Source library:** `wiki/raw-data/llm-training/papers/ccnet.md`
**Paper:** Wenzek, Lachaux, Conneau et al. 2019, "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data" (Facebook AI).

---

## Why this source anchors ch-11

CCNet is five years older than Dolma and predates every other pipeline in the library. It matters for ch-11 because it is the earliest operational demonstration of **hash-based document lineage at Common Crawl scale**. Every subsequent pipeline — C4, The Pile, Dolma, FineWeb — inherits CCNet's three-stage skeleton and its hash-primary-key convention. Ch-11's insistence on content-addressed doc-ids (§3) is a direct descendant.

Ch-11 cites CCNet once, as the hash-based lineage ancestor in §3. This excerpt expands that brief citation with the operational context that the raw-data page compresses to one paragraph.

---

## The three-stage skeleton

From the source (line 8):

> **Guideline:** For Common Crawl preprocessing, run language ID and exact/near dedup first, then score documents by similarity to a clean anchor corpus before building the final shard.

And the key line from the technical details (lines 22-26):

> - Input is raw Common Crawl snapshots.
> - Pipeline stages: text extraction, language identification, deduplication, quality scoring, shard export.
> - Quality scoring is based on language-model perplexity / similarity relative to cleaner reference text.
> - CCNet is multilingual, which matters because later open stacks often copied only the English slice of the recipe.

The three stages — **language ID → deduplication → quality scoring** — became the template. [[excerpts/dolma]] extended this to six stages; [[excerpts/fineweb]] replaced CCNet's perplexity-based quality with an LLM-classifier-based one; both preserved the stage ordering CCNet introduced.

For ch-11 §3's lineage argument, the critical detail is in **what CCNet dedups on**: hashes of the document text, computed before any filter runs. At CC scale (~3B documents per snapshot), exact dedup via hash is the only approach fast enough: 16-byte hashes, bloom filters, one-pass streaming. CCNet was the first paper to demonstrate this at scale.

---

## Hashes as primary keys — the lineage argument

CCNet does not have Dolma's attribute-file architecture (that's a 2024 invention). But it has the underlying primitive: **a hash that uniquely identifies a document**. Once you have that, you can:

1. Dedup in a single pass (bloom filter on hash).
2. Track which documents survived each filter (attach stage flags keyed by hash).
3. Reproduce a subset months later (re-query against the hash list).

Ch-11 §3 takes this implicit pattern and makes it explicit: every raw document gets a BLAKE3 hash; every attribute file references that hash; the filtered corpus is a query over hashes. CCNet gave the operational precedent; Dolma and FineWeb generalized it to multi-attribute lineage.

The key CCNet-era insight, still load-bearing: **hashing is commutative with pipeline stages**. If you hash at stage 0 (raw ingest) and carry the hash through every subsequent stage, every stage can reference documents by their hash without re-computing. If you hash at stage 5, you can't — upstream transformations have already changed the bytes. This is why ch-11 §3's recipe insists BLAKE3 runs on the **original raw bytes** before any filter.

---

## Perplexity-based quality scoring — the ancestor of FineWeb-Edu

From the source (line 25):

> Quality scoring is based on language-model perplexity / similarity relative to cleaner reference text.

CCNet's quality filter runs a small n-gram language model trained on Wikipedia, then scores every CC document by its perplexity under that model. Low perplexity (text "looks like Wikipedia") = high quality. This was the 2019 state-of-the-art — and is the direct ancestor of FineWeb-Edu's classifier-based quality filter.

The operational difference matters for ch-11 §3:

| Generation | Quality signal | Attribute shape |
|---|---|---|
| CCNet (2019) | log-perplexity of doc under Wikipedia LM | one float per doc |
| C4 (2020) | heuristic pass/fail (symbol ratio, line rules) | boolean array |
| Dolma (2024) | Gopher/C4 heuristics + fastText toxic classifier | multiple scores per doc |
| FineWeb (2024) | single LLM-annotated classifier score (0–5) | one integer per doc |

Each generation adds richness but keeps the same *shape*: per-document scalar(s) that the downstream mix query filters on. CCNet established the pattern. Ch-11's attribute-vector model is this pattern formalized.

A subtlety CCNet exposes: **the reference corpus is load-bearing**. CCNet's Wikipedia-anchored perplexity is a specific assumption about what "good" means. A document that looks like forum chat will always score poorly — even if it's high-quality conversational data. This is why FineWeb's classifier-based approach (where the annotator is a large LM, not a small n-gram model) is richer: the annotator knows about more genres. Ch-11 §3 does not belabor this, but it is the implicit reason why modern attribute schemes carry classifier provenance.

---

## Multilingual from day 1

From the source (line 27):

> CCNet is multilingual, which matters because later open stacks often copied only the English slice of the recipe.

C4, Dolma, and FineWeb are English-only (or English-dominant). CCNet explicitly supported multiple languages via fastText language ID + per-language perplexity models. This is the operational precedent for [[excerpts/qwen-3]]'s 119-language pipeline and [[excerpts/deepseek-v3]]'s Chinese-English tokenizer.

The per-language quality scoring is critical: a Wikipedia-English-trained LM cannot score Hindi documents. CCNet trained one perplexity scorer per language, a small operational detail that generalizes: **every pipeline stage that is language-dependent must be parameterized by language**. Ch-11 §1's multilingual vocab discussion (32K English vs 151K for 119 languages) is the tokenizer-side consequence; the quality-classifier side is the CCNet pattern applied to LLM classifiers.

---

## What CCNet does not solve — paragraph-level dedup, PII, code

CCNet is a document-level pipeline. It does not:

- **Paragraph-level dedup** — two documents sharing a common boilerplate paragraph both survive CCNet. Dolma added paragraph-level bloom-filter dedup as stage 6 precisely to address this.
- **PII removal** — CCNet has no PII filter. This is a 2023+ concern that Dolma and FineWeb added.
- **Code-specific filtering** — CCNet is text-only; The Stack and its descendants (CodeParrot, BigCode) built separate code pipelines because CCNet's heuristics don't apply.
- **Opt-out registries** — pre-regulatory; CCNet doesn't have this either.

Ch-11 inherits from CCNet the *lineage-via-hash* primitive and expands outward to cover what CCNet punted on. Reading CCNet after Dolma and FineWeb gives the archaeological view: you can see where each successor filled a specific gap.

---

## Text extraction — the underrated first stage

CCNet's pipeline starts with *text extraction* from WARC files. This is the stage most pipelines gloss over — but it's the gate that determines what the rest of the pipeline even sees.

CCNet used a custom WARC→text extractor. FineWeb later replaced this with Trafilatura (line 30 of the FineWeb raw-data file), which is substantially higher quality for HTML-heavy pages. The operational consequence: rerunning the pipeline with Trafilatura on the same CC snapshots produces ~15–20% more extracted tokens at comparable quality, because Trafilatura better handles JS-rendered content, inline scripts, and CSS boilerplate.

Ch-11 §2 treats text extraction as part of stage "raw" (the first stage). The lineage implication: if the extractor changes, the doc-ids change (different bytes → different BLAKE3). This is why ch-11 §3 emphasizes that the raw hash is over *the extracted text*, not the WARC bytes — extractor versioning is part of the pipeline's primary key.

---

## What to take from CCNet for ch-11

1. **Hash-based dedup at CC scale works** — 16-byte hashes + bloom filters + single-pass streaming. The operational ancestor of every modern content-addressed lineage system.
2. **The three-stage skeleton (lang-ID → dedup → quality)** is the backbone every subsequent pipeline inherits. Ch-11 §2's eight-stage pipeline is CCNet plus filler.
3. **Quality scoring is reference-corpus-dependent.** CCNet's Wikipedia anchor makes sense in 2019; FineWeb-Edu's Llama-3 annotator makes sense in 2024. The pattern — "classifier output as attribute" — is invariant.
4. **Multilingual from the start** is the operational discipline that later English-only pipelines lost; Qwen 3 / DeepSeek-V3 had to rebuild it.
5. **Text extraction is part of the primary key.** Re-running with a different extractor invalidates all downstream hashes.

---

## Connections

- [[excerpts/dolma]] — six-stage successor; adds paragraph-level dedup, PII, content filter.
- [[excerpts/fineweb]] — classifier-based quality filter replacing CCNet's perplexity scorer.
- [[excerpts/llama-3]] — 15.6T token pretrain inherits CCNet-line pipelines at Meta scale.
- [[excerpts/qwen-3]] — multilingual pipeline inheriting CCNet's per-language quality-scoring discipline.
- [[ch-11]] — §3 (content-addressed hashing as the lineage primitive), §4 (the three-stage skeleton that the modern pipeline elaborates on).
