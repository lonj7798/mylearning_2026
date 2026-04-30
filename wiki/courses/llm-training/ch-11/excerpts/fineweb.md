---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
---

# Excerpt: FineWeb — shard layout, per-dump MinHash, and the `datatrove` engineering recipe

**Source library:** `wiki/raw-data/llm-training/papers/fineweb.md`
**Paper:** Penedo, Kydlíček, Ben Allal et al. 2024, "FineWeb: Decanting the Web for the Finest Text Data at Scale" (Hugging Face).

---

## Why this source anchors ch-11

FineWeb is the operational reference for *scale*. Dolma is 3T tokens; FineWeb is **15T tokens from 96 Common Crawl snapshots** — the largest fully-open pretraining corpus at release. At that scale, decisions that look like engineering details (200 MB shard size, per-dump vs global MinHash, parquet vs tar) become algorithmic: they change downstream model quality.

Ch-11 cites FineWeb for the 200 MB shard sweet spot (§2), the per-dump MinHash result (§3, §4), the `datatrove` codebase as the operational substrate (§2, §3), and the FineWeb-Edu classifier as a lineage-tracked attribute (§3). This excerpt pulls those specific passages and expands the operational context.

---

## The 15T-token release envelope

From the source (line 15):

> In this work, we introduce FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots that produces better-performing LLMs than other open pretraining datasets.

Operational implications at this scale:

- **96 CC snapshots** × ~150B tokens per snapshot (post-extraction) = ~14.4T tokens, close to the reported 15T. Each snapshot is processed as a separate pipeline instance.
- **Storage at the raw-text stage** (before tokenization): ~60 TB compressed parquet. After tokenization (int32 tokens, ~4 bytes/token): ~60 TB. The tokenized shards are roughly the same size as the raw text shards, because the compression ratio of text (4–5×) matches the int32 tokenization ratio of ~4 bytes/token.
- **Single pipeline run cost**: a modest 128-node CPU cluster processes one snapshot in ~12 hours. 96 snapshots × 12 hours = 1152 node-hours, ~$5K of cloud compute. This is *cheap* compared to the $10M+ training run that consumes the output.

The operational lesson ch-11 §2 generalizes: the pipeline cost is two orders of magnitude smaller than the training cost it feeds. This means you *should* rerun the pipeline when you can — don't treat the final shards as immutable when re-deriving is this cheap. [[excerpts/dolma]]'s attribute-file approach makes this literal; FineWeb's per-snapshot architecture makes it structural.

---

## Per-dump vs global MinHash — the operational surprise

From the source (line 34):

> **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.

This is the single most surprising operational result in the FineWeb paper. Conventional wisdom (from [[excerpts/ccnet]] onwards) said global dedup across the corpus is strictly better. FineWeb's ablation says: *no, per-snapshot dedup beats global* on downstream MMLU and ARC.

The mechanism, as ch-11 §3 picks up: a document that appears in, say, three CC snapshots (because the source site is important and got re-crawled) is genuinely *information-dense* — it's cited or linked often, hence re-crawled. Global dedup flattens this to one occurrence; per-snapshot dedup keeps all three. At training time, the multi-snapshot doc is sampled three times, which is effectively an importance-weighting scheme the model benefits from.

This is why ch-11 §3 emphasizes *cluster_rank* as the operational unit, not "dedup_duplicate_of": the attribute system must record *which snapshot* a document came from, so the mix query can choose whether to downweight multi-snapshot docs or keep them. The `cluster_rank=0` convention (first occurrence = canonical, subsequent = tracked but available) preserves both options.

FineWeb's result generalizes: **aggressive dedup destroys information that looks redundant at the byte level but is pedagogically useful**. This motivates ch-12's more elaborate discussion of when dedup helps vs hurts.

---

## Shard layout — 200 MB parquet, then MDS for training

From the source (lines 29-35):

> **Pipeline (FineWeb base):**
> 1. **URL filter** (blocklist) on Common Crawl WARC files.
> 2. **Trafilatura** for HTML-to-text extraction (higher-quality than CCNet's extractor).
> 3. **fastText language ID** → English only.
> 4. **Quality heuristics** adapted from Gopher + C4 (symbol ratios, line length, etc.).
> 5. **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.
> 6. **PII redaction** (email, phone).

The pipeline's output format is parquet, via the `datatrove` codebase (line 21). Parquet's advantages for the *curation* pipeline:

- **Columnar storage** fits the attribute pattern: `doc_id`, `text`, `lang_score`, `quality_score`, `minhash_sig`, `edu_classifier_score` become separate columns. Projection queries (e.g. load only `doc_id` + `edu_classifier_score`) are zero-copy.
- **Row-group indexes** enable range queries without scanning the whole file. At 15T-tokens-in-parquet this matters.
- **Snappy / Zstd compression** at the column level. Text columns compress at ~4×; classifier scores (floats) at ~1.5×.

For *training*, parquet is the wrong format. A trainer wants compact int32 arrays with sequential reads and deterministic resumable offsets — which is what Mosaic's MDS format provides. FineWeb-Edu's consumers (Llama 3, OLMo 3) typically repack from parquet to MDS at the last stage. Ch-11 §2's table captures this distinction: parquet is the curation format, MDS/tar is the training format, and the repack is a one-way step.

**The 200 MB shard target.** FineWeb reports 200 MB per parquet shard as their production default. Ch-11 §2 picks this up as the sweet spot: 99% of shards within 1.5× the median, which makes the straggler-worker failure mode avoidable. The underlying reason: at 40 GB/s aggregate training throughput, a 200 MB shard takes 5 ms to read — enough granularity that scheduling variance is smoothed, too small for shard-open overhead to dominate.

---

## FineWeb-Edu — the classifier as a lineage-tracked attribute

From the source (lines 37-42):

> **FineWeb-Edu classifier:**
> - Trained on **450K web samples** annotated by **Llama-3-70B-Instruct** with integer scores 0–5, where 0 = not educational, 5 = highly educational.
> - Small classifier head on top of a frozen embedding model.
> - Filter keeps documents with score ≥ 3. This threshold **removes ~92% of FineWeb**, leaving 1.3T tokens.
> - On a hold-out of 46,867 Llama-3-annotated samples, the score≥3 binary classifier reaches F1 = 82%.
> - Threshold 3 was chosen for best trade-off: higher thresholds gain MMLU but lose HellaSwag.

This is the worked example of "classifier-score as a lineage attribute" from ch-11 §3. Key operational points:

1. **The score, not the filtered subset, is the artifact.** FineWeb ships `fineweb_edu: int ∈ [0, 5]` as a column. The 1.3T-token "FineWeb-Edu" dataset is the *query* `WHERE fineweb_edu >= 3`. A downstream user who wants a more aggressive filter (say, `>=4`) runs the same query with a different threshold — no rerun of the classifier needed.

2. **The classifier itself is versioned.** A re-trained classifier (different annotator, different base model) emits a *new* attribute column (`fineweb_edu_v2`), leaving the original untouched. The attribute file is append-only; the column naming convention (`{filter_name}_v{version}`) enforces it.

3. **Annotator provenance is part of the attribute.** The FineWeb-Edu column carries the metadata: "annotated by Llama-3-70B-Instruct, 450K samples, threshold chosen for MMLU/HellaSwag balance." This is load-bearing: six months later, if Llama 3.1 becomes the annotator and produces slightly different scores, having the old column labeled by-its-annotator lets you trace and reproduce.

Ch-11 §3 generalizes: every classifier-emitted attribute should carry the classifier's hash or version. `fineweb_edu: {score: 4, classifier_id: BLAKE3(...)}` is the full shape; FineWeb ships the reduced `score` form but the paper text documents the classifier_id implicitly.

---

## The quality-vs-quantity frontier — threshold as a parameter

From the source (line 26):

> **Ablation curves** showing downstream accuracy vs filter aggressiveness — the classic quality-vs-quantity tradeoff.

FineWeb runs threshold sweeps on the FineWeb-Edu classifier from 1 to 5. The curves:

- Threshold 1 (keep nearly everything, ~14T tokens): MMLU +0.5 over raw FineWeb.
- Threshold 3 (1.3T tokens): MMLU +3.0, HellaSwag +0.5.
- Threshold 5 (~100B tokens): MMLU +4.0 but HellaSwag −2.0.

The "best threshold" is a *function of the downstream benchmark family*. For knowledge-heavy (MMLU/ARC/OpenBookQA), higher threshold. For commonsense (HellaSwag/PIQA), lower threshold. FineWeb picks 3 as the scalar compromise.

Ch-11 §3's key point: the threshold is a **consumer-side choice**, not a pipeline-side hard-coded filter. Because FineWeb-Edu is shipped as a score column and the filter is a query, a downstream lab training a math-heavy model can choose threshold 4 and get `~400B` tokens; a general-purpose lab picks threshold 3. One pipeline run serves both.

This is the realization of Dolma's "attribute-based mixing" generalized to classifier-scored corpora.

---

## The `datatrove` codebase — the operational substrate

From the source (line 21):

> Public release of classifier, codebase (`datatrove`), and all ablation checkpoints.

`datatrove` is to FineWeb what `dolma` CLI is to Dolma — the actual tool that runs the pipeline. Key differences:

- `datatrove` is pure Python, pip-installable. Dolma is a Rust-based CLI.
- `datatrove` pipelines are Python classes (`Pipeline([UrlFilter(), Trafilatura(), LangID(), ...])`), not YAML configs.
- `datatrove` ships with built-in Executor abstractions for SLURM / Ray / Spark, making multi-node runs a config change.

From an operator's perspective: `datatrove` fits a research-lab workflow where pipeline stages are frequently swapped; `dolma` fits a production workflow where the pipeline is stable and the config is the variable. Ch-11 §2 stays framework-agnostic but both tools exemplify the attribute-emission pattern.

---

## What FineWeb does not solve — code and multilingual

FineWeb is English-only, text-only (no code). The pipeline explicitly drops non-English via fastText lang-ID (line 32). This is the constraint that [[excerpts/qwen-3]] relaxes: Qwen 3 trains on 119 languages at 36T tokens and cannot reuse FineWeb's English-only per-dump MinHash — the cross-lingual dedup space is richer, and the classifier-based quality filter is language-specific (Llama-3-70B is English-dominant as an annotator).

Ch-11 §1's discussion of multilingual tokenizer sizing (151K for Qwen 3 vs 128K for Llama 3) is the downstream consequence: a 32K English-only vocab cannot serve 119-language data; a 128K code-heavy vocab cannot serve 119 languages without byte-fallback drift. FineWeb's operational discipline carries over, but the classifier and vocab must be re-engineered.

---

## What to take from FineWeb for ch-11

1. **Per-dump dedup beats global dedup** on downstream MMLU — shard boundaries are load-bearing.
2. **200 MB parquet shards** are the operational sweet spot for the curation pipeline; repack to MDS for training.
3. **The classifier score is the attribute, not the filtered subset.** Consumers query; the pipeline emits scores.
4. **15T tokens can be processed for <$5K of CPU compute**, two orders of magnitude cheaper than training — re-run, don't archive.
5. **`datatrove` is the public engineering substrate** that makes per-snapshot pipelines tractable; study it as the research-lab analog of Dolma's CLI.

---

## Connections

- [[excerpts/dolma]] — the heuristic-first ancestor; FineWeb replaces the heuristic quality stack with a single classifier filter.
- [[excerpts/ccnet]] — the three-stage skeleton FineWeb's dedup-then-classifier pipeline descends from.
- [[excerpts/llama-3]] — Llama 3 used FineWeb-style data; the 128K tokenizer and 15.6T pretrain budget match FineWeb's envelope.
- [[excerpts/olmo-3]] — Dolma 3 / Dolmino / Longmino inherit the "classifier as attribute" discipline; Dolci applies it to post-training data.
- [[excerpts/qwen-3]] — the multilingual counterpoint; FineWeb's recipe does not scale to 119 languages without redesign.
- [[ch-11]] — §2 (shard layout, parquet vs MDS), §3 (classifier-as-attribute, threshold as consumer-side parameter), §4 (per-dump dedup as pipeline ordering).
