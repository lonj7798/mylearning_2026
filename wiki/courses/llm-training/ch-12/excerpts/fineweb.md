---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md (card verified 2026-09-14)
source_url: https://arxiv.org/abs/2406.17557
primary_version: arXiv:2406.17557v2 (2024-10-31; v1 2024-06); §3.4 and App. E re-read 2026-09-15
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the verified card; the earlier version explained the global-dedup result as loss of re-crawled high-quality pages, which contradicts the paper's reported diagnosis, and quoted unsupported claims about FineWeb-Edu and pipeline order
---

# Excerpt: The FineWeb Datasets (deduplication sections)

Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, et al. (Hugging Face). Library card: [[fineweb]]. This excerpt covers §3.4 and App. E; filtering is covered in ch-10 and ch-10a.

## MinHash setting (§3.4, App. E.1)
- 5-grams from an English word tokenizer; 112 hash functions in 14 buckets of 8 hashes, "targeting documents that are at least 75% similar".
- "Documents with the same 8 MinHashes in any bucket are considered duplicates of each other." There is no separate Jaccard verification step.
- Transitive clustering: A, B, C share a cluster if A–C and B–C are duplicates, even when A and B do not match; one randomly chosen document per cluster is kept.
- Match probability 1 − (1 − s⁸)¹⁴: 56%, 77%, 92%, 98.8% at s = 0.7, 0.75, 0.8, 0.85 (App. E.1).
- Compared with RefinedWeb's 9,000 hashes in 450 buckets of 20, the larger hash count gives "a steeper, more well-defined cut off" but needs more compute and storage; the authors "believe the compute and storage savings make up for the higher uncertainty on documents near the threshold" (App. E.1, Fig. 13).
- Released config, github.com/huggingface/datatrove@acd431b examples/fineweb.py L80–88: `num_buckets=14`, `hashes_per_bucket=8`, `n_grams=5`, `hash_fc="sha1"`, `precision=64`.

## Global versus per-snapshot (§3.4, Fig. 3–5)
1. Global MinHash over 96 snapshots, iterating from the newest (2023-50) to the oldest, removed as much as 90% of the base-filtered data of old snapshots and left 4T tokens. A 350B-token run showed little improvement over non-deduplicated data and scored far below RefinedWeb.
2. Snapshot 2013-48: the ~31B tokens kept by global dedup trained a worse model than 171B tokens obtained by deduplicating the ~460B removed tokens on their own. By visual inspection, kept data "contains more ads, incoherent lists of keywords and generally badly formatted text".
3. Deduplicating each snapshot independently gave 20T tokens and matched RefinedWeb.
4. Hypothesis (Interpretation): the main gain comes from removing very large duplicate clusters present in all crawls; further deduplicating clusters with fewer than about 100 duplicates (the number of crawls) can hurt; filtering targeted at the long tail of quality may suit that subset better.

## Lighter global methods after per-snapshot MinHash (App. E.3, Fig. 15)
URL dedup (71.5% of tokens removed), line dedup (77.8%), line dedup with minimum words (85%), 3-line dedup (80.9%): all performed worse than per-snapshot MinHash alone.

## Measuring deduplication (App. E.2, Fig. 14)
Simulation with 100 identical snapshots of 200B unique tokens each: a 1B-token sample is almost all unique although every document occurs 100 times in the full set; at 1T tokens most documents occur up to 8 times. Dedup ablations were therefore run at 350B tokens.

## Ablation setting (§3.1)
1.71B Llama-architecture models, GPT-2 tokenizer, sequence length 2,048, about 2M tokens per batch; two runs per data version; benchmarks CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC, MMLU.
