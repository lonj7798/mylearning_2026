---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the card verified on 2026-09-14; the earlier version gave wrong MinHash settings, wrong PII categories, a reconstructed threshold table, and a reason for per-snapshot deduplication that the paper does not give
---

# Excerpt: The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

Penedo, Kydlíček, Ben allal, Lozhkov, Mitchell, Raffel, et al. (Hugging Face), arXiv 2024-06 (v2 2024-10), NeurIPS 2024 D&B. Library card: [[fineweb]]. Loci refer to arXiv v2.

## Ablation protocol (§3.1)
1.71B-parameter Llama-architecture models, sequence length 2,048, global batch about 2M tokens, GPT-2 tokenizer. About 28B tokens for filtering ablations; 350B tokens for some deduplication and cumulative runs. Two runs per data version (different data subset and seed), averaged. Benchmarks: CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC, MMLU, chosen for low run-to-run variance, near-monotonic improvement, and above-random scores at this scale.

## Stages
- **Extraction** (§3.2, Figure 1): trafilatura on WARC outperformed WET text at 28B tokens.
- **Base filtering** (§3.3): URL blocklist (adult content); fastText English score ≥ 0.65; MassiveText quality and repetition filters with original thresholds. About 36T tokens from 96 snapshots.
- **MinHash** (§3.4, App. E.1): word 5-grams; 112 hash functions in 14 buckets of 8, "targeting documents that are at least 75% similar"; match if all 8 minhashes agree in any bucket; transitive clusters; one random document kept. P(s) = 1 − (1 − s^8)^14 = 56%, 77%, 92%, 98.8% at s = 0.7, 0.75, 0.8, 0.85.
- **Global vs per-snapshot** (§3.4, Figures 3–5): global dedup from the newest snapshot to the oldest removed up to 90% of old snapshots' data and left 4T tokens, with little improvement at 350B tokens. In 2013-48, the ~31B kept tokens trained a worse model than 171B tokens rebuilt from the ~460B removed tokens; kept data "contains more ads, incoherent lists of keywords and generally badly formatted text". Per-snapshot dedup gave 20T tokens and matched RefinedWeb. Hypothesis (Interpretation): the gain comes from removing large clusters present in all crawls; removing clusters with fewer than ~100 duplicates can harm performance.
- **Lighter global methods** (App. E.3, Figure 15): URL dedup (71.5% of tokens removed), line dedup (77.8%), line dedup with min words (85%), 3-line dedup (80.9%) all worse than per-snapshot MinHash.
- **Measurement** (App. E.2, Figure 14): in a simulation of 100 identical 200B-token snapshots, a 1B-token sample is almost all unique.
- **C4 rules** (§3.5, Figure 6, 2019-18 crawl): terminal punctuation gives the largest individual boost but removes ~30% of tokens; curly bracket 2.8%; word length 4.3%; lorem ipsum, javascript, and policy < 0.5% each; all but terminal punctuation (~7%) beats terminal punctuation alone. Adopted: all except terminal punctuation.
- **Custom rules** (§3.6, Figure 7, App. E.4 Table 2): from over 50 statistics compared between per-snapshot (higher quality) and global (lower quality) dedup of 2013-48, 16 candidates, 3 kept: fraction of lines ending in punctuation ≤ 0.12 (10.14% of tokens); fraction of characters in duplicated lines ≥ 0.1 (12.47%; Table 2 lists ≤ 0.01 as the keep condition); fraction of lines shorter than 30 characters ≥ 0.67 (3.73%). Together ~22% removed, aggregate +~1% at 28B tokens.
- **Final** (§3.7): WARC extraction → base filtering → per-snapshot MinHash → C4 rules → custom rules; PII: anonymize email and public IP addresses. The App. A datasheet lists MinHash after the custom filters; the released `datatrove@acd431b examples/fineweb.py` runs all filters before MinHash.

## Topic and domain analyses (§4.1–§4.2; FineWeb vs FineWeb-Edu)
- Topic clusters (50k + 50k samples): FineWeb-Edu gains "Education, Learning, Teaching" and "History, Culture, Politics"; down-samples "Business, Finance, Law", "Entertainment, Film, Theater", "Places, Travel, Real Estate".
- Paloma macro perplexity without decontamination: FineWeb lower on broad web sources and on Twitter AAE, Manosphere, Gab, 100 Subreddits, 4chan; FineWeb-Edu lower on WikiText-103, M2D2 Wikipedia, M2D2 S2ORC, RedPajama ArXiv, and 100 PLs.

## Limits (§6)
Most experiments at 1.71B scale; academic benchmarks without instruction tuning; web data only.
