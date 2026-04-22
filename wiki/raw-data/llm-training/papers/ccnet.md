<!-- scope: CCNet web-corpus extraction pipeline from Common Crawl
     deps: [[deduplicating-training-data]]
     see-also: [[c4]], [[dolma]], [[fineweb]]
-->

# CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data
- **Core Insight:** Large web crawls become useful pretraining corpora only after language ID, deduplication, and quality filtering against a trusted reference like Wikipedia.
- **Guideline:** For Common Crawl preprocessing, run language ID and exact/near dedup first, then score documents by similarity to a clean anchor corpus before building the final shard.
- **Authors:** Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzman, Armand Joulin, Edouard Grave
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1911.00359
- **Relevant topics:** web filtering, language ID, deduplication, quality scoring, Common Crawl

## Abstract
CCNet proposes an automatic pipeline for extracting high-quality monolingual corpora from Common Crawl. The pipeline combines language identification, deduplication, and quality filtering based on perplexity against high-quality text such as Wikipedia. The result is a multilingual web corpus that became a template for later open pretraining pipelines.

## Key Contributions
- Established the standard crawl-cleaning stack later reused in C4-, Dolma-, and FineWeb-style pipelines.
- Combined fastText language ID with deduplication and quality scoring instead of only heuristic URL/domain filters.
- Showed that filtering toward Wikipedia-like distributions improves downstream model quality.

## Technical Details
- Input is raw Common Crawl snapshots.
- Pipeline stages: text extraction, language identification, deduplication, quality scoring, shard export.
- Quality scoring is based on language-model perplexity / similarity relative to cleaner reference text.
- CCNet is multilingual, which matters because later open stacks often copied only the English slice of the recipe.

## Connections
- Direct ancestor of [[c4]], [[dolma]], and [[fineweb]].
- Pairs with [[deduplicating-training-data]] and [[minhash-lsh]] as the practical dedup lineage.

