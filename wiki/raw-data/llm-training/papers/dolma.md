<!-- scope: Allen AI's open 3T-token pretraining corpus with fully documented filter pipeline
     deps: [[ccnet]], [[c4]], [[the-pile]]
     see-also: [[fineweb]], [[deduplicating-training-data]]
-->

# Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- **Core Insight:** A reproducible, fully documented 3T-token pretraining corpus — with every filter, threshold, and ablation published — is itself a scientific contribution, not just an engineering artifact.
- **Guideline:** Apply filters in the order URL → document-dedup → language → quality → content → paragraph-dedup, and ablate each stage against a downstream task eval rather than guessing thresholds.
- **Authors:** Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, and ~30 others (Allen AI)
- **Year:** 2024 (ACL 2024)
- **URL:** https://arxiv.org/abs/2402.00159
- **Relevant topics:** pretraining data curation, web filtering, deduplication cascade, open-science datasets

## Abstract
Information about pretraining corpora used to train the current best-performing language models is seldom discussed: commercial models rarely detail their data, and even open models are often released without accompanying training data or recipes to reproduce them. As a result, it is challenging to conduct and advance scientific research on language modeling, such as understanding how training data impacts model capabilities and limitations. To facilitate scientific research on language model pretraining, we curate and release Dolma, a three-trillion-token English corpus, built from a diverse mixture of web content, scientific papers, code, public-domain books, social media, and encyclopedic materials. We extensively document Dolma, including its design principles, details about its construction, and a summary of its contents. We present analyses and experimental results on intermediate states of Dolma to share what we have learned about important data curation practices. Finally, we open-source our data curation toolkit to enable reproduction of our work as well as support further research in large-scale data curation.

## Key Contributions
- A 3T-token English corpus mixing web (Common Crawl), scientific papers (peS2o), code (The Stack), books (Project Gutenberg), social media (Reddit), and encyclopedias (Wikipedia, Wikibooks).
- An open-source `dolma` toolkit for running large-scale filter + dedup pipelines; every stage's config is published.
- Ablation-driven defense of the filter order: each stage is shown to improve downstream OLMo eval scores.
- Dolma was the training set for OLMo 1, making the entire pretraining recipe reproducible end-to-end.

## Key Figures/Tables to Study
- **Pipeline diagram** showing the six-stage cascade (URL dedup → doc dedup → language ID → quality filter → content/PII filter → paragraph dedup).
- **Source mixture table** — token counts by source after filtering; useful to compare against FineWeb/The Pile domain mix.
- **Ablation table** — effect of removing each stage on downstream task accuracy.

## Technical Details
**Filter cascade (order matters):**
1. **URL / document-level deduplication first** using Bloom filters — remove exact-URL repeats across CC snapshots.
2. **Language identification** via fastText; keep English (lang score ≥ threshold).
3. **Quality filters** adapted from Gopher/C4 heuristics (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction).
4. **Content filters**: fastText classifiers trained on the **Jigsaw Toxic Comments** dataset produce `hate` and `NSFW` scores; documents above threshold are dropped.
5. **PII filtering** targets three high-precision categories: email addresses, IP addresses, phone numbers.
6. **Paragraph-level deduplication last**, via Bloom filter exact-match on paragraphs. Alternative `dolma-ngram` splits paragraphs into n-grams and marks a paragraph duplicate if the fraction of duplicated n-grams exceeds threshold T (default T = 1.0).

**Why this order:** doing paragraph dedup last is deliberate — earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

**Source-specific quirks:**
- `peS2o` (scientific) uses different quality filters than web — it trusts publication structure.
- `The Stack` code uses near-dedup via MinHash on code tokens.
- Social media (Reddit) is filtered by subreddit-level quality lists.

**Tooling:** the `dolma` CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document `attribute` files (one score per filter) so the final keep/drop decision is a separate, cheap pass.

## Connections
- Direct successor to [[ccnet]] + [[c4]] pipelines; same lineage, more transparency.
- Paired with OLMo model reports — see `[[olmo-2]]`.
- Compared against [[fineweb]] which uses a classifier-driven rather than heuristic-driven quality filter.
- Dedup methods connect to [[deduplicating-training-data]] and [[minhash-lsh]].
