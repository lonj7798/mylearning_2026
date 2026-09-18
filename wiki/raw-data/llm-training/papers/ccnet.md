<!-- scope: CCNet (Wenzek et al., LREC 2020): paragraph dedup, fastText language ID, and Wikipedia-trained 5-gram perplexity bucketing of Common Crawl
     deps: none
     see-also: [[fineweb]], [[c4]], [[dolma]], [[deduplicating-training-data]], [[minhash-lsh]], [[the-pile]]
-->

# CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data
- **Core Insight:** On the February 2019 Common Crawl snapshot, paragraph deduplication followed by fastText language identification and Wikipedia-LM perplexity bucketing yields 3.2 TB of compressed documents in 174 languages (532B English tokens), and BERT-BASE trained on the lowest-perplexity third of this data averages 75.9 XNLI dev accuracy over en/ru/zh/ur versus 72.6 for BERT-BASE trained on Wikipedia (§5.1; §5.3 Table 2).
- **Guideline:** When building a multilingual corpus from Common Crawl, deduplicate paragraphs before language identification, because removing repeated English boilerplate first increases the number of documents assigned to low-resource languages instead of English or "no language" (§4.1, Figure 3).
- **Authors:** Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, et al. (Facebook AI)
- **Year:** 2019 (arXiv v1 2019-11; LREC 2020, pp. 4003–4012)
- **URL:** https://arxiv.org/abs/1911.00359
- **Source type:** paper
- **Relevant topics:** Common Crawl processing, paragraph deduplication, language identification, perplexity-based quality scoring, multilingual pretraining data

## Abstract
The paper describes an automatic pipeline that extracts large monolingual corpora from Common Crawl for many languages. The pipeline follows the fastText data processing of Grave et al. (2018), which deduplicates documents and identifies their language. It adds a filtering step that selects documents close to high-quality corpora such as Wikipedia. The introduction states two differences from Grave et al.: document-level structure is preserved so that paragraph-level models such as BERT can be trained, and the similarity filter is optional (§1). Code and models are released at github.com/facebookresearch/cc_net (§1, footnote 2).

## Key Contributions
- A per-snapshot pipeline over Common Crawl WET text: paragraph hashing → deduplication → fastText language identification → SentencePiece tokenization and 5-gram language-model perplexity → regrouping by language and perplexity bucket (§3, Figure 1).
- An ablation of step order showing that deduplication before language identification recovers low-resource-language documents (§4.1, Figure 3).
- Wikipedia-trained SentencePiece and Kneser–Ney models for 48 languages, plus code to train models on other reference text and compute tercile thresholds (§3.4).
- Per-language statistics for the Feb. 2019 snapshot, for the 130 languages with more than 1,000 documents (§5.1, Table 3).
- Quality checks with fastText word embeddings (head/middle/tail) and BERT-BASE (CC head vs Wikipedia) (§5.3, Tables 1–2).
- A tool that rebuilds the output from a URL list without rerunning the pipeline (§3.5).

## Key Figures/Tables to Study
- Figure 1: pipeline for one snapshot. Figure 3: per-language document ratio, "dedup then LID" vs "LID then dedup" (estimated on 1% of Feb. 2019).
- Figures 4–5: characters remaining and RAM as a function of the fraction of hashes used.
- Figure 7: perplexity histograms for English and Gujarati with head/middle/tail thresholds.
- Table 1 (fastText analogies by bucket), Table 2 (XNLI), Table 3 (documents, sentences, tokens per language; the text calls it "table 6").

## Technical Details
**Input and sharding.** Each snapshot holds 20–30 TB of uncompressed text, "approximately 3 billion web pages"; Feb. 2019 has 24 TB (§3.1). WET files are grouped into 5 GB shards, 1,600 shards for Feb. 2019, "around 1.6M documents" each (§3.1, §4.3).

**Deduplication (exact, paragraph level).**
1. Duplicated paragraphs "represent 70% of the text" (§3.2).
2. Normalize each paragraph: lower-case, replace digits with 0, remove Unicode punctuation and accent marks (§3.2).
3. Key = first 64 bits of the SHA-1 digest of the normalized paragraph; hashes per shard are saved to binary files (§3.2).
4. Each shard is compared against 1 shard, a subset, or all hash files (§3.2). For one shard, 42% of characters remain after dedup across 1 shard and 28% across 100 shards (§4.2, Figure 4).
5. Hashes from 50 shards = 1.5B unique hashes, 13.5 GB on disk, 40 GB of RAM; the authors choose 50 shards, "blocks corresponding to 3% of the corpus" (§4.2).
The paper uses no fuzzy (MinHash-style) matching.

**Language identification.** fastText classifier trained on Wikipedia, Tatoeba, and SETimes; character n-gram features; 176 languages; 1k documents/s on one CPU core. A page is kept in its top language if the score is above 0.5 and discarded otherwise (§3.3).

**Perplexity scoring.** For each language, a SentencePiece tokenizer and a 5-gram Kneser–Ney model (KenLM) are trained on Wikipedia, for 48 languages (§3.4). Perplexity is computed per paragraph; lower perplexity means closer to the reference domain (§3.4). Each language is split into three parts of equal size (head, middle, tail) with per-language thresholds (§3.4, §5.2). The English LM was trained on "534M of text" and the Gujarati LM on "12M" (§5.2).

**Compute.** Hashing runs at about 600 documents/s per core, 45 min per shard, all hashes in 45 min on 1,600 CPUs (§4.3). The mining step uses 17 processes per shard, about 40 documents/s per worker and about 40 min per shard; paragraph removal takes 40% of the time, language ID 12.5% of CPU time, SentencePiece 33%, the LM 13% (§4.3). Total per snapshot: "about 9 hours using 5000 CPU cores" (§4.3); the introduction gives 8.5 hours (§1).

**Output for Feb. 2019.** 1.5 billion documents in 174 languages (§1); 3.2 TB compressed (§5.1). English: 706M documents and 532B tokens; Russian 101B tokens; Chinese 92B tokens; 11 languages above 10B tokens and 27 above 1B; 12 languages above 10M documents and 29 above 1M (§5.1). Afrikaans, Gujarati, Khmer, and Burmese have 160 MB, 190 MB, 154 MB, and 440 MB, versus 103 MB, 88 MB, 71 MB, and 153 MB in Wikipedia (§5.1).

**Evaluations (§5.3).**
- Table 1, fastText 300-d word embeddings, analogy accuracy Total/Sem/Syn. English: head 77.9/81.2/75.3, middle 74.2/79.0/70.4, tail 62.0/68.1/57.3. Polish: head 65.3/66.5/64.1, middle 62.8/62.7/63.0, tail 59.9/59.8/60.1.
- Table 2, BERT-BASE without next-sentence prediction, XNLI dev accuracy (en/ru/zh/ur/average): Wikipedia 82.8/73.3/77.0/57.3/72.6; CC head 85.0/76.4/77.9/64.3/75.9. The paper reports this as +3.3% on average and a 7-point gain for Urdu, where the Wikipedia model is "similar" to a randomly initialized one.
- Table 2 compares CC head against Wikipedia. It does not compare CC head against unfiltered, middle, or tail CC data.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| BERT-BASE (CCNet XNLI probe, en/ru/zh/ur) | BERT-BASE | pretrain-stable | training data | Wikipedia 16G/5G/1.1G/106M (full); CC head capped at 21G/21G/17G/2.2G | arXiv:1911.00359v2 §5.3 | verified 2026-09-14 | no ablation reported |
| BERT-BASE (CCNet XNLI probe) | BERT-BASE | pretrain-stable | objective | no next-sentence prediction | arXiv:1911.00359v2 §5.3 | verified 2026-09-14 | no ablation reported |
| BERT-BASE (CCNet XNLI probe) | BERT-BASE | pretrain-stable | stopping rule; hardware | early-stopped after two days on 16 Volta32 GPUs; same number of steps for each model | arXiv:1911.00359v2 §5.3 | verified 2026-09-14 | no ablation reported |
| BERT-BASE (CCNet XNLI probe) | BERT-BASE | pretrain-stable | LR, batch, sequence length, step count | not reported | checked body and Table 2 caption | not reported | — |

## Findings relevant to generality
- The perplexity score measures similarity to Wikipedia, not quality in general. The authors report that valid text with vocabulary different from Wikipedia, such as "blog comments with spoken-like text, or very specialized forums with specific jargon", ends up in the tail, and that keyword-list pages also receive high perplexity (§5.2).
- For this reason the authors do not remove content based on the LM score; they release head, middle, and tail buckets "because we think that some of it could be useful for specific applications" (§5.2).
- Thresholds are set per language because perplexity distributions differ across languages; the authors attribute the difference to the size of the Wikipedia used to train each LM rather than to less high-quality content (§5.2, Figure 7) (Interpretation by the authors).
- Language coverage: without deduplication first, many low-resource documents were classified as English or discarded (§4.1). Urdu BERT gains 7 points when trained on CC head instead of Wikipedia (§5.3).
- Measurement limits in the paper: one snapshot; Table 1 evaluates word embeddings, not language models; no seeds or variance are reported (§5.3).

## Connections
- [[fineweb]] — its §2 (Background) states that CC-100 and RedPajama use the cc_net pipeline and describes CC-100 as retaining "only the text that is assigned a low perplexity", a stricter use than CCNet's bucket release.
- [[c4]] — cited by CCNet §2 as concurrent Common Crawl work; CCNet contrasts itself with English-specific preprocessing such as hand-crafted filtering rules.
- [[dolma]] — later open web corpus; compare its rule-based quality filters with CCNet's reference-perplexity buckets.
- [[deduplicating-training-data]], [[minhash-lsh]] — near-duplicate and substring deduplication; CCNet uses exact hashes of normalized paragraphs only.
- [[the-pile]] — Pile-CC uses a classifier against WebText (per [[fineweb]] §2), another similarity-to-reference filter.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/1911.00359 (arXiv v2, 15 Nov 2019); venue and pages from the citation in github.com/facebookresearch/cc_net README.
- Corrections to the previous card version:
  - "Large web crawls become useful pretraining corpora only after language ID, deduplication, and quality filtering" → the similarity filter is described as optional, and documents are bucketed rather than removed by LM score (§1, §5.2).
  - "run language ID and exact/near dedup first" → deduplication runs before language ID (§4.1), and it is exact matching of normalized paragraph hashes, with no near-duplicate step (§3.2).
  - "Pipeline stages: text extraction, language identification, deduplication, quality scoring, shard export" → input is WET text; order is paragraph dedup → language ID → SentencePiece + KenLM perplexity → regroup by language and perplexity (§3, Figure 1).
  - "Showed that filtering toward Wikipedia-like distributions improves downstream model quality" → Table 1 shows fastText embedding accuracy rising from tail to head; Table 2 compares CC head with Wikipedia, not with unfiltered CC (§5.3).
  - "Direct ancestor of [[c4]]" → CCNet cites C4 (Raffel et al., 2019) as concurrent work (§2).
  - "instead of only heuristic URL/domain filters" → the paper contrasts with "keeping URLs shared on Reddit or using hand-crafted filtering rules" (§2).
  - "Pairs with [[minhash-lsh]] as the practical dedup lineage" → CCNet uses SHA-1 paragraph hashes; MinHash is not used (§3.2).
  - Author "Francisco Guzman" → "Francisco Guzmán"; year given as arXiv v1 2019-11 with LREC 2020 venue.
- Removed as unsupported by the source: "became a template for later open pretraining pipelines"; "Established the standard crawl-cleaning stack later reused in C4-, Dolma-, and FineWeb-style pipelines"; "later open stacks often copied only the English slice of the recipe".
- Not reported by the source: how paragraph perplexities are aggregated into a document score; any evaluation of generative LMs trained on the buckets; BERT learning rate, batch size, and step count; run-to-run variance.
