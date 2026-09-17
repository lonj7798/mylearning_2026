---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ccnet.md
source_url: https://arxiv.org/abs/1911.00359
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the card verified on 2026-09-14; the earlier version stated the stage order as language ID before deduplication and attributed downstream practices to the paper
---

# Excerpt: CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data

Wenzek, Lachaux, Conneau, Chaudhary, Guzmán, Joulin, et al. (Facebook AI), arXiv 2019-11, LREC 2020. Library card: [[ccnet]]. Loci refer to arXiv v2.

## Pipeline for one snapshot (§3, Figure 1)
1. **Input** (§3.1): Common Crawl WET text, 20–30 TB uncompressed per snapshot, about 3 billion pages; February 2019 = 24 TB, grouped into 1,600 shards of 5 GB.
2. **Paragraph deduplication** (§3.2): duplicated paragraphs "represent 70% of the text". Normalize (lower-case, digits → 0, remove Unicode punctuation and accents), hash with the first 64 bits of SHA-1, and remove paragraphs whose hash appears in the comparison set. Exact matching only.
3. **Language identification** (§3.3): fastText, 176 languages; keep the page in its top language if the score is higher than 0.5, otherwise discard.
4. **Perplexity** (§3.4): per language, a SentencePiece tokenizer and a 5-gram Kneser–Ney model (KenLM) trained on Wikipedia, for 48 languages; perplexity per paragraph; each language split into three equal parts (head, middle, tail).

## Order and scope ablations (§4)
- "Contrarily to (Grave et al., 2018), we have chosen to deduplicate the data before language identification, because a lot of English boilerplate, such as cookie warnings, is present in pages of other languages" (§4.1). Figure 3 (1% of Feb. 2019) shows per-language document ratios; low-resource languages benefit most.
- Deduplication scope (§4.2, Figure 4): 42% of a shard's characters remain after comparing with 1 shard, 28% with 100 shards; 50 shards chosen (1.5B hashes, 13.5 GB on disk, 40 GB RAM), "blocks corresponding to 3% of the corpus".
- Cost (§4.3): about 9 hours on 5,000 CPU cores per snapshot.

## Buckets are labels (§5.2)
"Some documents despite being valid text ends up in the tail because they have a vocabulary very different from Wikipedia. This includes blog comments with spoken-like text, or very specialized forums with specific jargon. We decided to not remove content based on the LM score because we think that some of it could be useful for specific applications." Per-language thresholds are used because perplexity distributions differ; the authors attribute this to Wikipedia size (Interpretation).

## Evaluations (§5.3)
- Table 1, fastText analogy accuracy (Total), English: head 77.9, middle 74.2, tail 62.0.
- Table 2, BERT-BASE without NSP, XNLI dev (en/ru/zh/ur/avg): Wikipedia 82.8/73.3/77.0/57.3/72.6; CC head 85.0/76.4/77.9/64.3/75.9.
- Table 2 compares CC head with Wikipedia only; there is no comparison with unfiltered, middle, or tail Common Crawl.

## Output (§5.1)
February 2019: 3.2 TB compressed, 174 languages; English 706M documents and 532B tokens.
