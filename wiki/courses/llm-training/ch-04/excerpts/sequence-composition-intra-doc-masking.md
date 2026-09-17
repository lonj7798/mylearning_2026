---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Zhao, Qu, Staniszewski, Tworkowski, Liu, Miłoś, Wu, Minervini — "Analysing The Impact of Sequence Composition on Language Model Pre-Training"
source_url: https://arxiv.org/abs/2402.13991
created_at: "2026-09-15"
note: "No library card for this artifact on 2026-09-15; written from the primary PDF (arXiv v1, 2024-02-21)."
---

# Excerpt: Zhao et al. 2024 — Sequence composition and intra-document causal masking

**Artifact:** arXiv:2402.13991 (v1 2024-02). Authors: Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miłoś, Yuxiang Wu, Pasquale Minervini (University of Edinburgh, Xiaomi AI Lab, University of Warsaw, Weco AI). Code: github.com/yuzhaouoe/pretraining-data-packing.

## Definitions (§2)

- A chunk is `C = (d_1 [EOS] d_2 [EOS] … SPLIT(d_n))`, where SPLIT truncates the last document so that |C| = L (Eq. 1).
- Packing strategies: MIXChunk samples documents uniformly from all corpora; UNIChunk samples each chunk from a single source corpus; BM25Chunk retrieves related documents with BM25 from a buffer (§2.1).
- Causal masking: `P(C) = Π_{i=1..|C|} P(x_i | x_1, …, x_{i−1})`, conditioning on earlier documents in the chunk.
- Intra-document causal masking (INTRADoc): `P(C) = Π_{i=1..n} Π_j P(d_ij | d_i1, …, d_i(j−1))`, conditioning only on the same document (§2.2).

## Setup (§3.1, App. B)

LLaMA-architecture 1.3B models, context 2,048 (2K) or 8,192 (8K), 150B tokens sampled from SlimPajama with source proportions kept. All models see the same documents; only composition and masking differ. Intra-document masking cost 4.0% efficiency in the authors' implementation (§3.1, App. A).

## Results

- Perplexity on SlimPajama test documents, average (Table 1): 2K MIXChunk 9.172, UNIChunk 8.831, BM25Chunk 8.550, INTRADoc 8.410; 8K 9.065, 8.851, 8.289, 8.079. On GitHub at 2K, 5.531 (MIXChunk) vs 4.252 (INTRADoc).
- In-context learning, average accuracy over 7 classification datasets, 16 demonstration seeds, 20 (2K) or 48 (8K) demonstrations (Table 2): 2K 63.54 / 65.02 / 70.91 / 70.52; 8K 62.43 / 65.96 / 69.47 / 71.23 (order MIXChunk / UNIChunk / BM25Chunk / INTRADoc).
- Closed-book QA exact match, average of NQ and TriviaQA (Table 3): 2K 10.33 / 11.12 / 11.34 / 11.60; 8K 7.99 / 7.92 / 8.23 / 10.99.
- The abstract's relative gains for BM25Chunk over MIXChunk: in-context learning +11.6%, knowledge memorisation +9.8%, context utilisation +7.2%.
- At 8K, MIXChunk and UNIChunk do not improve with more demonstrations, while INTRADoc does (Fig. 2).

## Interpretation by the authors

Causal masking over randomly packed chunks includes "distracting information from previous documents", and removing it with intra-document masking or reducing it with related-document packing improves language modelling and downstream abilities (abstract; §5.1 attention analysis).

## Limits

Single model size (1.3B); pre-training only; no SFT or chat evaluation.
