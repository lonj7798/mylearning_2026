---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: primary source arXiv:2303.09540v3 (no library card as of 2026-09-15; the library card [[d4]] describes the later D4 paper)
source_url: https://arxiv.org/abs/2303.09540
primary_version: arXiv:2303.09540v3 (2023-03-22), full text read 2026-09-15
created_at: "2026-09-15"
---

# Excerpt: SemDeDup: Data-efficient learning at web-scale through semantic deduplication

**Paper:** Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos (Meta AI FAIR; Stanford). arXiv 2023-03. Source type: paper.

## Taxonomy of removable data (§1)
1. Perceptual duplicates: identical to a human observer; exact duplicates are the simplest case.
2. Semantic duplicates: largely identical information content but perceptually distinct (for example, a sentence with some words replaced by synonyms).
3. Semantically redundant data: different underlying objects with overlapping information (two photos of different golden retrievers in parks).
4. Misleading data: data whose removal improves performance.
SemDeDup targets category 2.

## Method (§3, App. Algorithm A7)
1. Embed each data point with a pre-trained model: CLIP for images, OPT for language.
2. Cluster embeddings with k-means: k = 50,000 for LAION, k = 11,000 for C4.
3. Within each cluster compute pairwise cosine similarity; pairs with cosine similarity at least 1 − ε are semantic duplicates.
4. From each duplicate group keep the example with the lowest cosine similarity to the cluster centroid and remove the rest.

Pseudocode core (App., Algorithm A7, captioned Table A7):
```
pairwise_sim_matrix = cluster_i_embeddings @ cluster_i_embeddings.T
triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1)
M = torch.max(triu_sim_matrix, dim=0)[0]
points_to_keep_from_cluster_i = M <= 1 - epsilon
```
- Cost (§3): all pairs for LAION-440M need about 1.9 × 10¹⁷ similarity computations; within clusters about 4.6 × 10¹² (O(n²/k) for uniform cluster sizes).
- ε is tuned per dataset to reach a target dataset size, using 10% of clusters and linear interpolation between two trial values (§6.5). Fig. A17 plots the C4 kept fraction over ε from 0.100 to 0.250.

## LAION results (CLIP ViT-B/16, 32 epochs, §4)
- 30% of LAION-440M images have a semantic duplicate at ε = 0.00095 and 50% at ε = 0.03 (§4.2).
- Removing 37% of LAION-440M gave no drop in ImageNet zero-shot top-1, and removing 50% a drop under 0.5%; random removal gave larger drops (§4.4, Fig. 4).
- Average out-of-distribution accuracy over 6 ImageNet variants increased over baseline at 37% removed and matched it at 50% removed (§4.4, Fig. 5).
- Embeddings from OpenAI CLIP (trained on a different, private dataset) gave 66.96 top-1 vs 66.90 with LAION-trained CLIP at 40% kept (§6.2, Table 2).

## C4 language-model results (OPT 125M and 1.3B, §5)
- Embedding: last-layer embedding of the last token from the 125M OPT model; k = 11,000 (§5.1).
- Validation: OPT validation corpora ("opt_valid") and verbalized OPT-IML instruction data ("prompts_with_answers") (§5.1).
- 125M, 4% of examples removed (appendix captions this setting "96% pruning", Fig. A12): C4-validation perplexity Random 39.51, SemDeDup 39.35, NearDup (Lee et al., 3.9% removed) 39.46; no-pruning baseline 38.95 (column order in Fig. A12: Baseline, NearDup, Random, SemDedup).
- 125M, 20% of examples removed (captioned "80% pruning", Fig. A13; the Fig. A12 note, "Random and SemDeDup prune 4% of examples", shows that caption numbers give the share kept): opt_valid Random 50.66, SemDeDup 49.04, baseline 47.13; prompts_with_answers 31.65, 30.98, 29.60.
- Against random pruning at equal size, SemDeDup had lower perplexity on every opt_valid validation set (§5.2, Fig. A20).
- Multiple epochs over a pruned set reach the single-epoch full-data result with 10–15% less compute (§5.2, Fig. 8).
- What is pruned (§5.3): at low ε, templated text in which few words (a location, a name) change; at higher ε, semantically redundant clusters such as advertisements for one shoe brand.

## Limitations stated by the authors (§8)
The C4 gains were more modest than on LAION because C4 is partially curated and has fewer duplicates; models were small; a pre-trained embedding model relevant to the domain is required.
