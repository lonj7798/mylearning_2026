---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/how-much-do-lms-memorize.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2505.24832
created_at: "2026-09-15"
---

# Excerpt: How much do language models memorize?

**Authors:** John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, et al. (FAIR at Meta, Google DeepMind, Cornell University, NVIDIA)
**Version read:** arXiv:2505.24832v3 (18 Jun 2025); v1 May 2025.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v3 PDF text.

## Definitions (§2.1-§2.3)
- Unintended memorization is the information a model contains about a specific dataset; generalization is the information about the data-generating process (Abstract).
- Kolmogorov version (Definition 3): mem_U(x, θ, θ̂) = H^K(x | θ) − H^K(x | θ, θ̂), with θ a reference model and θ̂ the trained model.
- Estimation (§2.3): H^K(x | θ̂) ≈ −log p(x | θ̂); H^K(x | θ̂, θ) ≈ −log max{p(x | θ̂), p(x | θ)}. On synthetic data the reference is the true distribution; on text it is a larger model of the same family trained on a superset of the data (§2.3), or a same-size model trained on the full dataset (§4).

## Capacity on random data (§3.2, Table 1, Fig. 1, Fig. 6)
- Setup: GPT-2 architecture from scratch, 1-8 layers, hidden size 32-512, 100K-20M parameters; 10^6 steps, batch 2048, Adam, bfloat16, single A100; vocabulary V = 2048, sequence length S = 64; five seeds. Dataset entropy H = N·S·log2 V.
- Memorization plateaus at a capacity; models "consistently memorize between 3.5 and 3.6 bits per parameter" (§3.2). Abstract: "approximately 3.6 bits-per-parameter". Fig. 6: α = 3.64. Table 1 means: 3.51 (bf16) and 3.83 (fp32).
- Example row (Table 1): 8 layers, d_model 256, 6.86 × 10^6 parameters, capacity 2.51 × 10^7 bits (bf16), α = 3.65.
- The authors call their estimate "slightly larger" than the 2 bits per parameter of Allen-Zhu & Li (2024) (§3.2).

## Text (§4, FineWeb, 64-token sequences with extra deduplication)
- Unintended memorization increases with parameters and decreases with training-set size (Fig. 4).
- Double descent begins when dataset size (in bits) exceeds model capacity (Fig. 3, Fig. 4).
- Extraction: with large enough deduplicated datasets, the training extraction rate converges to the test extraction rate: "all successful training data extraction is attributable to generalization" (§4, Fig. 8).
- Extra deduplication step because 1-2% of sequences become duplicates after truncation to 64 tokens (§4).

## Membership inference (§5)
- Membership inference F1 is strictly higher than extraction in every case; F1 0.97 with extraction rate 0 in some cases (§5.1).
- Sigmoidal scaling law in capacity and dataset size, fit to within 1-2% of observations (§5.2.1). Validation (Table 2): GPT2-XL (1,556,075,200 parameters) with |D| = 170,654,583 predicted F1 0.55, observed 54.61 ± 1.3; |D| = 18,851,574 predicted 0.95, observed 95.85 ± 0.8.
- The law implies that models trained with 10^2 or more tokens per parameter have membership F1 of 0.5 (§5.2.2).

## Not reported
Downstream task effects; models above 1.5B parameters; natural (non-deduplicated) duplication rates.

## How ch-12a uses it
§1 (unintended memorization vs generalization, worked example), §3 (capacity), Generalization lens (c) (membership inference limits).
