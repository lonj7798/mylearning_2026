---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/vocabulary-scaling-laws.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2407.13623
created_at: "2026-09-15"
---

# Excerpt: Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies

**Authors:** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong (HKU, Sea AI Lab, Contextual AI, Stanford, Ohio State).
**Version read:** arXiv:2407.13623v3 (1 Nov 2024); v1 July 2024; NeurIPS 2024.
**Status:** no library card existed for this slug on 2026-09-15; every value below was read in the v3 PDF text at the stated locus.

## Setup
- Models with non-vocabulary parameters N_nv from 33M to 3B, trained on up to 500B characters of uniformly sampled SlimPajama (Abstract, §4.1).
- Vocabulary parameters are counted as N_v = V·d (output layer only; the embedding layer is excluded because "the main computational burden, as measured in FLOPs, is associated with the output layer", footnote 1).
- Data is measured in characters H. Tokens D = H·f(V), with f(V) = a·(log V)² + b·log V + c; a = 0.0064, b = −0.1581, c = 1.2047, fit on BPE tokenizers from 1K to 1024K (§2.2, Eq. 3, App. A.8). The natural logarithm reproduces Table 2 and Table 3 (D/H = 0.2524 for V = 32K) (this excerpt's check).
- Vocabulary-insensitive loss: L_u = −(1/T) Σ log[ p(w_i | w_<i, V) / p(w_i | V) ], where p(w_i | V) is the unigram frequency of the token (§2.2, Eq. 4).
- Compute: C ≈ 6·(N_nv + V·d)·H·f(V) (§3, Eq. 5).
- Training details (App. A.7): sequence length 2048; global batch 512; AdamW; peak LR 4e-4 decaying to 4e-5; bfloat16; A100 40GB; vocabulary sizes 4096 to 96256 in the IsoFLOP grid; the 2.87B runs took about 120 hours on 64 GPUs for over 500B characters.

## Fits
- Approach 1 (IsoFLOPs): N_nv = 0.08·C^0.50, N_v = 0.20·C^0.42, H = 6.42·C^0.50; γ = 0.42/0.50 = 0.84 (§4.1).
- Approach 2 (derivative of FLOPs with respect to V): γ = 0.83 (§4.2). The abstract and introduction state γ ≈ 0.83.
- Approach 3 (parametric): L_u = −E + A1/N_nv^α1 + A2/N_v^α2 + B/D^β with A1 = 1.831, A2 = 0.196, B = 2.124, E = 5.533, α1 = β = 0.447, α2 = 0.671 (§4.3, Eq. 8).

## Table 1 (predicted optimal V at compute-optimal data)
| N_nv | V_opt App1 | App2 | App3 | FLOPs |
|---|---|---|---|---|
| 3B | 39K | 43K | 37K | 1.3e21 |
| 7B | 62K | 67K | 60K | 7.1e21 |
| 13B | 83K | 91K | 81K | 2.4e22 |
| 30B | 142K | 154K | 142K | 1.3e23 |
| 70B | 212K | 231K | 218K | 7.1e23 |
Abstract: Llama2-70B's optimal vocabulary "should have been at least 216K, 7 times larger than its vocabulary of 32K".

## Tables 2-3 (N_nv = 2.87B, zero-shot accuracy, normalized)
| Budget | V | N_v | D | H | ARC-C | ARC-E | HellaSwag | OBQA | WG | PIQA | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.8e20 (undertraining) | 32K | 0.10B | 15.7B | 62.2B | 23.6 | 40.8 | 34.4 | 29.0 | 49.7 | 64.9 | 59.8 | 43.2 |
| 2.8e20 | 24K | 0.08B | 15.8B | 60.8B | 24.2 | 42.2 | 36.0 | 28.6 | 50.0 | 64.9 | 61.5 | 43.9 |
| 1.2e21 (compute-optimal) | 32K | 0.10B | 67.3B | 266.6B | 28.5 | 49.2 | 47.5 | 31.6 | 50.4 | 71.4 | 56.4 | 47.9 |
| 1.2e21 | 35K | 0.11B | 67.1B | 268.2B | 29.1 | 50.6 | 48.1 | 31.6 | 51.9 | 71.4 | 57.1 | 48.5 |
| 2.3e21 (overtraining) | 32K | 0.10B | 128.5B | 509.1B | 29.1 | 53.5 | 53.0 | 33.0 | 52.0 | 72.0 | 59.5 | 50.3 |
| 2.3e21 | 43K | 0.14B | 127.0B | 517.5B | 32.0 | 54.7 | 54.1 | 33.0 | 52.8 | 72.6 | 61.9 | 51.6 |
Standard deviations printed in the tables are 0.5-2.1 points per task; OBQA is identical in the overtraining pair.

## Data-amount effect (§5, Fig. 7)
At N_nv = 302M, the best vocabulary among {8K, 10K, 16K, 24K, 32K, 48K} moves from 16K to 10K when data is the bottleneck and from 16K to 24K with excess data. The authors still recommend the vocabulary for the compute-optimal allocation "even in scenarios where overtraining may occur", because a larger vocabulary increases inference compute.

## Limits stated by the authors
- Verified up to 3B parameters only; dense Transformers only (App. B.2).
- English corpora (§6 "we pre-train ... on English corpora"); multilingual extension is future work, and languages compete for capacity (App. B.4).
- Approach 2 relies on the FLOPs equation and on f(V) (App. B.1).

## How ch-11 uses it
§2 (vocabulary size and compute), worked example with f(V) and C, Recipe rows, Generalization lens.
