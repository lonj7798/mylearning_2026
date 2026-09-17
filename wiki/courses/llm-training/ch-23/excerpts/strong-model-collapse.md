---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/strong-model-collapse.md (card has no Verification section as of 2026-09-15; this excerpt was read against the primary source)
source_url: https://arxiv.org/abs/2410.04840
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Strong Model Collapse (Dohmatob, Feng, Subramonian, Kempe)

**Paper:** Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, Julia Kempe (FAIR Meta, NYU, UCLA). arXiv v1 2024-10; v2 2024-10-08 read. Source type: paper.

> This file replaces the 2026-04 excerpt. The formula `f(N) + c(p)·σ²_synth` and the claim "reproduced on GPT-2-scale LM training with 1% synthetic injection; the curve never recovers" are not in the paper. The library card still carries them.

## Setting (§2)
- Real sample D_1 of size n_1 from P_1; synthetic sample D_2 of size n_2 from a different distribution P_2; n = n_1 + n_2; p_2 = n_2/n.
- "Synthetic data" means any training data from a distribution that deviates from the test distribution (§1.1). It is a fixed shifted distribution, not a recursive loop.
- P_k: x ~ N(0, Σ), y = xᵀw_k* + ε. Label shift δ = w_2* − w_1* ~ N(0, Δ).
- Quality of synthetic data: c²(Δ) = (1/d) tr ΣΔ; smaller is better (Definition 1).
- Models: ridge regression (Eq. 3) and a random-projection model f(x) = vᵀSx of width m (Eq. 5). Proportionate limit d/n → ϕ (Eq. 4).

## Results
- Theorem 1: E_test ≃ B + V + ζ; ζ is the extra term from synthetic data (Eq. 9–10). As ϕ → 0, ζ ≃ p_2² tr Δ, which stays positive unless p_2 → 0 ("strong model collapse", §3.1).
- Corollary 1, isotropic, ϕ ∈ (0, 1), λ → 0: E_test ≃ σ²ϕ/(1 − ϕ) + (p_2² + p_2 p_1 ϕ/(1 − ϕ)) c² (Eq. 11); for small ϕ, E_test ≃ σ²d/n + p_2²c² + O(ϕ²) (Eq. 12). "The scaling law plateaus unless p_2 → 0+."
- Abstract: "as little as 1% of the total training dataset" can still lead to model collapse (theoretical statement).
- Model size (Theorem 2, §3.2): larger models can amplify collapse; beyond the interpolation threshold they may mitigate it without preventing it (Abstract).
- MNIST (§4.1, Fig. 7): random-feature (width 100,000) and two-layer (width 2000) networks; scaling slows and plateaus as p_2 grows; collapse subsides only as p_2 → 0.
- Language modeling (§4.2, Fig. 8, App. A.3): GPT-2-small (124M) generator trained on BabiStories (2.2M stories; Mixtral-generated reproduction of TinyStories); synthetic stories generated with temperature 1, top-p 1, then filtered for poor quality; models trained on mixes with p_2 ∈ {0, 0.001, …, 1.0}; evaluation on the TinyStories test set. The synthetic data are high quality (small c²), and "even moderate amounts of synthetic data delay the progression of the scaling laws"; the authors "expect this to eventually lead to plateaus". A plateau is not shown.
- Model size in LM (p_2 = 1, synthetic set 10× the original): 18-layer (166M) and 24-layer (204M) models have lower loss below 1×10¹⁰ tokens; smaller models have lower loss beyond 3×10¹⁰ tokens (§4.2).
- GPT-2 training settings: embedding 768, 12 heads, context 512, LR 5×10⁻³, dropout 0.05, weight decay 0.1, warm-up 2,000 iterations (App. A.3).
- Mixing (§5): weighted single-step mixing cannot avoid collapse for fixed weight α ∈ (0, 1] (Corollary 2). Iterative mixing with p_1 = p_2 = 0.5 over order log(n/d) iterations recovers a scaling law proportional to the clean error but underperforms training on the real half alone (Corollary 3, Fig. 10).
- Filtering and curation methods (Feng et al. 2024 and others) are named as beyond the paper's analysis (§1.2).

## Verification
- Read on 2026-09-15 against arXiv:2410.04840v2 PDF text (Abstract, §1–5, App. A.3).
