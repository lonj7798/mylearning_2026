---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/influence-functions-generalization.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2308.03296
created_at: "2026-09-15"
---

# Excerpt: Studying Large Language Model Generalization with Influence Functions

**Authors:** Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, et al. (Anthropic; Grosse and Bae also University of Toronto and Vector Institute)
**Version read:** arXiv:2308.03296v1 (7 Aug 2023).
**Status:** no library card existed for this slug on 2026-09-15; values read in the v1 PDF text.

## Method
- Influence of adding training sequence z_m on a measurement f (§2.1, Eq. 5): I_f(z_m) = −∇_θ f(θ*)ᵀ H⁻¹ ∇_θ L(z_m, θ*), with H the Hessian of the training cost; the change in f is approximately I_f(z_m)·ε with ε = 1/N (Eq. 6).
- In practice (§3, Eq. 24-25): f(θ) = log p(z_c | z_p; θ) for a prompt z_p and completion z_c; I_f(z_m) ≈ −∇_θ f(θ_s)ᵀ (G + λI)⁻¹ ∇_θ L(z_m, θ_s), with θ_s the final pretrained weights, G the Gauss-Newton Hessian approximated by EK-FAC, λ a damping term. Only MLP parameters are included (§3.1).
- Neural-network influence functions are interpreted as approximating the proximal Bregman response function (PBRF), a local formulation (§1, §2.1.1).
- Candidate gradients: TF-IDF filtering or query batching over unfiltered data (§3.2). About 5 million sequences were estimated to suffice for most queries; at least 10 million were scanned for the remaining experiments (§5.2.2).
- Models: pretrained models of 810M, 6.4B, 22B, 52B parameters (Fig. 16 and §5.3).

## Findings (§1 summary list and §5)
1. EK-FAC accuracy is competitive with LiSSA at much lower cost (§5.1).
2. The influence distribution is heavy-tailed; power laws fit the tail (top 0.01% of 5 million samples) for most queries; influence is spread over many sequences (§5.2.1, §5.3.3).
3. Larger models generalize at a more abstract level: for the shutdown query, 810M top sequences share tokens with the query; 52B top sequences share little token overlap but related themes (§1, Fig. 1, §5.3.1).
4. Cross-lingual influence increases with scale: influence of top English sequences on Korean and Turkish translations of two queries is negligible at 810M and increases with model size (§5.3.1, Fig. 16).
5. Word-order sensitivity: training sequences show significant influence only when phrases related to the prompt appear before phrases related to the completion; with synthetic sequences, flipping the order decays influence to near zero (§5.3.4, Fig. 24-25).
6. Memorization (§5.3.3): no clear instances of copying whole sentences in AI Assistant outputs apart from famous quotes; for six famous-passage queries the top influential sequences contained the exact passages.
7. Influence is roughly evenly spread across layers on average; middle layers carry more abstract patterns (§5.3.2).

## Limitations stated by the authors (§1)
PBRF may not capture nonlinear training phenomena; pretrained models only (no fine-tuning); models up to 52B; MLP parameters only; only a fraction of the pretraining corpus searched.

## How ch-12a uses it
§6 (order sensitivity as independent evidence), §8 (influence functions as a tracing tool), Generalization lens (c).
