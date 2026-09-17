---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/gradient-entanglement-margin-alignment.md on 2026-09-15)
source_url: https://arxiv.org/abs/2410.13828
source_version: arXiv v2 (2025-04-22); v1 2024-10-17
created_at: "2026-09-15"
---

# Excerpt: A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglement (Yuan, Zeng, Wu, Wang, Wang, Leqi)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Setting (§1, §3.1)
- Margin-based preference losses have the form `ℓ(x, y_w, y_l; θ) = m(h_w(log π_θ(y_w|x)) − h_l(log π_θ(y_l|x)))` (Eq. 1); DPO, IPO, SLiC, KTO, ORPO, SimPO, CPO, R-DPO are instances that differ in m, h_w, h_l (§3.2, Table 2).
- The DPO gradient is `∇_θ ℓ_DPO = −β c(θ) (∇_θ log π_w − ∇_θ log π_l)` (Eq. 3). With step size η and `C = ηβc(θ)`, a first-order Taylor expansion gives (Eqs. 4-5):
  `Δ log π_w ≈ C (‖∇ log π_w‖² − ⟨∇ log π_w, ∇ log π_l⟩)`
  `Δ log π_l ≈ C (⟨∇ log π_w, ∇ log π_l⟩ − ‖∇ log π_l‖²)`
- Gradient entanglement: each response's log-probability change depends on the other response's gradient through the inner product.

## Condition 1 and the three cases (§3.1.1, Table 1)
- `⟨∇ log π_w, ∇ log π_l⟩ ≤ ‖∇ log π_w‖²` ⇔ log π_w increases; `⟨∇ log π_w, ∇ log π_l⟩ ≤ ‖∇ log π_l‖²` ⇔ log π_l decreases.
- Case 1 (ideal, chosen up and rejected down) requires the inner product to be below min(‖∇ log π_w‖², ‖∇ log π_l‖²). Case 2 (both down) holds when ‖∇ log π_w‖² ≤ inner product ≤ ‖∇ log π_l‖². Case 3 (both up) is the reverse ordering.
- §3.1.1: "Empirically, we observe that many times, the rejected gradient has a larger norm compared to the chosen, resulting in simultaneous decrease of both the chosen and rejected log-probability."
- The margin itself always increases: `Δ(log π_w − log π_l) ≈ C(‖∇ log π_w‖² − 2⟨∇ log π_w, ∇ log π_l⟩ + ‖∇ log π_l‖²) ≥ 0` by Cauchy-Schwarz.

## Empirical observations
- Figure 1: DPO on the TL;DR summarization comparisons (CarperAI/openai_summarize_comparisons). With Mistral-7B-Instruct-v0.3 both chosen and rejected evaluation log-probabilities trend upward; with Meta-Llama-3-8B-Instruct both trend downward (App. C.2 lists the two checkpoints).
- Token-level study (§4.3, GPT-2 small on a curated sentiment preference set built from mteb/tweet_sentiment_extraction): with a single differing token the gradient cosine similarity "quickly declines and stays negative during training" and the chosen log-probability rises; with short and long identical suffixes the similarity is positive and grows with suffix length, and the chosen log-probability falls, more for the longer suffix (Figures 3-4a).
- Token-wise heat map (Figure 4b): the gradients of the contrasting tokens ("positive" vs "negative") have inner product below 0, while identical tokens shared by the two responses have cosine similarities "close to 1 for some tokens" (§4.3).
- Implication stated by the authors: contrastive tokens prevent entanglement, non-contrastive shared tokens cause it, which motivates token-level or sparse alignment methods (§4.3, §5.2).
