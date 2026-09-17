---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2402.13228
created_at: "2026-09-15"
verified_against: "arXiv:2402.13228v2 (3 Jul 2024), cached plain text"
---

# Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive (DPOP)

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, Colin White. arXiv v1 2024-02.

## Claim the chapter uses
- **Edit-distance failure mode (§3).** For a pair whose completions differ at one token position `m`, all
  token positions before `m` cancel in the DPO gradient, and for positions `k > m` the gradient with
  respect to the logits is `∇_{θ_j}[log π_θ(y_w|x) − log π_θ(y_l|x)] = s_j^{(y_l^{<k},x)} − s_j^{(y_w^{<k},x)}`,
  where `s_j^{(·)}` is the model probability of vocabulary item `j` in that context (Eq. 2).
  The paper argues that after SFT one expects `s_1^{(y_w^{<k})} ≥ s_1^{(y_l^{<k})}` for the correct token 1 and
  the reverse for other tokens, so the update lowers the logit of the correct token at every position after
  the differing token (§3).
- **DPOP loss (§4, Eq. 3).**
  `L_DPOP = −E log σ( β [ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) − λ · max(0, log π_ref(y_w|x)/π_θ(y_w|x)) ] )`,
  with `λ > 0`. The penalty is 0 while the chosen ratio is at least 1 and grows once it falls below 1.
- **Experiments (§5).** Preference versions of MetaMath (normalised edit distance 6.5%) and ARC-Challenge
  (90%), fine-tuning a Mistral-7B-based model, evaluated with the LM Evaluation Harness; defaults β = 0.3,
  λ = 50. On MetaMath, DPO and SLiC "catastrophically fail" and IPO does not improve over the base model,
  while DPOP improves it (Fig. 2, left). On ARC, DPO, SLiC and DPOP all improve, DPOP most (Fig. 2, right).
  λ ∈ {5, 50, 500} changed results little (Fig. 3). β ∈ {0.1, 0.3, 1.0} did not prevent the DPO failure (Fig. 3, left).
- **Token-level measurement (§5, Fig. 4).** On 900 MetaMath samples, the average log-probability of tokens
  after the differing index in the preferred completion is −0.37 for the reference model, −0.26 for DPOP and
  −1.82 for DPO.
- **Released models.** Smaug-34B and Smaug-72B; Smaug-72B reaches an average of 80.48% on the
  HuggingFace Open LLM Leaderboard (§1, Abstract).

## Not stated by the source
- No measurement of where the removed probability mass goes at the distribution level.
- No result above 72B, and no general-instruction-following evaluation of DPOP.
