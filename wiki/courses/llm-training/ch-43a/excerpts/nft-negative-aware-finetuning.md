---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/nft-negative-aware-finetuning.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.18116
source_version: arXiv v3 (2026-03-01, ICLR 2026); v1 2025-05-23
created_at: "2026-09-15"
---

# Excerpt: NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning (Chen, Zheng, Zhang, Cui, Yuan, Cui, et al.)

Facts used by [[read]], read in the arXiv v3 PDF on 2026-09-15. The same source is excerpted for [[ch-31a]]; the facts below are the ones this chapter uses.

## Method (§3.2-3.3, Algorithm 1)
- Policy splitting: `r_q π⁺(a|q) + (1 − r_q) π⁻(a|q) = π_old(a|q)` with r_q the old policy's correctness rate on q (Eq. 7).
- Implicit negative policy: `π_θ⁻ = (π_old − r_q π_θ⁺)/(1 − r_q)` (Eq. 8). Maximizing the likelihood of π_θ⁻ on wrong answers therefore lowers π_θ⁺ on them; with unlimited data and capacity the optimum is π_θ⁺ = π⁺ (Theorem 3.1). No explicit negative gradient term is written; the push-down is implied by the parameterisation.
- Token-level loss (Eq. 10): `−Σ ω(q) Σ_t [ r log R + (1 − r) log max_v((1 − r̂_q R)/(1 − r̂_q), ε) ]` with `R = π_θ⁺(a_t|·)/π_old(a_t|·)`, r ∈ {0,1} the verifier reward, r̂_q the measured correctness rate of the K samples, and `max_v(x, ε) = stop_gradient[max(x, ε) − x] + x`.
- Relation to GRPO (§4, Props. 4.1-4.2): with ω(q) = (1 − r̂_q)/r̂_q, ε ≤ 1 and on-policy data (R = 1) the NFT and GRPO gradients are equal; with ω(q) = 1 − r̂_q, NFT aligns with Dr. GRPO.

## Setup (§5.1; App. C)
- Qwen2.5-Math-7B and Qwen2.5-32B on DAPO-Math-17k; 16 answers for each of 512 questions per rollout step, 16 gradient steps per rollout step, 320 rollout steps, learning rate 1e-6 with linear warm-up, temperature 1.0, DAPO defaults (dynamic sampling, token-level normalisation, no KL); NFT uses ε = 1.0 and ω(q) = 1 − r_q; truncated answers are treated as negative; RFT is the same pipeline with the negative-data loss zeroed and ω(q) = 1.

## Results (Table 1, six-benchmark average; avg@32 for AIME24, AIME25, AMC23 and avg@1 for MATH500, OlympiadBench, Minerva)
- Qwen2.5-Math-7B: base 31.6; RFT 48.3; GRPO 49.5; Dr. GRPO 49.8; DAPO 51.2; NFT 51.7.
- Qwen2.5-32B: base 29.6; RFT 52.8; DAPO 59.9; NFT 59.2.

## Findings used here
- Entropy (§5.3, Figure 8): "Across both 7B and 32B settings, RFT tends to reduce entropy over time, whereas NFT and RL methods like DAPO encourage increasing entropy."
- Share of the gain (§5.3, Figure 11): "In 32B settings (Figure 11), learning from positive data (RFT) contributes to 80% of the total gain achieved by our best-performing model, while negative data only accounts for the remaining 20%."
- Scale (§5.3): the RFT-NFT gap widens faster over training at 32B than at 7B.
- Clip value (§5.4, Figure 10): ε bounds the penalty weight on a rising wrong answer; "overly aggressive penalization with ε → 0 degrades overall performance"; default ε = 1.0.
- Prompt weighting (§5.4, Figure 9): ω(q) = 1 − r̂_q and (1 − r̂_q)/r̂_q perform similarly and both beat ω(q) = 1, so the NFT-RFT comparison also contains a weighting difference.
