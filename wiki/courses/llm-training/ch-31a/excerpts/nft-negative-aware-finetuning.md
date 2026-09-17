---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/nft-negative-aware-finetuning.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.18116
source_version: arXiv v3 (2026-03-01, ICLR 2026 camera-ready); v1 2025-05-23
created_at: "2026-09-15"
---

# Excerpt: NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning (Chen, Zheng, Zhang, Cui, Yuan, Cui, et al.; Tsinghua, NVIDIA, UIUC, Stanford)

Facts used by [[read]], read in the arXiv v3 PDF on 2026-09-15. Section and equation numbers refer to v3.

## Method (§3.2-3.3, Algorithm 1)
- Policy splitting: r_q·π⁺(a|q) + (1 − r_q)·π⁻(a|q) = π_old(a|q), where r_q = p(r = 1 | q) is the old policy's correctness rate on question q, estimated as the mean of K rewards (Eq. 7).
- Implicit negative policy: π_θ⁻(a|q) := [π_old(a|q) − r_q·π_θ⁺(a|q)] / (1 − r_q). Maximizing its likelihood on negative answers optimizes π_θ⁺; with unlimited data and capacity the optimum is π_θ⁺ = π⁺ (Theorem 3.1, Eq. 8).
- Practical token-level loss (Eq. 10): L = −Σ ω(q) Σ_t [ r·log R_θ^t(q,a) + (1 − r)·log max_v((1 − r̂_q·R_θ^t(q,a)) / (1 − r̂_q), ε) ], with R_θ^t = π_θ⁺(a_t | q, a_<t) / π_old(a_t | q, a_<t).
- max_v(x, ε) = stop_gradient[max(x, ε) − x] + x: the value is clipped at ε while the gradient of x is kept (Algorithm 1, lines 2-3). The negative argument must stay positive; without the clip, training can collapse (§3.3).
- Prompts with 0 < r_q < 1 are kept (Algorithm 1, line 8). Prompt weight ω(q) gives hard questions more weight (§3.3).

## Relation to GRPO (§4)
- Proposition 4.1: with ω(q) = (1 − r̂_q)/r̂_q, NFT and GRPO gradients differ only in how they clip once training is off-policy. Proposition 4.2: with ε ≤ 1 and R_θ^t = 1 (on-policy), the gradients are equal. With ω(q) = 1 − r̂_q, NFT aligns with Dr. GRPO (§4, App. A).

## Setup (§5.1; App. C)
- Qwen2.5-Math-7B and Qwen2.5-32B; DAPO-Math-17k; generation temperature 1.0; about 5,000 gradient steps, batch size 512.
- App. C: LR 1e−6 with linear warm-up; each rollout step generates 16 answers for each of 512 questions, split into 16 mini-batches for 16 gradient steps; 320 rollout steps; DAPO defaults (dynamic sampling, token-level normalization, no KL); context 4K (7B) and 16K (32B NFT; 32K for DAPO); 64 H100 (7B), 128-256 H100 (32B).
- NFT: ε = 1.0, ω(q) = 1 − r_q; DAPO's overlong reward shaping removed; "truncated answers are treated as negative". RFT: negative-data loss zeroed and constant ω(q) = 1.

## Results (Table 1; avg@32 for AIME24, AIME25, AMC23 and avg@1 for others; six-benchmark average)
- Qwen2.5-Math-7B: base 31.6; DPO 48.9; GRPO 49.5; Dr. GRPO 49.8; DAPO 51.2; RFT 48.3; NFT 51.7.
- Qwen2.5-32B: base 29.6; DAPO 59.9; RFT 52.8; NFT 59.2.
- Entropy (§5.3, Fig. 8): "RFT tends to reduce entropy over time, whereas NFT and RL methods like DAPO encourage increasing entropy."
- Share of gain (§5.3): "In 32B settings (Figure 11), learning from positive data (RFT) contributes to 80% of the total gain achieved by our best-performing model, while negative data only accounts for the remaining 20%."
- Larger models (§5.3): the RFT-NFT gap widens faster over training at 32B than at 7B (Fig. 11).
- Prompt weighting (§5.4, Fig. 9): ω(q) = 1 − r_q and (1 − r_q)/r_q perform similarly and both beat ω(q) = 1.
- Clip value (§5.4, Fig. 10): ε sets an upper bound on the penalty weight; "overly aggressive penalization with ε → 0 degrades overall performance"; default ε = 1.0.
- Variance: 3-4 independent runs per algorithm for the 7B curves (Fig. 6).
