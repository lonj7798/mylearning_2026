---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "Online Merging Optimizers for Boosting Rewards and Mitigating Tax in Alignment (Lu et al., Qwen Team)"
source_url: https://arxiv.org/abs/2405.17931
created_at: "2026-09-15"
---

# Excerpt: Online Merging Optimizer (OnDARE, OnTIES)

This excerpt stands in for the library card `online-merging-optimizer`, which did not exist when ch-34 was written.
Every value below was read in arXiv:2405.17931v1 (2024-05-28, "Preprint. Under review.") at the stated locus.
The Qwen2.5 report names this optimizer for its DPO stage but does not describe it ([[qwen-2.5]] §4.2).

- **Authors:** Keming Lu, Bowen Yu, Fei Huang, Yang Fan, Runji Lin, Chang Zhou (Qwen Team, Alibaba)
- **Source type:** paper; code at github.com/QwenLM/online_merging_optimizers (footnote 1)

## Problem (§1, §3)
- "Alignment tax": abilities from pre-training and SFT decline during RLHF; responses also show code-switching and
  longer outputs (§1).
- A one-time (offline) merge of the RLHF model with its SFT reference restores benchmark scores but lowers MT-Bench
  and AlpacaEval 2.0 (§3, "Offline Merging Mitigates Alignment Tax at Cost").

## Mechanism (§4.1-4.2)
- Delta parameters: τ_r = θ_r − θ_b, where θ_b is the pre-trained model and θ_r the SFT reference.
- Eq. 1 merges the policy and the reference at each step; the authors report it as unstable and relax it to
  Eq. 2: θ^(t+1) ≈ θ^(t) + F(Δθ^(t)) ⊕ F(τ_r), with Δθ^(t) the optimizer update (Adam), F a sparsification operator
  and ⊕ a consensus rule.
- OnDARE (Eq. 3): θ_m^(t) = θ_m^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r); F_R keeps each entry with Bernoulli
  probability p (Eq. 4). "Larger α introduces stronger regularization."
- OnTIES (Eqs. 5-7): top-p magnitude sparsification with sign-based consensus.
- α = 0 makes OnDARE equivalent to gradient-dropout regularization such as ChildTuning (§5.3).

## Evidence (Table 1: change versus AdamW DPO on UltraFeedback; benchmark average / MT-Bench / AlpacaEval 2.0 LC)
| Backbone | Offline linear merge | OnDARE |
|---|---|---|
| Qwen1.5-1.8B-Chat | +0.2 / −0.03 / −0.96 | +0.5 / +0.24 / +0.05 |
| Qwen1.5-7B-Chat | −0.2 / −0.28 / −1.65 | +1.1 / +0.12 / +0.28 |
| LLaMa-3-8B-Instruct | −1.2 / +0.11 / +0.02 | +1.3 / +0.19 / +1.57 |

- Benchmark average covers math, code, instruction following, reading comprehension, knowledge, agent, and
  code-switching categories (§5.1, Table 1 header).
- Reserve rate p: training remains stable down to p = 5e−4 (§5.3, Fig. 2).
- Merging weight α (Table 2, OnDARE): benchmark average 7.4 / 37.2 / 40.1 / 41.8 / 41.6 / 40.1 / 39.2 for
  α = 1e−4 / 5e−5 / 1e−5 / 5e−6 / 1e−6 / 5e−7 / 1e−7. The §5.3 prose says the average peaks at α = 5e−7, which does not
  match Table 2 (peak 41.8 at 5e−6). The authors suggest starting the α search near 1e−7 (§5.3).
- Scale tested: 1.8B to 8B parameters, with DPO, IPO, and KTO (Abstract, §5.4).

## Not reported for Qwen2.5
- α, p, variant (OnDARE or OnTIES), and any ablation of the optimizer inside Qwen2.5 post-training ([[qwen-2.5]] §4.2).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2405.17931v1: Abstract, §1, §3, §4.1-4.2, §5.1-5.4, Tables 1-2.
