---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/online-merging-optimizer.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2405.17931
created_at: "2026-09-15"
---

# Excerpt: Online Merging Optimizers for Boosting Rewards and Mitigating Tax in Alignment

**Authors:** Keming Lu, Bowen Yu, Fei Huang, Yang Fan, Runji Lin, Chang Zhou (Qwen Team, Alibaba)
**Version read:** arXiv:2405.17931v1 (2024-05-28). Source type: paper (preprint; the authors trained the Qwen models it is later used for).
**Status:** no library card existed for this slug on 2026-09-15. The verified card [[qwen-2.5-recipe]] records its use in Qwen2.5 DPO (§4.2 of that report).

## Mechanism (§4.1-§4.2)
- Delta parameters: θ_b pre-trained model, θ_r reference SFT model, τ_r = θ_r − θ_b. Δθ^(t) is the update a gradient-based optimizer (Adam) would apply at step t.
- Relaxed online merge (Eq. 2): θ^(t+1) ≈ θ^(t) + F(Δθ^(t)) ⊕ F(τ_r).
- OnDARE (Eq. 3): θ^(t) = θ^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r), with F_R a Bernoulli random sparsification that keeps each entry with probability p (Eq. 4) and α the merging weight. "Larger α introduces stronger regularization on the training." α = 0 reduces OnDARE to gradient dropout (ChildTuning) (§5.3).
- OnTIES (Eq. 5-7): top-p magnitude sparsification with sign-based consensus.
- The authors neglect the rescaling of sparsified deltas used in offline DARE/TIES because "rescaling harms the numeric stabilities in multi-step optimization" (§4.2).

## Setting (§5.1, App. C)
Off-policy DPO on UltraFeedback-binarized (about 61K training pairs) from Qwen1.5-1.8B-SFT, Qwen1.5-7B-SFT, and LLaMa-3-8B-it. AdamW baseline LR searched over {1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7}; batch 128; DPO β = 0.1. Evaluation: 12 benchmarks in 7 categories plus MT-Bench and AlpacaEval 2.0; the authors state MT-Bench has a standard deviation of about 0.05 and AlpacaEval (LC) about 0.5 (App. B).

## Results (Table 1, improvement over vanilla AdamW)
| Backbone | AdamW benchmark avg | OnDARE Δ avg | OnDARE Δ MT-Bench | KL penalty Δ avg | LoRA Δ avg |
|---|---|---|---|---|---|
| Qwen1.5-1.8B-Chat | 41.8 | +0.5 | +0.24 | +0.4 | −0.1 |
| Qwen1.5-7B-Chat | 58.4 | +1.1 | +0.12 | +0.1 | +0.2 |
| LLaMa-3-8B-Instruct | 58.0 | +1.3 | +0.19 | +1.0 | +0.6 |

The reference SFT models' benchmark averages were 41.8, 57.3, and 57.7 respectively. Number of seeds per configuration is not reported.

## Hyperparameter effects (§5.3)
- Parameter reserve rate p: stable down to 5e-4 ("discarding 99.95% of gradient-based parameter modifications at each RLHF step still results in stable training", Fig. 2, Qwen1.5-1.8B).
- Merging weight α (Table 2, OnDARE): benchmark average 7.4 / 37.2 / 40.1 / 41.8 / 41.6 / 40.1 / 39.2 and MT-Bench 1.00 / 3.13 / 4.37 / 4.66 / 4.76 / 4.79 / 4.85 for α = 1e-4 / 5e-5 / 1e-5 / 5e-6 / 1e-6 / 5e-7 / 1e-7. The §5.3 prose says the benchmark average peaks at α = 5e−7, while Table 2 prints the highest average at 5e-6; ch-01 uses the table values.

## Verification
- Checked on 2026-09-15 against arXiv:2405.17931v1 (Abstract, §1, §4.1-§4.2, §5.1-§5.3, Tables 1-2, App. B-C).
