---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2310.02743v2 (Reward Model Ensembles Help Mitigate Overoptimization); library card [[reward-ensembling]]
source_url: https://arxiv.org/abs/2310.02743
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: conservative reward-model ensembles (Coste, Anwar, Kirk, Krueger)

Used by [[read]] Guideline, §3.1, the negatives section, the Recipe, and the Generalization lens. ICLR 2024. Checked against arXiv v2 (2024-03-10) on 2026-09-15.

The library card for this slug has no Verification section, and several of its statements are not in the paper. The earlier version of this excerpt repeated them. Corrections:
- "Mean gives the highest peak; LCB and min are safer but cap lower" → worst-case (WCO) and uncertainty-weighted (UWO) optimization avoid over-optimization in best-of-n and reduce it in PPO, while mean optimization over-optimizes with noisy labels (§5, Fig. 3, Fig. 5). The paper studies mean, WCO, and UWO; it does not define an "LCB" aggregator.
- "Peak moves from d ≈ 3 to d ≈ 5–8" → no such values are reported.
- "Shared blind spots demonstrated on adversarial prompts" and "data-shard diversity matters more than seeds" → not in this paper. Ensemble members use the same data and differ only in random seed (§4.3). The shared-error finding is from [[helping-or-herding]], which found pretraining-seed diversity more useful than fine-tuning-seed diversity.
- "3–5 RMs is usually enough" → performance is similar for 4 and 5 members, with a gap from 3 to 4 (§5.4, Fig. 11).
- UWO uses the intra-ensemble variance, not the standard deviation (Eq. 5).

## Objectives (§3)
- `R_μ(q,a) = (1/k) Σ_i R_i(q,a)` (Eq. 3); not conservative: one overestimating member can be exploited.
- `R_WCO(q,a) = min_i R_i(q,a)` (Eq. 4); no hyperparameters.
- `R_UWO(q,a) = (1/k) Σ_i R_i − λ · (1/k) Σ_i (R_i − (1/k) Σ_i R_i)²` (Eq. 5).

## Setup (§4)
- Pythia 1.4B policy; proxy RMs of 7M, 44M, 1.3B (Pythia 14M, 70M, 1.4B with unembedding removed); AlpacaFarm 7B human-preference RM as gold.
- SFT on 10k AlpacaFarm demonstrations; two responses per instruction labelled by the gold RM; optional 25% label noise; RM training set fixed at 46k samples (§5.3); 5 epochs; RMs reach 60–75% validation accuracy.
- Ensembles of five RMs trained on identical data with different random seeds (head initialization and data order).
- BoN up to n = 12,500 (≈ 8.4 nats); PPO for 3000 steps.
- Hyperparameters (App. D.1): SFT LR 8e-6, 3 epochs, batch 4; RM LR 1e-5, 5 epochs, batch 32; PPO LR 1e-6, 4 PPO epochs, batch 32, 256 rollouts, clip 0.2, GAE λ 0.95.

## Results (§5)
- BoN: up to ~30% (no noise) and ~75% (25% noise) improvement over the average single-RM result; no over-optimization with WCO and UWO; mean optimization over-optimizes with noise (Fig. 3). UWO shown with λ = 0.5.
- PPO without KL penalty: WCO and UWO reduce but do not eliminate over-optimization (Fig. 5). With KL penalty 0.01 they prevent it without notable performance loss; a single RM needs 0.2 and loses performance (Figs. 4, 6).
- Gains are orthogonal to RM size and data size (Figs. 8, 9).
- Intra-ensemble variance under PPO: rises almost 3× (no noise) and ~2.5× (25% noise) with mean optimization; ~20% with UWO (λ = 0.1) under noise (§5.5, Fig. 10).

## Limits (§6)
One environment and model family; offline RLHF with no RM refresh during policy optimization.
