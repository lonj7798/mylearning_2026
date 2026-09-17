---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/ngrpo.md on 2026-09-15)
source_url: https://arxiv.org/abs/2509.18851
source_version: arXiv v1 (2025-09-23)
created_at: "2026-09-15"
---

# Excerpt: NGRPO — Negative-enhanced Group Relative Policy Optimization (Nan, Chen, Huang, Lu, Wang, Xie, et al.; Ant Group)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15.

## Problem
- In GRPO a group whose responses are all incorrect has zero advantage for every response and therefore contributes no gradient; the same holds for all-correct groups. NGRPO turns homogeneous-incorrect groups into a learning signal (§1, §4).

## Advantage calibration (§4.1, Eqs. 5-6)
- The group's reward set R = {r_1, …, r_G} is augmented with one virtual sample whose reward is the maximum possible r_max; the virtual sample is never generated, only its reward enters the statistics. Advantages are standardized over the augmented set: `A'_i = (r_i − µ'_R)/(σ'_R + ε_std)`.
- Consequences stated: the baseline is always above the observed rewards, so an all-incorrect group receives a uniform negative advantage, and the sum of advantages in a group is negative, which "effectively increas[es] policy entropy and encourag[es] exploration" (§4.2).
- Figure 3 (G = 8) prints three cases: all-incorrect (GRPO 0.00 for every sample, NGRPO a uniform negative advantage); a group with one correct response (GRPO +2.47 for the correct and −0.35 for each incorrect; NGRPO +1.76 and −0.50); a group with seven correct responses (adjustments described as minimal, with a moderately larger penalty on the single failure).

## Asymmetric clipping (§4.2, Eq. 8)
- Separate clip bounds by advantage sign: `min(ρ_t A'_i, (1+ε_pos) A'_i)` for A'_i ≥ 0 and `max(ρ_t A'_i, (1−ε_neg) A'_i)` for A'_i < 0, with ε_neg = 0.16 and ε_pos = 0.24 ("a tighter constraint on negative updates … while allowing more latitude for positive updates", following Yu et al. 2025).

## Setup and results
- Qwen2.5-Math-7B trained for 20 epochs on MATH, 8 × H100, learning rate 1e-6, global batch 1024, maximum prompt 1024 tokens and maximum response 3072 tokens (§5.1.1). Evaluation uses the unbiased pass@k estimator with N = 256 samples per problem at temperature 0.6 and top-p 0.95, summarised as the pass@k AUC over k ∈ {1, …, 256} (§5.1.2).
- Table 1 (pass@k AUC): AIME2025 — PPO 27.17, GRPO 28.33, PSR-NSR 28.85, DAPO 30.27, NGRPO 31.28. AMC — PPO 83.24, GRPO 81.04, PSR-NSR 85.04, DAPO 83.40, NGRPO 86.09. AIME2025 pass@256: GRPO 53.33, DAPO 53.33, NGRPO 60.00.
- Ablation (Table 2, pass@k AUC on AIME2025 / AMC): GRPO baseline 28.33 / 81.04; asymmetric clipping only 28.48 / 81.22; advantage calibration only 29.56 / 83.85; calibration + asymmetric clipping 30.54 / 84.37; full NGRPO, which also keeps homogeneous-incorrect groups, 31.28 / 86.09.
- §6/Appendix: adding more than one virtual reward "leads to excessive and unproductive exploration, degrading performance".
