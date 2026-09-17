---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2405.19534
created_at: "2026-09-15"
verified_against: "arXiv:2405.19534v4 (31 Oct 2024), cached plain text"
---

# Preference Learning Algorithms Do Not Learn Preference Rankings

Angelica Chen, Sadhika Malladi, Lily H. Zhang, Xinyi Chen, Qiuyi Zhang, Rajesh Ranganath, Kyunghyun Cho.

## Definitions
- **Ranking accuracy (Def. 2.3).** `R(x, y_w, y_l; π_θ) = 1[π_θ(y_w|x) ≥ π_θ(y_l|x)]`, averaged over a
  dataset. `R̃` denotes the same quantity on length-normalised likelihoods. It is not the reward accuracy
  of RewardBench and not the DPO implicit-reward accuracy (Remark 2.5).

## Claims the chapter uses
- **Measured ranking accuracy (Table 1, AlpacaFarm validation split).** Length-normalised `R̃` /
  non-normalised `R`: Zephyr-7B-DPO 54% / 42%; Tülu-2-DPO-7B 53% / 42%; Gemma-7B-IT 54% / 40%;
  Llama-2-7B-Chat-HF 53% / 40%. The idealised ranking accuracy computed from Corollary 3.3 over a range
  of β has medians 97–99% for those models (73%/93% for Gemma), so the measured gap is 19 to 59 points.
- **Rankings rarely flip (§4.1, Fig. 2).** Training Pythia-2.8B with DPO on Anthropic HH for 5 epochs:
  at the point of lowest validation loss, fewer than 10% of the initially incorrectly ranked training
  examples have been flipped to the correct ranking, while the loss falls and the reward margin rises.
- **Why (Theorem 4.1).** For a pair whose reference log-ratio is `log π_ref(y_l|x)/π_ref(y_w|x) = c`,
  the model ranks the pair correctly if and only if `L_DPO(x, y_w, y_l) ≤ −log σ(βc)`. Pairs the reference
  model ranks wrongly (large `c`) require the per-pair loss to fall to a very small value before the
  ranking flips.
- **Relation to win rate (§5).** Ranking accuracy and GPT-4 win rate (AlpacaEval, 500 training prompts)
  move together early in training and become anti-correlated once the policy moves away from the
  reference model; with the reference-attenuating objective `L^γ_DPO` they trend together more strongly
  for larger γ.

## Conditions and limits
- Training experiments use GPT-2, Pythia-2.8B and Llama-2-7B on Anthropic HH; measurements are on the
  training distribution, which the authors name as a limitation (§7).
