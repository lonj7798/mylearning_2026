---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/learning-dynamics-llm-finetuning.md on 2026-09-15)
source_url: https://arxiv.org/abs/2407.10490
source_version: arXiv v4 (2025-06-29); v1 2024-07-15; ICLR 2025
created_at: "2026-09-15"
---

# Excerpt: Learning Dynamics of LLM Finetuning (Ren, Sutherland; UBC and Amii)

Facts used by [[read]], read in the arXiv v4 PDF on 2026-09-15.

## Framework (§2, Proposition 1)
- One gradient step on (x_u, y_u) changes the prediction on any x_o by
  `Δ log π^t(y | x_o) = −η · A^t(x_o) · K^t(x_o, x_u) · G^t(x_u, y_u) + O(η² ‖∇_θ z(x_u)‖²_op)`,
  with `A^t(x_o) = I − 1 π_θ^t(x_o)ᵀ` (the softmax Jacobian term), `K^t` the empirical neural tangent kernel of the logit network, and `G^t = ∇_z L(x_u, y_u)`.

## Squeezing effect (§3.3)
For a negative gradient on label y⁻_u with a softmax output head, the paper states (L = 1 case):
- Guarantee: π(y⁻_u) decreases.
- Guarantee: the decreased mass is "largely squeezed" into the label that was most confident before the update, y* = argmax_{i ≠ y⁻_u} π^t(i).
- Trend: "the rich get richer and the poor get poorer": dimensions with high π^t tend to increase and those with low π^t tend to decrease.
- Trend: "peakier π^t squeezes more": if the mass is concentrated on a few dimensions, all π^{t+1}(y ≠ y*) decrease.
- Trend: "smaller π^t(y⁻_u) exacerbate the squeezing effect": if y⁻_u is unlikely, all other π^{t+1}(y ≠ y*) fall more and π^{t+1}(y*) rises more.
- App. E proves Claims 1-5 for multi-class logistic regression by computing π^{t+1}/π^t (Claim 3C: for a very peaky p^t, all p_i other than the max decrease; Claim 4: a smaller p_y makes the effect stronger; Claim 5: larger |η| and larger feature norm amplify it).
- §3.3 states the effect "can become more complicated" when positive and negative pressures and the autoregressive structure are all considered, and that Razin et al. (2025) analyse a similar problem at token level with conclusions that "align with ours well".

## Off-policy DPO experiment (§4.2, Figure 4)
- Setup (§4): 5,000 training examples from Anthropic-HH and from UltraFeedback; models pythia-410M/1B/1.4B/2.8B and Qwen1.5-0.5B/1.8B; a 500-example probing set with generated response types (rephrases of y⁺, y⁻, answers to other questions, random sentences).
- During off-policy DPO the log-probability of almost every probed response falls, while the log-probability of the greedy-decoded ("teacher forcing") response rises from about −113 to −63 within 8 epochs, which is faster than the rise of π(y⁺) during SFT (about −130 to −90).
- §4.2 interpretation: as π becomes peakier at its most confident predictions, repeated phrases are easier to sample, which the authors connect to the degeneration reported by Holtzman et al. (2020).

## "Extend" mitigation (§4.3, App. F)
- Method: add (x, y⁻_u) pairs to the SFT dataset so that y⁻ is no longer in a low-probability region before DPO; all other settings unchanged.
- Result (App. F.3): Qwen1.5-1.8B, Anthropic-HH subset of 5,000 examples, SFT 2 epochs then identical DPO. Win rate of the extend pipeline against the baseline pipeline, judged by GPT-3.5-Turbo and Claude3-Haiku on 1,000 test prompts: after SFT 0.4729 / 0.4679; after 2 DPO epochs 0.6518 / 0.5151; after 4 epochs 0.6928 / 0.6045; after 6 epochs 0.6667 / 0.5432 (Table 1).
- Learning-dynamics evidence (Figure 5): with the extend pipeline, other responses decay more slowly during DPO and the greedy-decoding log-probability rises more slowly.
