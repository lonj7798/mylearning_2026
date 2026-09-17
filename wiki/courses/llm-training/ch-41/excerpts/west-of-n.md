---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2401.12086v2 (West-of-N: Synthetic Preferences for Self-Improving Reward Models); library card [[west-of-n]]
source_url: https://arxiv.org/abs/2401.12086
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library card has no Verification section)"
---

# Excerpt: West-of-N pseudo-preferences

Used by [[read]] the negatives section, the Recipe, and the Generalization lens. Authors: Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn (ETH Zürich, MPI-IS, Google Research). Checked against arXiv v2 (2024-10-25) on 2026-09-15.

Notes on the library card: the default is N = 64, not N = 16 (§5 "Methods"); the card's "argmax-vs-random loses 4 points", "≈20K seed pairs", "10K prompts", and "Table 1: +6% per iteration" were not found in v2 and are not used in the chapter.

## Method (§4-5)
- Sample N responses per unlabeled prompt from the SFT policy (temperature 0.7); a base preference model picks the best and worst; the pair becomes a pseudo-preference. Pairwise base models use a tournament to find best and worst.
- Self-trained RMs are trained pointwise on a 1:1 mixture of base (50% of human data, HF50%) and West-of-N preferences.
- Policies and RMs: T5-XXL (11B) for Reddit TL;DR and Anthropic HH; Gemma 2B for UltraFeedback.
- Optional filter: keep pairs with high `P_θ(y+ ≻ y− | x)` (§4, §5.2).

## Findings
- Doubling human data gives about a 1% RM accuracy increase (citing Stiennon et al.); West-of-N self-training gives up to about 2.3% (§5.1).
- N = 2 (label every pair) harms the RM because pseudo-labels are noisy; performance surpasses the base model as N grows; pseudo-preference accuracy increases with N, beyond high-confidence human data (§5.2, Fig. 4a-b).
- Confidence filtering gives further gains in accuracy and best-of-N win rate (§5.2, Fig. 5).
- West-of-N responses are orders of magnitude more likely under the policy than human-dataset responses; their likelihood decreases as N grows, which can make them out of distribution (§5.2, Fig. 4c).
- Base pairwise preference models: 72.5% (TL;DR), 66.9% (Anthropic Helpful), 71.4% (Harmless) test accuracy (§5.2).
