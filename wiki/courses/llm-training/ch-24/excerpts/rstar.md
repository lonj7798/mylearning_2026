---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rstar.md
source_url: https://arxiv.org/abs/2408.06195
created_at: "2026-09-15"
---

# Excerpt: rStar — MCTS with five reasoning actions and a second-model consistency check

**Checked on 2026-09-15 against arXiv:2408.06195v1.** The library card `papers/rstar.md` had no verification section at that date and contains the errors listed at the end; this excerpt is the checked extract used by ch-24 §5.

## Method (§3)

- Actions (§3.2): A1 propose a one-step thought; A2 propose the remaining thought steps; A3 propose the next sub-question with its answer; A4 answer the sub-question again with few-shot CoT; A5 rephrase the question or sub-question by listing its conditions. A4 follows A3; A5 follows the root.
- Reward (§3.2): no self-rewarding of intermediate nodes and no external tools or trained value models; the terminal node's score is the confidence of self-consistency majority voting, added to each node on the path.
- Selection: `UCT(s, a) = Q(s, a)/N(s, a) + c·sqrt(ln N_parent(s) / N(s, a))`.
- Mutual consistency (§3.3): mask the trajectory from a random step; a second model completes it from the earlier steps; trajectories whose answers agree are kept; the final trajectory maximizes reward × terminal confidence.

## Settings (§4.1)

32 rollouts; depth 8 for MATH, 5 otherwise; A1 and A3 up to 5 nodes per depth; discriminator Phi3-mini-4k (3.8B) for all generators (self-discrimination when Phi3 generates); split between 20% and 80% of steps.

## Results

- Table 2, GSM8K few-shot CoT → rStar: LLaMA2-7B 12.51 → 63.91; Mistral-7B 36.46 → 81.88; LLaMA3-8B 47.23 → 85.52; LLaMA3-8B-Instruct 74.53 → 91.13.
- StrategyQA (commonsense): LLaMA2-7B 58.82 → 67.25.
- Table 3, MATH-500: LLaMA3-8B-Instruct 17.80 → 42.94; Phi3-mini-4k 32.20 → 48.60.
- Table 1, action ablation (LLaMA3-8B, 200 GSM8K questions): A3 only 70.5; all actions 75.0.

## Errors in the earlier chapter and library card

- Action list "decompose / answer then verify / rephrase / propose new sub-question" in the wrong order and meaning; "discriminator is the same base model with a different prompt"; "12.5 → 63.1"; "Mistral-7B MATH 10.2 → 25.4" (MATH-500 is reported only for LLaMA3-8B-Instruct and Phi3-mini).
