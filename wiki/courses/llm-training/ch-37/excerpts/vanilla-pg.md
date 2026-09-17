---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/vanilla-pg.md (card unverified on 2026-09-15)
source_url: https://link.springer.com/article/10.1007/BF00992696
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; previous version replaced)"
---

# Excerpt: Williams (1992), REINFORCE

- **Artifact:** Ronald J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", Machine Learning 8, 229–256 (1992).
- **Source type:** paper
- **Used in:** ch-37 §1–§2 (log-derivative estimator, baseline).
- **Verification note:** the text layer of the openly hosted PDF could not be extracted on 2026-09-15, so no equation number or theorem number from the paper is quoted in ch-37. The chapter uses the estimator and the baseline identity as mathematical derivations, which the reader can check line by line in ch-37 §1–§2.

## What ch-37 takes from this source
- The REINFORCE family of updates for stochastic units: Δθ ∝ (r − b) · ∂ log π_θ(a) / ∂θ, where r is the reinforcement signal and b a reinforcement baseline that does not depend on the sampled action a.
- The property that such updates follow the gradient of expected reinforcement in expectation.

## What ch-37 does not take from the card
The card's baseline table (RLOO, GRPO, PPO rows) and its statement that all listed baselines are unbiased describe later work, not this paper. ch-37 §2 shows that GRPO's group statistics do not satisfy the action-independence condition.
