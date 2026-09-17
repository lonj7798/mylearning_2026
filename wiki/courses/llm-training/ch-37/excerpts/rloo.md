---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rloo.md (card unverified on 2026-09-15; items below checked against the arXiv:2402.14740v2 PDF)
source_url: https://arxiv.org/abs/2402.14740
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; previous version replaced)"
---

# Excerpt: Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO)

- **Authors:** Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, et al. (Cohere For AI, Cohere)
- **Year:** 2024 (arXiv v1 2024-02; v2 2024-02-26)
- **Source type:** paper
- **Used in:** ch-37 §2 (leave-one-out baseline), §3 (bias–variance argument), Recipe.

## Items used by ch-37, with loci
- **Sequence as one action (§2.2).** Because the reward is given only for the full generation, the paper models the entire generation as a single action and uses E[R(y, x) ∇ log π_θ(y|x)] (Eq. 6), with a baseline b (Eq. 7) and a moving-average baseline b_MA = (1/S) Σ_s R(x^s, y^s) (Eq. 8).
- **RLOO estimator (§2.3, from Kool et al. 2019).** (1/k) Σ_i [R(y^(i), x) − (1/(k−1)) Σ_{j≠i} R(y^(j), x)] ∇ log π(y^(i)|x), with y^(1..k) i.i.d. from π_θ(·|x). The remaining k − 1 samples give "an unbiased estimate of the expected return for the prompt".
- **Bias versus variance (§3.1, Fig. 1).** For PPO on Llama-7B with Anthropic-HH, training reward decreased monotonically as GAE λ was lowered from 1.0 to 0.95, 0.5, 0.0. The authors attribute this to the strong pre-trained initialization concentrating probability mass on a few tokens, so that added bias is not worth the variance reduction (Interpretation).
- **Win rates versus reference completions (Table 1, GPT-4 judge, held-out prompts).** TL;DR (Pythia-6.9B) / HH (Pythia-6.9B) / HH (Llama-7B): RLOO k = 4 77.9 / 43.7 / 64.1; RLOO k = 2 74.2 / 47.6 / 62.2; RAFT k = 4 73.2 / 42.1 / 63.3; REINFORCE with baseline 70.7 / 37.9 / 55.3; Vanilla PG 70.4 / 36.4 / 52.3; PPO 67.6 / 29.2 / 32.0; DPO 66.6 / 39.0 / 61.9.
- **Training settings (App. C).** TL;DR: 600 steps, rollout batch 512, step batch 256, β = 0.03. HH (Pythia): 393 steps, same batch sizes; HH (Llama): rollout and step batch 2048 over 2 epochs; β = 0.10 for HH. Constant LR 1 × 10⁻⁶ with 3% linear warm-up for all models, selected from sweeps; "For all algorithms, we take 2 gradient steps for each batch."

## Not used (card values not found in the paper)
"~50% memory footprint of PPO", "5–20% win rate at comparable KL", "KL coef β 0.05", "LR 1e-6 to 3e-6", "batch size 32–64", "epochs per rollout K = 1". The paper reports β = 0.03 / 0.10, LR 1 × 10⁻⁶, and 2 gradient steps per batch (App. C).
