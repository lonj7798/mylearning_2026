---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/dapo.md on 2026-09-15)
source_url: https://arxiv.org/abs/2503.14476
source_version: arXiv v2 (2025-05-20); v1 2025-03-18
created_at: "2026-09-15"
---

# Excerpt: DAPO — An Open-Source LLM Reinforcement Learning System at Scale (ByteDance Seed and AIR, Tsinghua)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15. Only the parts this chapter uses are recorded; the algorithm as a whole is taught in [[ch-40]].

## Clip-Higher (§3.1, Eq. 10)
- The clip range is decoupled into ε_low and ε_high. The motivation is stated for positive advantages: with ε = 0.2 and π_old = 0.01 the updated probability is bounded by 0.012, while for π_old = 0.9 the bound is 1.08, so low-probability "exploration" tokens cannot be raised much; empirically "the mean probability of up-clipped tokens is low: π_θ(o_i|q) < 0.2" (Figure 3a).
- ε_low is left unchanged, "because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space" (§3.1). Run values: ε_low = 0.2, ε_high = 0.28 (§4.1).

## Dynamic sampling (§3.2, Eq. 11)
- Prompts whose sampled responses are all correct or all incorrect give zero advantage and therefore no gradient; the share of all-correct prompts grows during training (Figure 3b). DAPO oversamples and filters so that every prompt in a batch has `0 < |{o_i : is_equivalent(a, o_i)}| < G`.

## Overlong reward shaping (§3.4, Eq. 13)
- By default truncated samples get a punitive reward, which "may introduce noise into the training process, as a sound reasoning process can be penalized solely due to its excessive length".
- Overlong Filtering masks the loss of truncated samples and "significantly stabilizes training and enhances performance" (Figure 5).
- Soft Overlong Punishment: 0 for |y| ≤ L_max − L_cache, `((L_max − L_cache) − |y|)/L_cache` for lengths between L_max − L_cache and L_max, and −1 beyond L_max. Run values: L_max − L_cache = 16,384 with L_cache = 4,096, so generation is capped at 20,480 tokens (§4.1).

## Setup and ablation (§4.1-4.2, Table 1)
- Qwen2.5-32B base, verl, AdamW at a constant 1e-6 with 20 rollout steps of linear warm-up, prompt batch 512 with 16 responses per prompt, mini-batch 512 (16 gradient updates per rollout step); AIME 2024 evaluated as avg@32 at temperature 1.0 and top-p 0.7.
- Progressive ablation on AIME 2024 (avg@32): naive GRPO 30; + overlong filtering 36; + clip-higher 38; + soft overlong punishment 41; + token-level loss 42; + dynamic sampling (full DAPO) 50. DeepSeek-R1-Zero-Qwen-32B is listed at 47.
