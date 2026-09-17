---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/raft-reinforce-rej-minimalist.md on 2026-09-15)
source_url: https://arxiv.org/abs/2504.11343
source_version: arXiv v2 (2025-06-12); v1 2025-04-15
created_at: "2026-09-15"
---

# Excerpt: A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (Xiong, Yao, Xu, Pang, Wang, Sahoo, et al.; Salesforce AI Research and UIUC)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Algorithms (§3)
- RAFT keeps only the highest-reward responses per prompt and fine-tunes on them. RAFT++ adds the importance ratio and PPO-style clipping to RAFT (Eq. 6): `min(s_t(θ), clip(s_t(θ), 1−ε, 1+ε)) · 1[r(x,a) = max_i r(x,a_i)]`.
- Reinforce-Rej: policy gradient that filters out prompts whose sampled responses are entirely correct and prompts whose responses are entirely incorrect.

## Setup (§4)
- verl; prompt set Numina-Math (about 860k problems); Qwen2.5-Math-7B-base and LLaMA-3.2-3B-instruct; AdamW, learning rate 1e-6, 1,024 prompts per iteration, n = 4 responses per prompt for RAFT and GRPO, mini-batch 512, maximum generation 4,096 tokens.
- Metric: average@16 at temperature 1.0 on Math500, Minerva Math and Olympiad Bench.

## Results (Table 1, three-benchmark average)
- Qwen2.5-Math-7B-base: base 23.6; RAFT 52.3; RAFT++ 56.1; iterative DPO 48.8; Reinforce 53.9; GRPO 56.3; PPO 52.5; Reinforce-Rej 56.4.
- LLaMA-3.2-3B-instruct: base 13.1; RAFT 25.9; RAFT++ 27.6; Reinforce 24.2; GRPO 28.4; PPO 26.9; Reinforce-Rej 28.5.

## Entropy and the role of negatives (§5)
- Figure 3: RAFT++, which trains only on positive samples, shows "a much more rapid decline in policy entropy compared to GRPO" on both Qwen and LLaMA; once entropy stabilises at a low level, RAFT++'s improvement slows and GRPO overtakes it. KL from the initial policy grows faster for RAFT++ early in training.
- Adding the clip-higher trick of [[dapo]] to RAFT++ stabilises entropy and lets that variant outperform the original RAFT++ in later training (§5, Figure 2 right).
- Ablation on LLaMA-3.2-3B-instruct (Figure 4): "Reinforce + Remove all wrong" (drop prompts whose responses are all incorrect) gives the largest reward gain over vanilla Reinforce; "Remove all correct" alone "does not help much"; removing both gives better-behaved entropy and slightly better reward; mean-zero normalisation alone raises KL and does not improve reward; dividing by the per-prompt standard deviation adds little. The authors conclude that GRPO's benefit comes from rejecting low-quality samples rather than from reward normalisation.
- Authors' interpretation of the LLaMA result (Reinforce 24.2 vs RAFT++ 27.6, §5): "defining negative samples solely based on final answer correctness may be too coarse"; with coarse negative signals "unlearning on the negative samples is more unstable than fine-tuning on the positive samples".
