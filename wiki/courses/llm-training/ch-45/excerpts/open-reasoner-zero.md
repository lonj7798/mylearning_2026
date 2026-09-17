---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: primary source (no library card at time of writing)
source_url: https://arxiv.org/abs/2503.24290
version: arXiv v2, 2025-07-05
verified: 2026-09-15 (read against the cached PDF text of v2)
---

# Excerpt: Open-Reasoner-Zero — An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

**Authors:** Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, Heung-Yeung Shum (StepFun; Tsinghua University)
**Source type:** paper with released code, data and weights
**Used by:** [[read]] §7.3, Recipe

## Why this source is used in ch-45

It is an open reproduction of R1-Zero-style RL that reports (a) the minimal algorithmic setting that worked, (b) results against DeepSeek-R1-Zero-Qwen-32B on the same base model, and (c) a generality check outside the RL domain. The chapter uses it instead of the older multi-source card `r1-zero-analysis`, which mixes several papers and is pending re-verification.

## Method as stated

- Vanilla PPO with GAE λ = 1 and γ = 1, and **no KL regularization** of any kind (Abstract, §2.2).
- Reward: a rule that extracts the content between `<answer>` and `</answer>` and compares it with the reference answer; 1 for an exact match, 0 otherwise. **No format reward** — unlike DeepSeek-R1-Zero — because the base model adopts the template quickly under the prompt (§2.3, Fig. 4 left).
- Data: tens of thousands of curated question–answer pairs from AIME (up to 2023), MATH, Numina-Math, Tülu 3 MATH, OpenR1-Math-220k and AoPS, plus programmatically synthesized logic and multi-step reasoning tasks. Proof-style problems are excluded because the rule reward cannot evaluate them; LLM-based difficulty filtering removes extreme pass rates (§2.3).
- Training loop (App. B): 128 unique prompts per generation step, 64 responses per prompt at temperature 1.0 and top-p 1.0; strictly on-policy for the policy (one optimization step per generation); the critic processes 12 mini-batches per iteration; batch-level advantage normalization; AdamW β = (0.9, 0.95), no weight decay; policy learning rate 1e−6, critic 5e−6, constant with 50-step linear warm-up; sample packing.
- 32B annealing stage (App. B): 13k prompts on which the model produced fewer than 4 correct answers out of 64 attempts during the first 1,100 steps are used for 100 additional steps with the learning rate decayed to 3e−7.

## Numbers quoted in the chapter

| Model | AIME 2024 | AIME 2025 | MATH500 | GPQA Diamond | Locus |
|---|---|---|---|---|---|
| DeepSeek-R1-Zero-Qwen-32B | 47.0 | — | 91.6 | 55.0 | Table 1 (values from the R1 paper) |
| DAPO-Qwen-32B (authors' re-evaluation) | 48.3 | 37.9 | 71.8 | 16.0 | Table 1 |
| Open-Reasoner-Zero-32B | 48.1 | 36.0 | 92.2 | 55.5 | Table 1 |
| ORZ-0.5B / 1.5B / 7B | 1.0 / 3.5 / 17.9 | 0.2 / 1.0 / 15.6 | 31.0 / 58.0 / 81.4 | 12.1 / 16.8 / 36.6 | Table 4 |

Generality check (Table 2): MMLU 83.3 (Qwen2.5-32B base) → 84.9 (ORZ-32B); MMLU-Pro 55.1 → 74.4. Qwen2.5-32B-Instruct scores 83.2 and 69.2 on the same two benchmarks. The RL data is reasoning-only and no instruction tuning is applied.

Efficiency claim: the same base model as DeepSeek-R1-Zero-Qwen-32B with "only 1/10 of the training steps" (Abstract, §1).

Ablations (§3.2, Fig. 3): GAE λ = 1.0 keeps reward and length growing where λ = 0.95 collapses the length dynamics; adding a KL loss or a KL reward penalty slows training and reduces length scaling; the 57k ORZ question set keeps improving where MATH-train 7.5k plateaus.

Reflection measurement (§3.1, Fig. 4 right): five keywords ("wait", "recheck", "retry", "alternatively", "however") are used to label a response as reflective; the average length of correct reflective responses stays above the overall average response length throughout training.

## Not reported

Instruction-following, safety or open-ended-writing evaluations; total optimizer steps for the 32B run; hardware and wall-clock cost.
