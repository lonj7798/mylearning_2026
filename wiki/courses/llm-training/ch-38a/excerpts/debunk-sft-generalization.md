---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/debunk-sft-generalization.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2510.00237
created_at: "2026-09-15"
---

# Excerpt: Debunk the Myth of SFT Generalization

**Authors:** Xiaofeng Lin, Hejian Sang, Zhipeng Wang, Xuezhou Zhang (Boston University; LinkedIn).
**Version read:** arXiv:2510.00237v1 (30 Sep 2025).
**Status:** no library card existed for this slug on 2026-09-15; numbers read from Table 1 and App. B of the v1 PDF.

## Claim (Abstract, §5)
"much of SFT's perceived failure stems from frozen-prompt artifacts: when trained on fixed instruction templates, SFT models cling to training semantics rather than adapting to new ones." Prompt diversity addresses instruction variants; chain-of-thought (CoT) supervision addresses difficulty variants; the two together "match or surpass RL baselines on our benchmarks".

## Setup (§4, App. B)
Tasks: Sokoban (6×6 grid, one box in training) and General Points (the environment of [[sft-memorizes-rl-generalizes]], training mapping J = Q = K = 10). Backbones: Qwen2.5-7B and Llama-3.1-8B-Instruct. Answer-only Sokoban data: 3,981 BFS state-action pairs. General Points: 10,000 demonstrations sampled from the 800,000 of Chu et al. CoT data: 16 samples per query from a task-specific RL-finetuned Qwen3-8B, rejection-sampled on correctness. RL baseline is GRPO warm-started from a 10-step SFT checkpoint. "Fake" environments present variant instructions but score with the training semantics.

## Table 1a — Sokoban success rates (0–1)
| Model | Method | ID | Alpha. | Num. | Rand. | Large | TwoBoxes | Complex | Fake |
|---|---|---|---|---|---|---|---|---|---|
| Qwen | Ans. | 0.98 | 0 | 0 | 0 | 0.64 | 0.35 | 0.19 | 0.93 |
| Qwen | Diver. + Ans. | 0.91 | 0.92 | 0.89 | 0.84 | 0.53 | 0.33 | 0.09 | 0 |
| Qwen | CoT | 1 | 0.73 | 0.8 | 0.22 | 0.53 | 0.57 | 0.28 | 0.03 |
| Qwen | Diver. + CoT | 1 | 0.97 | 0.98 | 1 | 0.74 | 0.58 | 0.4 | 0 |
| Qwen | RL (warm) | 0.9 | 0.76 | 0.69 | 0.55 | 0.34 | 0.36 | 0.1 | 0 |
| Llama | Ans. | 0.92 | 0 | 0 | 0 | 0.53 | 0.3 | 0.08 | 0.9 |
| Llama | Diver. + CoT | 0.99 | 0.95 | 0.95 | 0.99 | 0.67 | 0.58 | 0.3 | 0 |
| Llama | RL (warm) | 0.43 | 0.18 | 0.28 | 0.1 | 0.18 | 0.05 | 0.02 | 0 |

## Table 1b — General Points success rates (0–1)
| Model | Method | ID | All-5 | All-7 | All-12 | Regular | Large | Five | Fake |
|---|---|---|---|---|---|---|---|---|---|
| Qwen | Ans. | 0.61 | 0.23 | 0.25 | 0.07 | 0.06 | 0.02 | 0.00 | 0.01 |
| Qwen | Diver. + Ans. | 0.78 | 0.71 | 0.70 | 0.09 | 0.07 | 0.03 | 0.01 | 0 |
| Qwen | CoT | 0.95 | 0.90 | 0.90 | 0.85 | 0.86 | 0.80 | 0.26 | 0 |
| Qwen | Diver. + CoT | 0.96 | 0.93 | 0.93 | 0.9 | 0.89 | 0.84 | 0.29 | 0 |
| Qwen | RL (warm) | 0.93 | 0.80 | 0.87 | 0.82 | 0.80 | 0.69 | 0.22 | 0.05 |
| Llama | Ans. | 0.71 | 0.00 | 0.00 | 0 | 0 | 0.02 | 0.01 | 0.68 |
| Llama | Diver. + CoT | 0.97 | 0.93 | 0.93 | 0.9 | 0.89 | 0.82 | 0.23 | 0 |
| Llama | RL (warm) | 0.92 | 0.87 | 0.83 | 0.72 | 0.76 | 0.65 | 0.37 | 0 |

Caption: "We report the best checkpoints under the average of these metrics except fake."

## Proximity controls (App. C)
SFT + α·KL(π_θ ‖ π_ref) and SFT + α‖θ − θ_ref‖² with α ∈ {0.05, 0.1, 0.5}, 5 epochs. Result: "adding KL/L2 regularization curbs the model's tendency to cling to the frozen training prompt: instruction-variant scores rise and 'Fake' success collapses. But the same constraint also limits task-specific adaptation … in-distribution accuracy plateaus or drops, and performance on difficulty variants … degrades."

## Hyperparameters (App. B.3, Tables 4–5)
SFT: LR 1e−5, batch 128, micro-batch 1 per GPU, max response 7000, 15 epochs, AdamW, weight decay 0.01, gradient clipping 1.0. RL (GRPO): LR 1e−6, batch 256, mini-batch 256, micro-batch 64 per GPU, max prompt 1000, max response 7000, KL coefficient 0.0, entropy coefficient 0.0, gradient clipping 1.0, 8 rollout workers, 100 total epochs, warm-started from a 10-step CoT SFT checkpoint because "the base models struggle to achieve positive rewards without any supervised fine-tuning".

## Stated limits (§7)
"Our study only evaluates two tasks and two backbones"; the evaluation is limited to decision-making tasks and does not cover open-ended or creative generation.

## How ch-38a uses it
§3 (the frozen-prompt account of Chu et al.'s SFT failures and the Fake-environment test), §3 (proximity controls as a partial fix), Common mistakes, Recipe.
