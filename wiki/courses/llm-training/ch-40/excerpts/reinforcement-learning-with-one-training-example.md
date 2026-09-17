---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2504.20571v3 (Reinforcement Learning for Reasoning in Large Language Models with One Training Example); library card [[reinforcement-learning-with-one-training-example]]
source_url: https://arxiv.org/abs/2504.20571
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library card has no Verification section)"
---

# Excerpt: 1-shot RLVR, and how large the effect is per model family

Used by [[read]] §9 and the Generalization lens. Authors: Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, et al. Text read from arXiv v3 (24 Oct 2025) on 2026-09-15.

## Headline (Abstract)
Qwen2.5-Math-1.5B with one selected training example: MATH500 36.0% → 73.6% (the paper states 8.6% of improvement beyond format correction) and a six-benchmark math average of 17.6% → 35.7% (7.0% beyond format correction). The 1,209-example DeepScaleR subset that contains the example gives 73.6% / 35.9%; two examples give 74.8% / 36.6%.

## Setup (§3.1)
verl; GRPO by default (PPO also tested); binary outcome reward; KL coefficient β = 0.001 and entropy-loss coefficient α = −0.001; rollout temperature 0.6; training batch and mini-batch 128 with 8 responses per prompt (8 gradient updates per rollout step); max prompt 1024 and max response 3072 tokens. The one-shot dataset is duplicated to fill a batch. Example selection uses a historical-variance score of training accuracy.

## Cross-model results (Table 4, six-benchmark average)
| Model | no RL | full DSR-sub (1209) | 1 example | 2 examples ({π₁, π₁₃}) |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B (GRPO) | 17.6 | 35.9 | 35.7 | 36.6 |
| Llama-3.2-3B-Instruct (GRPO) | 17.5 | 19.8 | 19.0 | 21.0 |
| Qwen2.5-Math-1.5B (PPO) | 17.6 | 35.4 | 33.8 | — |
| DeepSeek-R1-Distill-Qwen-1.5B (GRPO, 32k eval) | 44.9 | 48.6 | 46.3 | — |

The format-reward baseline for Qwen2.5-Math-7B reaches an average of 34.3, which is why the paper reports gains "beyond format correction".

## Statements the chapter uses
- Post-saturation generalization: test accuracy keeps rising after training accuracy on the single example saturates (§3.2, Fig. 2).
- The entropy term matters: adding entropy loss improves the average by 2.3 points in their ablation, and too large a coefficient destabilizes training (§4, Table rows 5–6).
- The effect is smaller in absolute terms on Llama-3.2-3B-Instruct, and RLVR on that model is described as unstable (§3.3, App. C.1).

## Limits
Mathematics only; the absolute gains are concentrated in Qwen2.5-Math models, which [[spurious-rewards-rlvr]] shows respond to uninformative rewards as well. Use this result as evidence about data efficiency in one family, not as a general statement about RLVR data requirements.
