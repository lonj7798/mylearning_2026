---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2505.09388v1 §4 (the library card qwen-3 has no Verification section and lacks these numbers; chapter-local verified extract)
source_url: https://arxiv.org/abs/2505.09388
created_at: "2026-09-15"
---

# Excerpt: Qwen3 Technical Report — post-training and strong-to-weak distillation

- **Authors:** Qwen Team
- **Year:** 2025 (arXiv v1 2025-05-14)
- **Source type:** official technical report
- **Used in:** ch-35 §5.1–5.4, Negative samples and negative feedback, Recipe, Generalization lens

## Pipeline (§4, Figure 1)
Flagship models (Qwen3-235B-A22B, Qwen3-32B): Stage 1 long-CoT cold start → Stage 2 reasoning RL → Stage 3 thinking mode fusion → Stage 4 general RL. Lightweight models (Qwen3-30B-A3B, 14B, 8B, 4B, 1.7B, 0.6B): strong-to-weak distillation from the flagship models. §4 states that distilling teacher logits gives higher Pass@1 and Pass@64 than the four-stage process for small models and needs "only 1/10 of the GPU hours".

## Stage 1 cold start (§4.1)
- Query filtering with Qwen2.5-72B-Instruct: remove queries that are not easily verifiable (multiple sub-questions, general text generation); remove queries Qwen2.5-72B-Instruct answers correctly without CoT; annotate domain for balance.
- After reserving a validation set, QwQ-32B generates N candidate responses per query (N not printed). When QwQ-32B consistently fails, human annotators assess accuracy.
- For queries with positive Pass@N, responses are removed if they (1) have incorrect final answers, (2) contain substantial repetition, (3) indicate guesswork, (4) have thinking and summary that disagree, (5) mix languages or shift style, (6) are suspected of being similar to validation items.
- "It is preferable to minimize both the number of training samples and the training steps during this preparatory phase."

## Stage 2 reasoning RL (§4.2)
3,995 query-verifier pairs (not used in cold start, learnable, challenging, broad); GRPO; Qwen3-235B-A22B AIME'24 70.1 → 85.1 over 170 steps.

## Stage 3 thinking mode fusion (§4.3)
Continual SFT on the reasoning-RL model. "The 'thinking' data is generated via rejection sampling on Stage 1 queries using the Stage 2 model itself." Non-thinking data covers coding, math, instruction following, multilingual tasks, creative writing, QA, role-play, checked with automatically generated checklists. Flags /think and /no_think; non-thinking responses keep an empty think block.

## Strong-to-weak distillation (§4.5)
1. "Off-policy Distillation: ... we combine the outputs of teacher models generated with both /think and /no_think modes for response distillation."
2. "On-policy Distillation: ... the student model produces responses in either /think or /no_think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence." The KL direction is not stated.

## Table 21 (Qwen3-8B, math and code queries only, from the same off-policy distilled checkpoint; pass@64 in parentheses)
| Method | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU hours |
|---|---|---|---|---|---|---|---|
| Off-policy distillation | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | – |
| + Reinforcement learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy distillation | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

## Table 22 (Qwen3-32B thinking mode: Stage 2 → Stage 3 → Stage 4)
IFEval strict prompt 73.0 → 78.4 → 85.0; Arena-Hard 86.8 → 89.4 → 93.8; ToolUse (in-house) 63.3 → 70.4 → 85.5; AIME'24 83.8 → 81.9 → 81.4; LiveCodeBench v5 68.4 → 67.2 → 65.7; GPQA-Diamond 68.8 → 69.0 → 68.4. The authors "choose to accept this performance trade-off to enhance the model's overall versatility" (§4.7).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2505.09388 (v1): §4 introduction, §4.1–4.5, §4.7, Tables 9, 21, 22.
- Not reported by the source: cold-start set size and N; SFT learning rates and epochs; on-policy distillation steps, sampling settings, and KL direction; which teacher was used for Qwen3-8B.
