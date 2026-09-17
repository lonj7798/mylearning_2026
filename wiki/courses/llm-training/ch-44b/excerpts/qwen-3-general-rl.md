---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: "Qwen3 Technical Report (arXiv:2505.09388v1), §4.2, §4.4, §4.5, §4.7"
source_url: https://arxiv.org/abs/2505.09388
created_at: "2026-09-15"
---

# Excerpt: Qwen3 — general RL over 20+ tasks, stage effects, and on-policy distillation

The library card [[qwen-3]] carries no Verification section, so every value below was read in the cached full text of arXiv:2505.09388v1 (2025-05-14) on 2026-09-15.

## Reasoning RL (§4.2)
3,995 query-verifier pairs that are unused in cold start, learnable, challenging, and broad; GRPO. Qwen3-235B-A22B AIME'24 rises from 70.1 to 85.1 over 170 RL steps.

## General RL (§4.4)
- "a sophisticated reward system covering over 20 distinct tasks, each with customized scoring criteria", targeting instruction following, format following (including `/think` and `/no_think` switching), preference alignment, agent ability (multi-turn interaction with real environment execution feedback), and specialized scenarios such as RAG.
- Three reward types: (1) rule-based rewards, which "assess the correctness of model outputs with high precision, preventing issues like reward hacking"; (2) model-based rewards with a reference answer, scored by Qwen2.5-72B-Instruct, "avoiding false negatives that can occur with purely rule-based rewards"; (3) model-based rewards without a reference, from a reward model trained on human preference data, which "can handle a broader range of queries while effectively enhancing the model's engagement and helpfulness".
- Mixing weights, prompt counts, and RL hyperparameters for this stage are not reported.

## Stage effects on Qwen3-32B (§4.7, Table 22; thinking mode unless marked NT)
| Benchmark | Stage 2 (Reasoning RL) | Stage 3 (Thinking-Mode Fusion) | Stage 4 (General RL) | Stage 4 NT |
|---|---|---|---|---|
| LiveBench 2024-11-25 | 68.6 | 70.9 | 74.9 | 59.8 |
| Arena-Hard | 86.8 | 89.4 | 93.8 | 92.8 |
| CounterFactQA* | 50.4 | 61.3 | 68.1 | 66.4 |
| IFEval strict prompt | 73.0 | 78.4 | 85.0 | 83.2 |
| Multi-IF | 61.4 | 64.6 | 73.0 | 70.7 |
| ThinkFollow* | – | 88.7 | 98.9 | (shared) |
| BFCL v3 | 69.0 | 68.4 | 70.3 | 63.0 |
| ToolUse* | 63.3 | 70.4 | 85.5 | 86.5 |
| MMLU-Redux | 91.4 | 91.0 | 90.9 | 85.7 |
| GPQA-Diamond | 68.8 | 69.0 | 68.4 | 54.6 |
| AIME'24 | 83.8 | 81.9 | 81.4 | 31.0 |
| LiveCodeBench v5 | 68.4 | 67.2 | 65.7 | 31.3 |

\* in-house benchmarks. Authors' conclusion (3): "for challenging tasks like AIME'24 and LiveCodeBench, the performance in thinking mode actually decreases after these two training stages. We conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall versatility." One score per stage; no variance reported. AIME scores average 64 samples per question (§4.6).

## Strong-to-weak distillation (§4.5, §4.7, Table 21; Qwen3-8B, math and code queries, both runs from the same off-policy distilled checkpoint)
| Method | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU hours |
|---|---|---|---|---|---|---|---|
| Off-policy distillation | 55.0 | 42.8 | 92.4 | 42.0 | 86.4 | 55.6 | – |
| + Reinforcement learning | 67.6 | 55.5 | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy distillation | 74.4 | 65.5 | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |
On-policy distillation aligns the student's logits with a teacher's (Qwen3-32B or Qwen3-235B-A22B) by minimizing KL divergence on student-generated responses (§4.5).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2505.09388v1: §4.2, §4.4, §4.5, §4.6, §4.7, Tables 21–22.
- Not reported: General RL prompt counts, reward weights, steps, and algorithm settings; KL direction and vocabulary coverage for on-policy distillation; the RL settings of the comparison run in Table 21.
