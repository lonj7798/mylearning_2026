---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "Qwen3 Technical Report (arXiv:2505.09388v1)"
source_url: https://arxiv.org/abs/2505.09388
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the primary text; the earlier version reconstructed the chat template and misdescribed on-policy distillation)"
---

# Excerpt: Qwen3 — four-stage hybrid-thinking post-training, stage trade-offs, and strong-to-weak distillation

The library card [[qwen-3]] has not been verified. Every value below was read in arXiv:2505.09388v1 (2025-05-14).

## Pre-training and context (§2, §3.2, Tables 1-2, App. A.1.1)
- 36T tokens, 119 languages and dialects (§3.1). Stages: S1 general, over 30T tokens at 4,096; S2 reasoning, about 5T
  tokens at 4,096 with more STEM, coding, reasoning, and synthetic data; long-context stage, "hundreds of billions of
  tokens" at 32,768 with 75% of text 16,384-32,768 tokens and 25% 4,096-16,384; RoPE base 10,000 → 1,000,000 by ABF;
  YaRN and DCA for "a four-fold increase in sequence length capacity during inference" (§3.2).
- Context length in Tables 1-2: 32K for Qwen3-0.6B and 1.7B; 128K for 4B, 8B, 14B, 32B, 30B-A3B, 235B-A22B.
- RULER (App. A.1.1, Table 23) is run with YaRN scaling factor 4 and an 8,192-token thinking budget. Qwen3-8B average:
  89.1 non-thinking, 84.4 thinking; Qwen3-235B-A22B: 95.0 non-thinking, 92.2 thinking. The authors hypothesize that
  thinking content "may instead interfere with the retrieval process".
- Qwen3-8B model card (huggingface.co/Qwen/Qwen3-8B, "Processing Long Texts", read 2026-09-15): "Qwen3 natively supports
  context lengths of up to 32,768 tokens"; validated to 131,072 tokens with YaRN (`factor` 4.0). "All the notable
  open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input
  length, potentially impacting performance on shorter texts. We advise adding the rope_scaling configuration only when
  processing long contexts is required." Default `max_position_embeddings` is 40,960 (32,768 output + 8,192 prompt).

## Stage 1: long-CoT cold start (§4.1)
- Query filters with Qwen2.5-72B-Instruct: remove queries that are not easily verifiable (multiple sub-questions,
  general text generation); remove queries it answers correctly without CoT; annotate domains for balance.
- A validation query set is reserved. QwQ-32B generates N candidates per query; when it consistently fails, human
  annotators assess accuracy.
- Responses are removed if they (1) have incorrect final answers, (2) contain substantial repetition, (3) indicate
  guesswork without adequate reasoning, (4) show thinking/summary inconsistency, (5) mix languages or shift style, or
  (6) are suspected of being overly similar to validation items.
- "it is preferable to minimize both the number of training samples and the training steps during this preparatory
  phase." Sample counts and hyperparameters are not reported.

## Stage 2: reasoning RL (§4.2)
- 3,995 query-verifier pairs that were not used in cold start, are learnable, challenging, and broad; GRPO.
- Large batch, many rollouts per query, and off-policy training are described as beneficial; entropy is controlled to
  "increase steadily or remain stable". Values not reported.
- Qwen3-235B-A22B AIME'24: 70.1 → 85.1 over 170 RL steps.

## Stage 3: thinking-mode fusion (§4.3, Table 9)
- Continual SFT on the Stage-2 model. Thinking data: rejection sampling on Stage 1 queries with the Stage 2 model.
  Non-thinking data: coding, mathematics, instruction following, multilingual, creative writing, QA, role-play, checked
  with automatically generated checklists; more translation data for low-resource languages.
- Template (Table 9): user turn ends with `/think` or `/no_think`; the non-thinking assistant turn keeps an empty
  `<think>\n\n</think>` block. `/think` may be omitted because thinking is the default; in multi-turn data flags are
  inserted at random and the response follows the last flag.
- Thinking budget: when thinking reaches a user threshold, the stop-thinking instruction "Considering the limited time by
  the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n" is inserted. The report
  states this ability "is not explicitly trained but emerges naturally" from fusion.

## Stage 4: general RL (§4.4)
- Reward system over "over 20 distinct tasks": instruction following, format following (including `/think` and
  `/no_think`), preference alignment, agent ability, RAG. Agent RL: "the model is allowed to perform complete multi-turn
  interaction cycles with real environment execution feedback".
- Rewards: rule-based; model-based with reference answer (Qwen2.5-72B-Instruct scores, "avoiding false negatives");
  model-based without reference (reward model trained on human preference data).

## Stage effects on Qwen3-32B (§4.7, Table 22; thinking mode unless marked NT = non-thinking)
| Benchmark | Stage 2 | Stage 3 | Stage 3 NT | Stage 4 | Stage 4 NT |
|---|---|---|---|---|---|
| LiveBench 2024-11-25 | 68.6 | 70.9 | 57.1 | 74.9 | 59.8 |
| Arena-Hard | 86.8 | 89.4 | 88.5 | 93.8 | 92.8 |
| CounterFactQA* | 50.4 | 61.3 | 64.3 | 68.1 | 66.4 |
| IFEval strict prompt | 73.0 | 78.4 | 78.4 | 85.0 | 83.2 |
| Multi-IF | 61.4 | 64.6 | 65.2 | 73.0 | 70.7 |
| LengthCtrl* | 62.6 | 70.6 | 84.9 | 73.5 | 87.3 |
| ThinkFollow* | - | 88.7 | (shared) | 98.9 | (shared) |
| BFCL v3 | 69.0 | 68.4 | 61.5 | 70.3 | 63.0 |
| ToolUse* | 63.3 | 70.4 | 73.2 | 85.5 | 86.5 |
| MMLU-Redux | 91.4 | 91.0 | 86.7 | 90.9 | 85.7 |
| GPQA-Diamond | 68.8 | 69.0 | 50.4 | 68.4 | 54.6 |
| AIME'24 | 83.8 | 81.9 | 28.5 | 81.4 | 31.0 |
| LiveCodeBench v5 | 68.4 | 67.2 | 31.1 | 65.7 | 31.3 |

\* in-house benchmarks. Authors' conclusion (3): "for challenging tasks like AIME'24 and LiveCodeBench, the performance in
thinking mode actually decreases after these two training stages. We conjecture this degradation is due to the model
being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex
problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall
versatility." One score per stage; no variance reported. AIME scores are averages of 64 samples per question (§4.6).

## Strong-to-weak distillation (§4, §4.5, Table 21)
- Students: Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B. (1) Off-policy: teacher outputs in `/think` and `/no_think` modes
  as response distillation targets. (2) On-policy: "the student model produces responses in either /think or /no_think
  mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or
  Qwen3-235B-A22B) to minimize the KL divergence." The section describes no reward model for this phase.
- Qwen3-8B, math and code queries only, both runs start from the same off-policy distilled checkpoint (§4.7, Table 21):

| Method | AIME'24 (pass@64) | AIME'25 (pass@64) | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU hours |
|---|---|---|---|---|---|---|---|
| Off-policy distillation | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | - |
| + Reinforcement learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy distillation | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

- Not reported: KL direction and whether the KL is computed on full vocabularies or top-k logits, number of prompts,
  steps, learning rates, and the RL algorithm settings of the comparison run.

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2505.09388v1: §2-§4.7, Tables 1-2, 9, 21-23.
- Flag spelling: the PDF text layer drops underscores in the template font (it also renders `<|im_start|>` as
  `<|im start|>`); the glyph spacing matches an underscore, and the Qwen3-8B model card uses `/no_think`. This excerpt
  writes `/no_think` (checked 2026-09-15).
