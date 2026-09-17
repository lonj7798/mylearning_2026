---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/small-models-learnability-gap.md on 2026-09-15)
source_url: https://arxiv.org/abs/2502.12143
source_version: arXiv v3 (2025-11-13); v1 2025-02
created_at: "2026-09-15"
---

# Excerpt: Small Models Struggle to Learn from Strong Reasoners (Li, Yue, Xu, Jiang, Niu, Lin, et al.; University of Washington, CMU, Western Washington University)

Facts used by [[read]], read in the arXiv v3 PDF text on 2026-09-15.

## Setup (§3.1)
- Prompts: the 7,500-problem MATH training set.
- Students: instruct versions of Qwen2.5-0.5B, 1.5B, 3B, 7B, 14B, 32B and Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Llama-3.3-70B.
- Long vs short CoT: QwQ-32B-Preview generates long CoT; Qwen2.5-32B-Instruct generates short CoT.
- Large vs small teacher: Qwen2.5-72B-Instruct vs Qwen2.5-3B-Instruct (also Llama3.1-70B vs 8B, Gemma2-27B vs 9B in App. B). Both generate short CoT; average lengths 432.98 (72B) and 440.70 (3B) tokens.
- Teachers use rejection sampling with greedy decoding. Students under 14B: full SFT, cosine schedule, max LR 1e-5, two epochs. Students above 14B: LoRA, LR 1e-4, two epochs. Evaluation: MATH, GSM8K, AMC 2023, AIME 2024, OlympiadBench (English math), zero-shot, greedy, 16k maximum generation tokens; score = average of five benchmarks.

## Definitions and results (§3.2-3.3)
- ∆Long = P_Long − P_Short, where P_Long and P_Short are the student's average score after SFT on long CoT and on short CoT.
- ∆Large = P_Large − P_Small, for SFT on large-teacher and small-teacher CoT.
- Table 1 (P_Long / P_Short / ∆Long): Qwen2.5-0.5B 14.8 / 19.5 / −4.7; 1.5B 27.0 / 34.2 / −7.1; 3B 40.3 / 43.4 / −3.1; 7B 48.9 / 47.2 / +1.7; 14B 59.2 / 54.3 / +4.9; 32B 73.0 / 59.3 / +13.7; Llama-3.2-1B −3.7; Llama-3.2-3B −0.6; Llama-3.1-8B +3.7; Llama-3.3-70B +3.8.
- Table 2 (∆Large): Qwen2.5-0.5B −3.5; 1.5B −0.8; 3B +0.3; 7B +6.6; 14B +3.0; 32B +6.5; Llama-3.2-1B −1.9.
- Takeaway 3: math-specialized small models (Qwen2.5-Math-1.5B-Instruct) show a smaller gap than general models of the same size (Fig. 4).

## Mix Distillation (Table 3)
- Mix-Long combines long CoT and short CoT in a 1:4 ratio; Mix-Large combines large-teacher and small-teacher responses in equal proportion.
- Qwen2.5-3B average: long CoT 40.3; short CoT 43.4; strong-model CoT 39.7; weak-model CoT 39.4; DeepSeek-R1-32B long CoT 33.5; Mix-Long 45.9; Mix-Large 45.8.
- Llama3.2-3B average: long CoT 32.5; short CoT 33.1; Mix-Long 35.1; Mix-Large 34.7.

## Limits of the evidence
Math only; one prompt set of 7,500 problems; greedy teacher decoding; no seed count is printed; LoRA above 14B changes the training method at the large end.
