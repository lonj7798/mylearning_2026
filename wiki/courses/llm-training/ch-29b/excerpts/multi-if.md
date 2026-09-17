---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/multi-if.md (library card not present on 2026-09-15; content taken from the primary source)
source_url: https://arxiv.org/abs/2410.15553
primary_version: arXiv:2410.15553v2 (2024-11; v1 2024-10)
created_at: "2026-09-15"
---

# Excerpt: Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following

Authors: Yun He, Di Jin, Chaoqi Wang, Chloe Bi, Karishma Mandyam, Hejia Zhang, et al. (Meta GenAI). Checked against the v2 PDF on 2026-09-15.

## Construction (§3)
- Turn 1 is the original IFEval prompt. For each later turn, one of the 30 IFEval instruction types is sampled at random, and Llama 3.1 405B writes a natural-language user prompt for it given all previous turns (§3.1).
- Conflicts across turns (for example a short summary in turn 1 and at least 800 words in turn 2) are detected by Llama 3.1 405B and then filtered by human verification (§3.2).
- Llama 3.1 405B translates into French, Hindi, Italian, Portuguese, Chinese, Spanish, and Russian; annotators review, and "15% translations get rewritten in average" (§3.3). Sensitive prompts are flagged by an LLM and removed by annotators (§3.4).
- 4,501 conversations, three turns each (§3.5).

## Metrics (§4.1)
- The model sees the concatenation of earlier prompts and its own earlier responses.
- Final metric = mean of instruction-level strict, conversation-level strict, instruction-level loose, and conversation-level loose accuracy. Conversation-level accuracy requires every instruction from turn 1 to the current turn to be followed.
- Instruction Forgetting Ratio (§5.2, Eq. 1): IFR = (number of previously followed instructions not followed in subsequent turns) / (total number of instructions followed in previous turn) × 100.

## Results (Table 1)
- Average over languages, turn 1 → turn 3: o1-preview 0.877 → 0.707; Llama 3.1 405B 0.854 → 0.707; GPT-4o 0.843 → 0.631; Qwen-2.5 72B 0.837 → 0.609; Llama 3.1 8B 0.688 → 0.542.
- "All the models tested showed a higher rate of failure in executing instructions correctly with each additional turn" (Abstract). Non-Latin-script languages (Hindi, Russian, Chinese) show higher error rates (Abstract; §4.2).
- IFR decreases with Llama 3.1 size from 8B to 405B; Gemini 1.5 models have the highest IFR, "partly but not mainly due to the false refusals" (§5.2, Fig. 7; §4.2 notes Gemini refusing requests such as capitalizing responses).

## Limits
- Three turns only; constraints are IFEval's verifiable types, which are format and lexical constraints rather than task requirements. No training experiment.
