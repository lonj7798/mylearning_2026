---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/consistentchat.md (library card not present on 2026-09-15; content taken from the primary source)
source_url: https://arxiv.org/abs/2506.03558
primary_version: arXiv:2506.03558v2 (2025-09; v1 2025-06)
created_at: "2026-09-15"
---

# Excerpt: ConsistentChat: Building Skeleton-Guided Consistent Multi-Turn Dialogues for Large Language Models from Scratch

Authors: Jiawei Chen, Xinyan Guan, Qianhao Yuan, Guozhao Mo, Weixiang Zhou, Yaojie Lu, et al. (Institute of Software, Chinese Academy of Sciences). Checked against the v2 PDF on 2026-09-15.

## Preliminary analysis (§2)
- ShareGPT (GPT-4 version) dialogues scored 1-10 for chat consistency by Qwen-2.5-72B-Instruct; High (8-10), Low (4-6), and a sample group, about 10,000 dialogues each; LLaMA-3.1-8B fine-tuned on High scores best on consistency and multi-turn metrics (§2.2, Fig. 3).

## Method (§3)
- Nine conversational intent types (from Rapp et al. 2021) with information flows form a skeleton. Qwen-2.5-72B-Instruct first generates the full query sequence q_1..q_T conditioned on intent I and flow F (Eq. 1), then generates all responses in a single pass with chain-of-thought (Eq. 2). The authors motivate single-pass responses by causal masking: turn-by-turn generation "cannot adjust response on future user queries".
- About 100 scenarios per intent, temperature 0.9, more than 15 dialogues per scenario; about 15,000 conversations and 224,392 utterances (§4.1).

## Training and evaluation (§4)
- SFT of Qwen-2.5-7B, LLaMA-3.1-8B, and Mistral-7B-v0.3 base models: LR 1e-5, cosine, 3 epochs, per-device batch 1, gradient accumulation 2, LLaMA-Factory. Baselines: about 15,000 dialogues with more than 3 turns sampled from ShareGPT, ChatAlpaca, UltraChat, LMSYS-Chat.
- Judges: Qwen-2.5-72B-Instruct and LLaMA-3.1-70B-Instruct (LIGHT, TopDial consistency); Qwen-2.5-72B-Instruct (MT-Eval). Human-judge correlation on 50 dialogues per benchmark: Spearman 0.70 (LIGHT), 0.66 (TopDial) (Table 4).
- Table 1 average consistency: Qwen-2.5-7B-ConsistentChat 7.32 vs ShareGPT 7.10, UltraChat 7.01; LLaMA-3.1-8B-ConsistentChat 6.93 vs UltraChat 6.67.
- Table 2 MT-Eval ST / MT: Qwen-2.5-7B-ConsistentChat 8.07 / 8.38; ShareGPT 7.81 / 7.86.
- Table 3 (Qwen-2.5-7B base → ConsistentChat SFT): HellaSwag 80.20 → 83.09; MATH 49.80 → 64.96; GPQA 36.40 → 35.35; MMLU 74.20 → 74.02; Gaokao 61.05 → 75.54; TriviaQA 64.93 → 67.31; HumanEval 57.90 → 77.44; average 60.64 → 68.24. LMSYS-Chat SFT average 37.44.

## Measurement caveats (course reading, not stated by the authors)
- The generator (Qwen-2.5-72B-Instruct) is also a judge, and the Qwen-2.5-7B student shares its model family; both can favour ConsistentChat outputs.
- Responses generated with access to future queries do not match inference, where the model cannot see later turns.
