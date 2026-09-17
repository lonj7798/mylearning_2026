---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/limo.md
source_url: https://arxiv.org/abs/2502.03387
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: LIMO — small-set long-CoT SFT, version differences, and base-model dependence

**Checked on 2026-09-15 against arXiv:2502.03387v3 (COLM 2025) and v1 (2025-02-05).** The library card `papers/limo.md` had no verification section at that date; this excerpt is the checked extract used by ch-24 §4 and the Recipe.

## Version differences

| Version | Examples | AIME24 | MATH500 | Chain selection |
|---|---|---|---|---|
| v1 (2025-02) | 817 | 57.1 | 94.8 | "hybrid approach combining rule-based filtering and LLM-assisted curation" (§3.3.2) |
| v3 (2025-07, COLM 2025) | 800 | 63.3 | 95.6 | rule-based score (§3.1.2) |

## v3 pipeline (§3.1)

- Pool: NuminaMath-CoT, DeepScaleR (about 40,000 problems), AIME before 2024, MATH, and Chinese school and exam questions; tens of millions of problems.
- Difficulty: drop problems Qwen2.5-Math-7B-Instruct solves within 4 attempts; keep problems DeepSeek-R1-Distill-Qwen-32B solves 1–3 times in 32 → 2,125 problems (LIMO-Pool); n-gram deduplication against evaluation sets.
- Chains: sampled from DeepSeek R1, DeepSeek-R1-Distill-Qwen-32B, and QwQ-32B; scored by length (30%), verification words such as "check" (20%), tentative words such as "perhaps" (25%), connectives such as "therefore" (25%), normalized by length; best chain per problem; top 800 pairs.

## Training (v3 §4)

Qwen2.5-32B-Instruct; full fine-tuning with DeepSpeed ZeRO-3 and FlashAttention-2; all responses under 16,384 tokens (v1: "sequence length limit of 16,384 tokens"); LR 5.0e-6, cosine decay, no warmup; 15 epochs; batch 64.

## Results (v3)

- Table 1 (pass@1; AIME24, AMC23, CHMath use 4 samples at temperature 0.6, others greedy; 32,768 max output tokens, §5): base 16.5 AIME24 / 79.4 MATH500 / 48.0 GPQA / 49.9 average; NuminaMath-100k SFT 6.5 / 59.2 / 25.8 / 32.3; OpenThoughts-114k SFT 50.2 / 80.6 / 42.9 / 58.3; LIMO 63.3 / 95.6 / 70.7 / 78.1.
- Question difficulty (§6.3.2): Advanced-500 (AIME) data reaches 51.5 AIME24 and 91.2 MATH500.
- Base model (§6.3.3): LIMO data on Qwen1.5-32B-Chat gives 9.2 AIME24 vs 63.3 on Qwen2.5-32B-Instruct.
- Size (§6.3.4): AIME24 2.5 (3B) to 68.3 (72B); MATH500 95.6 (32B) vs 94.8 (72B).
- Data size (§6.3.5): LIMO-400 57.5 / 94.8; 800 → 1.2k +0.9 AIME24, −0.2 MATH500; 2k gives 69.6 / 95.8.

## Removed from the earlier excerpt

- "817 samples with 63.3 / 95.6" (version mix); "hand-filter removes subtly broken traces"; pool "GSM8K-hard, physics olympiad".
