---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "Qwen announcement of Qwen3-235B-A22B-2507 (X post) and the Instruct-2507 / Thinking-2507 model cards"
source_url: https://x.com/Alibaba_Qwen/status/1947344511988076547
created_at: "2026-09-15"
---

# Excerpt: Qwen3-2507 — separate Instruct and Thinking models replace one hybrid checkpoint

This excerpt stands in for the library card `qwen3-2507-instruct-thinking-split`, which did not exist when ch-34 was
written. Source type: official post and official model cards (Qwen Team). All three pages were read on 2026-09-15.

## Official statement (X post by @Alibaba_Qwen, 2025-07-21)
> "After talking with the community and thinking it through, we decided to stop using hybrid thinking mode. Instead,
> we'll train Instruct and Thinking models separately so we can get the best quality possible. Today, we're releasing
> Qwen3-235B-A22B-Instruct-2507 and its FP8 version for everyone."

The post gives no ablation, no benchmark comparison of hybrid versus separate training, and no training details.

## Model cards (huggingface.co, main branch)
- **Qwen3-235B-A22B-Instruct-2507** ("Model Overview"): "This model supports only non-thinking mode and does not
  generate `<think></think>` blocks in its output." Context length "262,144 natively and extendable up to
  1,010,000 tokens"; the 1M setting uses Dual Chunk Attention and MInference ("Processing Ultra-Long Texts").
- **Qwen3-235B-A22B-Thinking-2507** ("Model Overview"): "This model supports only thinking mode." Context length
  "262,144 natively". The default chat template inserts `<think>`. Recommended output length 32,768 tokens, and 81,920
  tokens for competition math and programming ("Best Practices").

## Benchmark table in the Instruct-2507 card ("Performance"; hybrid model in non-thinking mode → Instruct-2507)
| Benchmark | Qwen3-235B-A22B Non-thinking | Qwen3-235B-A22B-Instruct-2507 |
|---|---|---|
| MMLU-Pro | 75.2 | 83.0 |
| GPQA | 62.9 | 77.5 |
| SimpleQA | 12.2 | 54.3 |
| AIME25 | 24.7 | 70.3 |
| LiveCodeBench v6 (25.02-25.05) | 32.9 | 51.8 |
| IFEval | 83.2 | 88.7 |
| Arena-Hard v2 | 52.0 | 79.2 |
| BFCL-v3 | 68.0 | 70.9 |

The Instruct-2507 model differs from the original hybrid in more than the mode split (the card lists "long-tail
knowledge coverage" and 256K long-context improvements), so this table is not a controlled test of hybrid versus
separate training.

## Verification
- X post text read from the cached page on 2026-09-15.
- huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 and huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507 model cards
  read on 2026-09-15.
