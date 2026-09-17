---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: "Qwen announcement of Qwen3-235B-A22B-2507 (X post) and the Instruct-2507 / Thinking-2507 model cards"
source_url: https://x.com/Alibaba_Qwen/status/1947344511988076547
created_at: "2026-09-15"
---

# Excerpt: Qwen3-2507 — separate Instruct and Thinking models replace one hybrid checkpoint

Chapter-local extract; no library card existed when ch-44b was written. Source type: official post and official model cards (Qwen Team), read on 2026-09-15.

## Official statement (X post by @Alibaba_Qwen, 2025-07-21)
> "After talking with the community and thinking it through, we decided to stop using hybrid thinking mode. Instead, we'll train Instruct and Thinking models separately so we can get the best quality possible. Today, we're releasing Qwen3-235B-A22B-Instruct-2507 and its FP8 version for everyone."

The post gives no ablation and no comparison of hybrid versus separate training.

## Model cards (huggingface.co)
- Qwen3-235B-A22B-Instruct-2507: "This model supports only non-thinking mode and does not generate `<think></think>` blocks in its output."
- Qwen3-235B-A22B-Thinking-2507: "This model supports only thinking mode."

## Benchmark tables in the model cards (hybrid checkpoint in the matching mode → 2507 checkpoint)
| Benchmark | Hybrid non-thinking | Instruct-2507 | Hybrid thinking | Thinking-2507 |
|---|---|---|---|---|
| MMLU-Pro | 75.2 | 83.0 | 82.8 | 84.4 |
| AIME25 | 24.7 | 70.3 | 81.5 | 92.3 |
| IFEval | 83.2 | 88.7 | 83.4 | 87.8 |
| Arena-Hard v2 | 52.0 | 79.2 | 61.5 | 79.7 |
| SimpleQA | 12.2 | 54.3 | — | — |

The 2507 checkpoints differ from the hybrid release in more than the mode split (the Instruct card lists long-tail knowledge coverage and 256K long-context improvements), so these tables are not a controlled test of hybrid versus separate training.

## Verification
- X post text and both model cards read on 2026-09-15 from cached copies of the pages.
