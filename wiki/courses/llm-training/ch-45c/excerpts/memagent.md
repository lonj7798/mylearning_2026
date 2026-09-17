---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: arXiv:2507.02259v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2507.02259
created_at: "2026-09-15"
---

# Excerpt: MemAgent — Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

**Authors:** Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou (AIR, Tsinghua University; ByteDance Seed). arXiv v1 2025-07; version read v2 2026-07-29 (published at ICLR 2026). Source type: paper.
**Used in:** [[read]] §5, §8.

## Mechanism (§2, §3)
- The input document is "streamed through the model in K contiguous chunks c₁, …, c_K (each of length ≤ C). After chunk k is read, the model overwrites the panel with a new vector m_k", where the memory is a fixed-length token segment, not a hidden state.
- Overwrite, not append: "Because memory length never grows, the total compute per chunk stays O(1) and end-to-end complexity is strictly linear to the number of chunks."
- Training: "we extend the DAPO algorithm to directly optimize memory ability in an end-to-end fashion, facilitating training via independent-context multi-conversation generation" (Multi-Conv DAPO). Each sample generates multiple conversations; the answer in the final conversation supplies the reward for all of them.

## Training context budget (§4)
"During training, we intentionally limit the model to an 8K context window to demonstrate its extrapolation capabilities. This 8K window is allocated as follows: 1024 tokens for the query, 5000 tokens for the context chunk, 1024 tokens for the memory, and 1024 tokens for the output, with the remaining tokens reserved for the chat template." Training documents are 60K tokens long; training runs in two stages (stage I acquires memory behaviour, stage II transfers it to more diverse contexts).

## Table 1 — RULER-HQA accuracy (%) by document length

| Model | 7K | 14K | 28K | 56K | 112K | 224K | 448K | 896K | 1.75M | 3.5M |
|---|---|---|---|---|---|---|---|---|---|---|
| QwenLong-L1-32B | 72.66 | 75.00 | 72.66 | 60.94 | 31.25 | 17.19 | 13.28 | 11.72 | N/A | N/A |
| Qwen2.5-Instruct-14B-1M | 60.16 | 60.94 | 50.00 | 57.03 | 50.00 | 37.50 | 8.59 | 0.00 | N/A | N/A |
| Qwen2.5-Instruct-7B-1M | 61.72 | 56.25 | 53.91 | 55.47 | 51.56 | 33.59 | 12.50 | 0.00 | N/A | N/A |
| DS-Distill-Qwen-32B | 70.31 | 66.41 | 65.62 | 46.88 | 23.44 | 13.28 | 7.81 | 7.03 | N/A | N/A |
| RL-MemAgent-14B | 80.47 | 82.03 | 82.03 | 83.59 | 81.25 | 77.34 | 79.69 | 75.78 | 78.91 | 71.09 |
| RL-MemAgent-7B | 81.25 | 81.25 | 82.03 | 80.47 | 79.69 | 75.78 | 76.56 | 74.22 | 77.34 | 71.88 |

Abstract claim matching this table: "being able to extrapolate from an 8K context to a 3.5M QA task with a performance loss of less than 10% and achieving over 95% on the 512K NIAH test."

## Other benchmarks (§4)
RULER-HQA (synthetic, moderate information density, controllable length), LongBench-QA (NarrativeQA, Qasper, HotpotQA, 2WikiMultihopQA, MuSiQue — short but information-dense), and NIAH (very low information density).

## Not reported in the extract used here
Per-chunk latency, the memory prompt template token cost in evaluation, and any agentic tool-use evaluation — the setting is document reading, not tool calling.

## Verification
- Read on 2026-09-15 against the cached PDF text of arXiv:2507.02259v2 (Abstract, §1–§2.2, §3, §4, Table 1, Figure 1).
