---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2507.02259v2 (MemAgent, ICLR 2026), Abstract, §2.1-§2.2, §3.1-§3.3, Table 1, App. A.3 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2507.02259
created_at: "2026-09-15"
---

# Excerpt: MemAgent — Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

- **Authors:** Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, et al. (Institute for AI Industry Research (AIR), Tsinghua University; ByteDance Seed)
- **Year:** 2025 (arXiv v1 2025-07; this extract read from v2, 2026-07-29, ICLR 2026 version)
- **Source type:** paper
- **Used in:** ch-44a §5, Negative samples, Recipe.

## Workflow (§2.1)
The document is streamed in K chunks of length ≤ C. After reading chunk k the model overwrites the memory with a new
fixed-length memory `m_k`; the answer is produced from `m_K` and the question. "Because memory length never grows, the total
compute per chunk stays O(1) and end-to-end complexity is strictly linear to the number of chunks."

## Training window (§3.1)
"During training, we intentionally limit the model to an 8K context window to demonstrate its extrapolation capabilities.
This 8K window is allocated as follows: 1024 tokens for the query, 5000 tokens for the context chunk, 1024 tokens for the
memory, and 1024 tokens for the output", with the remainder for the chat template.

## Multi-Conv DAPO (§2.2, Eq. 1-2)
Each sample produces several independent conversations (one per chunk plus the answer conversation). The reward of the
answer conversation defines a single advantage that is applied to every conversation of that sample:
`Â_{i,j,t} = R_i − mean({R_i}_{i=1..G})`. "Following Dr. GRPO, we do not devide the advantage by the standard deviation."

## Training settings (App. A.3)
DAPO with KL factor 1 × 10⁻³ and entropy loss disabled; AdamW, constant learning rate 1 × 10⁻⁶, linear warm-up over 20
steps; rollout batch size 256, group size 16; off-policy training with a fixed 16:1 ratio of sample batch to
backpropagation batch. Stage I: 32,768 synthetic QA instances of about 32K tokens, built from HotpotQA with the RULER
method ("embedding golden paragraphs ... within extensive distractor content sampled from the same dataset"); convergence
takes about 400 steps. Stage II: 2,560 instances with a maximum length of 60K tokens from DocQA-RL-1.6K.

## Results (Table 1, RULER-HotpotQA accuracy %)
| Model | 7K | 112K | 896K | 1.75M | 3.5M |
|---|---|---|---|---|---|
| RL-MemAgent-14B | 80.47 | 81.25 | 75.78 | 78.91 | 71.09 |
| QwenLong-L1-32B | 72.66 | 31.25 | 11.72 | N/A | N/A |
| Qwen2.5-Instruct-14B-1M | 60.16 | 50.00 | 0.00 | N/A | N/A |
| DS-Distill-Qwen-32B | 70.31 | 23.44 | 7.03 | N/A | N/A |

Abstract: "able to extrapolate from an 8K context to a 3.5M QA task with a performance loss of less than 10% and achieving
over 95% on the 512K NIAH test". LongBench-QA averages (Table 3): RL-MemAgent-14B 51.0, QwenLong-L1-32B 50.7,
Qwen2.5-Instruct-14B 42.0.

## Not reported
An SFT baseline on memory-update trajectories, clip range, truncation behaviour when a memory update exceeds the 1024-token
output budget, and short-context benchmark scores for the trained models.
