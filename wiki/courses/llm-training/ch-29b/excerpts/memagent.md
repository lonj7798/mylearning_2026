---
chapter: ch-29b
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/memagent.md (library card not present on 2026-09-15; content taken from the primary source)
source_url: https://arxiv.org/abs/2507.02259
primary_version: arXiv:2507.02259v2 (ICLR 2026 version; v1 2025-07)
created_at: "2026-09-15"
---

# Excerpt: MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

Authors: Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, et al. (Tsinghua AIR; ByteDance Seed). Checked against the v2 PDF on 2026-09-15.

## Workflow (§2.1, App. A.1 Table 5)
- The document is read in chunks; at each step the model sees the question, the previous memory, and one chunk, and writes a new memory that overwrites the old one. After the last chunk an answer call sees only the question and the memory.
- Context-processing template (Table 5, top): "You are presented with a problem, a section of an article that may contain the answer, and a previous memory. Please read the section carefully and update the memory with new information that helps to answer the problem, while retaining all relevant details from the previous memory. <problem> {prompt} </problem> <memory> {memory} </memory> <section> {chunk} </section> Updated memory:"
- Factorization (Eq. 4): p(x_{1:N}) = Σ_{m_{1:K−1}} Π_{k=1..K} p(c_k | m_{k−1}) · p(m_k | c_k, m_{k−1}), with chunks c_k of length ≤ C, memory m_k of fixed length M, and m_0 = ∅; per-step cost O(C + M), total O(N).
- 8K window during training: 1,024 tokens query, 5,000 chunk, 1,024 memory, 1,024 output, remainder for the chat template (§3.1).

## Training (§2.2, App. A.3)
- Multi-Conv DAPO: each sample produces several context-independent conversations; the reward of the final answer conversation gives the advantage Â = R_i − mean({R_i}) (no std division, following Dr. GRPO), applied to all conversations of that sample (Eq. 1-2). Reward R = 1 if the answer is equivalent to the ground truth (Eq. 3).
- KL factor 1e-3; entropy loss disabled; AdamW, constant LR 1e-6, linear warmup 20 steps; rollout batch 256, group size 16 (App. A.3).
- Stage I: 32,768 synthetic QA instances of about 32K tokens built from HotpotQA with the RULER method ("embedding golden paragraphs ... within extensive distractor content sampled from the same dataset"); about 400 steps to convergence. Stage II: 2,560 instances up to 60K tokens from DocQA-RL-1.6K mixed with stage-I data.
- Filter: "filtering out questions where Qwen2.5-7B-Base or Qwen2.5-7B-Instruct achieves 100% Best-Of-2 score without given any context"; about 50% of 80,000 processed HotpotQA samples were removed. Test sets use 50 to 6,400 wiki items, about 7K to 3.5M tokens.

## Results (Table 1, RULER-HotpotQA accuracy %)
| Model | 7K | 112K | 896K | 3.5M |
|---|---|---|---|---|
| RL-MemAgent-14B | 80.47 | 81.25 | 75.78 | 71.09 |
| RL-MemAgent-7B | 81.25 | 79.69 | 74.22 | 71.88 |
| Qwen2.5-Instruct-14B-1M | 60.16 | 50.00 | 0.00 | N/A |
| QwenLong-L1-32B | 72.66 | 31.25 | 11.72 | N/A |

- LongBench-QA average (Table 3): MemAgent-14B 51.0, QwenLong-L1-32B 50.7, Qwen2.5-Instruct-14B 42.0.
- Ablation (§3.3.1, Figs. 5-7): MemAgent without RL (same workflow, untrained) outperforms the backbones on RULER-HQA "however, it still exhibits a substantial decline in performance as the input length increases"; on LongBench-QA it gives "only marginal or even negative improvements". NIAH average at 512K: MemAgent-14B w/o RL 40.10 vs RL-MemAgent-14B 98.18 (Fig. 5).

## Not tested
- SFT on memory-update trajectories (for example rejection-sampled successful traces) is not compared with RL. Multi-session dialogue and tool-output histories are not evaluated.
