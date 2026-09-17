---
chapter: ch-32c
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ruler.md (library card has no Verification section and no Table 3 values on 2026-09-15)
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-09-15"
---

# Excerpt: RULER: What's the Real Context Size of Your Long-Context Language Models?

**Authors:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, et al. (NVIDIA)
**Version read:** arXiv:2404.06654v3 (6 Aug 2024; COLM 2024), PDF text. Leaderboard: github.com/NVIDIA/RULER README.md at commit ab17b78 (2025-10-09).
**Source type:** paper; the README leaderboard is released code/results from the same authors.

## Protocol (§4; App. D)
- 13 tasks in four categories: retrieval (S-NIAH, MK-NIAH, MV-NIAH, MQ-NIAH), multi-hop tracing (variable tracking, VT), aggregation (common-words and frequent-words extraction, CWE and FWE), and QA (SQuAD, HotpotQA with distractor paragraphs) (§3; Table 2).
- 500 generated examples per task per length; lengths 4K, 8K, 16K, 32K, 64K, 128K; each model's chat template; an answer prefix appended "to prevent the model from refusing to answer a query or generating explanations"; recall-based accuracy (§4).
- Inference: vLLM, BFloat16, 8 NVIDIA A100 GPUs, greedy decoding (§4).
- For VT and CWE, one task sample is used as an in-context demonstration (App. D).
- Effective length: "We use the performance of Llama2-7b model at the 4K context length as the threshold. We report in Table 3 the maximum length exceeding the threshold as the 'effective length'" (§4). The threshold is 85.6 (Table 3 caption).
- wAvg. (inc) and wAvg. (dec): weights increase or decrease linearly with sequence length (§4). A check with weights 1-6 reproduces the printed Llama3.1 (70B) values 85.5 and 93.7 (derived in read.md §1).

## Table 3 (13-task average, %; claimed / effective; 4K / 8K / 16K / 32K / 64K / 128K)
| Model | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K | Avg | wAvg inc | wAvg dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama2 (7B) | 4K | - | 85.6 | | | | | | | | |
| Gemini-1.5-Pro | 1M | >128K | 96.7 | 95.8 | 96.0 | 95.9 | 95.9 | 94.4 | 95.8 | 95.5 | 96.1 |
| GPT-4 | 128K | 64K | 96.6 | 96.3 | 95.2 | 93.2 | 87.0 | 81.2 | 91.6 | 89.0 | 94.1 |
| Llama3.1 (70B) | 128K | 64K | 96.5 | 95.8 | 95.4 | 94.8 | 88.4 | 66.6 | 89.6 | 85.5 | 93.7 |
| Qwen2 (72B) | 128K | 32K | 96.9 | 96.1 | 94.9 | 94.1 | 79.8 | 53.7 | 85.9 | 79.6 | 92.3 |
| Command-R-plus (104B) | 128K | 32K | 95.6 | 95.2 | 94.2 | 92.0 | 84.3 | 63.1 | 87.4 | 82.7 | 92.1 |
| GLM4 (9B) | 1M | 64K | 94.7 | 92.8 | 92.1 | 89.9 | 86.7 | 83.1 | 89.9 | 88.0 | 91.7 |
| Llama3.1 (8B) | 128K | 32K | 95.5 | 93.8 | 91.6 | 87.4 | 84.7 | 77.0 | 88.3 | 85.4 | 91.3 |
| GradientAI/Llama3 (70B) | 1M | 16K | 95.1 | 94.4 | 90.8 | 85.4 | 80.9 | 72.1 | 86.5 | 82.6 | 90.3 |
| Mixtral-8x22B (39B/141B) | 64K | 32K | 95.6 | 94.9 | 93.4 | 90.9 | 84.7 | 31.7 | 81.9 | 73.5 | 90.3 |
| Yi (34B) | 200K | 32K | 93.3 | 92.2 | 91.3 | 87.5 | 83.2 | 77.3 | 87.5 | 84.8 | 90.1 |
| Phi3-medium (14B) | 128K | 32K | 93.3 | 93.2 | 91.1 | 86.8 | 78.6 | 46.1 | 81.5 | 74.8 | 88.3 |
| Mistral-v0.2 (7B) | 32K | 16K | 93.6 | 91.2 | 87.2 | 75.4 | 49.0 | 13.8 | 68.4 | 55.6 | 81.2 |
| LWM (7B) | 1M | <4K | 82.3 | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 | 72.8 | 69.9 | 75.7 |
| DBRX (36B/132B) | 32K | 8K | 95.1 | 93.8 | 83.6 | 63.1 | 2.4 | 0.0 | 56.3 | 38.0 | 74.7 |
| Together (7B) | 32K | 4K | 88.2 | 81.1 | 69.4 | 63.0 | 0.0 | 0.0 | 50.3 | 33.8 | 66.7 |
| LongChat (7B) | 32K | <4K | 84.7 | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 | 49.1 | 33.1 | 65.2 |
| LongAlpaca (13B) | 32K | <4K | 60.6 | 57.0 | 56.6 | 43.6 | 0.0 | 0.0 | 36.3 | 24.7 | 47.9 |

## Statements used in ch-32c
- "Despite achieving nearly perfect accuracy in the vanilla NIAH test, almost all models exhibit large performance drops as the context length increases. While these models all claim context sizes of 32K tokens or greater, only half of them can maintain satisfactory performance at the length of 32K" (Abstract).
- Top open models "contain both brute-force context scaling (Llama3.1 trained on 128K context length) and inference-time length extrapolation (Qwen2 trained on 32K context length)"; "less performant models also include those trained on much larger context size (e.g., LWM and GradientAI/Llama3 both on 1M context length)" (§4).
- LWM "performs worse than Llama2-7B even at 4K", which the authors describe as "a trade-off in evaluation between absolute performance on short sequences and the relative degradation" (§4).
- Failure modes at long lengths: failure to ignore distractors, copying from context, and reliance on parametric knowledge (§5; §7). Yi-34B's QA accuracy "approaches its no-context baseline" (§5).
- Training length: LWM-1M is worse than LWM-512K at 256K; abrupt drops appear when extrapolating beyond training length (§6). Model size: Yi-34B-200K is better than Yi-6B-200K at 4K and in relative degradation (§6).

## README leaderboard rows added after the paper (commit ab17b78)
- Llama3.1 (8B) 128K / 32K; ProLong (8B) 512K / 32K (128K: 81.6); GradientAI/Llama3 (8B) 1M / 16K (128K: 69.5); Qwen3-8B* 128K / 64K (64K: 82.1, below 85.6 and not underlined; 32K: 91.2; same inconsistency for Qwen3-4B*, listed 64K with 64K 77.8); Qwen2.5-7B-Instruct-1M* 1M / >128K (128K: 84.4, which is below 85.6 and not underlined; the row's effective-length entry is inconsistent with the stated threshold). Rows marked * are "reported by authors" of the respective model reports, not run by NVIDIA (README Notes).
