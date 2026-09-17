---
chapter: ch-32f
course: llm-training
phase: read
excerpt_of: primary source arXiv:2404.06654v3 (the library card [[ruler]] describes the task generators but does not print Table 3)
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-09-15"
---

# Excerpt: RULER effective-length rule and Table 3

**Paper:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, et al. (NVIDIA), "RULER: What's the Real Context Size of Your Long-Context Language Models?". arXiv v1 2024-04; read at v3 (2024-08-06), COLM 2024. Source type: paper.

## Protocol (§4)

- 13 tasks from four categories; "500 examples generated for each length from the series (4K, 8K, 16K, 32K, 64K, 128K)"; each model's chat template; "an answer prefix" appended; "recall-based accuracy"; vLLM, BFloat16, greedy decoding.
- The 13 configurations were selected so that "most models perform decently at short context size of 4K tokens".
- Effective length: "We use the performance of Llama2-7b model at the 4K context length as the threshold. We report in Table 3 the maximum length exceeding the threshold as the 'effective length'". The threshold is 85.6% (Table 3 caption).
- wAvg. (inc) and wAvg. (dec): weights that increase or decrease linearly with length.
- Base models (App. A): "We use the performance of Llama2-7b (base) and Llama2-7b (chat) at context length of 4K as the threshold for determining effective context size." The main-text Table 3 contains aligned models only; base models are in Table 12 with Llama2-7B (base) at 79.4.

## Table 3 rows used in ch-32f (aligned models)

| Model | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|---|
| Llama2 (7B) | 4K | - | 85.6 | | | | | |
| Llama3.1 (8B) | 128K | 32K | 95.5 | 93.8 | 91.6 | 87.4 | 84.7 | 77.0 |
| Mistral-v0.2 (7B) | 32K | 16K | 93.6 | 91.2 | 87.2 | 75.4 | 49.0 | 13.8 |
| LWM (7B) | 1M | <4K | 82.3 | 78.4 | 73.7 | 69.1 | 68.1 | 65.0 |
| Together (7B) | 32K | 4K | 88.2 | 81.1 | 69.4 | 63.0 | 0.0 | 0.0 |

## Table 12 rows (base models, App. F)

| Model | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|---|
| Llama2-7B (base) | 4K | - | 79.4 | | | | | |
| Mistral-base (7B) | 32K | 16K | 91.6 | 89.8 | 86.3 | 77.2 | 52.3 | 8.0 |
| Yarn-base (7B) | 128K | <4K | 77.3 | 67.5 | 59.0 | 47.3 | 38.6 | 13.9 |
| Together-base (7B) | 32K | 4K | 84.6 | 78.7 | 68.3 | 57.9 | 0.0 | 0.0 |

- On LWM: it "performs worse than Llama2-7B even at 4K. This result suggests a trade-off in evaluation between absolute performance on short sequences and the relative degradation with the scaling of context size" (§4).
- Model analysis: "abrupt performance drops when models need to extrapolate to unseen lengths (e.g., LMW-128K given input of 256K), and almost linear degradation with input length on log scale within the max training context size" (§6).

## Used in

ch-32f §6 (RULER effective length for small checkpoints).
