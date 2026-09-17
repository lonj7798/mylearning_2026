---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Hsieh et al. — "RULER: What's the Real Context Size of Your Long-Context Language Models?"
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against arXiv v3, COLM 2024; previous version relied on an unverified card)"
---

# Excerpt: RULER — task generators, the effective-length threshold, and Table 3

**Paper:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA), arXiv 2404.06654 v3 (2024-08-06), COLM 2024.
**Library card:** [[ruler]] (task list and protocol consistent with the paper; numbers below are read from the paper).

## What the paper states (abstract, §1)

> "We evaluate 17 long-context LMs with 13 representative tasks in RULER. Despite achieving nearly perfect accuracy in the vanilla NIAH test, almost all models exhibit large performance drops as the context length increases. While these models all claim context sizes of 32K tokens or greater, only half of them can maintain satisfactory performance at the length of 32K." (Abstract)

## Task categories and generator knobs (§3, App. B Table 5)

| Category | Task | Generator knob |
|---|---|---|
| Retrieval | S-NIAH (single key-value needle) | needle type: word, 7-digit number, 32-digit UUID; haystack: repeated noise sentences or Paul Graham essays |
| Retrieval | MK-NIAH (several needles, one queried) | number of distractor keys, from 4 to a haystack filled with needles ("FULL") |
| Retrieval | MV-NIAH (one key, several values) | number of values |
| Retrieval | MQ-NIAH (several keys queried) | number of queries |
| Multi-hop tracing | VT (variable tracking, `X2 = X1`, `X3 = X2`, ...) | number of chains, hops per chain |
| Aggregation | CWE (common words extraction) | frequency of common vs uncommon words |
| Aggregation | FWE (frequent words extraction) | Zeta distribution parameter α |
| QA | SQuAD / HotpotQA with distractor paragraphs | number of distractor paragraphs |

The 13 configurations used for the main table were chosen after a task-correlation study (App. C).

## Protocol (§4)

> "For each task, we evaluate each model with 500 examples generated for each length from the series (4K, 8K, 16K, 32K, 64K, 128K), while complying with each model's necessary chat template. To prevent the model from refusing to answer a query or generating explanations, we append the task input with an answer prefix and check the presence of the target output with recall-based accuracy." (§4)

> "We use the performance of Llama2-7b model at the 4K context length as the threshold. We report in Table 3 the maximum length exceeding the threshold as the 'effective length' along with the 'claimed length'." (§4)

The threshold value is 85.6 (Table 3, first row).

## Table 3 rows used in ch-28 (average of 13 tasks, %)

| Model | Claimed | Effective | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|---|
| Gemini-1.5-Pro | 1M | >128K | 96.7 | 95.8 | 96.0 | 95.9 | 95.9 | 94.4 |
| GPT-4 | 128K | 64K | 96.6 | 96.3 | 95.2 | 93.2 | 87.0 | 81.2 |
| Llama3.1 (70B) | 128K | 64K | 96.5 | 95.8 | 95.4 | 94.8 | 88.4 | 66.6 |
| Qwen2 (72B) | 128K | 32K | 96.9 | 96.1 | 94.9 | 94.1 | 79.8 | 53.7 |
| Llama3.1 (8B) | 128K | 32K | 95.5 | 93.8 | 91.6 | 87.4 | 84.7 | 77.0 |
| GradientAI/Llama3 (70B) | 1M | 16K | 95.1 | 94.4 | 90.8 | 85.4 | 80.9 | 72.1 |
| LongChat (7B) | 32K | <4K | 84.7 | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 |
| LongAlpaca (13B) | 32K | <4K | 60.6 | 57.0 | 56.6 | 43.6 | 0.0 | 0.0 |

Note: Llama 3.1 appears in v3 of the paper and on the NVIDIA/RULER GitHub leaderboard with the same 70B and 8B rows. The Llama 3 technical report does not report RULER.

## Failure analysis of Yi-34B-200K (§5)

- Needle type: performance degrades when the needle is a UUID rather than a word-number pair (§5, Fig. 2).
- Distractors: "increasing the number of distracting needles steadily lowers performance, with Yi dropping by ∼40 points at 256K in the extreme version" (§5).
- Multiple items: "increasing the number of queries from 1 to 8 drops the performance by ∼15 points" (§5).
- Copying: "Over 80% of Yi's output in the CWE task at 128K is simply a string copied from the one-shot example, whereas the copying is nonexistent for short sequences." The same behavior is reported for LWM and LongAlpaca (§5).

## Model analysis (§6)

- Training length: larger training context sizes overall help, "but the ranking can be inconsistent for long sequences"; LWM-1M is worse than LWM-512K at 256K (§6, Fig. 4).
- Model size: Yi-34B-200k is better than Yi-6B-200k at 4K and in relative degradation, with the same training length and data (§6).

## Connections

- [[needle-in-haystack-data]] — the single-needle test RULER extends.
- [[babilong]] — reasoning tasks over facts hidden in PG19 text.
- [[helmet]] — correlation of RULER tasks with application categories.
- Chapter: ch-28 §1; evaluation depth in ch-32c.
