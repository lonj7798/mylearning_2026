---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/flan-collection.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2301.13688
primary_version: arXiv:2301.13688v2 (v1 2023-01)
created_at: "2026-09-15"
---

# Excerpt: The Flan Collection: Designing Data and Methods for Effective Instruction Tuning

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Checked against the v2 PDF on 2026-09-15.

## Setup (§3)
> "We finetune on the prefix language model adapted T5-LM (Lester et al., 2021), using the XL (3B) size for all models for consistency, unless otherwise stated."

> "We evaluate on (a) a suite of 8 'Held-In' tasks represented within the 1800+ training task collection (4 question answering and 4 natural language inference validation sets), (b) Chain-of-Thought (CoT) tasks (5 validation sets), and (c) the MMLU (Hendrycks et al., 2020) and BBH (Suzgun et al., 2022) benchmarks as our set of 'Held-Out' tasks"

## Method ablations (Table 1, zero-shot / few-shot)
| Model | Held-In | CoT | MMLU | BBH | BBH-CoT |
|---|---|---|---|---|---|
| T5-XL Flan 2022 | 73.8 / 74.8 | 35.8 / 34.1 | 50.3 / 52.4 | 26.2 / 39.3 | 33.9 / 35.2 |
| - CoT | 73.3 / 73.2 | 28.8 / 24.6 | 47.5 / 46.9 | 18.2 / 30.0 | 18.2 / 12.0 |
| - Input Inversion | 73.8 / 74.1 | 32.2 / 23.5 | 41.7 / 41.2 | 18.4 / 24.2 | 15.7 / 13.0 |
| - Mixture Balancing | 71.2 / 73.1 | 32.3 / 30.5 | 45.4 / 45.8 | 15.1 / 24.3 | 13.8 / 15.4 |
| - Few Shot Templates | 72.5 / 62.2 | 38.9 / 28.6 | 47.3 / 38.7 | 27.6 / 30.8 | 18.6 / 23.3 |
| T5-XL Flan 2021 | 68.4 / 56.3 | 24.6 / 22.7 | 41.4 / 34.8 | 28.1 / 28.3 | 26.0 / 26.9 |
| T5-XL P3++ | 70.5 / 62.8 | 25.6 / 25.6 | 46.1 / 34.1 | 26.0 / 30.8 | 23.4 / 26.1 |
| T5-XL Super-Natural Inst. | 50.3 / 42.2 | 13.8 / 14.3 | 35.6 / 31.1 | 10.4 / 15.6 | 8.0 / 12.5 |

## Mixed prompt settings (§3.2)
> "Both Held-In and Held-Out tasks peak anywhere between 10-90% of few-shot data, but this range is consistently higher than training with only one prompt setting."

## Task scaling (§3.3)
> "Held-in task evaluations peak around 200 total tasks, and diminish in performance as more tasks are added, though larger models peak later and diminish less. Held-out task performance increases log-linearly with the number of tasks, achieving the highest performances with all 1836 tasks."

> "Surprisingly, only T5-Small appears to exceed its Held-Out task performance before 1836 tasks, while larger model sizes continue to improve."

## Input inversion (§3.4)
> "In Table 1 we find this is not beneficial for Held-In performance, but strongly beneficial for Held-Out performance."

## Source balancing (§3.5, Table 2)
| Train mixture | Held-In | CoT | MMLU |
|---|---|---|---|
| All (Equal) | 64.9 | 41.4 | 47.3 |
| All - Flan 2021 | 55.3 | 38.6 | 45.7 |
| All - T0-SF | 63.2 | 43.4 | 44.7 |
| All - Super-Nat. Inst. | 65.9 | 42.2 | 46.8 |
| All - CoT | 65.6 | 29.1 | 46.8 |
| All - Prog. Synth. | 66.9 | 42.3 | 46.8 |
| All - Dialog | 65.4 | 40.3 | 47.1 |
| All (Weighted) | 66.4 | 40.1 | 48.1 |

> "These findings are corroborated by Iyer et al. (2022) who extensively test data mixing proportions, and also determine their Flan 2021, T0-SF, and T5 mixtures are the most broadly beneficial."
