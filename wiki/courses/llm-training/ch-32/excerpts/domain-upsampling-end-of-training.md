---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.03476v1 (planned library card papers/domain-upsampling-end-of-training.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2406.03476
created_at: "2026-09-15"
---

# Excerpt: Does your data spark joy? Performance gains from domain upsampling at the end of training

**Authors:** Cody Blakeney, Mansheej Paul, Brett W. Larsen, Sean Owen, Jonathan Frankle (Databricks Mosaic Research). arXiv v1 2024-06-05. Source type: paper.

## Setup (§2, Table 1, §3.1 Table 2)

- 7B decoder-only MPT-architecture models trained for 1T tokens; LionW, LR 0.00012, betas (0.9, 0.95), weight decay 0.00012, max sequence 4096, batch 960, GPT-4 tiktoken tokenizer, ALiBi, "inverse square root learning schedule similar to (Zhai et al., 2022)" (§2, Table 1). How the LR schedule behaves inside the upsampling window is not stated.
- Evaluation: MosaicML Eval Gauntlet v0.3 (35 in-context-learning tasks in 6 categories), plus MMLU 5-shot, GSM8K 8-shot, HumanEval pass@1 (§2, Table 3). One run per condition; no seeds reported.
- Baseline mix (Table 2): Large-Scale Common Crawl 34.35% (343.5B), Small-Scale Common Crawl 36.70% (367.0B), Domain Specific 7.17% (71.7B), Code 21.78% (217.8B).

## Domain upsampling, final 20% (§3.2, Table 4, Table 5)

- From the 0.8T checkpoint, the mix changes to Large-Scale CC 0%, Small-Scale CC 30% (60B), Domain Specific 35% (70B), Code 35% (70B) for the last 0.2T tokens. Small-Scale CC is kept "to prevent a large distribution shift" (§3.2).
- No DU → 20% DU: MMLU 35.69 → 42.59; GSM8K 14.71 → 22.97; HumanEval 17.23 → 23.40; Gauntlet Core Average 35.37 → 39.32; Language Understanding 61.52 → 60.08 (Table 5).

## Duration ablation (§3.3, Table 6; final 5%, 10%, 20%, 30% = 50B, 100B, 200B, 300B tokens)

| Benchmark | 0% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|
| MMLU (5-shot) | 35.69 | 40.20 | 43.19 | 42.59 | 41.78 |
| GSM8K (8-shot) | 14.71 | 16.98 | 20.47 | 22.97 | 24.56 |
| HumanEval (pass@1) | 17.23 | 18.50 | 20.39 | 23.40 | 24.17 |
| Gauntlet Core Average | 35.37 | 37.63 | 38.46 | 39.32 | 38.89 |
| Language Understanding | 61.52 | 61.05 | 60.41 | 60.08 | 60.35 |
| Reading Comprehension | 37.02 | 41.23 | 43.35 | 45.45 | 42.50 |

- Authors' reading: GSM8K and HumanEval keep improving with longer DU; MMLU peaks at 10% and Core Average at 20%; "DU for the final 10%-20% of training provides the best trade-off for this set up" and "the mix used for DU should not be used for the entire duration of training" (Figure 3 caption, §3.3).

## Dataset attribution by removal (§3.4, Table 7; 10% DU)

| Benchmark | No DU | 10% DU with math | 10% DU sans math |
|---|---|---|---|
| MMLU | 35.69 | 43.19 | 29.71 |
| GSM8K | 14.71 | 20.47 | 11.37 |
| HumanEval | 17.23 | 20.39 | 21.15 |
| Core Average | 35.37 | 38.46 | 32.54 |

- The authors state that experiments at the end of training measure dataset impact "at an order of magnitude fewer training FLOPS" than full pre-training runs (§3.4).

## Used in

ch-32 §2.2 (duration trade-off), §3.1 (annealing as a probe), figure panel B, Recipe rows.
