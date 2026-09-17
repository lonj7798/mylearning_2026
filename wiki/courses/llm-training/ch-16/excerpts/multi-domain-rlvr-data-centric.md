---
chapter: ch-16
course: llm-training
phase: read
artifact: "Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning"
source_url: https://arxiv.org/abs/2507.17512
version: arXiv v1 (2025-07-23)
verified_on: "2026-09-15"
note: no library card exists for this slug yet; this file is the chapter's verified extract
---

# Excerpt: multi-domain RLVR — composition, curriculum, and policy refresh

Used by [[read]] §4, §5 and the Generalization lens. Loci refer to the cached primary text
(`scratchpad/sources/multi-domain-rlvr-data-centric.txt`, arXiv:2507.17512v1).

## Setup (§2.1, Table 1)

GRPO on Qwen2.5-7B-Base and Qwen2.5-7B-Instruct. Domains and prompt counts after sampling to equal
scale: Math — DeepScaleR 10k and CountDown 10k; Code — CodeR1-12k (2K LeetCode plus 10K verified
from 26K TACO); Puzzle — Knights-and-Knaves 5.4k and Logic Puzzle Baron 2.4k. Rewards are binary
0–1 on final-answer correctness for all sets except Logic Puzzle Baron, which uses a proportional
reward over correctly predicted cells because the model rarely solves it in one attempt at the
start of training. Logic Puzzle Baron ground truth is annotated by DeepSeek-R1 and treated as
pseudo ground truth.

## Domain combinations (Table 9, Qwen2.5-7B-Base; Δ against the base row)

| Training data | Math Avg | Code Avg | Puzzle Avg | All Avg |
|---|---|---|---|---|
| Base | 22.48 | 67.46 | 9.07 | 31.50 |
| Math | 47.48 | 64.23 | 22.42 | 45.11 |
| Puzzle | 29.47 | 71.35 | 61.98 | 50.72 |
| Code | 19.17 | 73.95 | 22.55 | 35.78 |
| Math + Puzzle | 49.72 | 44.90 | 49.78 | 48.36 |
| Puzzle + Code | 32.06 | 74.88 | 55.15 | 50.89 |
| Math + Code | 47.22 | 75.06 | 25.34 | 48.92 |

Triple domain (§4.2, Figure 3): Math + Code + Puzzle gives an overall average of 56.57, above the
best pair (Puzzle + Code, 50.89), with Math at its highest (49.75) and Puzzle at 49.73, below the
55.15 of Puzzle + Code. The authors read the puzzle drop as evidence that the puzzle task requires
specialization and is hurt by out-of-domain data.

Stated takeaways (§4.2 box): multi-domain training improves overall performance, with the triple
combination showing moderate gains; it improves task balance and prevents extreme drops in a single
area. Overall takeaways (§1 box) also include: puzzle and math support each other; code has mixed
cross-domain effects; template consistency between training and evaluation is decisive; RLVR is
language-sensitive (Chinese training underperforms English).

## Curriculum and policy refresh (§6, Figure 6)

Difficulty for Knights-and-Knaves is the number of sub-questions per problem (PPL), 3PPL to 8PPL,
six levels, trained in sequence with 175 steps per stage. Standard curriculum reaches a final
average of 97.29; mixed (non-curriculum) training reaches 94.29. "Policy refresh" replaces the
reference model with the latest actor and resets the optimizer state at each stage boundary; it
reaches 97.43 already at the 6PPL stage and 99.71 at the end.

## Status and limits

Result (single study): one model family (Qwen2.5-7B), one seed per configuration, no variance
reported; the puzzle-only overall average is inflated by one sub-task (KK 94.29), which the authors
flag themselves (§4.2). The mechanisms of transfer and interference are not isolated.
