---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2410.07137
created_at: "2026-09-15"
---

# Excerpt: Cheating Automatic LLM Benchmarks — Null Models Achieve High Win Rates

**Authors:** Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Jing Jiang, Min Lin (Sea AI Lab; Singapore Management University)
**Year:** 2024 (arXiv v1 2024-10; v2 2025-03-02; ICLR 2025)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v2 PDF text.

## Why ch-49 uses it

It bounds how much of a judge-scored benchmark number can be produced with zero capability, and it shows that length control and style control do not prevent that.

## Setup (§3, Table 1)

A "null model" is a callable that returns one constant string for every instruction, independent of the input. Benchmarks and configuration:

| Benchmark | Instructions | Type | Metric | Reference model |
|---|---|---|---|---|
| AlpacaEval 2.0 | 805 | pairwise | length-controlled win rate | GPT-4-1106-Preview |
| Arena-Hard-Auto | 500 | pairwise | win rate | GPT-4-0314 |
| MT-Bench | 80 | single | score 1–10 | — |

GPT-4-1106-Preview is the auto-annotator for all three.

## Two mechanisms

1. **Structured cheating response** (§3, Fig. 2). The constant string is written so the annotator parses it as the end of the evaluation instructions followed by a fresh, empty comparison; when the annotator is led to treat both outputs as identical and empty, it returns the first identifier. This alone reaches 76.8% LC win rate on AlpacaEval 2.0.
2. **Transferable adversarial prefix by random search** (§3). The benchmark instructions are assumed private, so the prefix is optimized on a separate public instruction set and then transferred to all three benchmarks.

The 16 "persuasive response" baselines crafted with ChatGPT achieved win rates below 1% (§1), so the effect comes from the template exploit, not from persuasive writing.

## Table 2 — results (annotator GPT-4-1106-Preview; SOTA recorded before 2024-10-01)

| Target model | AlpacaEval 2.0 LC | AlpacaEval 2.0 raw | AlpacaEval 2.0 discrete | Arena-Hard-Auto | Arena-Hard 95% CI | avg #tokens | MT-Bench |
|---|---|---|---|---|---|---|---|
| Verified SOTA | 57.5 | 51.3 | 53.8 | 82.6 | (−1.9, +2.0) | 662 | 8.96 |
| Community SOTA | 78.5 | 77.6 | 79.5 | — | — | — | — |
| Structured (theirs) | 76.8 | 59.5 | 64.2 | 67.2 | (−1.7, 1.2) | 198 | 7.75 |
| Structured + random search (theirs) | 86.5 | 76.9 | 84.0 | 83.0 | (−1.1, 1.5) | 205 | 9.55 |

The winning responses average 198–205 tokens, so the AlpacaEval length control is active and does not prevent the result.

## Framing stated by the authors

The experiments are described as proof-of-concept; the authors note an adversary could use an LLM to generate less conspicuous cheating responses, and call for anti-cheating mechanisms in automatic benchmarks (Abstract). Code: github.com/sail-sg/Cheating-LLM-Benchmarks.

## Connections

[[length-controlled-alpacaeval]] and [[lmsys-style-control]] (the controls this defeats), [[arena-hard-benchbuilder]] (one of the three benchmarks attacked), [[judge-llm-bias]] (MT-Bench, the third), [[leaderboard-illusion]] (a non-adversarial route to the same inflation).
