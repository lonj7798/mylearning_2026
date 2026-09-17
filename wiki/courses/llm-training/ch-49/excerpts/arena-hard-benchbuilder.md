---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2406.11939
created_at: "2026-09-15"
---

# Excerpt: From Crowdsourced Data to High-Quality Benchmarks — Arena-Hard and BenchBuilder Pipeline

**Authors:** Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. González, Ion Stoica (UC Berkeley)
**Year:** 2024 (arXiv v1 2024-06; v2 2024-10-14)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v2 PDF text.

## Why ch-49 uses it

It supplies the construction criteria for a judge-scored benchmark (separability, agreement with confidence, Brier score), the numbers that show how weakly MT-Bench separates close models, and the measurement of how much the choice of judge changes a ranking.

## Definitions (§3)

- **Separability with Confidence** — percentage of model pairs whose bootstrapped benchmark-score confidence intervals do not overlap.
- **Agreement with Confidence Interval** — for two benchmarks A and B and a model pair: +1 if both separate the pair confidently and agree on the order, −1 if both separate it and disagree, 0 if either cannot separate it.
- **Pair Rank Brier Score** — squared error of the predicted pairwise ordering probability.

The paper states the two requirements it is measuring: a benchmark should separate models with high confidence, and it should agree with human preference (§3).

## Benchmark configuration (§4–§5, §6.1)

- 500 prompts curated by BenchBuilder from Chatbot Arena data.
- Judge: gpt-4-1106-preview ("GPT4-T" in Table 4). Baseline model for pairwise comparison: gpt-4-0314.
- Aggregation: Bradley–Terry fit over all pairwise comparisons against the baseline; 100 rounds of bootstrapping give 95% confidence intervals.
- Reported evaluation cost: $20 per model.

## Table 1 — 20-model comparison against the Chatbot Arena (English) ranking of 2024-04-13

| Metric | Arena-Hard-Auto | MT-Bench | AlpacaEval 2.0 LC | Chatbot Arena |
|---|---|---|---|---|
| Confidence agreement | 90.9% | 26.6% | 82.5% | n.a. |
| Separability | 87.4% | 22.6% | 83.2% | 85.8% |
| Spearman correlation | 93.2% | 89.9% | 91.9% | n.a. |
| Kendall tau | 80.0% | 64.2% | 77.9% | n.a. |
| Brier score | 0.069 | 0.09 | 0.11 | n.a. |
| Prompts per model | 500 | 160 | 800 | 10,000+ |
| Eval cost per model | $20 | $10 | $10 | very high |

## Table 2 — pipeline robustness on WildChat (150,000 prompts)

Wild-Hard-Auto vs a baseline of 250 randomly selected WildChat prompts: confidence agreement 88.6% vs 36.4%; separability 86.7% vs 75.6%; Spearman 91.5% vs 45.5%.

## Table 3 — style control (§6.5)

Style features added to the Bradley–Terry fit: answer token length, density of markdown headers, markdown bold elements, markdown lists (App. A.2). Compared against a style-controlled Chatbot Arena (English Hard Prompts) ranking:

| Metric | Arena-Hard-Auto (style control) | Arena-Hard-Auto | AlpacaEval 2.0 LC | MT-Bench |
|---|---|---|---|---|
| Confidence agreement | 98.6% | 94.4% | 83.8% | 30.3% |
| Separability | 86.8% | 87.4% | 83.2% | 22.6% |
| Spearman | 98.6% | 94.9% | 88.1% | 90.7% |
| Kendall tau | 93.7% | 85.3% | 70.5% | 77.9% |

The paper also instructed GPT-3.5-Turbo, Llama-3.1-70b-instruct, and Gemini-1.5-Flash to increase verbosity and markdown usage: raw scores rose, and style control neutralized the advantage (§6.5, App. Table 12).

## Table 4 — the judge is part of the result

| Judge | Confidence agreement | Separability | Spearman | Brier |
|---|---|---|---|---|
| GPT4-T (gpt-4-1106-preview) | 90.9% | 87.4% | 93.2% | 0.069 |
| Claude-3-Opus | 66.7% | 83.68% | 77.0% | 0.170 |
| Gemini1.5-Pro (gemini-1.5-pro-0514) | 84.8% | 82.11% | 95.2% | 0.064 |
| Llama3-70B (llama-3-70b-instruct) | 65.6% | 81.6% | 70.5% | 0.196 |
| Ensemble-as-Judges (GPT-4-Turbo + Gemini-1.5-Pro) | 91.5% | 89.5% | 96.5% | 0.065 |

Self-bias check (§6.6, App. Table 10): with GPT-4-Turbo as the default judge, GPT models receive slightly higher average rankings than human preference gives them and Claude models rank lower; the two-model ensemble reduces this.

## Limits stated by the paper

Biases may remain in the curation pipeline (§7). All alignment metrics are computed against one Chatbot Arena snapshot over 20 released models, not over checkpoints from a single training run.

## Connections

[[chatbot-arena]] (prompt source and reference ranking), [[length-controlled-alpacaeval]] (the length-control method it compares against), [[lmsys-style-control]] (the style features it reuses), [[judge-llm-bias]] (MT-Bench, the benchmark it is measured against).
