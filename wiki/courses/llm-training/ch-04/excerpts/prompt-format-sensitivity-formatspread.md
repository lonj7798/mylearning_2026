---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Sclar, Choi, Tsvetkov, Suhr — "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting" (ICLR 2024)
source_url: https://arxiv.org/abs/2310.11324
created_at: "2026-09-15"
note: "Library card for this slug was planned but not present on 2026-09-15; written from the primary PDF (arXiv v2, 2024-07-01)."
---

# Excerpt: Sclar et al. 2023 — FormatSpread

**Artifact:** arXiv:2310.11324 (v1 2023-10; read as v2). Authors: Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr (University of Washington, Allen Institute for AI, UC Berkeley). Code: github.com/msclar/formatspread.

## Definitions (§3)

- A grammar of prompt formats built from descriptors, separators, spacing, casing, and item enumerations; two formats are equivalent when they apply the same rules with the same descriptors, so they preserve meaning (§3.1).
- Performance spread: for formats p_1..p_n, dataset D, and metric m, spread = max_i m(p_i, D) − min_i m(p_i, D) (§3.2).
- FormatSpread estimates the minimum and maximum with Thompson sampling over formats under an evaluation budget, without access to model weights (§3.2).

## Setup (§4.1)

53 Super-NaturalInstructions tasks (19 multiple choice, 34 classification); LLaMA-2 7B/13B/70B, Falcon-7B, Falcon-7B-Instruct, GPT-3.5-Turbo; few-shot examples fixed per task and shot count; ranking accuracy unless stated.

## Results

- Spread up to 76 accuracy points for LLaMA-2-13B; about 10 points on average across 50+ tasks and several models (§1, abstract).
- With 10 sampled formats per task, the median spread is 7.5 points across model and shot choices; 20% of tasks have a spread of at least 15 points for all LLaMA-2 settings and at least 9 points for all Falcon settings; several tasks exceed 70 points. Because only 10 formats were sampled, these are lower bounds (§4.2).
- Spread remains with larger models, instruction tuning (Falcon-7B vs Falcon-7B-Instruct), and more few-shot examples (§4.2, Fig. 2).
- Model comparisons reverse under another format: LLaMA-2-13B vs 70B reverse by at least 0.02 accuracy with probability 0.141; LLaMA-2-7B vs Falcon-7B with probability 0.140 (§4.2, Fig. 4).
- GPT-3.5: spread up to 56 points, median 6.4 points, across 320 formats and 53 tasks, at under 10 USD per task on average (§1).
- Format performance correlates only weakly between models (abstract).

## Recommendation by the authors

Report a range of performance across plausible formats instead of a single format, especially when comparing models (abstract, §1).

## Limits

Classification and multiple-choice tasks in few-shot prompting; chat-template changes for instruction-tuned chat models are not the object of study.
