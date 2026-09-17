---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: "primary source arXiv:2310.11324 (v1 2023-10; read as v2), ICLR 2024 (planned library card prompt-format-sensitivity-formatspread; not present on 2026-09-15)"
source_url: https://arxiv.org/abs/2310.11324
created_at: "2026-09-15"
note: "Read from the cached text on 2026-09-15. ch-04/excerpts/prompt-format-sensitivity-formatspread.md covers the same artifact for the data track."
---

# Excerpt: FormatSpread (Sclar et al. 2023) — format as a measured noise source

**Paper:** Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr (University of Washington; Allen Institute for AI;
UC Berkeley). "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to
start worrying about prompt formatting." Code: github.com/msclar/formatspread. Source type: paper.

## Definitions (§3)

- A grammar of prompt formats over descriptors, separators, spacing, casing, and enumerations. Two formats built
  from the same rules and descriptors are meaning-equivalent, so a score difference between them is a property of
  the model, not of the task (§3.1).
- Performance spread for formats p₁ … p_n, dataset D, metric m: `spread = max_i m(p_i, D) − min_i m(p_i, D)` (§3.2).
- FormatSpread searches for the minimum and maximum with Thompson sampling under an evaluation budget and without
  access to model weights (§3.2).

## Setup (§4.1)

53 Super-NaturalInstructions tasks (19 multiple choice, 34 classification); LLaMA-2 7B/13B/70B, Falcon-7B,
Falcon-7B-Instruct, GPT-3.5-Turbo; few-shot examples fixed per task and shot count; 1,000 samples per evaluation.

## Measured spread (§4.2)

- With 10 randomly sampled plausible formats per task: median spread 7.5 accuracy points across model and shot
  choices; 20% of tasks have a spread of at least 15 points for all LLaMA-2 settings and at least 9 points for all
  Falcon settings; several tasks exceed 70 points. Because only 10 formats are sampled, these are lower bounds.
- Reported maxima: up to 76 accuracy points for LLaMA-2-13B (abstract); GPT-3.5 spread up to 56 points with a
  median of 6.4 points across 320 formats and 53 tasks (§1, §4.2).
- Spread is not removed by larger models (LLaMA-2-70B), by instruction tuning (Falcon-7B vs Falcon-7B-Instruct), or
  by more few-shot examples (§4.2, Fig. 2).

## Rank reversals under a different format (§4.2, Fig. 4)

Taking model M better than M′ by at least d = 0.02 accuracy under format p, the probability that M′ is better than
M by at least d under another format p′ is 0.141 for LLaMA-2-13B vs LLaMA-2-70B and 0.140 for LLaMA-2-7B vs
Falcon-7B. The paper adds: "often both experiments (first using p, and then p′) were statistically significant
(p-value < 0.05) on 1000 samples: 76% and 47% respectively for the two model comparisons". Significance is tested
with one-sided McNemar tests (paired χ² tests) because both models are evaluated on the same samples (§4.2,
footnote 2).

## Recommendation by the authors

Report a range of performance across plausible formats rather than a single format, especially when comparing
models (abstract, §1).

## Limits

Classification and multiple-choice tasks in few-shot prompting. Chat-template variation for instruction-tuned chat
models is not the object of study.

## Used in

ch-51 §1.4 (format noise and rank reversal), §5 (what a significant result does and does not license), Recipe,
Common mistakes.
