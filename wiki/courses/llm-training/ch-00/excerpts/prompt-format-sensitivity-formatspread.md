---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:2310.11324v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2310.11324
created_at: "2026-09-15"
---

# Excerpt: Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design (FormatSpread)

**Paper:** Sclar, Choi, Tsvetkov, Suhr (University of Washington; Allen Institute for AI; UC Berkeley). arXiv v1 2023-10, read at v2 (2024-07-01); ICLR 2024. Source type: paper. Code: https://github.com/msclar/formatspread.

## Definitions (§3.1-3.2)
- A manually built grammar generates prompt formats (descriptors, separators, casing, spacing, enumeration) and defines which formats are semantically equivalent to a task's original format.
- Performance spread for formats p_1…p_n, dataset D, metric m: max_i m(p_i, D) − min_i m(p_i, D).

## Procedure (§3.2)
- Formats are arms of a multi-armed bandit; pulling an arm evaluates a mini-batch of B unevaluated items.
- Budget E: E/2 evaluations to find the best format, E/2 to find the worst; Thompson sampling with Beta priors (UCB with c = 2 and naive sampling as alternatives). No access to model weights is required.

## Setup (§4.1)
- 53 Super-NaturalInstructions tasks (19 multiple-choice, 34 classification); LLaMA-2 7B/13B/70B, Falcon-7B, Falcon-7B-Instruct, GPT-3.5-Turbo; few-shot examples fixed per task; ranking accuracy unless stated.

## Results
- Spread up to 76 accuracy points for LLaMA-2-13B (Abstract); about 10 points on average across 50+ tasks and several models (§1).
- With 10 randomly sampled formats per task, median spread 7.5 points across model and shot settings; 20% of tasks have spread of at least 15 points in all LLaMA-2 settings; spread is a lower bound because only 10 formats are sampled (§4.2).
- Spread persists with larger models, more few-shot examples, and instruction tuning (§4.2, Fig. 2).
- Ranking reversals: LLaMA-2-13B and -70B reverse by at least d = 0.02 with probability 0.141; LLaMA-2-7B and Falcon-7B with probability 0.140 (§4.2, Fig. 4).
- GPT-3.5: spread up to 56 points, median 6.4 points across 320 formats and 53 tasks (§1).
- 24% of single atomic format changes change accuracy by at least 5 points under exact prefix matching, 11% under probability ranking (§4.3, Fig. 6).
- Recommendation: report a range of performance across plausible formats instead of one format (Abstract).

## Verification
- Read on 2026-09-15 against arXiv:2310.11324v2 PDF text (Abstract, §1-4.3).
