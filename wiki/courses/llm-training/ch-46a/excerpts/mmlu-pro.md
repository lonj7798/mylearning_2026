---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark (arXiv:2406.01574v6, 2024-11-06; NeurIPS 2024 Datasets and Benchmarks)"
source_url: https://arxiv.org/abs/2406.01574
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug mmlu-pro). Values read from the v6 PDF on 2026-09-15."
---

# Excerpt: MMLU-Pro — size, format, and prompt robustness

**Authors:** Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, et al. (University of Waterloo; University of Toronto; Carnegie Mellon University).

## Composition (§3, Fig. 3)

- 12,032 questions across 14 discipline subsets, built from original MMLU questions plus other sources (§3).
- Ten answer options per question instead of MMLU's four, described as "3x more distractors" (§1).
- Stated design changes relative to MMLU: more college-level reasoning problems, removal of trivial and noisy questions (§1).

## Evaluation protocol (§4)

- 5-shot chain-of-thought prompting, with five demonstration examples selected per discipline.
- Answer extraction uses the regular expression `answer is \(?\([A-J]\)?\)`, then `\.*\[aA\]nswer:\s*\([A-J]\)`; if both fail, a random option is selected, "ensuring consistent answer provision across all evaluations" (§4). A failed extraction is therefore scored at chance, not as an error.
- CoT versus direct answering is compared in §6.2; §1 states CoT raises GPT-4o by 19% on MMLU-Pro while it lowers performance on MMLU.

## Prompt robustness (§6.3)

- Models were evaluated with "24 different but reasonable prompts".
- "On the MMLU benchmark, the influence of these prompts generally ranges between 4-5%, with peaks up to 10.98%. In contrast, on the MMLU-Pro benchmark, the impact of prompt changes is generally around 2%, with a maximum of 3.74%."

## Limits (§7)

- The multiple-choice format is stated as a limitation: it does not capture open-ended comprehension or generation.
- The paper does not report per-item standard errors or a recommended sample size for subsets.

## Used in

ch-46a §7 (the non-agentic regression suite: item count, prompt-variation floor, and why the harness settings are frozen before the run).
