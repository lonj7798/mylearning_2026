---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2406.01574 (primary text, arXiv v6; no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2406.01574
created_at: "2026-09-15"
---

# Excerpt: MMLU-Pro — ten-option knowledge and reasoning questions

**Artifact:** Wang, Ma, Zhang, Ni, Chandra, Guo, et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark" (arXiv v1 2024-06; v6 2024-11; NeurIPS 2024 Datasets and Benchmarks Track). University of Waterloo, University of Toronto, Carnegie Mellon University.

## Construction (Abstract, §3)

- 14 discipline subsets, 12,032 questions (§3.1, Figure 3).
- Sources: original MMLU questions (with trivial and noisy questions removed), STEM websites, TheoremQA, and SciBench (§3.1, Figure 3b).
- The choice set grows from four to ten options (Abstract). 83% of questions have ten options, 17% fewer; the mean is 9.47 options (§3.2).

## Findings (Abstract, §1, §6)

- Accuracy drops by 16% to 33% compared with MMLU (Abstract).
- Across 24 prompt styles, score sensitivity is generally 4–5% on MMLU (peaks up to 10.98%) and generally about 2% on MMLU-Pro (maximum 3.74%) (§6.3).
- Chain-of-thought improves performance on MMLU-Pro relative to direct answering, unlike on MMLU; CoT raises GPT-4o by 19% on MMLU-Pro (§1 item 4).
- Evaluation protocol in the paper: 5-shot chain-of-thought (§4).

## Limits stated by the authors (§7)

- Multiple-choice format only; it may not capture comprehension and creative generation as open-ended answers do.
- Text only.

## Use in ch-29

- MMLU-Pro is the knowledge-and-reasoning column of the lab's held-out suite. The SFT pool in the lab does not target it, so a drop against the base model is a forgetting signal rather than a missed target.
- A subset keeps compute low. Stratify the subset by the 14 disciplines so that per-discipline shares match the full set, and report per-discipline accuracy next to the average.
- Instance-level noise for a subset of n questions at accuracy p: standard error √(p(1 − p)/n). With n = 1,400 and p = 0.30, SE = √(0.21/1400) = 0.0122, a 95% interval of about ±2.4 points.
- Prompt format matters: Tülu 3 reports MMLU-Pro with a 0-shot CoT prompt and compares it with the 5-shot prompt used for Llama 3.1 (Tülu 3 Table 27, [[tulu-3-sft-and-eval]]). Use one prompt format for every arm of the lab.

## Connections

- [[tulu-3-sft-and-eval]] — MMLU-Pro is in the Tülu 3 unseen evaluation suite.
- [[signal-and-noise-eval]] — MMLU-Pro is among the 30 tasks whose signal-to-noise ratio is measured.
