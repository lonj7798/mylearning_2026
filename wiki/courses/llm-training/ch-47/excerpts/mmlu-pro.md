---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: primary source arXiv:2406.01574v6 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2406.01574
created_at: "2026-09-15"
---

# Excerpt: MMLU-Pro — A More Robust and Challenging Multi-Task Language Understanding Benchmark

**Paper:** Wang, Ma, Zhang, Ni, Chandra, Guo, et al. (University of Waterloo; University of Toronto; Carnegie Mellon University). arXiv v1 2024-06, read at v6 (2024-11-06); NeurIPS 2024 Datasets and Benchmarks Track. Source type: paper.

## Dataset (§1, §3, Table 1)
- 12,032 questions over 14 discipline subsets.
- Ten answer options instead of MMLU's four: "3x more distractors than MMLU" (§1). 83% of questions have ten options, 17% fewer, average 9.47 options per question (§3).
- Two rounds of expert review to remove noisy items; a larger share of college-level exam problems (§1, §3).

## Evaluation protocol (§4)
- 5-shot chain-of-thought prompting, adapted from the chain-of-thought prompt library; direct answering is compared against it in §6.2.

## Results
- Accuracy falls by 16% to 33% relative to MMLU (Abstract). GPT-4o scores 72.6% and GPT-4-Turbo 63.7% (§1, Table 2).
- Discrimination: the GPT-4o to GPT-4-Turbo gap is 1% on MMLU and 9% on MMLU-Pro (§1).
- Chain of thought raises GPT-4o by 19% on MMLU-Pro and lowers scores on MMLU (§1; §6.2, Table 3).
- Prompt robustness (§6.3, Fig. 5): with 24 different reasonable prompts, the score range is generally 4-5% on MMLU with a peak of 10.98%, and generally about 2% on MMLU-Pro with a maximum of 3.74%.
- Error analysis of 120 GPT-4o errors: 39% reasoning flaws, 35% missing domain knowledge, 12% computation errors (§1).

## Limits (§7)
- The benchmark remains multiple choice and knowledge-and-reasoning oriented; it does not cover generation, tool use, or long inputs.

## Verification
- Read on 2026-09-15 against arXiv:2406.01574v6 PDF text (Abstract, §1, §3-4, §6.2-6.3, §7).
