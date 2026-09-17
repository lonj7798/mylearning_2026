---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2410.05229
source_url: https://arxiv.org/abs/2410.05229
created_at: "2026-09-15"
---

# Excerpt: GSM-Symbolic — templated and irrelevant-clause variants of GSM8K

**Authors:** Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar (Apple; Washington State University). ICLR 2025.
**Checked on 2026-09-15 against arXiv:2410.05229v2 (2025-08-27).** No library card existed at that date; this excerpt is the checked extract used by ch-24 §3 and the Generalization lens.

## Construction (§3.1–3.2)

- 100 GSM8K test questions are annotated as templates: variables for names and numbers, value ranges, and conditions that keep the answer valid (Fig. 1).
- Automated checks: original values must not appear in the template; original values must satisfy the conditions and reproduce the original answer; 10 samples per template are reviewed manually; questions that fewer than two models answer are reviewed again.
- 50 samples per template → 50 datasets of 100 questions (5,000 examples per benchmark). Default evaluation: 8-shot CoT, greedy.
- Variants: names only, numbers only, both; GSM-M1, GSM-P1, GSM-P2 remove or add clauses; GSM-NoOp adds a clause that seems relevant but does not change the answer.

## Results used in ch-24

- Variance across the 50 sets: gap between worst and best set above 12% for Gemma2-9B and about 15% for Phi-3.5-mini (§4.1).
- Gemma2-9b-it: GSM8K 87.0 on the 100 originals vs 79.1 (±3.0) on GSM-Symbolic; Llama3-8b-instruct 74.0 vs 74.6 (±2.9); Phi-3-medium 89.0 vs 82.5 (±2.9) (Fig. 2).
- For 21 of 25 models, the original-question score lies more than one standard deviation from the center of the variant distribution, frequently on the right side; the authors name data contamination as one explanation (§4.1).
- Changing names moves scores less than changing numbers, e.g. Gemma2-9b-it names 88.6, numbers 83.1, both 79.1 (Fig. 4).
- GSM-NoOp accuracy drops (Fig. 8a, reported in %): o1-preview −17.5, o1-mini −29.1, GPT-4o −32.0, GPT-4o-mini −40.0, Llama3-8b-instruct −57.4, Gemma2-9b-it −63.0; Phi-3-mini over 65% (§4.4).
- Eight shots of variants of the same question did not remove the NoOp drop for most models (§4.4, Fig. 8b).

## Use as a measurement

Report the distribution over template sets rather than one score, and compare the original-question score with that distribution. The paper evaluates released models; it does not test whether any augmentation dataset reduces the gap.
