---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2310.20689
source_url: https://arxiv.org/abs/2310.20689
created_at: "2026-09-15"
---

# Excerpt: Learning From Mistakes Makes LLM Better Reasoner (LEMA)

**Authors:** Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, Weizhu Chen (Xi'an Jiaotong University; Microsoft; Peking University).
**Checked on 2026-09-15 against arXiv:2310.20689v4 (2024-03-29).** No library card existed at that date; this excerpt is the checked extract used by ch-24's negative-samples section and Recipe. Course standard category: negative as content (a failure placed in the input, with a corrected target trained by cross-entropy).

## Method (§2)

- Wrong paths: sample reasoning paths from LLaMA-2, WizardLM, WizardMath, Text-Davinci-003, GPT-3.5-Turbo, and GPT-4; keep paths whose final answer differs from the reference (Eq. 1, §3.2).
- Corrections: GPT-4 with 4 annotated examples outputs (1) the incorrect step, (2) an explanation, (3) a corrected solution from that step; corrections with a wrong final answer are dropped (Eq. 2, Fig. 3).
- Human check of 50 GPT-4 corrections on GSM8K paths from LLaMA-2-70B: 35 excellent, 11 good, 4 poor (§2.1).
- Fine-tuning format: input = question + wrong path; output = correction; "only the loss in the output part participates in the back-propagation" (App. B.2).
- Correction data: 12,523 (GSM8K), 6,306 (MATH), 7,241 (CSQA) pairs; CoT data: GSM8K training set + 24,948 GPT-4 paths (§3.2).

## Settings (§3.3)

QLoRA, rank 64, dropout 0.05; LR 1e-4 for models ≥ 34B and 2e-4 below; batch 96; 2,000 steps; checkpoints every 100 steps; the accuracy of the best checkpoint on the test set is reported; greedy decoding, max 2,048 tokens.

## Results

- Table 1, CoT-only → LEMA, GSM8K / MATH / CSQA: LLaMA-2-70B 81.4 → 83.5 / 23.6 → 25.0 / 84.2 → 85.3; LLaMA-65B 76.2 → 77.9; CodeLLaMA-34B 68.8 → 71.7 (SVAMP 67.4 → 72.0); LLaMA-2-13B 62.9 → 65.7; LLaMA-2-7B 52.6 → 54.1.
- SVAMP and ASDiv use GSM8K training data and are treated by the authors as out-of-distribution (§4.1).
- Matched data size (32K and 45K): LEMA keeps a gain for four of five models; LLaMA-2-7B is the exception (Fig. 4, §4.1).
- Matched training tokens (5.8M, GSM8K): LLaMA-2-70B 82.1 → 83.5; LLaMA-2-13B 64.2 → 65.7 (Table 2). Adding more CoT paths to 6.8M gave 82.2.
- Specialized models (Table 3): WizardMath-70B 81.6 → 84.2; MetaMath-70B 82.3 → 85.4.
- Correction-centric evolution raises LLaMA-2-70B MATH from 25.0 to 29.3 (§1).
- Authors' reading: "a stronger backbone model can more effectively learn from mistakes" (§4.1).

## Limits relevant to ch-24

- Best-checkpoint-on-test reporting can inflate gains; the paper shows the three best checkpoints and training curves in App. D.1–D.2.
- No instruction-following, calibration, or non-reasoning evaluation beyond CSQA.
