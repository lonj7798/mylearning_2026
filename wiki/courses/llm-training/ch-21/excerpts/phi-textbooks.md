---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/phi-textbooks.md
source_url: https://arxiv.org/abs/2306.11644
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: "Textbooks Are All You Need" (phi-1)

This excerpt was rewritten on 2026-09-15 to match the verified library card `phi-textbooks` (checked 2026-09-14
against arXiv:2306.11644v2). The April 2026 version said the quality classifier was trained on hand-labeled seeds,
that later analyses found HumanEval overlap, and that the model trained on 800 GPU-hours; these are corrected below.

## Data (§2)
- **Filtered code-language data, about 6B tokens (§2.1).** Source pool: Python subset of deduplicated The Stack plus
  StackOverflow (over 35M files, over 35B tokens). GPT-4 labels about 100k samples with the prompt "determine its
  educational value for a student whose goal is to learn basic coding concepts". A random forest on the output
  embedding of a pretrained codegen model extends the label to the pool. The threshold is not reported.
- **Synthetic textbooks, under 1B tokens (§2.2).** GPT-3.5 text interleaved with code; diversity from constraints on
  topics and target audience.
- **CodeExercises, under 180M tokens, 879.5K problems (§2.2, §5.2).** Docstring-completion exercises; diversity from
  constraining function names.
- Filter effect at 350M (§2.1): unfiltered 12.19% HumanEval after 96k steps; filtered 17.68% after 36k steps;
  filtered plus synthetic textbooks 20.12%.

## Training (§2.3)
- phi-1-base: sequence length 2048; AdamW; LR 1e-3, 750 warm-up steps, weight decay 0.1; effective batch 1024;
  checkpoint at 24,000 of 36,000 planned steps, "∼ 8 epochs", "little over 50B total training tokens".
- phi-1 (finetuning on CodeExercises): batch 256; LR 1e-4; 50 warm-up steps; weight decay 0.01; 6,000 steps.

## Results
- phi-1-base 29% HumanEval (§2); phi-1 50.6% HumanEval, 55.5% MBPP (Table 1).
- 50 new problems graded by GPT-4: phi-1 52%, StarCoder 51%, phi-1-base 37% (Table 2).

## Contamination and similarity (§5)
- 4 HumanEval problems share a 13-gram with an exercise; all 4 are false positives (§5.1).
- Similarity split at τ = 0.95 (Table 3): phi-1 solves 81.7% of 71 HumanEval problems with close matches in
  CodeExercises and 26.9% of 93 without; StarCoder-Prompted 57.7% and 29.0%.
- Retrained on exercise sets pruned at τ from 0.95 to 0.8 (42.5K to 354K exercises removed), phi-1 scores 45.1–50.6%.

## Limitations stated by the authors (§1, §6, App. B)
Narrow task by design; Python only; weaker knowledge of specific APIs; lower robustness to stylistic variation and
grammatical errors in prompts; GPT-3.5 data "has a high error rate" (§6).

## Verification
- Values match the verified card `phi-textbooks`.
