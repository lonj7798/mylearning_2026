---
chapter: ch-46a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2311.07911v1 (planned library card papers/ifeval.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2311.07911
created_at: "2026-09-15"
---

# Excerpt: Instruction-Following Evaluation for Large Language Models (IFEval)

**Paper:** Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou (Google; Yale University). arXiv v1 2023-11-14. Source type: paper. Code and data: github.com/google-research/google-research/tree/master/instruction_following_eval.

## Benchmark (§1, Table 1, §2.1)

- "We create a list of 25 verifiable instructions. We further create a set of 541 prompts, with each prompt containing one or multiple verifiable instructions" (§1). The abstract says "around 500 prompts".
- The 25 instruction types are grouped as Keywords (4), Language (1), Length Constraints (4), Detectable Content (2), Detectable Format (6), Combination (2), Change Cases (3), Start with / End with (2), Punctuation (1) (Table 1; counts by row).
- Prompt construction: base prompts with one to three randomly selected verifiable instructions appended; few-shot prompting removes illogical prompts; few-shot rephrasing increases phrasing diversity; each prompt is then checked and edited manually (§2.1).

## Metrics (§2.2, §3)

1. Prompt-level strict accuracy: the percentage of prompts for which all verifiable instructions are followed.
2. Instruction-level strict accuracy: the percentage of verifiable instructions followed.
3. Prompt-level loose accuracy and 4. instruction-level loose accuracy: a response counts as following an instruction if any of eight transformed responses does. The transformations remove markdown font modifiers (`*`, `**`), remove the first line, remove the last line, their pairwise and triple combinations, and the identity (§2.2).
- The authors state that the loose criterion reduces false negatives but can introduce false positives, for example a word-count check passing after the first line is removed, so it is "a complement to the original criterion" (§2.2).

## Reported scores (Table 3)

| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose |
|---|---|---|---|---|
| GPT-4 (responses collected Nov 2023) | 76.89 | 83.57 | 79.30 | 85.37 |
| PaLM 2 S (Aug 2023) | 43.07 | 55.76 | 46.95 | 59.11 |

## Used in

ch-46a §7 (the instruction-following regression check of the generality gate; 541 prompts and prompt-level strict accuracy as the reported metric).
