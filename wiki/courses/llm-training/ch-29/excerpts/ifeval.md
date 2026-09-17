---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2311.07911 (primary text; no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2311.07911
created_at: "2026-09-15"
---

# Excerpt: IFEval — verifiable instructions and its four accuracy scores

**Artifact:** Zhou, Lu, Mishra, Brahma, Basu, Luan, Zhou, Hou, "Instruction-Following Evaluation for Large Language Models" (arXiv v1 2023-11; Google and Yale University). Loci are section and table numbers of arXiv v1.

## What it measures (Abstract, §2)

- A **verifiable instruction** is an atomic instruction whose compliance a "simple, interpretable, and deterministic program" can check, for example "write in more than 400 words" or "mention the keyword of AI at least 3 times" (Abstract, §2.1).
- 25 types of verifiable instructions; 541 prompts, each with one or more verifiable instructions (§2.1). Each instruction type has variants in its parameters (450 to 500 words vs 350 to 400 words) and in its phrasing (§2.1).
- Stated motivation: human evaluation is expensive and not reproducible, and LLM-judge evaluation "is potentially biased or limited by the ability of the evaluator LLM" (Abstract).

## Metrics (§2.2, §3)

- Strict check: `is_followed(resp, inst)` returns True if the instruction is followed (Eq. 1).
- Loose check: `is_followed_loose(resp, inst) = Any(is_followed(transform_t(resp), inst))` over 8 transformations built from three functions: remove markdown font modifiers "*" and "**", remove the first line, remove the last line, plus their pairwise and triple combinations and the identity (Eq. 2).
- The loose check reduces false negatives (for example "P.S. **I do like the cake**" failing a string match) and can add false positives (removing the first line can make a word-count check pass) (§2.2).
- Four reported scores (§3): prompt-level strict accuracy (share of prompts with all instructions followed), instruction-level strict accuracy, and the two loose versions.

## Reported results (Table 3)

| Model | Prompt strict | Inst strict | Prompt loose | Inst loose |
|---|---|---|---|---|
| GPT-4 (Nov 2023) | 76.89 | 83.57 | 79.30 | 85.37 |
| PaLM 2 S (Aug 2023) | 43.07 | 55.76 | 46.95 | 59.11 |

GPT-4 prompt-level loose minus strict is 2.41 points; for PaLM 2 S it is 3.88 points. This gap bounds how much a formatting change (for example bolding a required phrase) can move the strict score without any change in compliance.

## Limits stated by the authors (§4)

- The set of verifiable instructions should be made more diverse and larger.
- Only text; no multi-modal instructions.

## Use in ch-29

- IFEval is the instruction-following column of the lab's held-out suite. Report prompt-level strict and loose accuracy together.
- The lab's Evol-Instruct "add constraints" operator produces constraint-style instructions. Constraint types that match IFEval's 25 types can raise IFEval without broader instruction following; [[ifbench]] reports that models scoring well on IFEval's templates score lower on 58 unseen constraints (§1, Fig. 1). The lab therefore runs the contamination check of ch-29 §5 against IFEval prompts and, where possible, adds IFBench as an unseen instruction-following check.
- Instance-level noise: with 541 prompts and accuracy near 0.40, the binomial standard error is √(0.40 × 0.60 / 541) = 0.021, a 95% interval of about ±4.1 points (ch-29 §7).

## Connections

- [[ifbench]] — unseen verifiable constraints; generalization of instruction following beyond IFEval's types.
- [[tulu-3-sft-and-eval]] — IFEval (dev) and IFEval-OOD (unseen) in the Tülu 3 evaluation suite.
