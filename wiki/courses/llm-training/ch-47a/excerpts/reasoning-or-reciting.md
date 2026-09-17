---
chapter: ch-47a
course: llm-training
phase: read
excerpt_of: "Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks (arXiv:2307.02477v3, v1 2023-07-05; NAACL 2024)"
source_url: https://arxiv.org/abs/2307.02477
created_at: "2026-09-15"
note: "No library card exists for this source on 2026-09-15 (planned slug reasoning-or-reciting). Values read from the v3 PDF on 2026-09-15."
---

# Excerpt: counterfactual task variants and the comprehension check

**Authors:** Zhaofeng Wu, Linlu Qiu, Alexis Ross, Ekin Akyürek, Boyuan Chen, Bailin Wang, Najoung Kim, Jacob Andreas, Yoon Kim (MIT; Boston University).

## Design (§2, §2.1, §3)

- A **counterfactual task variant** keeps the abstract task and changes one default assumption of the world in which it is performed: arithmetic in base 8/9/11/16 instead of base 10, Python with 1-based indexing, sorting by a different key, transposed musical keys, rotated spatial axes, altered chess rules, altered SET rules, swapped premises in syllogisms.
- 11 tasks in total. The point of the design is that a general procedure should transfer, because the counterfactual world is fully specified in the prompt.
- Each task is paired with a **counterfactual comprehension check (CCC)**: a question that tests only whether the model understood the stated counterfactual world, asked in a separate query (§2.1). A low counterfactual score with a high CCC score is evidence that the deficit is in executing the task, not in reading the instruction.
- Models: GPT-4 (gpt-4-0314), GPT-3.5, Claude (claude-v1.3), PaLM-2 (text-bison-001, not the largest version). Evaluated with and without 0-shot chain-of-thought (§4).

## Arithmetic results (Table 18; two-digit addition, accuracy %, 1,000 test instances, 200 CCC instances)

| Base | 8 | 9 | 10 | 11 | 16 |
|---|---|---|---|---|---|
| GPT-4, no 0-CoT | 82.3 | 23.4 | 100.0 | 38.4 | 63.0 |
| GPT-4, with 0-CoT | 60.2 | 38.6 | **98.2** | 56.5 | 74.0 |
| GPT-3.5, with 0-CoT | 12.6 | 9.8 | 99.0 | 2.7 | 17.7 |
| Claude, with 0-CoT | 1.4 | 0.9 | 98.7 | 4.0 | 6.6 |
| PaLM-2, with 0-CoT | 1.1 | 0.6 | 82.2 | 0.5 | 1.2 |
| GPT-4 CCC | 98.0 | 90.0 | 100.0 | 91.0 | 100.0 |

GPT-4 answers 98.2% of base-10 two-digit additions and 38.6% of base-9 additions under the same prompt and 0-shot CoT, while answering 90.0% of the base-9 comprehension check.

## Few-shot recovery (Table 19; GPT-4, 0-shot CoT)

| Digits | Shots | base 8 | base 9 | base 10 | base 11 | base 16 |
|---|---|---|---|---|---|---|
| 2 | 0 | 60.2 | 38.6 | 98.2 | 56.5 | 74.0 |
| 3 | 0 | 56.8 | 32.2 | 87.1 | 24.2 | 33.2 |
| 4 | 0 | 24.0 | 14.6 | 83.4 | 8.9 | 9.1 |
| 2 | 1 | 97.3 | 48.1 | 99.7 | 25.7 | 49.1 |
| 2 | 8 | 99.7 | 85.8 | 100.0 | 79.6 | 83.5 |
| 2 | 16 | 99.9 | 88.4 | 99.9 | 86.9 | 88.7 |

The counterfactual gap narrows with in-context examples but does not close at 16 shots, and it widens with digit count at 0 shots.

## Interpretation stated by the authors (§4, §5.1, §5.2)

- Counterfactual performance is above chance on most tasks and consistently below default performance (Figs. 2–3).
- §5.1: counterfactual performance rises with how common the counterfactual world is in text. Swapping north and south in spatial reasoning gives the smallest drop, and for PaLM-2 exceeds the default, which the authors attribute to plotting libraries with an inverted y-axis (matplotlib, ggplot, D3). Drop-D guitar tuning (DADGBE) gives the highest counterfactual score for chord fingering. They describe this as a memorization-like effect.
- The authors conclude that models hold abstract task-solving skills to an extent but also rely on narrow, non-transferable procedures.
- The authors note (§6) that a low CCC score does not perfectly separate "did not understand the world" from "cannot execute in it".

## Used in

ch-47a §3 (counterfactual variants and the comprehension check), §7 (audit protocol).
