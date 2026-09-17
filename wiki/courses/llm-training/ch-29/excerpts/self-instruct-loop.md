---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2212.10560 (primary text; the library card [[self-instruct]] had no Verification section on 2026-09-15)
source_url: https://arxiv.org/abs/2212.10560
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; re-read against the arXiv PDF)"
---

# Excerpt: Self-Instruct generation and filtering, as used in ch-29 §2

**Artifact:** Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (arXiv v1 2022-12; ACL 2023). Loci are section and table numbers of the arXiv PDF.

## Pipeline (§2.2)

1. **Task pool.** The pool starts with 175 tasks, each with 1 instruction and 1 instance. The seed tasks were written by the authors and their labmates at UW (§2.2, footnote 3).
2. **Instruction generation.** Each step samples 8 task instructions from the pool as in-context examples: 6 from the human-written tasks and 2 from model-generated tasks of previous steps (§2.2). The prompt is "Come up with a series of tasks:" followed by "Task 1: …" to "Task 8: …" and "Task 9:" (Table 5). Generation stops at the model's length limit or when it produces "Task 16" tokens (Table 5 caption).
3. **Classification identification.** A few-shot prompt asks "Can the following task be regarded as a classification task with finite output labels?" (§2.2, Table 6). Classification tasks are those with a small, limited output label space (footnote 4).
4. **Instance generation.** Non-classification tasks use the **input-first** approach: generate the input, then the output. Classification tasks use the **output-first** approach: generate the possible class labels first, then an input conditioned on each label, because input-first generation produced inputs biased toward one label, for example grammatical inputs for grammar-error detection (§2.2, Tables 7–8).
5. **Filtering and postprocessing (§2.2).**
   - A new instruction enters the pool only when its ROUGE-L similarity with every existing instruction is below 0.7.
   - Instructions containing keywords such as "image", "picture", or "graph" are excluded.
   - Instances that are exactly the same, or that have the same input and different outputs, are removed.
   - Invalid generations are removed by heuristics, for example an instruction that is too long or too short, or an output that repeats the input. The paper text gives no numeric length thresholds.

## Statistics (§3.1, Table 1)

| Statistic | Value |
|---|---|
| instructions | 52,445 (11,584 classification, 40,861 non-classification) |
| instances | 82,439 (35,878 with empty input) |
| mean instruction / non-empty input / output length (words) | 15.9 / 12.7 / 18.9 |
| generator | GPT-3 "davinci" engine through the OpenAI API (§3) |

The number of raw generations before filtering and the survival rate of each filter are not reported in the paper text.

## Quality of the kept data (§3.3, Table 2)

An author labeled 200 random instructions, 1 instance each:

| Question | Yes |
|---|---|
| Does the instruction describe a valid task? | 92% |
| Is the input appropriate for the instruction? | 79% |
| Is the output a correct and acceptable response? | 58% |
| All fields are valid | 54% |

The authors note that most erroneous instances are still in the correct format or partially correct (§3.3). The filters above check format and similarity, not correctness.

## Diversity measurement (§3.2)

- Verb–noun structure: 26,559 of 52,445 instructions have a parseable root verb with a direct noun object; the top 20 verbs with their top 4 objects cover 14% of the set (Figure 3).
- Novelty against seeds: for each generated instruction, the highest ROUGE-L against the 175 seeds is plotted (Figure 4).

## Use in ch-29

- The lab keeps the 6 + 2 in-context sampling, the ROUGE-L < 0.7 pool rule, and the output-first branch for classification.
- The keyword and heuristic filters become the lab's format filter; the lab records the survival rate that the paper does not report.
- Table 2 is the reason the lab adds a verifier for checkable subsets: format filters kept a set in which 58% of sampled outputs were correct.

## Connections

- [[self-instruct]] — library card for the same paper (not verified at the time of this revision; its sentence "input-first for classification" is reversed relative to §2.2).
- [[evol-instruct]] — WizardLM uses the 52K Alpaca data, produced with this method, as seed.
- [[alpaca]] — re-run of this pipeline with a different generator.
