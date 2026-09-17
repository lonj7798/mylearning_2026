---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-instruct.md (the library card has no Verification section and states the classification branch in reverse; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2212.10560
primary_version: arXiv:2212.10560v2 (2023-05, ACL 2023); v1 2022-12
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; rewritten from the primary source)"
---

# Excerpt: Self-Instruct: Aligning Language Models with Self-Generated Instructions

Authors: Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. Checked against the v2 PDF on 2026-09-15. Used by ch-19 `read.md` §2.

## Pipeline (§2.2)
> "Our pipeline for data generation consists of four steps: 1) generating task instructions, 2) determining if the instruction represents a classification task, 3) instance generation with either an input-first or output-first approach, and 4) filtering low-quality data."

> "We initiate the task pool with 175 tasks (1 instruction and 1 instance for each task). For every step, we sample 8 task instructions from this pool as in-context examples. Of the 8 instructions, 6 are from the human-written tasks, and 2 are from the model-generated tasks in previous steps"

## Classification branch (§2.2, footnote 4, Tables 7-8)
> "However, we found that this approach can generate inputs biased toward one label, especially for classification tasks (e.g., for grammar error detection, it usually generates grammatical input). Therefore, we additionally propose an Output-first Approach for classification tasks, where we first generate the possible class labels, and then condition the input generation on each class label."

> "We apply the output-first approach to the classification tasks identified in the former step, and the input-first approach to the remaining non-classification tasks."

Footnote 4: "we regard tasks that have a small limited output label space as classification tasks."

## Filtering (§2.2)
> "a new instruction is added to the task pool only when its ROUGE-L similarity with any existing instruction is less than 0.7. We also exclude instructions that contain some specific keywords (e.g., image, picture, graph) that usually can not be processed by LMs. When generating new instances for each instruction, we filter out instances that are exactly the same or those with the same input but different outputs. Invalid generations are identified and filtered out based on heuristics (e.g., instruction is too long or too short, instance output is a repetition of the input)."

## Model and data (§3, Table 1)
> "We use the largest GPT3 LM ("davinci" engine) accessed through the OpenAI API."

| Statistic (Table 1) | Value |
|---|---|
| instructions | 52,445 |
| classification instructions | 11,584 |
| non-classification instructions | 40,861 |
| instances | 82,439 |
| instances with empty input | 35,878 |

No count of raw generations before filtering is reported.

## Quality audit (§3.3, Table 2; 200 sampled instructions, 1 instance each, one expert annotator)
| Question | Yes |
|---|---|
| Does the instruction describe a valid task? | 92% |
| Is the input appropriate for the instruction? | 79% |
| Is the output a correct and acceptable response? | 58% |
| All fields are valid | 54% |

## Results
- SuperNI, 119 unseen tasks, zero-shot ROUGE-L (Table 3): GPT3 6.8; GPT3 + T0 Training 37.9; GPT3-SELF-INST 39.9; InstructGPT001 40.8.
- 252 user-oriented instructions rated by the instruction authors on a four-level scale (§4.4, Fig. 6, κ = 0.57): "if we count acceptable response with minor imperfections (RATING-B) as valid, GPT3SELF-INST is only 5% behind InstructGPT001."
- Data size (§4.5, Fig. 7): "this improvement almost plateaus after 16K" (human evaluation); on SuperNI "the model's performance gain plateaus earlier at around hundreds of instructions."
- Seed-test overlap (App. A.1): average ROUGE-L between seed instructions and their most similar test instruction is 0.21 (SuperNI) and 0.34 (user-oriented set); one seed instruction is identical to a user-oriented test instruction.

## Settings (App. A.2-A.3, Table 4)
| Step | Temp. | Top_P | Presence penalty | Max length |
|---|---|---|---|---|
| Generating instructions | 0.7 | 0.5 | 2 | 1024 |
| Identifying classification tasks | 0 | 0 | 0 | 3 |
| Generating instances | 0 | 0 | 1.5 | 300 |

- Generation cost "around $600" at $0.02 per 1,000 davinci tokens (December 2022).
- Fine-tuning via the OpenAI API with default hyperparameters, "prompt_loss_weight" 0, two epochs; cost $338 for the full data (App. A.3).
