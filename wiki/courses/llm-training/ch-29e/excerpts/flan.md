---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/flan.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2109.01652
primary_version: arXiv:2109.01652v5 (v1 2021-09; ICLR 2022)
created_at: "2026-09-15"
---

# Excerpt: Finetuned Language Models Are Zero-Shot Learners (FLAN)

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Checked against the v5 PDF on 2026-09-15.

## Task clusters and templates (§2.1)
> "We aggregate 62 text datasets that are publicly available on Tensorflow Datasets, including both language understanding and language generation tasks, into a single mixture. [...] each dataset is categorized into one of twelve task clusters"

> "For each dataset, we manually compose ten unique templates that use natural language instructions to describe the task for that dataset. While most of the ten templates describe the original task, to increase diversity, for each dataset we also include up to three templates that 'turned the task around'"

## Evaluation split (§2.2 and footnote 1)
> "In this work, we only consider dataset D unseen at evaluation time if no datasets from any task clusters that D belongs to were seen during instruction tuning."

> "Hence, to evaluate zero-shot FLAN on c task clusters, we instruction tune c models, where each model holds out a different task cluster for evaluation."

> "We also drop the paraphrase cluster from instruction tuning when evaluating on NLI tasks and vice-versa."

## Training details (§2.4 and footnote 2)
> "we limit the number of training examples per dataset to 30k and follow the examples-proportional mixing scheme (Raffel et al., 2020) with a mixing rate maximum of 3k."

> "In this mixing scheme, a mixing rate maximum of 3,000 means that a dataset does not receive additional sampling weight for examples in excess of 3,000."

> "We finetune all models for 30k gradient steps with a batch size of 8,192 tokens using the Adafactor Optimizer (Shazeer & Stern, 2018) with a learning rate of 3e-5. The input and target sequence lengths used in finetuning are 1024 and 256, respectively."

> "This instruction tuning takes around 60 hours on a TPUv3 with 128 cores. For all evaluations, we report results on the final checkpoint trained for 30k steps."

## Results (§3, §4.1-4.3)
> "With the best dev template, zero-shot FLAN outperforms zero-shot GPT-3 on 20 of 25 datasets"

Figure 6 data labels (average over the held-out NLI, closed-book QA, and commonsense clusters), read from the rendered page 6:

| Training clusters | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Datasets | 11 | 20 | 26 | 30 | 34 | 37 | 39 |
| Cluster added | summarization | translation | read. comp. | sentiment | data to text | coreference | conv. QA |
| Held-out average | 49.9 | 55.0 | 59.3 | 59.2 | 60.8 | 61.9 | 63.5 |

> "The behavior on held-out tasks for the 8B and smaller models, however, is thought-provoking—instruction tuning actually hurts performance on held-out tasks." (§4.2; Figure 7 plots 422M, 2B, 8B, 68B, 137B without printed values)

Table 3 (App. B.2), four-task-cluster average: natural instructions / natural instructions (FLAN) 55.2; no template / natural instructions 37.3; task/dataset name / natural instructions 46.6; task/dataset name / task/dataset name 47.0.

## Datasets versus templates (App. B.1)
> "Using more datasets per cluster improved performance by almost 10% on average across the three held-out clusters. Using more templates per dataset, however, had a comparatively negligible effect on performance when there was one task per cluster, which disappeared when there were four tasks per cluster."

## Contamination (App. C)
> "Of the remaining datasets, only ReCoRD and PIQA had a clean subset performance that was lower than the overall evaluation set performance by more than 1%."

## Completion-style tasks (§3)
> "For seven commonsense reasoning and coreference resolution tasks (see Table 2 in the Appendix), FLAN only outperforms LaMDA-PT on three of the seven tasks."
