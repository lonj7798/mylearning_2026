---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/flan-palm-scaling-instruction-finetuning.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2210.11416
primary_version: arXiv:2210.11416v5 (v1 2022-10)
created_at: "2026-09-15"
---

# Excerpt: Scaling Instruction-Finetuned Language Models (Flan-PaLM, Flan-T5)

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Checked against the v5 PDF on 2026-09-15.

## Data and held-out control (§2.1, §2.3)
> "Our finetuning data comprises 473 datasets, 146 task categories, and 1,836 total tasks." (Figure 2 caption)

> "We removed 44 tasks related to MMLU (Hendrycks et al., 2020), since MMLU is used for evaluation." (footnote 4)

> "We do not use the evaluation set from Brown et al. (2020) since almost all of those tasks have training sets that are included in our finetuning mixture." (§2.3)

> "if random guessing produces 50% accuracy and the max accuracy of 100%, then a raw accuracy of 55% would be be normalized to 10%, and a raw accuracy of 45% would be normalized to -10%" (footnote 5)

## Checkpoint selection and compute (§2.2)
> "For each model, we use a single checkpoint for all evaluations; the optimal step was chosen based on periodic evaluations (every 2k to 10k steps depending the model size) of the held-out tasks"

> "we only use 0.2% of the pre-training compute to instruction-finetune Flan-PaLM 540B (approximately 512 v4 TPU chips for 37 hours)."

## Scaling (§3, Table 3, normalized average)
| Model | 0 tasks | 9 | 89 | 282 | 1,836 |
|---|---|---|---|---|---|
| 8B | 6.4 | 8.3 | 14.8 | 20.5 | 21.9 |
| 62B | 28.4 | 29.0 | 33.4 | 37.9 | 38.8 |
| 540B | 49.1 | 52.6 | 57.0 | 57.5 | 58.5 |

MMLU-direct for the CoT-only (9-task) row: 62B 55.1 → 48.5; 540B 71.3 → 68.8 (Table 3).

> "increasing the number of finetuning tasks improves performance, although the majority of the improvement comes from using up to 282 tasks."

> "the pre-training data consists of 780B tokens, while instruction finetuning only uses 1.4B tokens (0.2% of the pre-training tokens)."

> "although the absolute gain was larger for the 8B model than the 540B model (15.5% for 8B vs. 9.4% for 540B), the relative reduction in error rate was larger for the 540B model (18.4% for 540B vs. 16.6% for 8B)."

## Chain-of-thought data (§4.1-4.2)
> "finetuning on only non-CoT degrades performance on CoT by a substantial amount"

> "instruction finetuning improves unseen tasks when the unseen tasks are in the same prompting paradigm as the finetuning tasks (i.e., non-CoT or CoT)."

> "Flan-PaLM with CoT + SC achieves a new state of the art of 83.9%, though note that the GSM8K training dataset is included in the instruction finetuning mixture."

## Usability (§6)
> "across 190 examples, Flan-PaLM generations were preferred 79% of the time."

## Hyperparameters (App. E, Tables 22-23)
| Model | Batch size | Dropout | LR | Steps |
|---|---|---|---|---|
| Flan-T5-XL (3B) | 64 | 0.05 | 5e-4 | 38k |
| Flan-T5-XXL (11B) | 64 | 0.05 | 5e-4 | 14k |
| Flan-PaLM 8B | 32 | 0.05 | 3e-3 | 40k |
| Flan-PaLM 62B | 32 | 0.05 | 3e-3 | 40k |
| Flan-PaLM 540B | 32 | 0.1 | 1e-3 | 21k |

| Mixture | Maximum cap | Proportion (A) | Proportion (B) |
|---|---|---|---|
| Muffin | 30,000 | 52% | 46.0% |
| T0-SF | 20,000 | 15% | 27.9% |
| CoT | 100,000 | 3% | 1.8% |
| NIV2 | 5,000 | 30% | 24.2% |

> "Proportion A was used in scaling and the ablation sections in Section 3 and Section 4. The rest of the experiments used Proportion B, since Proportion A signaled that the T0 mixture was good for performance."
