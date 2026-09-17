---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/super-natural-instructions.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2204.07705
primary_version: arXiv:2204.07705v3 (v1 2022-04; EMNLP 2022)
created_at: "2026-09-15"
---

# Excerpt: Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Checked against the v3 PDF on 2026-09-15.

## Benchmark (§3, Table 2)
- 1,616 tasks; 76 task types; 55 languages; 33 domains; 576 non-English tasks; average definition length 56.6 words; 2.8 positive and 2.4 negative examples per task; 3,106.0 instances per task on average (Table 2).

> "We limit the number of instances in each task to 6.5K to avoid an imbalance of instances between tasks."

## Evaluation split (§5.1 and footnote 6)
> "For evaluation tasks, we fix a manually-selected collection of 12 categories that represent 154 tasks. [...] we sample a maximum of 100 instances for each task, which results in 15,310 testing instances in total. The remaining tasks are used for training models."

> "To avoid data leakage, we exclude tasks from the training set if they are sourced from the same dataset as any test task. This results in 757 training tasks for the English track and 1271 training tasks for the cross-lingual track."

## Overall results (Table 3, ROUGE-L, English / cross-lingual)
| Method | En | X-lingual |
|---|---|---|
| Copying instance input | 14.2 | 5.4 |
| Copying demo output | 28.5 | 50.3 |
| T5-LM (11B) | 30.2 | – |
| GPT3 (175B) | 45.0 | 51.3 |
| T0 (11B) | 32.3 | – |
| InstructGPT (175B) | 52.1 | 52.8 |
| Tk-Instruct (11B) | 62.0 | – |
| mTk-Instruct (13B) | 57.1 | 66.1 |
| Supervised training (upper bound) | 74.3 | 94.0 |

## Scaling (§7.1)
> "The model generalization performance grows log-linearly as we increase the set of tasks used for training."

> "in our setup, the model's performance saturates when only 64 instances per task are used for training."

> "a T5-large model trained with 757 tasks can achieve comparable performance (48.0 ROUGE-L) to the T5-3B model trained with 128 tasks (48.4 ROUGE-L), indicating that increasing the diversity of training tasks is an alternative to scaling model sizes."

## Instruction elements (§7.2, Table 4 diagonal, T5-3B)
- Def 45.0; Def + Pos (2) 54.3; Def + Pos (2) + Neg (2) 54.3; Def + Pos (2) + Neg (2) + Expl 52.6.

> "Negative examples help a little bit; explanations decrease performance"

> "The negative result here is that definition-only models cannot generalize to example-only test encodings; and similarly, example-only models cannot generalize to definition-only test encodings."

## Implementation (App. D)
> "These experiments are run on Google V3-256 TPUs using a batch size of 1,048,576 tokens (1,024 examples), a constant learning rate of 1e-5 and a total of 1000 steps."

> "When fine-tuning models, we train them for two epochs with a batch size of 16 and a constant learning rate of 1e-5. The maximum input length is set to 1024, and the maximum output length is set to 128."
