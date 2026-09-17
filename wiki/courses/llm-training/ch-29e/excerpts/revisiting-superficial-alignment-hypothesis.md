---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/revisiting-superficial-alignment-hypothesis.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2410.03717
primary_version: arXiv:2410.03717v1 (2024-09)
created_at: "2026-09-15"
---

# Excerpt: Revisiting the Superficial Alignment Hypothesis

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Authors: Mohit Raghavendra, Vaskar Nath, Sean Hendryx. Checked against the v1 PDF on 2026-09-15.

## Claims examined (§1)
The paper lists the hypothesis's claims as (labels and order as printed):
- "C1: A model's knowledge is learned entirely during pre-training."
- "C3: Post-training is largely about style and doesn't does not teach a model new capabilities."
- "C2: A small number of examples can saturate a model's performance for a given task."

## Power law (§3.2, App. A.6)
> "Task accuracy P closely follows a power-law of the form P ∝ D^(1/b) with the number of finetuning examples D, for all the models on all the tasks."

> "Coefficients for the power law curves of the form P = aD^(1/b) are in Figure 1 are in Table 7."

Table 7 (selected): Math llama-3-8b a 19.47, b 7.66; Math mistral-7b 8.14, 4.97; Math llama-2-7b 4.21, 5.47; Multihop QnA llama-3-8b 11.54, 7.77; Coding llama-3-8b 27.14, 23.43; IF-Conifer llama-3-8b 31.06, 11.90; IF-Dolly llama-3-8b 30.94, 21.12.

## LIMA-style versus task data (§3.3, Table 2)
| Task | LIMA-1k accuracy | Task-1k accuracy | LIMA-1k win rate | Task-1k win rate | Neither |
|---|---|---|---|---|---|
| Math (GSM8k Test) | 14.7% | 46.5% | 84.4% | 0.24% | 14.2% |
| Multihop QnA (SubQA Test) | 21% | 36% | 20% | 57% | 23% |

> "We use the same prompt from LIMA to judge wins between responses and use GPT-4o (OpenAI, 2024) to predict wins."

## Style versus reasoning errors (§4.2)
> "From Figure 3 we see that the models get better at style and format errors with just 100 examples."

> "The total number of mistakes a model makes highly correlates with reasoning errors (r2 value of 0.98 for math and 0.99 for multihop QnA on Llama-3 8B)"

## New knowledge (§5, Tables 3-4; Facts100, events after March 2023)
| | Event SFT direct | Event SFT multihop | RAG-Oracle direct | RAG-Oracle multihop |
|---|---|---|---|---|
| Base Model | 65 | 37 | 49 | 34 |
| Post-trained Model | 81 | 55 | 86 | 71 |

## Training parameters (§3.1, App. A.5)
> "every model in a model family was trained with the same set of hyperparameters for a given task and dataset size, for 3 epochs over the base model with the default chat template from HuggingFace."

> "No PeFT methods were used, and the learning rate was set to 1e-5 with cosine decay to 0. We found that batch size has a big effect on model performance for smaller dataset sizes."

Table 6 batch sizes: 0-100 examples: 2 (8b and 13b), 16 (70b); 101-1000: 8, 32; 1001+: 16, 128.

## Limitations (§7)
> "frontier LLMs are trained to excel at multiple tasks, and we don't thoroughly understand how finetuning for one task or domain affects the performance on others."
