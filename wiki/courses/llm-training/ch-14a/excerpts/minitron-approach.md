---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: arXiv:2408.11796v4 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2408.11796
created_at: "2026-09-15"
---

# Excerpt: LLM Pruning and Distillation in Practice: The Minitron Approach

- **Authors:** Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Gerald Shen, et al. (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-08; v4 2024-12)
- **Source type:** paper

## Method (Methodology, Training Details)
1. **Teacher correction.** Llama 3.1 8B and Mistral NeMo 12B are used as teachers; because they "are pretrained on different proprietary datasets, which we do not have access to", each teacher is fine-tuned on the distillation dataset (Nemotron-4 curated continued-training data) "with ∼100B tokens", using "120 steps of warm-up and low learning rates: one-fifth the peak learning rate, identical batch size, minimum learning rate and decay schedule the original model was trained on".
2. **Pruning** of width (hidden, MLP) or depth (layers) of the corrected teacher.
3. **Distillation.** The student is trained with "forward KL Divergence loss on the teacher and student logits only".

## Table 4 (distillation-based retraining)
| | Llama-3.1-Minitron (4B) | MN-Minitron (8B) |
|---|---|---|
| Peak LR | 1e-4 | 1e-4 |
| Min LR | 1e-5 | 4.5e-7 |
| Warm-up | 40 steps | 60 steps |
| LR decay schedule | Cosine | Cosine |
| Global batch size | 1152 | 768 |
| Context length | 8192 | 8192 |
| Total tokens | 94B | 380B |

## Table 1 (base models) and budget statements
- Training tokens: Llama-3.1-Minitron 4B-Depth and 4B-Width 94B each; MN-Minitron 8B 380B; Llama 3.1 8B 15T. Total / non-embedding parameters: 4B-Width 4.5B / 3.7B; MN-Minitron 8.4B / 7.3B.
- MMLU (5-shot): Llama-3.1-Minitron 4B-Depth 58.7, 4B-Width 60.5, Llama 3.1 8B 65.3, MN-Minitron 8B 69.5, Mistral NeMo 12B-Base 69.0.
- "According to the Llama 3.1 tech report, the 8B model is pretrained on 15T tokens" (Training Details; a secondary statement about another model).
- MN-Minitron-8B uses "40× fewer training tokens (380B vs. 15T)" and Llama-3.1-Minitron-4B "150× fewer training tokens (94B vs. 15T)" than Llama 3.1 8B (Base Models).

## Verification
- Checked on 2026-09-15 against arXiv:2408.11796v4 PDF text: Methodology, Training Details, Table 1, Table 4, Base Models paragraph.
- Not reported: distillation temperature; compute spent on teacher forward passes; a from-scratch baseline trained on the same 94B or 380B tokens.
