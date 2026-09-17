---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2408.11796v4 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2408.11796
created_at: "2026-09-15"
---

# Excerpt: LLM Pruning and Distillation in Practice: The Minitron Approach

- **Authors:** Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Gerald Shen, et al. (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-08; v4 2024-12-09)
- **Source type:** paper
- **Used in:** ch-35 §1, §6.2, Recipe

## Method
1. **Teacher correction.** The unpruned teacher is fine-tuned on the distillation dataset for about 100B tokens with 120 warm-up steps and one-fifth of the original peak learning rate, because an uncorrected teacher "provides sub-optimal guidance if a different dataset is used to distill the knowledge". Teacher correction "yields over a 6% reduction in LM validation loss" (Insights, General 1).
2. **Pruning.** Width (hidden, MLP) or depth (contiguous layers) pruning of the corrected teacher.
3. **Distillation.** "We opt for logit-only distillation, minimizing the forward KL Divergence loss across the teacher and student probabilities, and ignore the LM cross-entropy loss altogether."

## Settings (Table 4; Nemotron-4 curated continued-training dataset; 32 DGX H100 nodes)
| Student | Peak LR | Min LR | Warm-up | Schedule | Global batch | Context | Tokens |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Minitron-4B (from Llama 3.1 8B) | 1e-4 | 1e-5 | 40 steps | cosine | 1152 | 8192 | 94B |
| MN-Minitron-8B (from Mistral NeMo 12B) | 1e-4 | 4.5e-7 | 60 steps | cosine | 768 | 8192 | 380B |

## Results
- MN-Minitron-8B needs "up to 40×" fewer training tokens than training from scratch ("380B instead of 15T tokens").
- MN-Minitron-8B exceeds its teacher on GSM8K (55.7% → 58.5%) and HumanEval (23.8% → 36.2%); the authors write that this "is likely influenced by the dataset".
- Llama-3.1-Minitron-4B base: width pruning MMLU 60.5 and GSM8K 41.24; depth pruning 58.7 and 16.8 (Table 1, Insights).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2408.11796 (v4): Abstract, Methodology, Training Details, Table 4, Analysis, Insights, Tables 1–2.
- Not reported by the source: distillation temperature; optimizer betas.
