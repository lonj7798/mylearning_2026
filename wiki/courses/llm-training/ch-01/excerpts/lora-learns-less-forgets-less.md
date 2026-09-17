---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lora-learns-less-forgets-less.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2405.09673
created_at: "2026-09-15"
---

# Excerpt: LoRA Learns Less and Forgets Less

**Authors:** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, et al. (Columbia University, Databricks Mosaic Research)
**Version read:** arXiv:2405.09673v2 (2024-09-20); v1 2024-05. TMLR 08/2024. Source type: paper.
**Status:** no library card existed for this slug on 2026-09-15. The forgetting chapter ch-30a has its own excerpt.

## Setting (§3, App. A)
- Base model Llama-2-7B. Code: StarCoder-Python continued pretraining (up to 20B tokens) and Magicoder-Evol-Instruct-110K instruction fine-tuning (72.97M tokens). Math: OpenWebMath (14.7B tokens) and MetaMathQA (103M tokens) (Table 1).
- Learning: HumanEval pass@1 (code), GSM8K (math). Forgetting average: mean of HellaSwag, ARC-Challenge, WinoGrande accuracy; higher means less forgetting (§3.3).
- Optimizer in all main-text experiments: decoupled LionW, not AdamW (App. A). Code IFT: betas (0.9, 0.95), LoRA LR 2e-4 for r = 16 and 64 and 1e-4 for r = 256 (loss spikes at 2e-4), cosine with 10% warmup, weight decay 0, bf16, global batch 192, gradient clipping by norm with threshold 1, 32 GPUs.
- Learning-rate sweep (App. B, Fig. S1, 2 epochs of IFT): best LoRA LRs 5e-4 (code) and 2e-4 (math); best full fine-tuning LRs 5e-5 and 1e-5, "an order of magnitude smaller".

## Results used by ch-01
Table S5 (Magicoder IFT, HumanEval pass@1) and Table S6 (forgetting average), by epoch 1 / 2 / 4 / 8 / 16:

| Condition | HumanEval | Forgetting average |
|---|---|---|
| LoRA r = 16 | 0.197 / 0.275 / 0.358 / 0.338 / 0.324 | 0.653 / 0.648 / 0.652 / 0.646 / 0.609 |
| LoRA r = 256 | 0.299 / 0.385 / 0.498 / 0.437 / 0.466 | 0.655 / 0.659 / 0.631 / 0.552 / 0.517 |
| Full fine-tuning | 0.302 / 0.464 / 0.470 / 0.497 / 0.416 | 0.595 / 0.579 / 0.512 / 0.446 / 0.414 |

- §4.2: "(1) IFT induces more forgetting than than CPT, (2) programming induces more forgetting than math, and (3) forgetting tends to worsen with training duration."
- §4.5 (Fig. 4): weight decay (5e−5, 1e−4) and attention dropout (0.05, 0.1) "appear to learn and forget as much as full finetuning, except that weight decay starts to generally deteriorate at longer training durations (epochs 8 and 16)."
- §4.5 (Fig. 5): full fine-tuning gives fewer unique HumanEval generations (out of 50) than the base model; LoRA lies in between.
- §1 and §4.6 (Fig. 6): the rank of the full fine-tuning perturbation "grows as training progresses, with ranks 10-100× higher than typical LoRA configurations."

## Limits
One base model (Llama-2-7B); number of seeds not reported; forgetting suite is three multiple-choice benchmarks.

## Verification
- Checked on 2026-09-15 against arXiv:2405.09673v2 (Abstract, §3.3, §4.2, §4.5, App. A, App. B, Tables S5-S6).
