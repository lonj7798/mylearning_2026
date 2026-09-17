---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lora-learns-less-forgets-less.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2405.09673
created_at: "2026-09-15"
---

# Excerpt: LoRA Learns Less and Forgets Less

**Authors:** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, et al. (Columbia University; Databricks Mosaic Research)
**Version read:** arXiv:2405.09673v2 (20 Sep 2024), "Published in Transactions on Machine Learning Research (08/2024)"; v1 May 2024.
**Status:** no library card existed for this slug on 2026-09-15; tables checked against the v2 PDF text.

## Setup (§3, App. A)
Llama-2-7B base. Code and math, each in continued pretraining (StarCoder-Python up to 20B tokens; OpenWebMath 14.7B tokens) and instruction fine-tuning (Magicoder-Evol-Instruct-110K, 72.97M tokens; MetaMathQA, 395K pairs, ~103M tokens). Learning: HumanEval pass@1, GSM8K. Forgetting: "the average of HellaSwag, ARC-Challenge and Winogrande". LoRA on all modules, r = 16, 64, 256, α = 2r, lora_dropout 0.05.
Code IFT settings: max_seq_len 4096; decoupled LionW (0.9, 0.95); "learning_rate: 2e-4 for rank r = 16, 64 and 1e-4 for r = 256 α = 2r = 512 (due to instabilities/loss spikes at 2e-4)"; cosine with warmup 0.1 dur; global batch 192. Math IFT: seq_len 1024; full FT LR 1e-5; LoRA 1e-4 (r = 16, 64), 5e-5 (r = 256); global batch 768.

## Tables used in ch-30a
Table S5 / S6 (Magicoder-Evol-Instruct-110K; HumanEval pass@1 / forgetting average), by epoch 1, 2, 4, 8, 16:
| Condition | HumanEval | Forgetting average |
|---|---|---|
| LoRA r=16 | .197 .275 .358 .338 .324 | .653 .648 .652 .646 .609 |
| LoRA r=256 | .299 .385 .498 .437 .466 | .655 .659 .631 .552 .517 |
| Full FT | .302 .464 .470 .497 .416 | .595 .579 .512 .446 .414 |

Table S2 (StarCoder-Python CPT, forgetting average at 20B tokens): LoRA r=256 0.617; full FT 0.545.
Table S8 (MetaMathQA, epoch 16): LoRA r=256 0.567; full FT 0.559.

## Statements used in ch-30a
- §4.2: "(1) IFT induces more forgetting than than CPT, (2) programming induces more forgetting than math, and (3) forgetting tends to worsen with training duration."
- §4.5: weight decay (5e-5, 1e-4) and attention dropout (0.05, 0.1) "appear to learn and forget as much as full finetuning"; "LoRA r = 256, on the other hand, learns as much as the other methods while forgetting less."
- §4.5: full fine-tuning gives fewer unique HumanEval generations than the base model, "with LoRA in between the two."
- §4.4: on Tülu-v2-mix, LoRA matches full fine-tuning on MT-bench, GSM8K, MMLU; "At longer training durations (6 epochs), LoRA also forgets less."
- App. B: best LoRA LRs 5e-4 (code) and 2e-4 (math); best full fine-tuning LRs 5e-5 and 1e-5.

## How ch-30a uses it
§2 (forgetting grows with epochs), §5.3 (LoRA as a control and its limits), Recipe rows.
