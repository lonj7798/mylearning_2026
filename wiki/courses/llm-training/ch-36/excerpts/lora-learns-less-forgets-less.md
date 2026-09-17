---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: primary source arXiv:2405.09673v2 (planned library card papers/lora-learns-less-forgets-less.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2405.09673
created_at: "2026-09-15"
---

# Excerpt: LoRA Learns Less and Forgets Less

**Paper:** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, et al. (Columbia University; Databricks Mosaic Research). arXiv v1 2024-05; read at v2, "Published in Transactions on Machine Learning Research (08/2024)". Source type: paper.

## Setup (§3, §4.1, App. A)

- Base model `meta-llama/Llama-2-7b-hf`. Instruction fine-tuning (IFT) on Magicoder-Evol-Instruct-110K (code) and MetaMathQA (math); continued pretraining on StarCoder-Python and OpenWebMath.
- "For IFT, we train separate models for 1, 2, 4, 8, and 16 epochs." "For each condition, we train one full finetuning model and three LoRA models with ranks r = 16, 64, 256" (§4.1). No seeds or run-to-run variance are reported.
- LoRA targets all transformer modules (W_q, W_k, W_v, W_o, W_gate, W_up, W_down), α = 2r, lora_dropout 0.05 (App. A).
- Learning metrics: HumanEval pass@1 (50 generations per problem, temperature 0.2, top_p 0.95); GSM8K test split, 1,319 samples, 5-shot (§3.2).
- Forgetting metric: the average of HellaSwag, WinoGrande, and ARC-Challenge accuracy, computed from logits in the MosaicML Gauntlet harness (§3.3).
- Code IFT settings: max_seq_len 4096; decoupled LionW (0.9, 0.95); LoRA LR 2e-4 (r = 16, 64) and 1e-4 (r = 256, "due to instabilities/loss spikes at 2e-4"); cosine with warmup 0.1 of duration; weight decay 0; global batch 192; gradient clipping norm 1; 32 GPUs (App. A). The full fine-tuning LR is not printed in this block. App. B (2-epoch sweep): best LoRA LR 5e-4 (code), 2e-4 (math); best full fine-tuning LR 5e-5 (code), 1e-5 (math).

## Tables S5 and S6 (Magicoder-Evol-Instruct-110K), epochs 1, 2, 4, 8, 16

| Condition | HumanEval pass@1 | Forgetting average |
|---|---|---|
| LoRA r = 16 | 0.197, 0.275, 0.358, 0.338, 0.324 | 0.653, 0.648, 0.652, 0.646, 0.609 |
| LoRA r = 64 | 0.249, 0.339, 0.417, 0.392, 0.405 | 0.652, 0.651, 0.632, 0.580, 0.510 |
| LoRA r = 256 | 0.299, 0.385, 0.498, 0.437, 0.466 | 0.655, 0.659, 0.631, 0.552, 0.517 |
| Full fine-tuning | 0.302, 0.464, 0.470, 0.497, 0.416 | 0.595, 0.579, 0.512, 0.446, 0.414 |

Table S8 (MetaMathQA forgetting average, epochs 1-16): full fine-tuning 0.598, 0.599, 0.590, 0.572, 0.559; LoRA r = 256 0.613, 0.607, 0.599, 0.584, 0.567.

§4.2 (verbatim): "(1) IFT induces more forgetting than than CPT, (2) programming induces more forgetting than math, and (3) forgetting tends to worsen with training duration."

## Diverse mixture: Tülu-v2-mix (App. C)

- About 326k samples; up to 6 epochs; evaluated after 2, 4, and 6 epochs. "Different from the main IFT experiments, the checkpoints evaluated are 'hot' and are not cooled down for each training duration" (App. C).
- Settings (App. C.1): max_seq_len 4096; decoupled LionW (0.9, 0.95); LR full fine-tuning 5e-6, LoRA 1e-4; cosine with warmup 0.1 of duration; weight decay 0; global batch 192; gradient clipping 1; 32 GPUs.
- Tables S10-S13, epochs 2, 4, 6:

| Condition | MT-Bench | MMLU | GSM8K | Forgetting average |
|---|---|---|---|---|
| LoRA r = 16 | 5.681, 5.997, 5.712 | 0.491, 0.502, 0.504 | 0.251, 0.275, 0.280 | 0.650, 0.657, 0.657 |
| LoRA r = 64 | 5.597, 5.725, 5.944 | 0.503, 0.509, 0.504 | 0.285, 0.270, 0.295 | 0.649, 0.655, 0.647 |
| LoRA r = 256 | 5.788, 5.834, 5.894 | 0.502, 0.496, 0.492 | 0.296, 0.335, 0.301 | 0.653, 0.649, 0.629 |
| Full fine-tuning | 5.825, 5.838, 5.862 | 0.507, 0.504, 0.502 | 0.324, 0.291, 0.303 | 0.660, 0.652, 0.621 |

- App. C.2: base MT-Bench 2.74 and base GSM8K 0.145. "All LoRA models are within one standard error of the mean of the full finetuning model (computed with 160 datapoints = 80 questions × 2 turns)"; "only 80 questions appear in this benchmark and ... the variance, within model, is high." "At two epochs, full finetuning does better than LoRA" on forgetting; at epoch 6 "full finetuning forgets the most and we find a clear ordering of forgetting by rank."

## Used in

ch-36 §3 (narrow code data), §4.1 (epochs and hot checkpoints), §4.3 (LoRA versus full fine-tuning), §6.1 (small evaluation sets), Recipe rows.
