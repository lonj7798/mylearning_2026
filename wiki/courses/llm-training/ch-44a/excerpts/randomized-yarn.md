---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2606.23687v2 (Randomized YaRN), §2.2, §3, §4, Tables 1-4 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2606.23687
created_at: "2026-09-15"
---

# Excerpt: Randomized YaRN Improves Length Generalization for Long-Context Reasoning

- **Authors:** Manas Mehta, Fangcong Yin, Greg Durrett (New York University)
- **Year:** 2026 (arXiv v1 2026-06; this extract read from v2, 2026-08-30)
- **Source type:** paper
- **Used in:** ch-44a §5, §6.

## Method (§2.2)
Training runs on short-context data with LoRA on all linear layers and YaRN encodings at scale factor `s`. For each batch a
sorted index set `I^(t) ∼ U(P^{L}_{L_t})` of size L is drawn from `{1, …, L_t}`, and token j receives `YaRN(i_j; s)`
instead of `YaRN(j; s)`; token order is preserved but the absolute indices span a much larger range than the sequence.
`L_t` grows across epochs as a length-generalization curriculum (for example `L_t ∈ {8K, 16K, 24K, …}`). At inference,
standard YaRN is used with scale `s'`; randomized sampling is training-only.

## Setup (§3)
Fine-tuning data is small and short: BABILong 4K examples from the 0-8K bin; MRCR 60 examples of 4-8K context; LongBench v2
all 116 examples of ≤32K context. Models: Qwen2.5-7B-Instruct (32K native), Olmo3-7B-Instruct (64K native, pretrained with
YaRN), Qwen3-14B (40K native, thinking mode disabled). Baselines: zero-shot, zero-shot + YaRN at inference, vanilla LoRA,
Trained YaRN, RPE, PoSE.

## Results
- BABILong out-of-distribution average (16K-128K), Table 1: Qwen2.5-7B-Instruct — LoRA 83.6, Trained YaRN 81.3, RPE 79.7,
  PoSE 78.1, Randomized YaRN 90.3; at 128K, 63.0 / 67.2 / 72.8 / 70.2 / 83.9. Olmo3-7B-Instruct 83.2 (Trained YaRN) vs 88.0;
  Qwen3-14B 88.8 vs 92.6.
- MRCR OOD average (Table 2): Qwen2.5-7B-Instruct 64.4 (LoRA), 59.3 (Trained YaRN), 47.2 (PoSE), 72.7 (Randomized YaRN);
  Qwen3-14B 78.3 vs 91.4.
- LongBench v2 (Table 3, both bins OOD): Qwen2.5-7B-Instruct 39.7 (Trained YaRN) vs 41.2; Olmo3-7B-Instruct 32.7 vs 37.1.
- Inference-time YaRN alone "only improves the average OOD accuracy by 1%" on BABILong.
- Curriculum ablation (Table 4, MRCR, Qwen2.5-7B-Instruct): without the curriculum the OOD average falls from 72.7 to 66.5
  for Randomized YaRN and from 61.1 to 57.0 for RPE.

## Stated limitations (Limitations section)
The method applies where the evaluation context is much longer than the training context; experiments cover 7B and 14B
models only; evaluation is English-only. The method is supervised fine-tuning with LoRA, not RL.

## Not reported
Short-context scores after training, LoRA rank and learning rate in the main text (they are in App. C), and interaction
with RL stages.
