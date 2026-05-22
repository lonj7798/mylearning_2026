---
chapter: ch-12
course: model-quantization
phase: read
excerpt_of: "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"
source_url: https://arxiv.org/abs/2309.14717
created_at: "2026-05-21"
---

# Excerpt: QA-LoRA — group-wise adapter that merges natively into INT4

**Authors:** Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian
**Year:** 2023 (ICLR 2024)
**URL:** https://arxiv.org/abs/2309.14717
**Raw-data source:** [[raw-data/qa-lora]]

---

## The QLoRA deployment problem

After QLoRA training you have **INT4 base + BF16 LoRA**. To deploy:
- **Keep both** → inference cost is ~FP16 (you carry around the LoRA matmul).
- **Merge LoRA into base** → re-quantize `dequant(W_q) + ΔW` → information loss, 0.5–2 PPL regression.

QA-LoRA traces this to an **imbalanced degrees-of-freedom problem**: LoRA's `B A` is fully dense per-element (`d_out × d_in` DOF); the INT4 base has only one scale per group (~`d_out × d_in / G` DOF). They don't merge cleanly.

---

## The fix — group-wise additive scalar

Replace LoRA's `B A` (full-rank-r dense) with a **group-wise additive scalar** `δ ∈ R^{d_out × (d_in/G)}` — one trainable scalar per `(output, group)` cell, **exactly matching the quantization grid granularity**:

```math
y = \mathrm{dequant}(W_q,\ s + \delta) \cdot x
```

(or equivalently, δ added to a per-group zero-point.)

This is a `(d_out × d_in/G)`-sized correction per layer. Total trainable parameters comparable to LoRA at rank `r ≈ d_in / G`.

---

## The merge step (lossless by construction)

Post-training:

```
s_new ← s + δ
```

The deployed weight is the **unchanged INT4 W_q** with the updated scales. No re-quantization, no FP16 LoRA branch at inference.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Base quant | INT4 group-wise (G = 32 or 128) |
| Adapter granularity | one scalar per `(d_out, group)` |
| Optimizer | AdamW, lr 2e-4 |
| Datasets tested | Alpaca, GSM8K |
| Deployment | **pure INT4** after merge |

---

## Why this is the right granularity

The INT4 representation has DOF concentrated in:
- per-group scales `s`: `(d_out × d_in/G)`
- per-weight 4-bit indices: dense but discrete (not continuously adjustable)

LoRA `BA` adds continuous-real DOF dense per element. Merging that into a discrete grid → information loss.

QA-LoRA's `δ` adds continuous-real DOF on **exactly the same grid as the scales**. Merging is just addition. Lossless.

---

## Empirical (Table 3)

| Method | Pre-merge MMLU | Post-merge (INT4) MMLU | Δ from merge |
|---|---|---|---|
| QLoRA | 56.0 | 54.1 | **-1.9** (re-quant loss) |
| QA-LoRA | 55.7 | 55.7 | **0.0** (lossless merge) |

QA-LoRA gives up a tiny amount of pre-merge quality for **zero merge loss** — and the deployed artifact is pure INT4, kernel-identical to a PTQ-only model.

---

## When to pick QA-LoRA over QLoRA

- Deployment must be pure INT4 (TensorRT-LLM, vLLM, llama.cpp): use QA-LoRA.
- Deployment can keep the BF16 LoRA branch (HF transformers, research): QLoRA is simpler.

---

## Connections

- Direct comparison: [[qlora]] (BF16 adapter, lossy merge).
- Joint quant + LoRA init alternative: [[loftq]].
- Scale-only PEFT alternative: [[peqa]] (no per-group δ, just scales).
- True QAT: [[llm-qat]].
- Outlier-aware PEFT alternative: [[owq]] Weak Column Tuning ([[ch-11]]).
