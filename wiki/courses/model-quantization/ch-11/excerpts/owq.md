---
chapter: ch-11
course: model-quantization
phase: read
excerpt_of: "OWQ: Outlier-aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models"
source_url: https://arxiv.org/abs/2306.02272
created_at: "2026-05-21"
---

# Excerpt: OWQ — structured whole-column outliers + Weak Column Tuning

**Authors:** Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, Eunhyeok Park
**Year:** 2023 (AAAI 2024 oral)
**URL:** https://arxiv.org/abs/2306.02272
**Raw-data source:** [[raw-data/owq]]

---

## The structural-outlier observation

In LLM weight matrices, certain **entire input columns** are disproportionately sensitive — specifically, the columns that multiply against high-magnitude activation channels. Per-weight outlier extraction ([[spqr]], [[squeezellm]]) catches these but requires CSR + SpMV. OWQ instead keeps whole columns in FP16 → simple concat at inference, no CSR.

This trades a small accuracy gap for kernel simplicity.

---

## Weak column score

For weight `W ∈ R^{d_out × d_in}` and calibration activations `X ∈ R^{N × d_in}`, per input channel j:

```math
\text{score}_j = \max_t |X_{t, j}| \cdot \lVert W_{\cdot, j} \rVert_2
```

Combines:
- **activation-outlier magnitude** `max_t |X_{t,j}|` — which channel hits hard.
- **weight contribution** `||W_{·,j}||_2` — does the column actually use that signal.

Pick the top `k = ⌈p · d_in⌉` columns (p = 1–5%) as the **weak set** W*.

---

## Mixed-precision storage

- Weak columns (1–5% of d_in): FP16, no quant.
- Dense columns (95–99%): GPTQ-quantized at b = 3 bits, group_size = 128, weak columns masked from the Hessian update.

```math
\text{bits/weight} \approx 0.95 \cdot 3.125 + 0.05 \cdot 16 \approx 3.77 \quad (p = 5\%)
```

Drops further at p=1% (~3.30).

---

## Inference (the simple concat)

```
y_dense   = (GPTQ-decoded GEMV)(W_dense_q, x)
y_outlier = FP16-GEMV(W_weak_fp16, x[weak_idx])
y = y_dense + y_outlier
```

Weak indices are fixed at quant time → simple gather + small dense matmul. No CSR, no SpMV.

---

## Weak Column Tuning (WCT) — the PEFT bonus

For fine-tuning:
- **Freeze** all dense (quantised) weights.
- **Trainable:** only `W_weak_fp16` (shape `d_out × k`, k ≈ 1–5% · d_in).
- Standard task-specific loss; AdamW.

Trainable parameter count ~3–5% of full fine-tune — comparable to LoRA at r=8.

**Key property:** WCT updates live natively in the quantised format. No merge step. No FP16 LoRA branch at inference. The OWQ-fine-tuned artifact is *just OWQ* with updated FP16 columns.

This makes OWQ both a quant scheme and a PEFT scheme in one package.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Weak fraction p | 0.01–0.05 |
| Dense bits | 3 |
| Group size | 128 |
| Score | `max|X_j| · ||W_{·,j}||_2` |
| Calibration | 128 sequences C4 |
| WCT optimizer | AdamW, lr 1e-4, batch 16 |

---

## Empirical (OPT / LLaMA WikiText-2 PPL, OWQ-3.01 vs GPTQ-4)

OWQ at ~3.01 average bits matches or beats GPTQ at 4 bits across LLaMA-7B/13B/30B/65B and OPT scales. Trades a slightly higher kernel cost (the FP16 weak-column matmul) for lower average bits.

WCT (Table 7) reaches comparable downstream accuracy to LoRA on instruction-tuning benchmarks with smaller trainable param count and zero merge overhead.

---

## The taxonomy this completes

| Outlier definition | Sparse format | Method | Best at |
|---|---|---|---|
| Per-weight | CSR | [[spqr]] | near-lossless 4-bit |
| Per-weight | CSR | [[squeezellm]] | aggressive 3-bit |
| Whole input column | concat | **OWQ** | quant + PEFT in one |
| Per-token activation | FP16 column path | [[llm-int8]] | INT8 inference |

---

## Connections

- Per-weight outlier rivals (finer-grained, harder to kernelise): [[spqr]], [[squeezellm]].
- Activation-side analogue (input X side, not weight side): [[llm-int8]].
- Activation-aware weight quant without FP16 storage: [[awq]].
- LoRA alternatives for quantised PEFT: [[qlora]], [[loftq]], [[peqa]], [[qa-lora]] (all in [[ch-12]]).
- Backend uses [[gptq]] for the dense path.
