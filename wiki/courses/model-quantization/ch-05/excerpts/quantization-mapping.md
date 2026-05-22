---
chapter: ch-05
course: model-quantization
phase: read
excerpt_of: "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper (Krishnamoorthi 2018)"
source_url: https://arxiv.org/abs/1806.08342
arxiv: 1806.08342
created_at: "2026-05-21"
---

# Excerpt: Krishnamoorthi 2018 — the PTQ taxonomy

**Author:** Raghuraman Krishnamoorthi (Google)
**Year:** 2018
**Raw-data source:** [[raw-data/classics/quantization-mapping]]

---

## The four orthogonal axes

The taxonomy every PTQ paper from 2018 onward inhabits. Memorise it.

| Axis | Options |
|---|---|
| Scale placement (W) | per-tensor / per-channel / per-group |
| Scale placement (A) | per-tensor / per-token / per-channel |
| Symmetry | symmetric (Z=0) / asymmetric (Z ∈ ℤ) |
| Training regime | PTQ / QAT |

The Krishnamoorthi sweet spot for CNNs: **per-channel symmetric weights + per-tensor asymmetric activations**. For LLMs (ch-07 onwards), the per-tensor activation cell turns out to be empty due to outliers — that's the central plot tension of this course.

---

## Affine quantization rule (asymmetric)

The canonical (S, Z) formulation every later framework cites:

```math
q \;=\; \text{clamp}\!\left( \text{round}(x / S) + Z,\; Q_{\min},\; Q_{\max} \right)
```

```math
\hat{x} \;=\; S \cdot (q - Z)
```

- `S ∈ ℝ₊` — the **scale** (fp32 at compile time).
- `Z ∈ ℤ` — the **zero-point** chosen so real 0.0 maps exactly to integer Z. Critical for ReLU and zero-padding correctness.

Symmetric variant: `Z = 0` (signed INT in range −128..127) or `Z = 2^{k−1}` (unsigned). Easier kernels (no Z subtraction in the GEMM); wastes resolution on one-sided distributions.

---

## Per-channel weights (the recipe)

For `W ∈ ℝ^{C_out × C_in}`: maintain `S ∈ ℝ^{C_out}`, one scale per output channel. Justification: weight ranges vary by 10× across filters; per-tensor scale wastes >3 bits of resolution.

Per-channel weight + per-tensor activation is **GEMM-friendly** because the per-channel weight scale folds into the output rescale via `M = S_w · S_x / S_y` — one extra multiply per output channel rather than per output element.

Per-channel activation, in contrast, would require per-output-element rescale at the GEMM boundary — incompatible with standard INT8 tensor-core layouts. This is why no production LLM quantizer uses per-channel activations until rotation methods ([[quip]], ch-13) make them effectively free.

---

## Calibration recipe

For weights (closed-form, no data needed):

```math
S_c \;=\; \max|W_c| \,/\, Q_{\max} \qquad \text{(symmetric per-channel)}
```

For activations (data-driven):

1. Histogram activation magnitudes over a few hundred unlabelled batches.
2. Minimise `KL(P_fp || P_quantized)` over candidate clip ranges (TensorRT-style search).
3. Set `S = (max − min) / (Q_max − Q_min)`, `Z = round(Q_min − min/S)`.

This is the "calibration" step every later paper extends. LLM-era calibration uses ~128 sequences × 2048 tokens instead of "a few hundred batches".

---

## Bias quantization (and the requantize trick)

Bias is the small term — keep as **int32** with `S_bias = S_w · S_x`. Requantization happens at the matmul output via:

```math
M \;=\; S_w \cdot S_x / S_y \;\approx\; M_0 \cdot 2^{-n}, \quad M_0 \in [0.5, 1)
```

`M_0` is stored as int32, `n` is a shift count. This is the **fixed-point requantization** that enables integer-only inference; ch-06 covers it in detail with Jacob 2018.

---

## PTQ vs QAT decision rule

The paper's pragmatic rule of thumb (still used in 2026):

```text
1. Try PTQ first with per-channel weights + per-tensor activations.
2. If Δaccuracy < 1% → ship it.
3. Else: insert fake-quant ops, train ~10% of original schedule with reduced LR (QAT).
```

For modern LLMs, the rule shifts to: try GPTQ/AWQ first; fall back to QLoRA or full QAT only if downstream task quality fails. PTQ has won the LLM era because retraining a 70B model is prohibitive.

---

## Common pitfalls

- **Confusing the four axes.** A "per-channel quantization" paper might mean per-output-channel weights, per-input-channel activations, or per-attention-head groups. Always pin down which dimension.
- **Symmetric INT8 has 255 levels.** Symmetric uses signed [−127, +127], not [−128, +127], to keep the grid symmetric around 0; you lose one level. Asymmetric uses all 256.
- **Per-tensor activation on transformers.** Works on BERT-Base at INT8 with QAT. Fails on any 7B+ LLM at PTQ-INT8 due to outliers — see [[llm-int8]] in ch-07.

---

## Connections

- [[excerpts/obc]] — the OBS-based PTQ that lives in the per-channel weight cell of this taxonomy.
- [[excerpts/data-free-quantization]] — the no-calibration corner.
- [[ch-05]] — parent synthesis with the full grid laid out.
- [[ch-06]] — Jacob 2018 integer-only inference uses the same (S, Z) convention.
- [[ch-07]] — [[llm-int8]] shows where the per-tensor activation cell breaks at LLM scale.
