---
chapter: ch-11
course: model-quantization
phase: read
excerpt_of: "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression"
source_url: https://arxiv.org/abs/2306.03078
created_at: "2026-05-21"
---

# Excerpt: SpQR — bilevel uniform + per-weight sparse outliers

**Authors:** Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh
**Year:** 2023
**URL:** https://arxiv.org/abs/2306.03078
**Raw-data source:** [[raw-data/spqr]]

---

## The two innovations

1. **Bilevel group-wise quantization** — extremely small inner groups (G_in = 16) for tight per-group scales, with the inner scales themselves quantized to 3 bits inside outer groups (G_out = 128). Keeps the average bits-per-weight near 4 despite the small inner group.
2. **Per-weight sparse outlier matrix** — ~1% of weights (by GPTQ sensitivity score) stay in FP16 in a CSR sidecar.

The combination closes the 4-bit-to-FP16 gap that plain group-wise GPTQ leaves.

---

## Bilevel quantization

For weight matrix `W ∈ R^{d_out × d_in}`, partition each row first into outer groups of size `G_out = 128`, then each outer group into inner groups of size `G_in = 16`.

For each inner group (G_in = 16 weights):

```math
s_{\text{inner}} = \frac{\max(|W_g|)}{2^{b-1} - 1}, \qquad W_q = \mathrm{round}\Big(\frac{W_g}{s_{\text{inner}}}\Big), \quad b = 3
```

The inner scales `{s_inner}` (8 per outer group) are themselves quantized to 3 bits within the outer group:

```math
s_{\text{outer}} = \frac{\max(s_{\text{inner-group}})}{2^3 - 1}, \qquad \hat{s}_{\text{inner}} = \mathrm{round}\Big(\frac{s_{\text{inner}}}{s_{\text{outer}}}\Big)
```

Per inner group: store INT3 weights + the INT3 inner scale. Per outer group of 128 weights = 8 inner groups: store one FP16 outer scale shared by all 8.

---

## Bit-budget arithmetic

```
bits/weight = b + 3/G_in + 16/G_out + 16/G_out
            = 3 + 3/16 + 32/128
            = 3 + 0.1875 + 0.25
            ≈ 3.44 effective bits/weight
```

Tight per-16-weight range without storing FP16 per group → near-4-bit quality at ~3.44 bits.

---

## Per-weight outlier extraction (the "Sp")

GPTQ-style per-weight sensitivity:

```math
\mathrm{sens}_{i, j} = \frac{(W_{i, j} - Q(W_{i, j}))^2}{[H^{-1}]_{j, j}}
```

Pick the top ~1% globally per layer (or threshold by τ). These go into a sparse CSR matrix `S` at FP16. The dense quantization is re-run with outlier indices masked to zero.

---

## Inference

```
y = (SpQR-decoded GEMV)(W_dense, x) + (CSR-SpMV)(S, x)
```

- Dense kernel: 2-level dequant (INT3 → FP16 via inner-scale × outer-scale) fused into matmul.
- Sparse kernel: standard CSR SpMV in FP16, ~1% nnz → negligible compute.
- Fused into one CUDA call.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Inner group G_in | 16 |
| Outer group G_out | 128 |
| Weight bits | 3 |
| Inner-scale bits | 3 |
| Outer-scale bits | 16 |
| Outlier fraction | ~1% |
| Outlier sensitivity | GPTQ-style |
| Average bits/weight | ~3.4–4.0 |
| Calibration | 128 × 2048 tokens |

---

## Empirical (LLaMA-2 WikiText-2 PPL)

| Model | FP16 | GPTQ (g=128) W4 | **SpQR** (~W4) | Δ vs FP16 |
|---|---|---|---|---|
| LLaMA-2-7B | 5.47 | 5.69 | **5.49** | **+0.02** |
| LLaMA-65B | 3.32 | 3.42 | **3.34** | **+0.02** |

**First near-lossless 4-bit on LLaMA-65B.** A 33B model in SpQR fits in a single 24 GB GPU with a 15% inference speedup.

---

## Why bilevel beats single-level

Single-level group-wise with G=128: 16 bits of FP16 scale per 128 weights = 0.125 bits/weight overhead, but the scale must cover a wide range → wasted INT-k resolution per group.

Bilevel G_in=16: 16-weight range is much tighter → fewer wasted bits in the INT-3 grid. The cost (the inner scale storage) is recouped by quantizing the inner scales themselves to 3 bits.

This is a *structured precision allocation* — same total bits, better distribution.

---

## Connections

- Direct ancestor: [[gptq]] (same Hessian sensitivity reused for outlier picking).
- Outlier-FP16 lineage rivals: [[squeezellm]] (Fisher + k-means + sparse), [[owq]] (whole-column outliers), [[llm-int8]] (activation outliers).
- Companion paper from Dettmers: [[qlora]] (NF4 + LoRA, see [[ch-12]]).
- Successor that drops sparse storage via rotations: [[quarot]], [[quip-sharp]] (both in [[ch-14]]).
- Framework integration: [[autogptq]] (SpQR back-end), [[bitsandbytes-nf4]].
