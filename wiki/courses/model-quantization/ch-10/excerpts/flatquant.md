---
chapter: ch-10
course: model-quantization
phase: read
excerpt_of: "FlatQuant: Flatness Matters for LLM Quantization"
source_url: https://arxiv.org/abs/2410.09426
created_at: "2026-05-21"
---

# Excerpt: FlatQuant — Kronecker affine targeting flatness

**Authors:** Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, Xin Jiang, Wulong Liu, Jun Yao
**Year:** 2024 (ICML 2025)
**URL:** https://arxiv.org/abs/2410.09426
**Raw-data source:** [[raw-data/flatquant]]

---

## The "rotation isn't enough" argument

Rotation-based methods (QuaRot, SpinQuant, DuQuant — all in [[ch-14]]) use orthogonal `R` to redistribute outliers. But **orthogonal preserves L²-norm** — a heavy-tailed distribution stays heavy-tailed, just rotated.

What uniform-interval INT-k round-to-nearest is *actually* optimal for (per the Bennett 1/12 noise model, [[uniform-quantization-noise]]) is a **uniform source**. The right transformation is one that **flattens** the distribution, which requires the freedom of non-orthogonal affine.

```
orthogonal R:   heavy-tail in → heavy-tail rotated out
affine A:       heavy-tail in → near-uniform out  ✓
```

Flatness — not just outlier elimination — is the right optimization target for uniform-interval quant.

---

## Affine transformation

For each Linear `y = Wx`, insert invertible affine A (and optional bias b):

```math
y = (W A^{-1})(A x) = W' x'
```

W' folded offline; x' computed online before quantization.

---

## Kronecker parametrization (the storage trick)

For `d = d_1 · d_2` (e.g. `4096 = 64 · 64`):

```math
A = A_1 \otimes A_2, \qquad A_1 \in \mathbb{R}^{d_1 \times d_1},\ A_2 \in \mathbb{R}^{d_2 \times d_2}
```

- **Storage:** `d_1² + d_2² ≪ d²` (e.g. `2 · 64² = 8192` vs `4096² ≈ 16M`).
- **Online cost:** `Ax = vec(A_2 · X · A_1^⊤)` with `X = reshape(x, d_2 × d_1)` — two small matmuls per token.

The Kronecker structure makes full-affine *cheaper than full A in AffineQuant* while retaining most of the expressivity.

---

## Calibration objective

Per-block reconstruction MSE:

```math
\mathcal{L}(A) = \lVert f_{FP}(x) - f_{\text{quant}}(x;\ A) \rVert^2
```

Quant = INT4 round-to-nearest on both W' and x'. Train A_1, A_2 with AdamW for a few hundred steps per block. No QAT of W is needed.

---

## Bias term (the +b)

Per-channel learned bias added before quantization to recentre the distribution; absorbed into the bias of the following Linear at fold time.

---

## Kernel fusion (the <5% overhead claim)

Forward for one Linear:

```
1. x' = A_2 · reshape(x) · A_1^⊤        (fused into one CUDA kernel)
2. quantize x' to INT4 (dynamic per-token, symmetric)
3. INT4 GEMM with W'                     (Marlin / TRT-LLM)
```

Reported overhead: <5% of prefill, <10% of decode vs raw INT4.

---

## Empirical results (LLaMA-3-70B W4A4)

| Method | Avg. acc | PPL gap |
|---|---|---|
| FP16 | 73.4 | — |
| QuaRot W4A4 | 65.1 | +8.3 |
| SpinQuant W4A4 | 65.9 | +7.5 |
| DuQuant W4A4 | 68.2 | +5.2 |
| **FlatQuant W4A4** | **72.4** | **+1.0** |

Plus 2.3× prefill / 1.7× decoding speedup vs FP16. **Beats SpinQuant by 7.5%** at LLaMA-3-70B W4A4.

---

## The flatness argument visualised (Figure 2)

Activation histograms (Figure 2 of the paper):

- **Original:** sharp peak near zero, heavy positive tail.
- **After orthogonal rotation:** still sharp peak; tail rotated but not flattened.
- **After learned affine:** near-uniform spread → ideal for INT4 round-to-nearest.

This is the single most convincing argument for why orthogonal-only methods leave PPL on the table at W4A4.

---

## Connections

- Rotation lineage it generalises: [[quarot]] → [[spinquant]] → [[duquant]] → [[flatquant]].
- Predecessors in equivalent-transformation: [[omniquant]] (learnable diag), [[affinequant]] (concurrent full affine).
- Theoretical motivation: [[uniform-quantization-noise]] (Bennett 1/12 noise — tight only for uniform sources).
- Weight quantizer paired: [[gptq]] or RTN.
