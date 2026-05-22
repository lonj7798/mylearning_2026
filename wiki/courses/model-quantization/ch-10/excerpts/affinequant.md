---
chapter: ch-10
course: model-quantization
phase: read
excerpt_of: "AffineQuant: Affine Transformation Quantization for Large Language Models"
source_url: https://arxiv.org/abs/2403.12544
created_at: "2026-05-21"
---

# Excerpt: AffineQuant — full invertible affine via gradual mask

**Authors:** Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, Rongrong Ji
**Year:** 2024 (ICLR 2024)
**URL:** https://arxiv.org/abs/2403.12544
**Raw-data source:** [[raw-data/affinequant]]

---

## The lift from diagonal to full affine

OmniQuant's LET is restricted to `A = diag(s)` (per-channel scaling). AffineQuant widens to **general invertible affine** `A ∈ R^{d × d}`:

```math
y = W x = (W A^{-1})(A x) = W' x'
```

- `W' = W A^{-1}` is folded into the weight offline.
- `x' = A x` is computed online (or A is folded into the previous Linear when possible).
- Quant operates on `x'` and `W'`, which are flatter than `x` and `W`.

---

## The problem: invertibility is not automatic

A general A trained by gradient descent can drift toward singularity. AffineQuant's solution is a **gradual mask schedule** `M_t ∈ {0, 1}^{d × d}`:

```math
A_t = I + M_t \odot (A - I)
```

At training step `t`:
- `t = 0`: `M_0 = I` → only diagonal entries trainable. `A_0 = diag` (SmoothQuant-equivalent start).
- `t = T/2`: M banded around diagonal.
- `t = T`: M = all-ones → full A trainable.

The schedule prevents early optimization from driving A into singular regions.

---

## Invertibility guarantee (Levy-Desplanques theorem, 1881)

A matrix is invertible if it is **strictly diagonally dominant**:

```math
|A_{ii}| > \sum_{j \ne i} |A_{ij}|
```

The gradual mask + diagonal init keeps `|A_{ii}| ≈ 1` while off-diagonal entries grow slowly. Dominance is preserved across training → A stays invertible.

---

## Loss

Per-block reconstruction MSE (same as OmniQuant):

```math
\mathcal{L}(A) = \lVert f_{FP}(x) - f_{\text{quant}}(x;\ A) \rVert^2
```

AdamW on a small calibration set for a few hundred steps per block.

---

## Quantization on top

After A is learned and folded:
- Per-channel weight RTN at 4-bit on W'.
- Per-token activation RTN at 4-bit on x'.

No GPTQ needed (though it can be added).

---

## Cost

- **Training:** same as OmniQuant (per-block on calibration, ~hours).
- **Inference:** A absorbed into W' offline (when possible) → zero added cost. Or one extra small matmul per Linear (Kronecker decomposition optional — see [[flatquant]]).

---

## Empirical results (LLaMA-family W4A4 C4 PPL)

| Method | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B |
|---|---|---|---|---|
| FP16 | 5.68 | 5.09 | 4.10 | 3.53 |
| SmoothQuant W4A4 | NaN | NaN | NaN | NaN |
| OmniQuant W4A4 | 18.02 | 14.61 | 12.30 | 10.42 |
| **AffineQuant W4A4** | **15.76** | **13.97** | **11.95** | **10.20** |

(Numbers from AffineQuant Table 2. Lower is better. SmoothQuant diverges at W4A4 across LLaMA.)

---

## Connections

- Direct predecessor: [[omniquant]] (diagonal equivalent transformations).
- Kronecker-decomposed sibling targeting flatness: [[flatquant]].
- Diagonal-only predecessors: [[smoothquant]], [[awq]].
- Orthogonal-only siblings: [[quarot]], [[spinquant]].
- Theoretical foundation: Levy-Desplanques (1881) diagonal dominance theorem.
