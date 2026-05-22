---
chapter: ch-14
course: model-quantization
phase: read
excerpt_of: "SpinQuant: LLM Quantization with Learned Rotations"
source_url: https://arxiv.org/abs/2405.16406
created_at: "2026-05-21"
---

# Excerpt: SpinQuant — learned rotations on the Stiefel manifold

**Authors:** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort (Meta)
**Year:** 2024
**Venue:** ICLR 2025
**URL:** https://arxiv.org/abs/2405.16406
**Raw-data source:** [[raw-data/spinquant]]

---

## The motivating observation

QuaRot uses a random Hadamard rotation Q. SpinQuant's first experiment: pick different random Hadamards (different sign vectors S) and measure downstream accuracy. The spread is **up to 13 points** on commonsense reasoning. Random Hadamard is not optimal — different rotations give wildly different post-quantization accuracy.

The right move: **treat the rotation as a learnable parameter** constrained to the orthogonal group.

---

## The constraint set — Stiefel manifold

Each rotation R must satisfy `R^⊤ R = I`. The set of such d×d matrices is the **compact Stiefel manifold**:

```math
\mathrm{St}(d, d) = \{ R \in \mathbb{R}^{d \times d} : R^\top R = I \} = O(d)
```

Vanilla SGD on R does not preserve orthogonality (perturbations push R off the manifold). You need either projection (QR retraction) or a parametrization that lives in St(d, d) by construction.

---

## Cayley parametrization

Parametrize R via a **skew-symmetric** matrix `A` (so `A = -A^⊤`):

```math
R = (I - A)(I + A)^{-1}
```

This is the **Cayley map**: a diffeomorphism from the space of skew-symmetric matrices (a linear subspace of ℝ^{d×d}) onto a dense subset of SO(d) (the connected component of the identity in O(d)).

**Properties:**
- For any skew A, the result R is orthogonal: trivial to verify since `(I-A)(I+A)^{-1} · ((I-A)(I+A)^{-1})^⊤ = I`.
- The Cayley map covers SO(d) minus a measure-zero set; perfectly fine for SGD.
- Gradients of A flow through `(I+A)^{-1}` via the implicit function theorem; PyTorch handles this with autograd on the linear solve.

```python
def cayley(A):
    """A is skew-symmetric; returns R = (I-A)(I+A)^{-1}."""
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return (I - A) @ torch.linalg.solve(I + A, I)

A = torch.nn.Parameter(0.01 * torch.randn(d, d))

def get_rotation():
    A_skew = (A - A.T) / 2          # project to skew at each step
    return cayley(A_skew)
```

The `(A - A^⊤) / 2` re-skew is the only constraint enforcement — cheap (one matrix subtract), idempotent.

---

## Loss

Per-block reconstruction loss:

```math
\mathcal{L}(R) = \sum_{\text{blocks } b} \| f_b^{\mathrm{FP}}(x) - f_b^{\mathrm{quant}}(x; R) \|^2
```

where `f_b^{quant}` is the quantized forward pass of block b given the current rotation R, weights GPTQ-quantized in the R-rotated frame, activations RTN-quantized per token.

Optimised over ~500 SGD steps on a small calibration corpus (typically WikiText-2 or C4 segments, 128–256 sequences).

---

## What's learned where

The four rotation slots from [[quarot]]:

| Slot | Original QuaRot | SpinQuant |
|------|-----------------|-----------|
| R1 (residual) | random Hadamard, dense | **learned dense Stiefel** (Cayley) |
| R2 (V → W_o) | random Hadamard, dense | **learned dense Stiefel** |
| R3 (FFN-down) | random Hadamard, block-diag | **block-diag Hadamard**, sign vectors learned |
| R4 (K after RoPE) | random Hadamard, block-diag | **block-diag Hadamard**, sign vectors learned |

R1 and R2 are dense d×d learnable (cost: `d² / 2` skew parameters each). R3/R4 are kept block-diagonal Hadamards so the inference FWHT speed is preserved; only the per-block sign vector S is learnable.

---

## Why learning beats random Hadamard

Random Hadamard uniformly spreads any single outlier coordinate over all d dimensions. This is optimal under a **uniform prior** on outlier location.

But real LLM activations have **structured outliers** — specific channels, head-correlated, persistent across tokens. Learned R can match this structure: concentrating rotation mass where outliers actually live, distributing into low-variance directions rather than uniformly across all.

Empirically: SpinQuant improves over QuaRot by **up to 45.1% on LLaMA-3-8B** at W4A4KV4. The gap is largest for LLaMA-3 (most structured outliers) and smallest for LLaMA-2-7B (least structured).

---

## Numbers

LLaMA-2-7B W4A4KV4 commonsense suite (avg of ARC, HellaSwag, PIQA, Winogrande):

| Method | Avg acc | Δ vs FP16 |
|--------|---------|-----------|
| FP16 | 67.8 | — |
| LLM-QAT | 45.8 | −22.0 |
| SmoothQuant | 40.6 | −27.2 |
| QuaRot | 62.7 | −5.1 |
| **SpinQuant** | **64.9** | **−2.9** |

LLaMA-3-8B W4A4KV4:

| Method | Avg acc | Δ vs FP16 |
|--------|---------|-----------|
| FP16 | 73.8 | — |
| QuaRot | 51.1 | −22.7 |
| **SpinQuant** | **63.5** | **−10.3** |

The 12.4-point recovery on LLaMA-3 is the headline — SpinQuant is what makes LLaMA-3 W4A4 viable.

---

## Inference cost

**Same as QuaRot.** R1 is folded offline (free). R2/R3/R4 are block-diagonal Hadamards with `O(d \log d)` cost per token, implemented with fused CUDA kernels. The learning happens entirely at calibration time; the deployed model has no extra runtime overhead vs QuaRot.

---

## Pitfalls

- **Initialization matters.** Random init of A produces an R far from any reasonable rotation; calibration often diverges. Initialise A from the matrix logarithm of a random Hadamard (so R starts at the QuaRot solution) and let SGD refine.
- **The Cayley map is a diffeomorphism onto SO(d), not O(d).** This means you can't reach rotations with determinant −1. For LLM quantization this is fine; the loss landscape is invariant under sign flips of arbitrary basis vectors.
- **Learning rate matters more than usual.** A's gradient flows through a matrix inverse `(I+A)^{-1}` which can amplify; use `lr = 1e-5` to `1e-4`, much smaller than typical fine-tuning.
- **Number of SGD steps.** 500 is the SpinQuant default; below ~200 the rotation under-fits and beats QuaRot by only a few points. Above ~1000 you risk overfitting to the calibration corpus.
- **R3 and R4 staying block-diagonal is a speed choice, not a quality choice.** If you can afford the online dense matmul cost, learning dense R3/R4 gives another ~1 point. Most production deployments don't.

---

## Connections

- [[excerpts/quarot]] — direct predecessor; same insertion graph, random Hadamard.
- [[excerpts/aqlm]] — orthogonal direction; AQLM tackles the codebook side, SpinQuant tackles the rotation side.
- [[ch-13]] — QuIP/QuIP# established the rotation idea SpinQuant generalises.
- Stiefel manifold optimisation lineage: Boumal *An Introduction to Optimization on Smooth Manifolds*; orthogonal fine-tuning papers BOFT / OFT.
