---
chapter: ch-04
course: model-quantization
phase: read
excerpt_of: "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction (Li, Gong, Tan, Yang, Hu, Zhang, Yu, Wang, Gu 2021)"
source_url: https://arxiv.org/abs/2102.05426
created_at: "2026-05-21"
raw_data_source: [[raw-data/brecq]]
---

# Excerpt: BRECQ — block-wise reconstruction with Fisher weighting

**Authors:** Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, Shi Gu.
**Year:** 2021 (ICLR).
**URL:** see source_url.

---

## The load-bearing block-wise objective

For a block `B_k` with FP function `f_k` and quantized version `f̂_k`:

```math
\boxed{\;\min_{W_k} \; \mathbb{E}_X \, \| f_k(X) - \hat f_k(X; W_k) \|_F^2\;}
```

where `‖·‖_F` is the **Fisher-information-weighted** Frobenius norm:

```math
\| y \|_F^2 \,=\, \sum_i \text{diag}(F)_i \cdot y_i^2, \qquad \text{diag}(F) \,=\, \mathbb{E}\!\big[(\partial L / \partial y)^2\big]
```

≈ Gauss-Newton diagonal approximation to the task Hessian w.r.t. the block output.

---

## Why per-weight isn't enough

AdaRound minimizes `ΔWᵀ (X Xᵀ) ΔW` per layer. If layer `L_k` is quantized first, layer `L_{k+1}` sees a perturbed input `X' ≠ X`; AdaRound for `L_{k+1}` optimizes against `X'` but **ignores how its own `ΔW_{k+1}` amplifies the upstream perturbation**. Block reconstruction folds both errors into one loss — it acts on the **block output** (an entire residual sub-graph including non-linearities), capturing all internal couplings.

---

## The granularity hierarchy

The paper sweeps `layer ⊂ block ⊂ stage ⊂ network` and shows **block is the empirical sweet spot** — small enough to optimize tractably, large enough to capture the dominant cross-layer dependencies.

| Granularity | Quality | Compute | Verdict |
|-------------|---------|---------|---------|
| layer | weak < 4-bit | cheap | AdaRound regime |
| **block** | **best 2–4-bit** | **moderate** | **BRECQ sweet spot** |
| stage | marginal gain | high | rarely worth |
| network | matches QAT | very high | not PTQ anymore |

---

## Soft assignment (inherited from AdaRound)

Per-weight rectified-sigmoid `h(V)` parameterization, optimized by Adam, with annealed sparsity regularization:

```math
L_{\text{total}} \,=\, L_{\text{recon}} + \lambda \sum \big( 1 - |2 h(V) - 1|^\beta \big)
```

---

## Mixed-precision bit allocation

For each block, compute the second-order sensitivity:

```math
\Omega_k \,=\, \text{trace}(H_k) \cdot \|\Delta W_k(b)\|^2
```

and allocate higher bit-widths to blocks with larger `Ω_k` subject to a global memory budget. **Same philosophy as HAWQ, applied at BRECQ's granularity.**

---

## Calibration data

- 1024 samples typical.
- Gradient w.r.t. block output computed once during a forward+backward of the FP model to estimate Fisher.

---

## Empirical impact

**First viable PTQ method at 2-bit for CNNs.** Matches FP32 at 4-bit on ResNet-50; small gap at 3-bit; usable at 2-bit (gap that QAT closes).

---

## Connections

- [[excerpts/adaround]] — direct predecessor; BRECQ generalizes layer → block.
- [[obs-obd]] — second-order foundations (same Taylor argument).
- [[qdrop]] — orthogonal: randomly drops quantization during BRECQ optimization to regularize (ch-05).
- [[hawq]] — same mixed-precision bit-allocation philosophy (ch-05).
- [[gptq]] — LLM-era descendant of the AdaRound / BRECQ line; operates per-layer with an exact sequential OBS update (ch-08).
- [[omniquant]] — block-wise LLM PTQ that follows BRECQ's grain choice but learns equivalent transformations instead of per-weight rounding (ch-10).
- [[ch-04]] — parent synthesis.
