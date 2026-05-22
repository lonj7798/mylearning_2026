---
chapter: ch-04
course: model-quantization
phase: read
excerpt_of: "Up or Down? Adaptive Rounding for Post-Training Quantization (Nagel, Amjad, van Baalen, Louizos, Blankevoort 2020)"
source_url: https://arxiv.org/abs/2004.10568
created_at: "2026-05-21"
raw_data_source: [[raw-data/adaround]]
---

# Excerpt: AdaRound — learned per-weight rounding direction via the Hessian objective

**Authors:** Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort.
**Year:** 2020 (ICML).
**URL:** see source_url.

---

## The load-bearing layer-wise objective

For a single layer with weights `W`, calibration inputs `X`, quantized weights `Ŵ`:

```math
\boxed{\;\min_{\hat W} \; \mathbb{E}_X \, \| W \cdot X - \hat W \cdot X \|^2 \,=\, \min_{\Delta W} \Delta W^T (X X^T) \Delta W \;}
```

with `ΔW = W − Ŵ`. This is the **Hessian-weighted reconstruction objective**; the Hessian is `H = X Xᵀ` (Gauss-Newton for squared-error reconstruction). **This is the parent objective of every later Hessian-aware PTQ method** — GPTQ solves it with a sequential OBS update, OmniQuant learns equivalent transformations against it, QuIP applies a random orthogonal rotation first.

---

## The Taylor argument — why nearest is suboptimal

For task loss `L` around the FP weights `W`:

```math
L(W + \Delta W) \,\approx\, L(W) + g^T \Delta W + \tfrac{1}{2} \Delta W^T H \Delta W
```

At a converged FP model `g ≈ 0`, so the loss is dominated by `ΔWᵀ H ΔW`. **Off-diagonal `H` means coupled weights** — flipping one weight's rounding can cancel another's error. Round-to-nearest minimizes `‖ΔW‖²` (i.e. assumes `H = I`); the true objective is the Hessian-weighted norm.

---

## Quantizer and the rounding decision

Standard uniform quantizer with step `Δ`:

```math
\hat W = \Delta \cdot \big( \lfloor W / \Delta \rfloor + b \big), \quad b \in \{0, 1\}
```

`b` is the per-weight rounding direction (down = 0, up = 1). Each weight has its own `b`.

---

## The rectified-sigmoid relaxation

Replace `b` with a continuous soft assignment `h(V) ∈ [0, 1]`:

```math
\boxed{\; h(V) \,=\, \text{clip}\!\big(\sigma(V)(\zeta - \gamma) + \gamma, \, 0, \, 1\big), \quad \gamma = -0.1, \, \zeta = 1.1 \;}
```

The "stretched" sigmoid saturates *exactly* at `{0, 1}`, so the final discrete decision is hard. After convergence, `b = round(h(V))` and freeze.

---

## Full optimization objective

```math
\min_V \; \big\| W \cdot X - \Delta \cdot (\lfloor W/\Delta \rfloor + h(V)) \cdot X \big\|^2 \,+\, \lambda \cdot f_{\text{reg}}(V)
```

```math
f_{\text{reg}}(V) \,=\, \sum \big( 1 - |2 h(V) - 1|^\beta \big)
```

`β` is annealed from 20 → 2 across iterations to encourage exploration then commitment.

---

## Training loop

- ~10k Adam steps per layer.
- ~1024 calibration samples.
- Quantize layer by layer in **forward order** — each subsequent layer sees the previous layer's *quantized* output, capturing error propagation.
- After convergence, freeze.

---

## Why this is the spiritual parent of GPTQ

AdaRound solves the per-layer Hessian-weighted reconstruction with soft relaxation + Adam + annealing. **GPTQ** (Frantar 2022, ch-08) solves the *same objective* with a sequential, exact OBS-style closed-form update: process columns of `W` in blocks, update remaining via inverse-Hessian Cholesky factor. The objectives are identical; GPTQ replaces the soft relaxation with the exact OBS step, making it efficient enough for 70B+ models in a few hours on a single GPU.

---

## Connections

- [[excerpts/straight-through-estimator]] — gradient through `round()` used inside the soft relaxation.
- [[obs-obd]] — uses the same Taylor argument for weight removal (pruning); shared mathematical scaffold.
- [[brecq]] — extends AdaRound from layer-wise to block-wise reconstruction.
- [[gptq]] — direct LLM-era descendant; sequential exact OBS update on the same objective (ch-08).
- [[omniquant]] — modern LLM heir: learns clip + equivalent transformations instead of per-weight rounding (ch-10).
- [[ch-04]] — parent synthesis.
