---
chapter: ch-05
course: model-quantization
phase: read
excerpt_of: "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning (Frantar, Singh, Alistarh 2022)"
source_url: https://arxiv.org/abs/2208.11580
arxiv: 2208.11580
created_at: "2026-05-21"
---

# Excerpt: OBC — pruning and quantization as one OBS

**Authors:** Elias Frantar, Sidak Pal Singh, Dan Alistarh
**Year:** 2022 (NeurIPS)
**Raw-data source:** [[raw-data/classics/obc]]

---

## The one-line unification

OBC's key conceptual move: pruning a weight to 0 and rounding a weight to its nearest quantizer grid point are the **same** problem under OBS. Both pick a perturbation `δw_q` to a single weight; the OBS Lagrange solution gives an identical closed-form correction to the surviving weights.

```math
\delta w_q \;=\; \begin{cases} -w_q & \text{(prune)} \\ Q(w_q) - w_q & \text{(quantize)} \end{cases}
```

```math
\delta L_q \;=\; \frac{(\delta w_q)^2}{2\,[H^{-1}]_{qq}}
\qquad
\delta w_{-q} \;=\; -\frac{\delta w_q}{[H^{-1}]_{qq}}\, H^{-1}_{-q,\, q}
```

Same `δw_{−q}` update for both — only the `δw_q` value differs. This is why the SparseGPT and GPTQ codebases share their inner loop.

---

## The Cholesky O(d³) implementation

A naive OBS sweep over a layer of dimension `d` costs `O(d⁴)` (matrix inversion per column). OBC's engineering contribution: Cholesky-factor `H = L Lᵀ`, then `H⁻¹ = L⁻ᵀ L⁻¹`. Two key identities:

- `[H⁻¹]_qq = 1 / L_qq²` after reordering for triangular structure.
- After processing column `q`, the residual sub-problem's `H̃⁻¹` is a **rank-1 down-date** of `L`, computable in `O(d²)`.

Total per-layer cost: `O(d³)` for the initial Cholesky + `d × O(d²)` per-column updates = `O(d³)`. This is what makes OBC tractable for BERT-Base (d ≈ 768) — and what fails for LLM-scale (d ≈ 16384), motivating GPTQ's *lazy block update*.

---

## Empirical: BERT-Base 4-bit PTQ

| Method | GLUE avg |
|---|---|
| FP32 baseline | 85.4 |
| AdaRound | 84.3 |
| BRECQ | 84.6 |
| **OBC** | **84.9** |

Within 0.5 of FP32, no QAT, ~minutes per layer. This is the first PTQ to match a strong block-reconstruction method (BRECQ) using pure closed-form per-column updates.

---

## Mixed compression (a feature inherited by SpQR)

At each OBC step, you compute saliency for **both** options (prune-to-0 vs quant-to-`Q(w_q)`) and pick the cheaper. The result: a per-weight pruning-or-quantization decision optimal under the local Hessian. SpQR's "dense and sparse" decomposition is a thin specialisation: top-0.5% outliers stay FP16 (pruning the quantization), the rest get quantized.

---

## What changes for LLMs (preview of GPTQ)

OBC processes columns one at a time, each touching every other column via `H⁻¹`. At `d = 16384`, that's `O(d³) = 4.4 × 10¹²` ops — feasible but slow when repeated per layer per model. GPTQ ([[gptq]] excerpt in ch-08) adds:

1. **Block-of-128 batching**: defer Hessian updates until B columns are processed; then apply a single dense GEMM correction.
2. **Cholesky factor reused**: `R` (upper triangular factor of `H⁻¹`) precomputed once; per-column quantities `[H⁻¹]_qq` and `[H⁻¹]_{q,:}` read from `R`.

Both make GPTQ ~10× faster than OBC at LLM scale. The mathematical algorithm is unchanged.

---

## Common pitfalls

- **`H` is near-singular.** Calibration `X` has rank ≤ N. Damp before Cholesky: `H ← H + percdamp · mean(diag(H)) · I` with `percdamp = 0.01`.
- **Ordering matters.** OBC processes columns in increasing saliency; GPTQ uses descending `diag(H)` (act_order) which empirically gives ~0.1–0.3 ppl on LLMs.
- **One H per layer.** Shared across output rows (W's rows are independent given X). Don't recompute per row.

---

## Connections

- [[excerpts/obs-obd]] — the 1993 mathematical foundation OBC re-derives.
- [[excerpts/brecq]] — alternative line: gradient-based block reconstruction instead of closed-form per-column.
- [[ch-05]] — parent synthesis of the classical PTQ playbook.
- [[ch-08]] — GPTQ ports OBC to LLM scale via Cholesky + lazy batched updates.
