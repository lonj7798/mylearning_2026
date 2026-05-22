---
chapter: ch-05
course: model-quantization
phase: read
excerpt_of: "Optimal Brain Damage (LeCun 1989) / Optimal Brain Surgeon (Hassibi & Stork 1993)"
source_url: https://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon
created_at: "2026-05-21"
---

# Excerpt: OBS — the two formulas every modern PTQ inherits

**Authors:** Yann LeCun, John S. Denker, Sara A. Solla (OBD, 1989); Babak Hassibi, David G. Stork (OBS, 1993)
**Year:** 1989 / 1993
**Raw-data source:** [[raw-data/classics/obs-obd]]

---

## Setup

A converged neural network with weights `w ∈ ℝⁿ` at a local minimum of loss `L`. Second-order Taylor:

```math
\delta L \;\approx\; g^\top \delta w \;+\; \tfrac{1}{2}\, \delta w^\top H\, \delta w \;\approx\; \tfrac{1}{2}\, \delta w^\top H\, \delta w
```

At convergence `g ≈ 0`, so only the quadratic term governs the cost of any edit. The OBS question: choose `δw` to force a constraint `e_qᵀ(w + δw) = 0` (i.e. remove weight `q`) while minimising `δL`.

---

## OBD — the diagonal approximation

Assume `H` is diagonal. With constraint `δw_q = −w_q` and `δw_{−q} = 0`:

```math
\delta L_q \;=\; \tfrac{1}{2}\, H_{qq}\, w_q^2
```

**Saliency**: `s_q = (1/2) H_qq · w_q²`. Prune the q with smallest s_q. This is the 1989 LeCun pruning criterion — first principled alternative to magnitude pruning.

---

## OBS — full-Hessian Lagrange solution (the load-bearing formulas)

With full `H`, use Lagrange multipliers on the constraint:

```math
\mathcal{L}_{\text{lag}} \;=\; \tfrac{1}{2}\, \delta w^\top H\, \delta w \;+\; \lambda \,(e_q^\top \delta w + w_q)
```

Solving:

```math
\boxed{\;\delta w \;=\; -\frac{w_q}{[H^{-1}]_{qq}}\, H^{-1}_{:,\,q}\;}
```

```math
\boxed{\;\delta L_q \;=\; \frac{w_q^2}{2\,[H^{-1}]_{qq}}\;}
```

That's it. Two formulas, six tokens each. Every modern LLM PTQ paper (GPTQ, SparseGPT, OBC, QuIP, SpQR) is an LLM-scale port of these two lines.

**What `δw` does.** It is the change to *all* surviving weights that exactly cancels the first-order effect of removing weight `q`. The remaining weights "redistribute" `w_q`'s lost contribution along the inverse-Hessian eigenvectors, which is provably optimal under the quadratic approximation.

---

## The procedure (gold standard for sparsification)

1. Compute or estimate `H⁻¹` once.
2. For each weight `q`: compute saliency `w_q² / (2[H⁻¹]_qq)`.
3. Prune the `q` with smallest saliency.
4. Update all remaining weights: `w ← w + δw`.
5. (Optional) update `H⁻¹` via Woodbury for the next round:

```math
H^{-1}_{\text{new}} \;=\; H^{-1} \;-\; \frac{H^{-1}_{:,\,q}\, H^{-1}_{q,\,:}}{[H^{-1}]_{qq}}
```

Loop until target sparsity / quantization is reached.

---

## Why GPTQ inherits this exactly

GPTQ takes one tiny step: replace "set `w_q = 0`" with "round `w_q` to its nearest quantizer grid point". The Lagrangian is unchanged — the constraint is now `w_q + δw_q = Q(w_q)` instead of `w_q + δw_q = 0`, giving:

```math
\delta w_q^{\text{quant}} \;=\; Q(w_q) - w_q
```

and the same `δw_{−q}` formula compensates the rounding error. This is what [[obc]] makes explicit and what GPTQ ports to billion-parameter weight matrices.

---

## Practical Hessian for PTQ

For a single linear layer with calibration activations `X ∈ ℝ^{d_in × N}`, the load-bearing Hessian of `‖W X − Ŵ X‖²` (with respect to W) is:

```math
H \;=\; 2\, X X^\top \;\in\; \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}
```

Properties:

- **Shared across output rows** of W → compute once, reuse `d_out` times.
- **Computable from forward only** (no backward), one outer product over the calibration batch.
- **Often rank-deficient** when N < d_in → must damp with `H ← H + λI` before inverting.

For d_in = 4096 (Llama-2-7B FFN intermediate), N = 128 × 2048 = 262144 → H rank ≈ 4096 (saturated), perfectly conditioned in practice. For attention output projection at d_in = 4096 the same logic holds.

---

## Empirical effect (CNN era — sets the bar for LLM era)

| Model | Pruning method | Accuracy retained at 50% sparsity |
|---|---|---|
| OBD diagonal | LeNet (MNIST) | matches magnitude pruning |
| OBS full | LeNet (MNIST) | ~+1% over OBD |
| Magnitude | LeNet | baseline |

The full-Hessian advantage is small at CNN scale. At LLM scale, the gap widens dramatically — and the *quantization* version of OBS (GPTQ) shows 3–10 ppl improvement over RTN at W4. The Hessian is doing real work when the bit budget is tight.

---

## Common pitfalls

- **Computing the full `H⁻¹` for d > 10k.** Use Cholesky-factor representation (OBC trick) — avoid forming `H⁻¹` explicitly.
- **Skipping the damping.** A rank-deficient `H` gives infinite `[H⁻¹]_qq` for some q. Always damp.
- **Forgetting H is per-layer.** The full network Hessian is intractable; OBS works because layer-MSE provides a tractable layer-local quadratic objective.

---

## Connections

- [[excerpts/obc]] — the unified pruning + quantization wrapper.
- [[excerpts/brecq]] — alternative gradient-based take on the same 2nd-order objective at block grain.
- [[ch-05]] — parent synthesis.
- [[ch-08]] — GPTQ ports OBS to LLM scale via Cholesky + lazy batched updates.
