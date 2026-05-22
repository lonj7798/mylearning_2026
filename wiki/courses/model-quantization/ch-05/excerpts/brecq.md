---
chapter: ch-05
course: model-quantization
phase: read
excerpt_of: "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction (Li et al. 2021)"
source_url: https://arxiv.org/abs/2102.05426
arxiv: 2102.05426
created_at: "2026-05-21"
---

# Excerpt: BRECQ — the block-reconstruction grain

**Authors:** Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, Shi Gu
**Year:** 2021 (ICLR)
**Raw-data source:** [[raw-data/classics/brecq]]

> BRECQ is covered as a QAT-flavoured method in [[ch-04]]. This excerpt focuses on the **PTQ-relevant** property: block grain as the optimal reconstruction unit.

---

## The reconstruction-grain hierarchy

BRECQ's empirical contribution: four candidate grains for PTQ reconstruction.

| Grain | What it optimises | Captures cross-layer coupling? | Accuracy at sub-4-bit |
|---|---|---|---|
| Per-layer (AdaRound) | `‖W X − Ŵ X‖²` for one layer | no | weak |
| **Per-block** | `‖f_block(X) − f̂_block(X)‖²` | **yes (residual sub-graph)** | **best** |
| Per-stage | one full residual stage | yes, but expensive | marginal gain over block |
| Per-network | end-to-end | yes, but intractable | impractical |

The result (Figure 2 of the paper): **block is the sweet spot**. Per-layer ignores how each layer's quantization error perturbs the next layer's input; per-block folds both into a single MSE; per-stage and per-network add compute without adding accuracy at the bit-widths PTQ targets.

---

## The block-wise objective (the load-bearing formula)

For block `B_k` with FP function `f_k` and quantized version `f̂_k`:

```math
\min_{W_k} \; \mathbb{E}_X \, \bigl\| f_k(X) - \hat{f}_k(X;\, W_k) \bigr\|^2_F
```

where `‖·‖_F` is the **Fisher-information-weighted Frobenius norm**:

```math
\|y\|^2_F \;=\; \sum_i \text{diag}(F)_i \cdot y_i^2, \qquad \text{diag}(F) \;=\; \mathbb{E}\bigl[(\partial L / \partial y)^2\bigr]
```

This is the **Gauss-Newton diagonal approximation** to the task Hessian with respect to block output — a cheap proxy that doesn't require forming the block Hessian. Computed from a single backward of the FP model.

---

## Why per-block beats per-layer (the load-bearing argument)

AdaRound minimises `ΔWᵀ (X Xᵀ) ΔW` per layer. If layer `L_k` is quantized first, layer `L_{k+1}` sees a *perturbed* input `X' ≠ X`; AdaRound for `L_{k+1}` optimises against `X'` but ignores how its own `ΔW_{k+1}` amplifies the upstream perturbation. Block reconstruction folds both errors into one loss:

```math
\bigl\| f_k(X) - \hat{f}_k(X) \bigr\|^2 \;\supseteq\; \text{(layer-}k\text{ error)} \otimes \text{(layer-}k+1\text{ amplification)}
```

The cross-term is what BRECQ captures and per-layer methods cannot.

---

## The connection to LLM PTQ — OmniQuant

[[omniquant]] (ch-10) adopts BRECQ's block grain verbatim — one transformer layer (attention + MLP residual) per reconstruction unit. The difference:

- BRECQ optimises per-weight rounding decisions (inherited from AdaRound's rectified-sigmoid soft assignment).
- OmniQuant optimises **learnable equivalent transformations** (LWC + LET): per-channel clipping bounds and per-channel scales, both trained via block-wise MSE.

Both share the block grain, the Fisher / MSE objective, and the gradient-based optimisation. They differ in *what* is parameterised.

GPTQ takes the **opposite** path: stays per-layer, replaces gradient descent with closed-form OBS updates, and pays the cross-layer coupling cost. Empirically the two methods are competitive at W4 on Llama-2-7B (within ~0.2 ppl), proving that block-grain reconstruction and OBS sequential update are roughly equally powerful tools for the same job.

---

## Empirical: sub-4-bit ResNet (the original target)

| Method | ResNet-18 ImageNet top-1 at 4-bit | at 2-bit |
|---|---|---|
| AdaRound | 70.18 | 51.32 |
| ZeroQ | 70.04 | — (fails) |
| **BRECQ** | **70.70** | **53.94** |
| LSQ-QAT (upper bound) | 70.95 | 65.40 |

At 4-bit BRECQ closes the gap to within 0.3% of QAT. At 2-bit, BRECQ becomes the first viable PTQ. This sub-4-bit viability is what motivates the LLM-era sub-4-bit push ([[gptq]], [[awq]], [[quip]], [[aqlm]]).

---

## Mixed-precision extension (HAWQ flavour)

For each block, compute the second-order sensitivity:

```math
\Omega_k \;=\; \text{trace}(H_k) \cdot \|\Delta W_k(b)\|^2
```

and allocate higher bit-widths to blocks with larger `Ω_k` subject to a global memory budget. This is HAWQ-style allocation operating on BRECQ's block granularity — the same idea covered in [[hawq]] but at the block level rather than the layer level.

---

## QDrop integration (the regulariser)

[[qdrop]] adds a Bernoulli mask on per-layer activation quantization during BRECQ's optimisation (p ≈ 0.5). Closes the calibration-vs-test distribution gap. Single line of code, monotonic improvement at sub-4-bit. Standard add-on for any BRECQ-derived recipe.

---

## Common pitfalls

- **Defining the "block" wrong.** For a ResNet block, include the residual addition — that's where the cross-layer coupling lives. For a transformer, include both attention output projection and the FFN; cutting at the residual loses half the coupling.
- **Forgetting Fisher weighting.** Plain MSE (no `diag(F)` weighting) underperforms by ~0.3% on ResNet-50 at 4-bit. The Fisher weight is cheap (one backward of FP model) and matters.

---

## Connections

- [[excerpts/obc]] — alternative branch: per-layer OBS instead of per-block reconstruction.
- [[ch-04]] — BRECQ as a QAT-survivor (the gradient-based PTQ branch).
- [[ch-05]] — parent synthesis: BRECQ is the block-grain pillar.
- [[ch-10]] — OmniQuant inherits the block grain for LLM PTQ.
