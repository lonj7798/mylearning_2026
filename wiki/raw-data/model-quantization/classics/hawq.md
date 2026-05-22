<!-- scope: HAWQ — Hessian-aware mixed-precision bit allocation
     deps: obs-obd, brecq
     see-also: zeroq, q-bert, adaround
-->

# HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision
- **Core Insight:** Layers with high top-eigenvalue of the loss Hessian are quantization-sensitive and should get more bits; the per-layer second-order sensitivity `Ω_ℓ = λ_ℓ · ‖ΔW_ℓ(b)‖²` provides a principled, scalable criterion for assigning per-layer (and later per-block) bit-widths under a global memory budget.
- **Guideline:** Estimate per-layer top eigenvalue λ_ℓ via Hutchinson power iteration on a small calibration set (~10 mini-batches); sort layers by Ω_ℓ; assign bit-widths in a Pareto-optimal way subject to a `Σ ℓ_size · b_ℓ ≤ budget` constraint (greedy or ILP).
- **Authors:** Zhen Dong, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer
- **Year:** 2019 (ICCV)
- **URL:** https://arxiv.org/abs/1905.03696
- **Relevant topics:** mixed precision, Hessian-aware, second-order sensitivity, bit allocation

## Abstract
HAWQ is the canonical reference for per-layer mixed-precision quantization. A second-order Taylor expansion of the task loss shows that the perturbation introduced by quantizing layer ℓ contributes `(1/2) ΔW_ℓᵀ H_ℓ ΔW_ℓ` to the loss; the top eigenvalue of H_ℓ dominates this term. HAWQ estimates λ_ℓ for every layer via a few power-iteration steps with Hutchinson's trace estimator, then ranks layers by sensitivity. Higher-sensitivity layers receive more bits in a constrained allocation problem solved greedily. On ImageNet ResNet-50 mixed at 2/4-bit, HAWQ delivers 8× compression at <1% accuracy drop — substantially better than uniform-bit baselines.

## Key Contributions
- Second-order sensitivity criterion `Ω_ℓ = λ_ℓ · ‖ΔW_ℓ(b)‖²` for per-layer bit assignment.
- Power-iteration / Hutchinson estimator for top eigenvalue without forming H_ℓ.
- Greedy / ILP bit-allocation algorithm under a memory budget.
- ImageNet mixed-precision SOTA at the time (8× compression, <1% drop).
- Sets up Q-BERT (mixed-precision transformer) and BRECQ (block-wise extension).

## Key Figures/Tables to Study
- **Figure 3** — per-layer λ_ℓ for ResNet-50: top-eigenvalue varies by 100× across layers — the empirical case for mixed precision.
- **Table 5** — ResNet-50 mixed-precision vs uniform-bit: Pareto front.

## Technical Details

### Second-order Taylor argument
Task loss around fp weights W:
`L(W + ΔW) ≈ L(W) + gᵀ ΔW + (1/2) ΔWᵀ H ΔW`
At convergence g ≈ 0; H is block-diagonal across layers (approximately). So the per-layer contribution is:
`δL_ℓ ≈ (1/2) ΔW_ℓᵀ H_ℓ ΔW_ℓ`

### Bounding by top eigenvalue
`δL_ℓ ≤ (1/2) λ_max(H_ℓ) · ‖ΔW_ℓ‖²`
HAWQ uses λ_ℓ := λ_max(H_ℓ) as the per-layer sensitivity proxy. ‖ΔW_ℓ‖² depends on bit-width b_ℓ (smaller b → larger ΔW); the per-layer perturbation norm is computed via uniform quantization error.

### Hutchinson + power iteration
Hessian-vector product `Hv = ∇(∇L · v)` via autograd's double backward, requires no explicit H.
Power iteration:
```
v ← random; for t in range(T): v ← Hv / ‖Hv‖; λ ← vᵀHv
```
T ≈ 50 power steps × ~10 minibatches per layer; ~minutes for ResNet-50.

### Bit allocation (greedy)
Given budget B, per-layer sensitivity Ω_ℓ(b) for each candidate bit-width b ∈ {2, 4, 8}:
- Start all layers at b=8.
- Repeatedly drop the layer-bit pair with the smallest Ω increment per byte saved until budget met.

ILP variant exists for optimal allocation but the greedy is within ~0.1% accuracy and 100× faster.

### Compatible with QAT and PTQ
- HAWQ-QAT: after allocation, run standard QAT (LSQ / DoReFa) at the assigned bit-widths.
- HAWQ-PTQ: combine with [[adaround]] or [[brecq]] for the per-layer/block rounding.

### Empirical effect
- ResNet-50 ImageNet, model size from 102 MB (fp32) → 12 MB (mixed 2/4): top-1 drops 0.8%.
- BERT-Base GLUE: applied as Q-BERT (mixed 2/3-bit) loses only 2.3 GLUE points.

## Connections
- [[obs-obd]] — same 2nd-order framework applied to pruning instead of bit allocation.
- [[brecq]] — extends HAWQ to block-wise granularity and Fisher-information approximation.
- [[zeroq]] — combines with HAWQ for data-free mixed-precision quantization.
- [[q-bert]] — Q-BERT is HAWQ applied to BERT (lives in `papers/q-bert.md`, bucket 5).
- [[adaround]] — drop-in PTQ rounding step paired with HAWQ's bit-width allocation.
- [[gptq]] — LLM-era PTQ that benefits from HAWQ-style per-layer bit allocation though typically used uniform 4-bit.
