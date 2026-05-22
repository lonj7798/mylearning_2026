<!-- scope: HQQ — half-quadratic splitting for data-free LLM weight quantization
     deps: [[straight-through-estimator]], [[lloyd-max-quantizer]]
     see-also: [[gptq]], [[awq]]
-->

# HQQ: Half-Quadratic Quantization
- **Core Insight:** The non-convex non-differentiable problem `min_z ||W − dequant(quant(W; z))||` can be split via half-quadratic optimization into two convex sub-problems (a continuous projection and a closed-form scale/zero update) that are alternated to convergence — yielding a data-free quantizer with no calibration data, no STE, and no gradient descent.
- **Guideline:** When you need fast (seconds-per-layer) data-free weight-only LLM quantization at 2/3/4/8 bits, use HQQ: solves the per-group quantization problem in 4–8 iterations of closed-form alternating updates; trivially parallelizable across all linears.
- **Authors:** Hicham Badri, Appu Shaji (Mobius Labs)
- **Year:** 2024
- **URL:** https://mobiusml.github.io/hqq_blog/ ; https://github.com/mobiusml/hqq
- **Relevant topics:** half-quadratic splitting, data-free PTQ, fast quantization, closed-form updates

## Abstract
HQQ formulates weight-only LLM quantization as a robust regression problem with a sparsity-promoting (lp, p<1) loss to handle outliers, and solves it via half-quadratic splitting. The split alternates between (1) a continuous proximal step for an auxiliary variable W_e (the "error projection") and (2) a closed-form update of the zero-point (and optionally scale). Requires no calibration data and no gradients. Quantizes Llama-2-70B at 4-bit in minutes on a single GPU with quality competitive with GPTQ / AWQ.

## Key Contributions
- Casts PTQ as `min_{z, s, W_q} ||W − s(W_q − z)||_p^p` with p ≈ 0.5–0.7 to be robust to outliers.
- Introduces auxiliary variable W_e = W − s(W_q − z) and applies half-quadratic splitting to decouple the two unknowns.
- Closed-form update for z: `z ← median(W_q − W/s)` (under p=0.5–1 the median is the proximal operator for the lp norm).
- Closed-form update for W_e: shrinkage / soft-thresholding.
- Completely data-free — no calibration corpus, no Hessian estimation.

## Key Figures/Tables to Study
- The HQQ blog Figure 1: comparison of HQQ vs GPTQ vs AWQ on wall-clock and PPL.
- Alternating update pseudocode (HQQ blog, Section "Algorithm").
- Per-bit PPL table on Llama-2 — HQQ at 4-bit ≈ GPTQ at 4-bit; at 2-bit HQQ trails AQLM but beats RTN.

## Technical Details

### Quantization rule
Per-group (e.g. group size 64) symmetric/asymmetric:
`W_q = round((W / s) + z)`,  `Ŵ = s · (W_q − z)`
where s ∈ ℝ (scale), z ∈ ℝ (zero-point), W_q ∈ {0, ..., 2^b − 1}.

### The lp objective (robust regression)
`min_{s, z, W_q} ||W − s(W_q − z)||_p^p`
with p ∈ (0, 1) (typically 0.5 or 0.7). Lp<1 down-weights outlier residuals — equivalent to assuming a heavy-tailed (generalized Gaussian) prior on the quantization error, which matches real LLM weight distributions.

### Half-quadratic splitting (HQS)
Introduce auxiliary W_e satisfying W_e = W − s(W_q − z):
`min ||W_e||_p^p + (β/2) ||W − s(W_q − z) − W_e||²`
β is a penalty parameter increased across iterations (continuation).

Alternating updates:
1. **W_e step** (fix z, s, W_q): closed-form proximal of lp:
   `W_e ← prox_{||·||_p^p / β} (W − s(W_q − z))`
   For p=0.5 this is the half-shrinkage operator; for p=2/3 a cubic-root closed form exists.
2. **z step** (fix W_e, s; recompute W_q implicitly):
   `z ← median_group(W_q − (W − W_e)/s)`
   Median is the L1-proximal — the natural counterpart to the robust-loss prior.
3. **(optional) s step**: closed-form least-squares update for s.
4. Refresh W_q = round((W − W_e)/s + z); increase β; repeat.

Convergence: 4–8 iterations per group.

### Why data-free
The objective minimizes raw weight reconstruction; no input distribution X is needed. Equivalent to assuming I·input — gives up the Hessian-weighting that GPTQ exploits, but the lp robustness mostly compensates for outlier-heavy real-world weights without needing calibration.

### Speed
All operations are elementwise / per-group; can be vectorized over an entire weight matrix. Per-layer cost is milliseconds-per-million-weights. Llama-2-70B at 4-bit: <10 minutes on a single A100.

### Bit-widths supported
8 / 4 / 3 / 2 / 1 bits (1-bit requires accepting larger PPL hit since no fine-tune).

## Connections
- Theoretical ancestor: half-quadratic optimization (Geman-Reynolds 1992) for robust image restoration.
- Quant-only baselines it competes with: [[gptq]] (Hessian-aware, needs calibration), [[awq]] (activation-aware, needs calibration), RTN (round-to-nearest, weakest).
- Robust loss ancestry: Lp-regression literature (Sparse PCA, Lp-norm regularization).
- Framework integration: HuggingFace transformers via `quantization_config = HqqConfig(...)`, llama.cpp, vLLM.
