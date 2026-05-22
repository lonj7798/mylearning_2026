<!-- scope: Error compounding across a transformer block — analytic bounds
     deps: uniform-quantization-noise, integer-only-inference
     see-also: brecq, qdrop, llm-int8
-->

# Quantization Error Propagation Across a Transformer Block
- **Core Insight:** Per-layer quantization noise compounds multiplicatively through depth — a transformer block introduces error of order `O(√L · σ²)` after L matmuls if errors are independent, but transformer residual streams and LayerNorm renormalisation break that independence in both directions (residual amplifies via accumulator drift; LayerNorm contracts via the 1/√Var rescale).
- **Guideline:** Bound per-layer error variance σ_ℓ² = Δ_ℓ²/12 · ‖input‖² (Bennett); track the accumulated variance through the block accounting for the residual sum (additive) and LayerNorm rescale (divisive); allocate bit-width inversely to the per-layer noise amplification factor when designing mixed-precision schemes.
- **Authors:** consolidation (Lin 2016, Choi 2020 study of compound error, Park 2023 quant-noise propagation in transformers)
- **Year:** 2016–2023 (consolidated)
- **URL:** https://arxiv.org/abs/1606.06160 (DoReFa noise discussion); https://arxiv.org/abs/2004.09602 (Wu integer quant survey, error-prop section)
- **Relevant topics:** error propagation, noise compounding, residual stream, LayerNorm scaling

## Abstract
This page consolidates the analytic framework for tracking quantization-noise propagation through a transformer block. The two key observations: (1) under independent uniform-noise assumptions, errors accumulate as a sum of variances across layers; (2) transformer-specific structure — residual streams (additive accumulators) and LayerNorm (divisive rescaling) — change the simple O(√depth) compounding bound into a depth-dependent expression that explains why deep transformers are quantization-sensitive in the middle of the stack (where the residual norm peaks) and tolerant at the embedding and output ends.

## Key Contributions
- Per-layer variance bound: `σ_ℓ² = Δ_ℓ²/12 · ‖x_ℓ‖²` (Bennett applied to per-tensor scale Δ_ℓ).
- Residual-stream accumulator analysis: variance sums across the L blocks in the residual path.
- LayerNorm contraction: post-LN noise is rescaled by `1/√Var(x)` — outlier channels amplify noise.
- Motivates per-block bit allocation (BRECQ, HAWQ) instead of uniform-bit.

## Key Figures/Tables to Study
- **Wu 2020 Figure** — per-layer quantization-noise contribution measured empirically across a CNN/transformer stack.
- **BRECQ Figure 1** — block-by-block reconstruction error growth, the empirical signature of compounding.

## Technical Details

### Per-layer noise variance (Bennett)
For a uniform quantizer of step Δ_ℓ on a tensor of size n_ℓ, the quantization noise n_ℓ has:
- mean 0
- variance Δ_ℓ²/12 per element (high-rate assumption)
- approximate independence from the signal

After a linear layer y = W·x, with x_quant = x + e_x, W_quant = W + e_W:
`y_quant = (W + e_W)(x + e_x) = W·x + e_W·x + W·e_x + e_W·e_x`
The dominant noise terms (ignoring second-order e_W·e_x):
`Var(y) ≈ ‖x‖² · σ_W² + ‖W‖_F² · σ_x²`

### Compounding without residuals
For L stacked linear layers each contributing σ_ℓ² output variance, independent errors give:
`σ_out² ≈ Σ_ℓ (Π_{k>ℓ} ‖W_k‖_F²) · σ_ℓ²`
The product factor is the "noise amplification" through subsequent layers; LayerNorm typically keeps this near 1.

### Residual stream (additive accumulator)
A transformer block: `x_{ℓ+1} = x_ℓ + Attn(LN(x_ℓ)) + FFN(LN(x_ℓ + Attn(...)))`.
Noise injected at block ℓ accumulates additively into x_{ℓ+L} for all L > 0. After D blocks:
`Var(residual noise) ≈ Σ_{ℓ=1}^D σ_ℓ_{block}²`
Independent of layer depth — pure additive sum. This is why mid-stack blocks dominate: their noise persists through every subsequent block.

### LayerNorm contraction
LN(x) = γ · (x − μ)/√(Var(x) + ε) + β.
Noise added to x is rescaled by 1/√Var(x); if x has outlier channels with large variance (the [[llm-int8]] regime), the bulk noise is suppressed but the outlier-channel noise is amplified (because their pre-LN magnitudes dominate Var).

This creates the per-channel noise asymmetry that motivates per-channel weight scaling ([[awq]], [[smoothquant]]).

### Attention compounding
Softmax(QKᵀ/√d) amplifies noise quadratically in the softmax temperature regime: a small noise δ on QKᵀ becomes ~e^δ in attention probabilities. This is why attention-input scale matters disproportionately in PTQ — see KL calibration in [[mse-vs-kl-calibration]].

### Mixed-precision implication
Layers whose noise amplification factor `Π_{k>ℓ} ‖W_k‖_F²` is large should get more bits. This is precisely the HAWQ Hessian-trace criterion (the Hessian encodes loss-sensitivity, which is the gradient of noise amplification).

### Empirical fingerprint
- Embedding & output layers: low amplification, can be aggressive (4-bit safe).
- Mid-stack residual MLP: high amplification, conservative (≥6-bit).
- Final-block attention: amplifies into the head; per-channel + percentile required.

## Connections
- [[uniform-quantization-noise]] — Bennett model that supplies the per-layer variance.
- [[integer-only-inference]] — the int-only pipeline where this noise budget is enforced.
- [[brecq]] — block-wise PTQ that directly minimises this compounded error.
- [[qdrop]] — regularises the PTQ search against the same compounding by randomising layer dropouts.
- [[hawq]] — Hessian-based bit allocation that responds to the propagation profile.
- [[llm-int8]] — empirical fingerprint of LayerNorm-contraction outliers at 6.7B scale.
- [[smoothquant]] — restructures the per-channel asymmetry the propagation analysis predicts.
