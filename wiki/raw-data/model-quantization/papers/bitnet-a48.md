<!-- scope: BitNet a4.8 — 4-bit activations on top of 1.58-bit ternary weights, with sparsification of intermediate values
     deps: [[bitnet-b158]], [[bitnet]]
     see-also: [[onebit]], [[era-of-1bit-llms]]
-->

# BitNet a4.8: 4-bit Activations for 1-bit LLMs
- **Core Insight:** BitNet b1.58 left activations at 8-bit (INT8) — pushing them to INT4 directly fails because of outlier channels, but a *hybrid* recipe — INT4 (or FP4) for attention and FFN inputs + sparsification + INT8 only for the intermediate states with outliers — captures the inference-throughput win of A4 without breaking the 1.58-bit weight model.
- **Guideline:** When building 1-bit-weight LLMs and targeting INT4 tensor-core throughput, use BitNet a4.8: absmean activation scaling, INT4 for attention/FFN inputs, INT8 + top-K sparsification for intermediate FFN states, and 3-bit KV cache; trained on equivalent compute to b1.58 with comparable end-task accuracy.
- **Authors:** Hongyu Wang, Shuming Ma, Furu Wei (Microsoft Research)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2411.04965
- **Relevant topics:** 1.58-bit weights, 4-bit activations, sparsification, hybrid INT4/INT8, 3-bit KV

## Abstract
BitNet a4.8 extends BitNet b1.58 (ternary weights) with 4-bit activations. The challenge: certain intermediate tensors (especially the FFN gate/down inputs) contain outlier channels that crush INT4 dynamic range. The fix is a hybrid quantization-and-sparsification strategy: A4 (INT4 or FP4) for attention and FFN inputs, INT8 with top-K sparsification for outlier-heavy intermediate states, and 3-bit KV cache. Performance matches BitNet b1.58 at the same training cost, with INT4 tensor-core kernels for faster inference and only ~55% of parameters active per token.

## Key Contributions
- Hybrid quantization-sparsification: route activations through INT4 vs INT8 vs sparse-FP based on outlier sensitivity per layer position.
- Absmean activation scaling: per-token scale derived from `mean(|x|)` instead of `max(|x|)` — smoother, more stable than absmax for native-trained INT4.
- 3-bit KV cache support, leveraging the KV-quant lineage on top of an already-quantized backbone.
- Demonstrates that the BitNet line can scale to A4 throughput without sacrificing the b1.58 quality.

## Key Figures/Tables to Study
- **Figure 1:** Architecture diagram — INT4 / INT8 / sparse routing per tensor position.
- **Figure 3:** Sparsification at the FFN intermediate — top-K element selection.
- **Table 2:** Perplexity and zero-shot accuracy — a4.8 vs b1.58 vs FP16 LLaMA-equivalent.

## Technical Details

### Weight quantization (inherited from b1.58)
Ternary `{−1, 0, +1}` with per-tensor absmean scale:
`scale_w = mean(|W|)`,  `W_q = round(W / scale_w)` clipped to `{−1, 0, +1}`.
Effective 1.58 bits/weight (log₂ 3).

### Activation quantization — absmean variant
Standard absmax INT4:
`scale = max(|x|) / 7`,  `x_q = round(x/scale)`, clip to `[−8, +7]`.
BitNet a4.8 uses absmean variant:
`scale = mean(|x|) · k`, with k a learned per-layer multiplier.
absmean is more robust to single-token outliers than absmax — important when the underlying weights are ternary and have no precision headroom to absorb activation quantization error.

### Hybrid routing per tensor position
- Attention Q, K, V inputs: INT4 absmean.
- FFN gate/down inputs: INT8 + sparsification (top-K elements per token kept, rest zeroed). The intermediate FFN tensor is the most outlier-prone position in any LLM; sparsifying past the top-K kills the outlier tail without crushing precision.
- Output projection inputs: INT4 absmean.
- KV cache: 3-bit per token, separate scales.

### Sparsification
For the intermediate FFN tensor h ∈ ℝ^d:
`h̃_i = h_i if |h_i| ≥ τ_k(|h|) else 0`
where τ_k is the K-th largest absolute value. Sparsity ratio ~45% (so ~55% of activations live). Multiply-accumulate skips zero entries.

### Training
Trained from scratch with the same compute budget as BitNet b1.58. Uses the b1.58 STE recipe for weights:
```
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale
```
plus a similar STE for activations with the absmean scale.

### Inference cost
- INT4 attention/FFN-input GEMMs run on tensor cores at 2× INT8 throughput.
- Sparse FFN intermediate: only ~55% of MACs executed.
- Net result: ~2× decode throughput vs b1.58 at the same memory footprint.

## Connections
- Direct predecessor: [[bitnet-b158]] (1.58-bit ternary weights, A8).
- Original 1-bit predecessor: [[bitnet]] (1-bit weights, scratch training).
- Sibling 1-bit weight schemes: [[onebit]] (SVID decomposition).
- Consolidated survey: [[era-of-1bit-llms]].
- Activation absmean / absmax scaling theory: [[uniform-quantization-noise]].
- KV-quant background for the 3-bit KV path: [[kivi]], [[kvquant]].
