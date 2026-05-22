<!-- scope: DoReFa-Net — arbitrary-bit weights, activations, and gradients
     deps: bnn, xnor-net, straight-through-estimator
     see-also: wage, pact, lsq
-->

# DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
- **Core Insight:** A single quantizer family `q_k(x) = (1/(2^k−1))·round((2^k−1)·x)` over [0,1] can quantize weights, activations, and gradients independently — and gradients can be quantized too if you add stochastic rounding.
- **Guideline:** Pick (Wbits, Abits, Gbits) per layer; preprocess weights with `tanh`/`max(|·|)` normalization before applying q_k; quantize gradients only with stochastic rounding (deterministic rounding biases the gradient and breaks SGD).
- **Authors:** Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, Yuheng Zou
- **Year:** 2016
- **URL:** https://arxiv.org/abs/1606.06160
- **Relevant topics:** k-bit QAT, low-bit gradients, stochastic rounding, ImageNet ResNet

## Abstract
DoReFa-Net generalises BNN / XNOR-Net to arbitrary bit-widths for weights, activations, and gradients (W/A/G). It defines a single quantizer family q_k(·) and shows that gradients — which span many orders of magnitude — can be quantized only if combined with stochastic rounding. On ImageNet AlexNet with (1, 2, 6) bits for W/A/G, DoReFa-Net achieves 46.1% top-1 (vs 55.9% FP), establishing the first practical low-bit training pipeline. The paper made (Wbits, Abits, Gbits) the standard way to describe a QAT recipe.

## Key Contributions
- Single uniform quantizer `q_k` parameterised by bit-width, reusable for W/A/G.
- Demonstrates low-bit gradients are feasible if and only if rounding is stochastic.
- Per-layer bit-width tagging convention `(W, A, G)`.
- Pre-normalisation `tanh(W)/(2·max|tanh(W)|) + 1/2` to map weights to [0,1] before quantization.
- ImageNet-scale validation at multiple bit budgets.

## Key Figures/Tables to Study
- **Algorithm 1** — full forward/backward pass with per-layer quantizers.
- **Table 1** — accuracy across (W, A, G) configurations; the canonical low-bit ablation table.

## Technical Details

### k-bit uniform quantizer on [0,1]
`q_k(x) = (1/(2^k − 1)) · round((2^k − 1) · x)`
Backward via STE: `∂q_k/∂x ≈ 1` inside [0,1].

### Weight pre-normalization
`W̃ = tanh(W) / (2 · max(|tanh(W)|)) + 1/2  ∈ [0,1]`
`W_q = 2·q_k(W̃) − 1  ∈ [−1, 1]`
For 1-bit: collapses to `sign(W)` × scale.

### Activation quantization
After a clipping nonlinearity (e.g. `clip(x, 0, 1)`) that ensures inputs are in [0,1]:
`A_q = q_k(clip(A, 0, 1))`

### Gradient quantization (the novel piece)
Let g = ∂L/∂A. Map to [−1,1] via per-tensor scale, then:
`g_q = 2 · max|g| · (q_k((g/(2·max|g|)) + 1/2 + N) − 1/2)`
where N is uniform noise in [−1/(2(2^k−1)), 1/(2(2^k−1))] — i.e. stochastic rounding. Without N, gradient quantization biases SGD and training diverges.

### Why stochastic rounding for gradients only
Weights/activations are quantized once per forward; deterministic rounding's bias is small relative to BN. Gradients are summed over millions of steps; any deterministic bias compounds. Stochastic rounding makes `E[g_q] = g`.

## Connections
- [[bnn]], [[xnor-net]] — 1-bit special cases of DoReFa's quantizer.
- [[straight-through-estimator]] — backward mechanism used identically.
- [[stochastic-rounding]] — the unbiased-rounding theory behind G-quant.
- [[wage]] — extends DoReFa to also quantize errors.
- [[lsq]] — replaces fixed step size with a learned one, dominating DoReFa as a QAT baseline.
