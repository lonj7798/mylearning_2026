<!-- scope: WAGE — quantize weights, activations, gradients, and errors all to integers
     deps: dorefa-net, stochastic-rounding
     see-also: bnn, lsq, integer-only-inference
-->

# WAGE: Training and Inference with Integers in Deep Neural Networks
- **Core Insight:** Integer-only training is possible if you quantize all four tensors — weights (W), activations (A), gradients (G), and errors (E) — to low bit-widths with shift-based scaling and stochastic rounding for the gradient/error paths.
- **Guideline:** Use 8-bit forward (W/A) and 8-bit backward (E/G); replace BN with constant scaling (BN is a fp accumulator that breaks integer training); ditch SGD momentum (it accumulates fp error) and use plain SGD with stochastic-rounded integer gradients.
- **Authors:** Shuang Wu, Guoqi Li, Feng Chen, Luping Shi
- **Year:** 2018
- **URL:** https://arxiv.org/abs/1802.04680
- **Relevant topics:** integer training, low-bit backprop, hardware-friendly NN, ASIC inference

## Abstract
WAGE extends low-bit training from inference-only (weights/activations) to backprop itself, quantizing errors and gradients in addition. The forward pass uses 8-bit weights and activations; the backward pass uses 8-bit errors and gradients, all rounded by power-of-two shifts and stochastic rounding. Batch normalisation and momentum are removed because they reintroduce floating-point accumulators. On CIFAR-10 and ImageNet, WAGE matches or comes within 1–2 points of FP32 baselines using purely integer arithmetic — proof that even training can be ASIC-friendly.

## Key Contributions
- Quantizes the four tensors W, A, E, G — not just W and A.
- Shift-based scale factors (powers of two) so all rescaling is bit-shift on hardware.
- Replaces batch normalisation with constant scaling derived analytically from the layer fan-in.
- Removes momentum (an fp accumulator) and shows plain SGD with stochastic-rounded ints works.
- ImageNet AlexNet 51.6% top-1 with 8-bit W/A/G/E (FP baseline 56.0%).

## Key Figures/Tables to Study
- **Figure 1** — diagram of the four quantizers Q_W, Q_A, Q_G, Q_E around a linear layer.
- **Table 2** — bit-width ablations across CIFAR/ImageNet.

## Technical Details

### Four quantizers
For bit-widths (k_W, k_A, k_E, k_G):
- `Q_W(W) = clip(σ(k_W) · round(W/σ(k_W)), −1+σ, 1−σ)` with σ = 2^{1−k_W}.
- `Q_A(a) = σ_A · round(a/σ_A)` (deterministic rounding).
- `Q_E(e) = σ_E · round(e/σ_E)` (deterministic; errors are dense enough).
- `Q_G(g) = σ_G · stoch_round(g/σ_G)` (stochastic rounding — non-negotiable for gradients).

### Shift-based scaling (replaces BN)
Per layer: `α_L = 2^{round(log2(√(6/n_in)))}` (Xavier init expressed as a shift).
Forward: `a = α_L · (Wᵢ ⨂ aᵢ₋₁)`; the scale is folded into the integer accumulator.

### Stochastic rounding
`stoch_round(x) = ⌊x⌋ + 1[u < x − ⌊x⌋],  u ~ U(0,1)`
Yields `E[stoch_round(x)] = x` — unbiased, mandatory for gradient quantization.

### Why no momentum
Momentum buffer m ← γm + g requires fp; quantizing m to 8-bit drifts due to repeated rounding. WAGE drops momentum entirely and shows plain SGD trains well at 8-bit.

### Why no BN
BN's running mean/var is a fp accumulator; the per-batch normalisation introduces non-integer scales. The fixed shift α_L approximates BN's variance-correction effect.

## Connections
- [[dorefa-net]] — direct predecessor; WAGE adds E quantization.
- [[stochastic-rounding]] — the unbiased-rounding result WAGE depends on.
- [[bnn]] — 1-bit extreme of the same idea.
- [[integer-only-inference]] — sister paper for inference-only integer pipelines.
- [[lsq]] — combines well: WAGE for backward, LSQ-style learned step on forward.
