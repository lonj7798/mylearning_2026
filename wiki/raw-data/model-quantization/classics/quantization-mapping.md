<!-- scope: Krishnamoorthi whitepaper — the canonical PTQ playbook + taxonomy
     deps: uniform-quantization-noise
     see-also: integer-only-inference, data-free-quantization, adaround, lsq
-->

# Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper
- **Core Insight:** Choosing a quantizer is really four orthogonal decisions — symmetric vs asymmetric, per-tensor vs per-channel, weight-only vs full INT8, PTQ vs QAT — and per-channel weight quantization + per-tensor activation quantization is the sweet spot for most CNNs.
- **Guideline:** Default to int8 with per-channel weights (signed symmetric) and per-tensor activations (asymmetric, calibrated by min/max percentiles); switch to QAT only when PTQ accuracy drop exceeds 1% on the target task.
- **Authors:** Raghuraman Krishnamoorthi (Google)
- **Year:** 2018
- **URL:** https://arxiv.org/abs/1806.08342
- **Relevant topics:** PTQ taxonomy, per-channel vs per-tensor, calibration, int8 inference, QAT recipe

## Abstract
A pragmatic Google whitepaper that consolidates int8 quantization practice circa 2018. It defines the canonical taxonomy of quantizer choices, walks through implementation in TensorFlow Lite, and presents the empirical evidence that per-channel weight quantization with per-tensor activation quantization recovers nearly all FP32 accuracy on standard ImageNet CNNs (ResNet, MobileNet, Inception). The whitepaper became the reference cited by virtually every later PTQ paper, including the ones aimed at LLMs.

## Key Contributions
- Canonical taxonomy: (symmetric vs asymmetric) × (per-tensor vs per-channel) × (weight vs activation) × (PTQ vs QAT).
- Empirical demonstration that per-channel weights + per-tensor activations is the production sweet spot.
- Calibration recipe using min/max percentiles + KL divergence vs FP distribution.
- TensorFlow Lite reference implementation alongside the paper.
- Sets the convention `S, Z` (scale, zero-point) used by every later quant framework.

## Key Figures/Tables to Study
- **Figure 1** — symmetric vs asymmetric quantizer diagrams; orientation for all later work.
- **Tables 3–6** — accuracy drop across CNNs for PTQ vs QAT vs per-tensor vs per-channel.

## Technical Details

### Affine quantization rule (asymmetric)
Real → integer:
`q = clamp(round(x/S) + Z, Qmin, Qmax)`
Integer → real (dequantize):
`x̂ = S · (q − Z)`
Where S ∈ ℝ₊ is scale, Z ∈ ℤ is zero-point such that real 0.0 maps exactly to integer Z (critical for padding and ReLU correctness).

### Symmetric variant
Z = 0 (signed int) or Z = 2^{k−1} (unsigned int). Easier kernels (no Z subtraction), slight wastage of resolution if distribution is one-sided.

### Per-channel weights
For W ∈ ℝ^{C_out × C_in}: maintain S ∈ ℝ^{C_out}, one scale per output channel. Justification: weight ranges vary by 10× across filters; per-tensor scale wastes >3 bits of resolution.

### Per-tensor activations
One S, Z per activation tensor. Per-channel activation scales would require per-element rescaling at each matmul output — not GEMM-friendly.

### Calibration
1. Collect a few hundred unlabelled batches.
2. For weights: S_c = max|W_c| / Qmax (symmetric per-channel).
3. For activations: histogram + minimise KL(P_fp || P_quantized) over candidate clip ranges (TensorRT-style search).

### PTQ vs QAT decision rule
- PTQ first: if Δacc < 1%, ship it.
- Else QAT: insert fake-quant ops, train for ~10% of original schedule with reduced LR.

### Bias quantization
Bias is the small term — keep as int32 (S_bias = S_w · S_x); requantization happens at the matmul output via the rescale `M = S_w · S_x / S_y`.

## Connections
- [[integer-only-inference]] — companion paper detailing the integer-only kernel using these conventions.
- [[uniform-quantization-noise]] — the Bennett noise model justifying MSE-minimising calibration.
- [[data-free-quantization]] — when calibration data is unavailable.
- [[adaround]] — improves PTQ rounding direction without needing QAT.
- [[lsq]] — the QAT method this whitepaper hands off to when PTQ fails.
- [[gptq]] — LLM-era heir: same per-channel weight quant idea, with Hessian-aware rounding.
