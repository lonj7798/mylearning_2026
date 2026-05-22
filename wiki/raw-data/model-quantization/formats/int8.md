<!-- scope: INT8 symmetric/asymmetric quantization; per-tensor vs per-channel scale; the LLM PTQ workhorse
     deps: [[uniform-quantization-noise]]
     see-also: [[int4]], [[llm-int8]], [[smoothquant]], [[quantization-mapping]]
-->

# INT8 (8-bit Integer Quantization)
- **Core Insight:** An 8-bit signed integer x_q ∈ [−128, +127] with a per-tensor or per-channel scale s ∈ FP32 (and optional zero-point z) gives x ≈ s · (x_q − z) — a uniform quantizer with 256 levels that recovers Bennett's Δ²/12 noise floor in regions where the input is well-modelled as smooth, and is the de facto inference precision for production deployment 2017–2023.
- **Guideline:** Default to **symmetric per-channel** for weights (sign-symmetric, no zero-point — cheaper kernels) and **asymmetric per-tensor (or per-token)** for activations (zero-point needed because activations are ReLU-positive); use SmoothQuant or LLM.int8() to handle outliers before quantization.
- **Authors:** Standardized as a hardware integer for decades; ML usage formalized in Jacob 2018 (integer-only inference whitepaper) and Krishnamoorthi 2018 (Google quantization whitepaper)
- **Year:** 2017–2018 (modern ML PTQ formulation)
- **URL:** https://arxiv.org/abs/1712.05877 (Jacob 2018); https://arxiv.org/abs/1806.08342 (Krishnamoorthi 2018)
- **Relevant topics:** INT8, symmetric/asymmetric quant, per-tensor/per-channel scale, zero-point, dequant

## Abstract
INT8 quantization represents a real tensor x ∈ ℝ as 8-bit integers x_q ∈ {−128, …, +127} (signed) or {0, …, 255} (unsigned) plus a scale factor s ∈ ℝ⁺ and optional zero-point z ∈ ℤ such that x ≈ s · (x_q − z). The quantization map is `x_q = round(x/s) + z`, clipped to the representable range. Choices of granularity (per-tensor, per-channel, per-group, per-token) and symmetry (z = 0 vs learned z) determine cost and accuracy. INT8 became the dominant deployment precision for inference 2017–2023 via TensorFlow Lite, ONNX Runtime, TensorRT, OpenVINO, and llama.cpp. For LLMs, naïve INT8 broke on outlier channels at 6.7B+ parameters, motivating LLM.int8() and SmoothQuant.

## Key Contributions
- Unified the symmetric/asymmetric, per-tensor/per-channel quantization taxonomy (Krishnamoorthi 2018).
- Demonstrated integer-only inference for mobile (Jacob 2018) — no FP32 ops at inference time.
- Establishes the canonical quant/dequant equations adopted by every subsequent framework.
- INT8 GEMM kernels (cuBLAS, MKL-DNN, llama.cpp ggml) defined the production baseline.
- Showed the outlier-channel failure mode at LLM scale, motivating LLM.int8() / SmoothQuant.

## Key Figures/Tables to Study
- **Per-tensor vs per-channel scale**: weight-quant error vs accuracy for a CNN — per-channel ≈ FP32 even at 8-bit, per-tensor drops 1–2% top-1.
- **Symmetric vs asymmetric activation quant**: distribution mismatch when activations are mostly positive (post-ReLU).

## Technical Details

### Bit representation
- **Signed INT8**: 8 bits, range [−128, +127]; two's complement.
- **Unsigned INT8** (UINT8): 8 bits, range [0, 255].

### Quantization map (asymmetric, general)
```
x_q = clip(round(x / s) + z,  q_min,  q_max)
x_hat = s · (x_q − z)
```
- **s** (scale): positive real; controls the step Δ.
- **z** (zero-point): integer; offset so that 0.0 maps to integer z (allows asymmetric ranges).
- For signed INT8: q_min = −128, q_max = +127.

### Symmetric quantization
Force z = 0; useful for sign-symmetric data (weights):
```
s = max(|x|) / 127           (signed, clipped to 127 to avoid -128 asymmetry)
x_q = clip(round(x / s), −127, +127)
x_hat = s · x_q
```
**Pros**: no zero-point ⇒ cheaper matmul (no cross-term s_w · z_w · x_q).
**Cons**: wastes half the range for non-symmetric data (e.g. post-ReLU activations).

### Asymmetric quantization
Use both s and z; fit min/max:
```
s = (max(x) − min(x)) / 255
z = round(−min(x) / s)        (clipped to [0, 255] for UINT8)
```
**Pros**: maximally uses the integer range for one-sided distributions.
**Cons**: matmul has cross-terms ⇒ ~30% kernel overhead.

### Per-tensor vs per-channel vs per-group
- **Per-tensor**: single (s, z) for whole tensor. Fastest, lowest accuracy.
- **Per-channel** (per-output-channel for weights): one (s, z) per row of W. ~1% accuracy improvement at negligible cost.
- **Per-group** (groups of 64/128 weights within a row): finer; standard for sub-INT8 (INT4 group-128).
- **Per-token** (activations): one (s, z) per batch element × sequence position. Standard for activation quant in LLM.int8() / SmoothQuant.

### Matmul with quantized weights and activations
For Y = X · Wᵀ with X_q = (X − z_x · 1) / s_x, W_q = W / s_w (symmetric weights):
```
Y ≈ s_x · s_w · (X_q · W_qᵀ)  +  s_x · s_w · z_x · (1 · W_qᵀ)
```
INT8 matmul produces INT32 accumulator → rescale by s_x · s_w → FP32 output (or requantize to INT8).

### Cost in modern hardware
- **NVIDIA**: cuBLAS / Marlin INT8 GEMM ~2× BF16 throughput on Hopper (1979 vs ~989 TFLOPS).
- **Intel Xeon (AMX)**: INT8 matmul instructions deliver ~2× BF16.
- **ARM Cortex (NEON, SVE)**: SDOT / SMMLA INT8 instructions.
- **CPU/edge**: llama.cpp's INT8 (q8_0) is the highest-quality llama.cpp quant tier.

### Bennett's noise floor
For a uniformly distributed source on [−A, +A] quantized to INT8:
- Δ = 2A / 256
- σ_q² = Δ²/12 = A² / (3 · 128²) = A² / 49152
- SNR ≈ 6.02 · 8 + 1.76 ≈ 49.9 dB (for sine input)

For outlier-heavy LLM activations the noise is dominated by clipping, not Δ²/12.

### Outlier breakdown in LLMs
At ~6.7B parameters, ~0.1% of activation channels exceed 6σ. INT8 per-tensor calibration either:
- Clips outliers ⇒ massive quantization error on those few channels ⇒ accuracy collapse.
- Includes outliers in range ⇒ Δ blows up to ~50× larger than needed for the bulk ⇒ bulk quant noise dominates.

Both fail beyond 6.7B; led to:
- [[llm-int8]]: route outlier channels through FP16 ("mixed-precision").
- [[smoothquant]]: migrate outliers from activations to weights via per-channel scaling.

## Connections
- [[uniform-quantization-noise]] — Bennett's Δ²/12 is the INT8 noise floor.
- [[quantization-mapping]] — Krishnamoorthi 2018 canonical PTQ taxonomy.
- [[int4]] — same template at 4 bits with group-wise scale.
- [[llm-int8]] — outlier-aware INT8 for LLMs.
- [[smoothquant]] — equalizes outliers to make plain INT8 work.
- [[bitsandbytes-int8]] — production INT8 kernel.
