<!-- scope: Jacob 2018 — int8-only mobile inference pipeline
     deps: quantization-mapping
     see-also: i-bert, q8bert, data-free-quantization
-->

# Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
- **Core Insight:** With careful per-tensor affine quantization (S, Z) and a fixed-point requantization multiplier `M = S_w · S_x / S_y ≈ M₀ · 2^{-n}` (M₀ ∈ [0.5, 1) stored as int32), an entire CNN forward pass can be executed in integer arithmetic only — no floating-point ops at runtime.
- **Guideline:** For each layer, compute int32 = int8·int8 GEMM + int32 bias, then requantize to int8 via `output = clamp((int32 · M₀) >> n + Z_y)`; train with simulated quantization (fake-quant ops in the forward pass, STE backward) so the model learns to tolerate the integer rounding.
- **Authors:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko
- **Year:** 2018 (CVPR)
- **URL:** https://arxiv.org/abs/1712.05877
- **Relevant topics:** integer-only inference, mobile CPU, fixed-point M, quantization-aware training, TFLite

## Abstract
The TensorFlow Lite reference paper. It defines the int8 quantization scheme that became the de facto industry standard for mobile / edge inference: per-tensor affine `q = round(x/S) + Z`, signed/unsigned int8 storage, int32 accumulator, fixed-point requantization to int8 between layers. The training side introduces "simulated quantization" (a.k.a. fake-quant) — quantization ops inserted into the FP training graph, transparent on the backward via STE — making QAT possible without specialised integer training kernels. On ImageNet MobileNet, the resulting int8 model runs ~2× faster on a Pixel 2 with <2% accuracy loss.

## Key Contributions
- Defines the affine quant scheme (S, Z) that anchors every later framework.
- Specifies the integer-only matmul + requantization pipeline reproducible on commodity ARM CPUs.
- Introduces simulated quantization for QAT — the forward-only fake-quant op.
- Solves the bias-scale problem: bias is int32 with S_bias = S_w·S_x (no separate calibration).
- Open-sourced as gemmlowp + TFLite, propagating the convention.

## Key Figures/Tables to Study
- **Figure 1.1** — the integer-only matmul + requantize diagram (the canonical figure cited everywhere).
- **Algorithm 1** — fake-quant forward + STE backward.
- **Table 4** — ImageNet MobileNet int8 vs fp32: 1.8% top-1 drop, 2× CPU speedup.

## Technical Details

### Affine quantization
`q = clamp(round(x/S) + Z, 0, 255)` for uint8
`x ≈ S · (q − Z)`
S ∈ ℝ₊ (typically fp32 at compile time), Z ∈ [0, 255] (int).

### Per-layer GEMM (the load-bearing pipeline)
For weights W (int8, scale S_w, zero Z_w=0 if symmetric), input X (uint8, S_x, Z_x):
1. **int32 accumulate**: `acc = Σ_k (W_{ik} − Z_w)(X_{kj} − Z_x)` in int32.
2. **Bias add**: `acc += b`, with b stored as int32, S_b = S_w · S_x, Z_b = 0.
3. **Requantize to uint8**:
   `output = clamp(round(acc · S_w · S_x / S_y) + Z_y, 0, 255)`

### The fixed-point multiplier (the canonical M₀·2^n trick)
The real-valued scalar `M = S_w · S_x / S_y` is in (0, 1) for any sane scale choice. Express it as:
`M = M₀ · 2^{−n}`  with M₀ ∈ [0.5, 1), n ∈ ℤ₊
Store M₀ as int32 (≈ 2³¹·M₀, gives 31-bit fractional precision); n is a shift count. Then:
`int32 · M = SaturatingRoundingDoublingHighMul(int32, M₀) >> n`
i.e. one multiply-high + one right-shift — purely integer.

### Cross-layer factoring
At each layer boundary, S_y is the next layer's S_x, so M chains across the whole network. Calibration determines all (S, Z) at compile time.

### Simulated quantization (QAT)
During training, insert fake-quant op:
`x_fq = S · round(clamp(x/S, qmin, qmax) − Z) + S·Z` (forward)
`∂x_fq/∂x = 1[qmin ≤ x/S − Z ≤ qmax]` (STE backward)
Train as normal FP32 SGD; the FP weights serve as latent shadows, quantization rounds them in the forward.

### Special-case ops
- **ReLU**: absorbed into the requantize clamp (set qmin = Z_y).
- **Element-wise add**: requires matching S; one operand is rescaled first.
- **Concat**: needs matching (S, Z) across all inputs.

## Connections
- [[quantization-mapping]] — the Google whitepaper sibling to this paper.
- [[i-bert]] — extends integer-only inference to transformer non-linearities (GELU/Softmax/LayerNorm).
- [[q8bert]] — applies the simulated-quantization training recipe to BERT.
- [[data-free-quantization]] — alternative when no QAT data is available.
- [[lsq]], [[adaround]] — improvements on the calibration/training side that feed into the same integer pipeline.
- [[gptq]] — LLM-era PTQ that targets the same int4/int8 GEMM kernel design.
