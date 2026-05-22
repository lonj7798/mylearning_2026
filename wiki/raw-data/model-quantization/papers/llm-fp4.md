<!-- scope: LLM-FP4 — FP4 (E2M1 vs E3M0) post-training quantization for LLM weights and activations
     deps: [[fp8-formats-paper]], [[smoothquant]]
     see-also: [[microscaling-formats]], [[mxfp-training]]
-->

# LLM-FP4: 4-Bit Floating-Point Quantized Transformers
- **Core Insight:** Integer INT4 quantization assumes uniform value density, but LLM activations and weights have bell-shaped / long-tail distributions where *floating-point* representation captures the dynamic range better at the same bit budget — and the right FP4 sub-format (E2M1 vs E3M0 vs E1M2) should be searched *per channel* because the optimal exponent/mantissa split depends on local distribution shape.
- **Guideline:** For FP4 weight + FP4 activation PTQ, use LLM-FP4: per-channel exponent-bias reparameterization for activation scales + per-channel format search across {E2M1, E3M0, E1M2}; achieves LLaMA-13B at 63.1 zero-shot average (only 5.8 points below FP).
- **Authors:** Shih-yang Liu, Zechun Liu, Xijie Huang, Pingcheng Dong, Kwang-Ting Cheng
- **Year:** 2023 (EMNLP 2023)
- **URL:** https://arxiv.org/abs/2310.16836
- **Relevant topics:** FP4, PTQ, per-channel format search, exponent-bias reparameterization

## Abstract
LLM-FP4 is the first systematic study of 4-bit floating-point post-training quantization for LLM weights *and* activations. The key empirical finding: FP4 (sign + exponent + mantissa) outperforms INT4 on long-tail LLM tensors because the exponent bits capture dynamic range that INT cannot. The choice of FP4 sub-format (how to split 3 non-sign bits into exponent and mantissa) is non-trivial and should be searched per channel. Combined with a reparameterization of per-channel activation scales as exponent biases (free fold-in to the FP4 representation), LLM-FP4 quantizes LLaMA-13B to W4A4 at 63.1 zero-shot avg vs FP 68.9.

## Key Contributions
- First post-training FP4 quantization for both LLM weights and activations.
- Per-channel format search: each channel picks its FP4 sub-format from {E2M1, E3M0, E1M2} based on calibration distribution fit.
- Exponent-bias reparameterization: per-channel activation scaling factors fold into the per-channel exponent bias of the FP representation, eliminating a separate scale multiplication.
- Demonstrates FP4 dominates INT4 on activation-heavy regimes (W4A4 > W4-INT-A4-INT by 5+ points).

## Key Figures/Tables to Study
- **Figure 2:** Comparison of representable values for E2M1, E3M0, INT4 — shows FP coverage of the long tail.
- **Figure 4:** Per-channel optimal format selection map across an LLaMA-13B layer.
- **Table 3:** W4A4 LLM-FP4 vs SmoothQuant W4A4 INT on common-sense reasoning.

## Technical Details

### The FP4 sub-formats
A 4-bit float has 1 sign bit + 3 value bits split into e exponent + m mantissa, with e + m = 3:
- **E2M1**: 2 exp + 1 mantissa. Values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Mid-range resolution.
- **E3M0**: 3 exp + 0 mantissa. Values ±{0, ±2^{-3}, ±2^{-2}, ..., ±2^3}. Widest range, no per-bin mantissa.
- **E1M2**: 1 exp + 2 mantissa. Narrowest range, finest resolution near zero.

Different channels have different distribution shapes:
- Outlier-heavy → E3M0 (need range).
- Concentrated → E1M2 (need resolution).
- Mixed → E2M1.

### Per-channel format search
For each output channel c, do a brute-force scan over {E2M1, E3M0, E1M2} on the calibration set and pick the format minimising MSE of dequantize(quantize(W_c)). Format choice stored as a 2-bit tag per channel (negligible overhead).

### Exponent-bias reparameterization (the trick that makes per-channel activation scaling free)
Standard per-channel activation quant: `x_q = quantize(x / s_c) · s_c`, where s_c is the per-channel FP scale.

LLM-FP4 observes that for FP4, the per-channel scale s_c can be folded into the *exponent bias* of the FP representation: replace s_c by an integer exponent shift e_c such that `s_c = 2^{e_c}`. Then quantize(x · 2^{−e_c}) is just a shift of the FP exponent bits — no multiplication needed at runtime.

This makes per-channel scaling free in FP4 (unlike in INT4 where a per-channel scale always requires an FP multiply).

### Calibration
- Weights: round-to-nearest in chosen per-channel format.
- Activations: dynamic per-token range estimation, per-channel exponent bias fixed from calibration.
- No fine-tuning required.

### Results
LLaMA-13B W4A4 LLM-FP4: WikiText-2 PPL 6.9 vs FP16 5.4. Zero-shot avg 63.1 vs FP 68.9. Beats INT-based W4A4 baselines (SmoothQuant-W4A4) by ~5 points.

### Cost
Forward pass: standard FP4 GEMM (e.g. on Hopper E4M3 unit + simulator for FP4). No runtime overhead beyond FP4 hardware.

## Connections
- FP8 ancestor: [[fp8-formats-paper]].
- Microscaling sibling: [[microscaling-formats]] (block-shared exponent generalization).
- INT4 competitors at the same bit budget: [[gptq]] W4A4, [[smoothquant]] W4A4.
- Activation-scaling lineage: [[smoothquant]], [[awq]].
- MX training studies that adopt FP4: [[mxfp-training]].
