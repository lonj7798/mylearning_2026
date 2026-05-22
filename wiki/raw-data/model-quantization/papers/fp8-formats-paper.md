<!-- scope: FP8 Formats for Deep Learning — joint NVIDIA/Arm/Intel spec defining E4M3 + E5M2
     deps: [[ieee-754]], [[bf16]], [[fp16]]
     see-also: [[fp8-e4m3]], [[fp8-e5m2]], [[fp8-lm]], [[deepseek-v3-fp8]]
-->

# FP8 Formats for Deep Learning
- **Core Insight:** A single FP8 format can't simultaneously have the dynamic range needed for gradients (~30+ orders of magnitude in training) and the precision needed for activations/weights; the right solution is two interchange formats — E4M3 (4 exp + 3 mantissa) for forward-pass values and E5M2 (5 exp + 2 mantissa) for gradients — with the GEMM hardware accepting both.
- **Guideline:** Use **E4M3 for weights and forward activations**, **E5M2 for backward gradients**; keep per-tensor scale management in higher precision and let the format's larger of (range, precision) match the variable's empirical distribution.
- **Authors:** Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, Hao Wu
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2209.05433
- **Relevant topics:** FP8 spec, E4M3, E5M2, NVIDIA H100 tensor cores, joint industry standard

## Abstract
A joint NVIDIA / Arm / Intel proposal for an industry-standard 8-bit floating-point interchange format with two encodings: **E4M3** (1 sign + 4 exponent + 3 mantissa) for forward-pass weights and activations, and **E5M2** (1 sign + 5 exponent + 2 mantissa) for backward gradients. The paper demonstrates that FP8 training preserves quality across CNNs, RNNs, and Transformers up to 175B parameters, provided E4M3 and E5M2 are used in the right places. The spec is what NVIDIA H100 and B100/B200 implement natively in tensor cores; it became the basis of the Open Compute Project (OCP) FP8 standard.

## Key Contributions
- Two-format design — different (range, precision) trade-off for different roles in training.
- Empirical demonstration across CNN/RNN/Transformer of training-quality parity with FP16/BF16 when E4M3/E5M2 placement follows the prescribed pattern.
- Standardised bias and special-value handling adopted by NVIDIA tensor cores.
- Defines the IEEE-754-like binary interchange that the OCP MX formats subsequently inherit.

## Key Figures/Tables to Study
- **Table 1:** E4M3 vs E5M2 spec (bias, max representable, NaN handling).
- **Figure 4–6:** training curves for Megatron-LM GPT-3, Transformer-XL — FP8 matches BF16.
- **Section 4:** placement guidance per layer type.

## Technical Details

### E4M3 (forward-pass format)
- 1 sign bit, 4 exponent bits, 3 mantissa bits.
- **Exponent bias = 7**.
- Max representable value = `1.111_2 × 2^{15-7} = 1.875 × 256 = 480`. Paper reports usable range up to ~448 because the all-ones exponent is partially reserved.
- **No infinities** (compromise to gain one more usable exponent value).
- **Single NaN bit-pattern** (mantissa all-ones with sign-1 exponent-all-ones) — frees 254 of the 256 codepoints for finite values.
- Subnormals supported.
- Useful range ≈ [2⁻⁹, 448], precision ≈ 12.5% relative.

### E5M2 (gradient format)
- 1 sign + 5 exp + 2 mantissa.
- **Exponent bias = 15** (same as FP16 — IEEE-754-like).
- Max = `1.11_2 × 2^{31-15} = 1.75 × 65536 = 114688` (≈ FP16 max).
- **Infinities supported** (IEEE-754 compliant).
- **NaN supported** (IEEE-754 compliant).
- Useful range ≈ [2⁻¹⁷, 57344], precision ≈ 25% relative.

### Why two formats
| Variable | Typical magnitude | Sensitivity to precision | Format |
|----------|-------------------|---------------------------|--------|
| Weight | O(1) | high | **E4M3** |
| Forward activation | O(1)–O(10) | high | **E4M3** |
| Gradient (weight) | O(10⁻⁶)–O(10⁻²) | low (averaged) | **E5M2** |
| Gradient (activation) | O(10⁻⁶)–O(10⁻²) | low | **E5M2** |

### Per-tensor scale
Both formats are used with a separate FP32 (or FP16) per-tensor scale `s` (sometimes per-block in MX). Tensor stored as `x_fp8 = round_fp(x / s)`; multiplied back as `x = s · x_fp8` on read.

### Hardware
- NVIDIA H100: native E4M3 and E5M2 tensor cores at 2× BF16 throughput.
- B100/B200 (Blackwell): same + native NVFP4.
- Arm / Intel: software emulation as of 2023; native hardware in subsequent generations.

### Bit-pattern summary
| Format | Sign | Exp | Mantissa | Bias | Max | Smallest normal | NaN |
|--------|------|-----|----------|------|-----|------------------|-----|
| E4M3 | 1 | 4 | 3 | 7 | 448 | 2⁻⁶ | single |
| E5M2 | 1 | 5 | 2 | 15 | 57344 | 2⁻¹⁴ | IEEE-754 |

## Connections
- Reference card pages: [[fp8-e4m3]], [[fp8-e5m2]].
- Earlier (FP16) precedent: [[fp16]], [[bf16]].
- FP8 LLM training using this spec: [[fp8-lm]], [[deepseek-v3-fp8]], [[transformer-engine]].
- FP8 PTQ inference: [[zeroquant-fp]].
- Successor block-scaled formats: [[microscaling-formats]] (MX), [[nvfp4]].
