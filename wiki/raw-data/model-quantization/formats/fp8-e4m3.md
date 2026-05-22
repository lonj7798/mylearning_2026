<!-- scope: OFP8 / NVIDIA H100 FP8 E4M3 — 4 exp + 3 mantissa; for activations and weights
     deps: [[ieee-754]]
     see-also: [[fp8-e5m2]], [[fp8-formats-paper]], [[fp8-lm]], [[deepseek-v3-fp8]]
-->

# FP8 E4M3 (OFP8 / NVIDIA Hopper)
- **Core Insight:** Allocating 4 exponent bits and 3 mantissa bits to an 8-bit float gives a 17-bin exponent range (E ∈ [−6, +8]) and ~12% relative precision — narrower range than E5M2 but finer precision, making E4M3 the natural choice for *weights and activations* in FP8 mixed-precision (paired with the wider-range E5M2 for gradients).
- **Guideline:** Use E4M3 for forward-pass tensors (activations, weights) and pair with E5M2 for backward-pass tensors (gradients); always apply per-tensor or per-block scale to fit the dynamic range of the actual tensor.
- **Authors:** Paulius Micikevicius et al. (joint NVIDIA + Arm + Intel "FP8 Formats for Deep Learning" proposal, 2022); standardized as OCP OFP8 (2023)
- **Year:** 2022 (paper); 2023 (OCP OFP8 spec); H100 hardware (2022)
- **URL:** https://arxiv.org/abs/2209.05433 ("FP8 Formats for Deep Learning"); https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
- **Relevant topics:** FP8, E4M3, mixed-precision training, Hopper, Transformer Engine

## Abstract
The 2022 joint NVIDIA/Arm/Intel proposal introduced two 8-bit floating-point formats: E4M3 (4 exponent + 3 mantissa) and E5M2 (5 exponent + 2 mantissa). E4M3 trades the IEEE-style ±∞ encoding for one extra finite value at the top of the range, reserving only S.1111.111 = NaN. This gives 254 finite values (vs 252 for a strict IEEE-style E4M3) and a max representable magnitude of 448. The format became the OCP OFP8 standard in 2023 and is the native FP8 weight/activation format on NVIDIA Hopper (H100) and Blackwell (B100) tensor cores, AMD MI300, and Intel Gaudi 2/3.

## Key Contributions
- Two-format FP8 system: E4M3 for weights/activations, E5M2 for gradients.
- Sacrifices ±∞ encoding in E4M3 to gain 2 extra representable values; only NaN reserved at the top.
- Demonstrates BERT-Large, GPT-3 175B, and image-model training matching BF16 with FP8 forward/backward.
- Drove H100 / B100 tensor-core design and the entire Transformer Engine library.
- Standardized as OCP OFP8 (2023).

## Key Figures/Tables to Study
- **Bit-layout side-by-side**: E4M3 vs E5M2 vs IEEE binary8 — shows the ±∞-sacrifice trick.
- **Per-tensor scale histograms** (Micikevicius Fig. 4): weight/activation magnitude distributions vs E4M3 / E5M2 representable ranges, motivating scale choice.

## Technical Details

### Bit layout
| Field | Bits | Width |
|-------|------|-------|
| sign s | 7 | 1 |
| exponent e | 6..3 | 4 |
| mantissa m | 2..0 | 3 |

- **Exponent bias:** 7. Unbiased E = e − 7.
- **Normal:** 1 ≤ e ≤ 15 → value = (−1)^s · (1 + m/8) · 2^{e−7}
  - **EXCEPTION at e=15, m=7** (S.1111.111): reserved as NaN, not finite.
  - **All other e=15 encodings are finite** (this is the non-IEEE departure).
- **Subnormal:** e = 0, m ≠ 0 → value = (−1)^s · (m/8) · 2^{−6}
- **Zero:** e = 0, m = 0 → ±0
- **No ±∞ encoding.** Only NaN reserved (one each for ±, technically).

### Key constants
- Smallest positive normal: 2^{−6} ≈ 1.5625e−2
- Smallest positive subnormal: 2^{−9} ≈ 1.9531e−3
- Largest finite: 1.75 · 2^{8} = **448** (= S.1111.110 = (1+6/8)·2^{15−7})
- Machine epsilon ε = 2^{−3} = 0.125
- Total finite values: 254 (with ±0 and NaN distinguished)

### Comparison with E5M2
| | E4M3 | E5M2 |
|---|---|---|
| Exponent bits | 4 | 5 |
| Mantissa bits | 3 | 2 |
| Bias | 7 | 15 |
| Max finite | 448 | 57344 |
| Min positive normal | 2^{−6} ≈ 0.0156 | 2^{−14} ≈ 6.1e−5 |
| ε | 0.125 | 0.25 |
| Special | NaN only | ±∞, NaN (IEEE-like) |

E4M3 → finer precision (12% vs 25% relative), narrower range (~10^{4.5} vs ~10^{9}).
**Use E4M3 for tensors with bounded magnitude after normalization (weights, activations); use E5M2 for tensors with heavy tails (gradients).**

### Why drop ±∞ in E4M3?
With only 256 codes, sacrificing 14 (S.1111.???) for ±∞+NaN is expensive. The OFP8 spec reserves only the two S.1111.111 encodings for NaN and lets all other S.1111.??? be finite, giving 6 extra finite values per sign (vs IEEE-style). This is the canonical "non-IEEE" departure of E4M3.

### Per-tensor / per-block scale
Raw E4M3 covers only [−448, +448]; LLM activations can reach 10^3 or more. A scale s ∈ FP32 (per-tensor) or per-block (in MX formats; see [[mx-formats]]) is multiplied in:
```
x_quant = round_to_e4m3(x_fp32 / s)
x_recon = x_quant · s
```
Scale chosen to map the per-tensor max-abs to the E4M3 max (448) — "amax" scaling. Dynamic-range-aware scale calibration is the core of NVIDIA Transformer Engine.

### Rounding mode
RNE by default; SR optional (NVIDIA Hopper supports both as instruction-level options).

### Hardware throughput (H100 SXM)
- FP8 tensor cores: ~1979 TFLOPS (E4M3 or E5M2) with FP32 accumulator.
- 2× BF16 throughput (989 TFLOPS), 4× FP32 (495 TFLOPS).

### Production usage
- [[fp8-lm]] (Peng 2023): FP8 LM training recipe; E4M3 for fwd weights/activations.
- [[deepseek-v3-fp8]]: first frontier-scale (671B) FP8 native training; E4M3 for most ops, per-block FP8 scale, FP32 accumulator.
- NVIDIA Transformer Engine library: automatic E4M3/E5M2 management.
- [[mx-formats]]: MXFP8 = E4M3 element + E8M0 block scale, 32-element blocks.

## Connections
- [[ieee-754]] — FP8 inherits sign/exponent/mantissa structure, but sacrifices ±∞ in E4M3.
- [[fp8-e5m2]] — gradient-side partner; IEEE-like ±∞/NaN semantics.
- [[fp8-formats-paper]] — Micikevicius et al. 2022, the defining paper.
- [[mx-formats]] — block-scaled FP8 (MXFP8).
- [[fp8-lm]] / [[deepseek-v3-fp8]] — production training recipes.
- [[transformer-engine]] — NVIDIA's reference FP8 software.
