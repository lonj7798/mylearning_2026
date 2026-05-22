<!-- scope: IEEE-754 floating-point standard; binary32 / binary16 bit layout, subnormals, special values, rounding
     deps:
     see-also: [[bf16]], [[fp16]], [[fp8-e4m3]], [[round-to-nearest-even]]
-->

# IEEE-754 Floating-Point Standard (binary32 / binary16)
- **Core Insight:** Every IEEE-754 number is represented as (−1)^s · 1.m · 2^{e − bias} (normal) or (−1)^s · 0.m · 2^{1 − bias} (subnormal), with reserved exponent encodings for ±0, ±∞ and NaN — giving a logarithmically-spaced grid that is the foundation every modern low-precision float (BF16, FP16, FP8, FP6, FP4, MX) inherits.
- **Guideline:** When designing or debugging a low-precision quantizer, always work out the exponent range, smallest normal, smallest subnormal, and machine epsilon first — these four numbers explain almost every numerical pathology (underflow, overflow, "0/0", silent loss of small updates).
- **Authors:** IEEE-754 working group (1985, 2008, 2019 revisions); William Kahan led the original committee
- **Year:** 1985 (original); 2008 (revised); 2019 (latest)
- **URL:** https://ieeexplore.ieee.org/document/8766229 ; Kahan lecture notes at https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
- **Relevant topics:** floating-point representation, bit layout, subnormals, NaN, machine epsilon

## Abstract
IEEE-754 defines binary floating-point formats with a sign bit, an unsigned biased exponent, and a fraction (mantissa). Normal numbers carry an implicit leading 1 bit; subnormal numbers (exponent field = 0, mantissa ≠ 0) extend the dynamic range smoothly to zero with a leading 0 bit and gradual underflow. Two exponent encodings are reserved: all-zeros for ±0 / subnormals, all-ones for ±∞ and NaN. The standard also fixes correctly-rounded basic arithmetic (under five rounding modes) and signaling semantics. binary32 (FP32) and binary16 (FP16) are the formats most widely used in deep learning; bfloat16 is a non-IEEE variant with FP32-compatible exponent.

## Key Contributions
- Universal bit layout: [sign | exponent | mantissa] with implicit leading 1.
- Subnormal numbers for graceful underflow (no sudden gap at zero).
- ±0, ±∞, signaling and quiet NaN as first-class values.
- Five rounding modes with RNE as default (see [[round-to-nearest-even]]).
- Correctly-rounded basic ops (+, −, ×, ÷, √) and required exception behaviour.

## Key Figures/Tables to Study
- **binary32 bit layout** [1 sign | 8 exp | 23 mantissa, bias 127] — the canonical figure.
- **Subnormal-to-normal transition diagram** at exponent boundary — explains the "smallest positive normal" vs "smallest positive subnormal" gap.

## Technical Details

### binary32 (FP32) bit layout
| Field | Bits | Width |
|-------|------|-------|
| sign s | 31 | 1 |
| exponent e | 30..23 | 8 |
| mantissa m | 22..0 | 23 |

- **Exponent bias:** 127. Stored e ∈ [0, 255]; unbiased E = e − 127.
- **Normal:** 1 ≤ e ≤ 254 → value = (−1)^s · (1 + m/2^23) · 2^{e−127}
- **Subnormal:** e = 0, m ≠ 0 → value = (−1)^s · (0 + m/2^23) · 2^{−126}
- **Zero:** e = 0, m = 0 → ±0
- **Infinity:** e = 255, m = 0 → ±∞
- **NaN:** e = 255, m ≠ 0 → quiet NaN (MSB of m = 1) or signaling NaN (MSB = 0)

### binary32 key constants
- Smallest positive normal: 2^{−126} ≈ 1.18e−38
- Smallest positive subnormal: 2^{−149} ≈ 1.40e−45
- Largest finite: (2 − 2^{−23}) · 2^{127} ≈ 3.40e+38
- Machine epsilon ε = 2^{−23} ≈ 1.19e−7

### binary16 (FP16) bit layout
| Field | Bits | Width |
|-------|------|-------|
| sign s | 15 | 1 |
| exponent e | 14..10 | 5 |
| mantissa m | 9..0 | 10 |

- **Exponent bias:** 15. Unbiased E = e − 15.
- Normal: 1 ≤ e ≤ 30 → (−1)^s · (1 + m/1024) · 2^{e−15}
- Subnormal: e = 0, m ≠ 0 → (−1)^s · (m/1024) · 2^{−14}
- Largest finite: (2 − 2^{−10}) · 2^{15} = 65504
- Smallest positive normal: 2^{−14} ≈ 6.10e−5
- Smallest positive subnormal: 2^{−24} ≈ 5.96e−8
- Machine epsilon ε = 2^{−10} ≈ 9.77e−4

### Why exponent + mantissa = piecewise-uniform on log axis
Between consecutive exponent boundaries [2^E, 2^{E+1}), the 2^M mantissa values divide the interval uniformly with step 2^{E−M}. So FP is a discrete log compander: 6 dB per exponent step, constant *relative* error 2^{−M} within each bin. This connects FP directly to [[companding-mu-law]].

### Rounding modes
1. **roundTiesToEven** (RNE) — default; bias-zero ([[round-to-nearest-even]]).
2. **roundTiesToAway** — symmetric magnitude; not IEEE-default.
3. **roundTowardPositive** (+∞), **roundTowardNegative** (−∞), **roundTowardZero** (truncate).

### Special-value arithmetic
- ±∞ + (∓∞) → NaN; ±∞ · 0 → NaN; 0/0 → NaN; √(−x) → NaN.
- ±∞ propagates through most ops; NaN is sticky and signals (or quietly propagates).
- Signaling NaN raises invalid-op exception; quiet NaN does not.
- −0 == +0 numerically but distinguishable by 1/x → ±∞.

### Subnormal performance
Subnormal results are produced when an operation underflows below 2^{1−bias}. On many CPUs/GPUs subnormals trigger microcode slow-paths (10–100× slowdown). FTZ/DAZ ("flush-to-zero" / "denormals-are-zero") modes disable subnormals; deep-learning kernels typically run FTZ for speed, sacrificing the smooth underflow.

### Operational use in LLM quant
- FP32: master weights, optimizer state, loss accumulator.
- FP16: ubiquitous activations/weights pre-2020; needs loss scaling because of tiny normal range.
- BF16 (see [[bf16]]): same exponent as FP32, no loss scaling needed → modern training default.
- FP8/FP6/FP4 (see [[fp8-e4m3]] etc.): tiny exponent ranges; require per-tensor or per-block scaling to fit.

## Connections
- [[bf16]] — same exponent layout as FP32, narrower mantissa; no loss scaling.
- [[fp16]] — IEEE binary16; the loss-scaling era.
- [[fp8-e4m3]] / [[fp8-e5m2]] — sub-IEEE FP8 specs.
- [[fp6]] / [[fp4-e2m1]] — sub-FP8 floats; built on the IEEE template.
- [[round-to-nearest-even]] — IEEE-754 default rounding mode.
- [[mx-formats]] — block-floating extension: shared block exponent + IEEE-style element.
