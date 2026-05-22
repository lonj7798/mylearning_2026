---
chapter: ch-02
course: model-quantization
phase: read
excerpt_of: "IEEE-754 Floating-Point Standard (binary32 / binary16)"
source_url: https://ieeexplore.ieee.org/document/8766229
created_at: "2026-05-21"
raw_data_source: [[raw-data/ieee-754]]
---

# Excerpt: IEEE-754 — the substrate every modern float inherits

**Standard:** IEEE-754 (1985 original, 2008 revision, 2019 latest); William Kahan led the original committee.
**Year:** 1985 / 2008 / 2019.
**URL:** see source_url; Kahan lecture notes at https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF

---

## The one-box value formula

For an IEEE-754 number with sign `s`, biased exponent `e`, mantissa bits `m` (width `M`), exponent bias `bias`:

```math
\text{value} =
\begin{cases}
(-1)^s \cdot (1 + m / 2^M) \cdot 2^{e - \text{bias}} & 1 \le e \le 2^E - 2 \quad\text{(normal)} \\
(-1)^s \cdot (0 + m / 2^M) \cdot 2^{1 - \text{bias}} & e = 0, m \ne 0 \quad\text{(subnormal)} \\
\pm 0 & e = 0, m = 0 \\
\pm \infty & e = 2^E - 1, m = 0 \\
\text{NaN} & e = 2^E - 1, m \ne 0
\end{cases}
```

The substrate every subsequent low-precision float (BF16, FP16, FP8 E4M3 / E5M2, FP6, FP4 E2M1, MX element formats) inherits. The only departures: BF16 has non-standard subnormal semantics, and FP8 E4M3 / FP4 E2M1 sacrifice `±∞` to gain finite codes.

---

## binary32 (FP32) and binary16 (FP16) — key constants

| Format | Sign | Exp | Mantissa | Bias | Max finite | Min positive normal | Min positive subnormal | ε |
|--------|------|-----|---------|------|------------|---------------------|------------------------|-----|
| FP32 (binary32) | 1 | 8 | 23 | 127 | `(2 − 2^{−23})·2^{127}` ≈ 3.40e+38 | `2^{−126}` ≈ 1.18e−38 | `2^{−149}` ≈ 1.40e−45 | `2^{−23}` ≈ 1.19e−7 |
| FP16 (binary16) | 1 | 5 | 10 | 15  | `(2 − 2^{−10})·2^{15}` = 65504    | `2^{−14}` ≈ 6.10e−5 | `2^{−24}` ≈ 5.96e−8 | `2^{−10}` ≈ 9.77e−4 |

---

## Floating-point as piecewise-uniform log quantization

Between consecutive exponent boundaries `[2^E, 2^{E+1})`, the `2^M` mantissa values divide the interval uniformly with step `2^{E−M}`. So FP is a **discrete log compander**: 6 dB per exponent step, constant *relative* error `2^{−M}` within each bin. This connects FP directly to [[companding-mu-law]] and to the Gish-Pierce optimal compander for log-distributed sources.

---

## Rounding modes

1. **roundTiesToEven (RNE)** — IEEE default; zero bias on symmetric distributions.
2. **roundTiesToAway** — symmetric magnitude; not default.
3. **roundTowardPositive (+∞)**, **roundTowardNegative (−∞)**, **roundTowardZero (truncate)**.

Stochastic rounding ([[excerpts/stochastic-rounding]]) is *not* in the standard; modern accelerators add it as a non-IEEE instruction-level option.

---

## Special-value arithmetic

```
±∞ + (∓∞) → NaN ;     ±∞ · 0 → NaN ;     0/0 → NaN ;     √(−x) → NaN
±∞ propagates through most ops
NaN is sticky and signals (sNaN) or quietly propagates (qNaN)
−0 == +0 numerically but distinguishable by  1/x → ±∞
```

---

## Subnormal performance pitfall

Subnormal operations trigger microcode slow-paths on most CPUs/GPUs (10–100× slowdown). Production training enables `FTZ` (flush-to-zero) and `DAZ` (denormals-are-zero), sacrificing gradual underflow for throughput. This is why BF16 is sometimes documented as "no subnormal guarantees" — vendors implement FTZ semantics by default.

---

## Operational use in LLM quantization

- **FP32:** master weights, optimizer state, loss accumulator.
- **FP16:** ubiquitous activations/weights pre-2020; needs loss scaling.
- **BF16:** modern training default ([[excerpts/bf16]]).
- **FP8 / FP6 / FP4:** sub-IEEE specifications; tiny exponent ranges; require per-tensor or per-block scaling to be deployable.

---

## Connections

- [[excerpts/bf16]] — same exponent layout as FP32, narrower mantissa.
- [[excerpts/fp8-e4m3]] / [[excerpts/fp8-e5m2]] — sub-IEEE FP8 specs.
- [[excerpts/fp4-e2m1]] — sub-FP8 float; built on the IEEE template.
- [[mx-formats]] — block-floating extension: shared block exponent + IEEE-style element.
- [[ch-02]] — parent synthesis.
