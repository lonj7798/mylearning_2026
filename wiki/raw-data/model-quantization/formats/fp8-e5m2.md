<!-- scope: OFP8 / NVIDIA H100 FP8 E5M2 — 5 exp + 2 mantissa; for gradients (wider range)
     deps: [[ieee-754]], [[fp16]]
     see-also: [[fp8-e4m3]], [[fp8-formats-paper]], [[fp8-lm]]
-->

# FP8 E5M2 (OFP8 / Gradient-Side FP8)
- **Core Insight:** With the same 5-bit exponent as IEEE FP16 but only 2 mantissa bits, E5M2 gives the full FP16 dynamic range (6e−5 to 5.7e4) in 8 bits — enough headroom for backward-pass gradients (which span many orders of magnitude) at the cost of coarse 25% relative precision that gradients can tolerate.
- **Guideline:** Use E5M2 for backward-pass tensors (gradient of weight / gradient of activation / optimizer-state intermediates); pair with E4M3 forward. The 25% precision is acceptable for gradients but unacceptable for activations, motivating the two-format split.
- **Authors:** Paulius Micikevicius et al. (joint NVIDIA + Arm + Intel, 2022); OCP OFP8 (2023)
- **Year:** 2022 (paper); 2023 (OFP8 standard)
- **URL:** https://arxiv.org/abs/2209.05433 ; https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
- **Relevant topics:** FP8, E5M2, gradient precision, IEEE-compatible FP8, mixed-precision training

## Abstract
E5M2 is the second half of the FP8 dual-format proposal. Unlike E4M3, E5M2 follows IEEE-754 conventions: 5-bit biased exponent, 2-bit mantissa, all-ones exponent reserved for ±∞ and NaN. This makes E5M2 essentially "FP16 with 8 fewer mantissa bits" — same exponent range, same dynamic range, but only 4 (= 2^2) mantissa values per exponent bin. The format is targeted at backward-pass gradients, where range matters more than precision, while E4M3 is targeted at forward-pass activations/weights, where precision matters more than range.

## Key Contributions
- 8-bit float with FP16-class dynamic range (~10^{9}).
- IEEE-compliant special-value handling (±∞, NaN).
- Pairs with E4M3 in NVIDIA Transformer Engine's two-format FP8 scheme.
- Standardized as OCP OFP8 (2023) alongside E4M3.

## Key Figures/Tables to Study
- **Gradient histogram pre- and post-quantization** (Micikevicius Fig. 5): pre-quant tails span 10^7 dynamic range → E5M2's exponent range required; E4M3 would clip.
- **Representable value map**: only ~120 finite values per sign — fewer than E4M3 because more exponent encodings, but spread over a wider range.

## Technical Details

### Bit layout
| Field | Bits | Width |
|-------|------|-------|
| sign s | 7 | 1 |
| exponent e | 6..2 | 5 |
| mantissa m | 1..0 | 2 |

- **Exponent bias:** 15 (same as FP16). Unbiased E = e − 15.
- **Normal:** 1 ≤ e ≤ 30 → value = (−1)^s · (1 + m/4) · 2^{e−15}
- **Subnormal:** e = 0, m ≠ 0 → value = (−1)^s · (m/4) · 2^{−14}
- **Zero:** e = 0, m = 0 → ±0
- **±∞:** e = 31, m = 0 (IEEE-compliant)
- **NaN:** e = 31, m ≠ 0 (IEEE-compliant)

### Key constants
- Smallest positive normal: 2^{−14} ≈ 6.10e−5 (same as FP16)
- Smallest positive subnormal: 2^{−16} ≈ 1.53e−5
- Largest finite: 1.75 · 2^{15} = **57344** (= 0.1110.11 = (1+3/4)·2^{30−15})
- Machine epsilon ε = 2^{−2} = 0.25
- Total finite values per sign: ~120

### Comparison with E4M3
| | E4M3 | E5M2 |
|---|---|---|
| Exponent bits | 4 | 5 |
| Mantissa bits | 3 | 2 |
| Bias | 7 | 15 |
| Max finite | 448 | 57344 |
| Min positive normal | 0.0156 | 6.1e−5 |
| ε | 0.125 (12.5%) | 0.25 (25%) |
| Special | NaN only | ±∞, NaN (IEEE) |
| Use | fwd weights/act | bwd gradients |

### Why gradients need E5M2
- Gradient magnitudes span ~7 orders of magnitude in a deep network (10^{−6} to 10^{+1}).
- E4M3's range (10^{−2} to 10^{2.5}) requires careful per-tensor scaling to fit gradients; E5M2's range (10^{−5} to 10^{4.7}) typically fits without scaling or with a single global scale.
- Gradient *direction* matters more than gradient *magnitude precision* (SGD/Adam noise dominates), so 25% precision is acceptable.

### Comparison with FP16
- E5M2 has exactly the same exponent range as FP16 (bias 15, max e = 30).
- E5M2 has 8× fewer values within each exponent bin (2² vs 2¹⁰).
- E5M2 is effectively "FP16 with low-mantissa bits stochastically rounded down to 2 bits."
- BF16 → FP16 cast loses range; BF16 → E5M2 loses both range and precision; FP16 → E5M2 loses only precision (cheap rounding).

### Use in mixed-precision training
Standard recipe (e.g. Transformer Engine):
- **Forward**: weights / activations cast to E4M3, accumulator FP32.
- **Backward**: input/output gradients cast to E5M2, accumulator FP32.
- **Master weights** stay in FP32 or BF16.
- **Optimizer state** stays in FP32.
- Per-tensor scale tracked online; updated each step from amax statistics.

### Rounding mode
RNE by default; SR available on Hopper / Blackwell. SR particularly important for gradient cast in long-running training to preserve update expectation (see [[stochastic-rounding]]).

### Hardware
- Same H100 / B100 tensor cores as E4M3 → identical throughput (~1979 TFLOPS on H100 SXM).
- AMD MI300, Intel Gaudi 3, NVIDIA Blackwell: native E5M2 support.

## Connections
- [[ieee-754]] — E5M2 is IEEE-compliant (unlike E4M3); preserves ±∞ / NaN semantics.
- [[fp16]] — same exponent as FP16; effectively a low-mantissa FP16.
- [[fp8-e4m3]] — partner format for forward-pass tensors.
- [[fp8-formats-paper]] — Micikevicius 2022, original proposal.
- [[stochastic-rounding]] — important for gradient cast.
- [[mx-formats]] — MXFP8 can be implemented with E5M2 elements for gradient blocks.
