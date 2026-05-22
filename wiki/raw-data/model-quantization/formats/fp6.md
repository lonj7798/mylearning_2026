<!-- scope: FP6 intermediate-precision floats (E3M2 / E2M3); academic and OCP proposals
     deps: [[fp8-e4m3]], [[ieee-754]]
     see-also: [[fp4-e2m1]], [[mx-formats]], [[llm-fp4]]
-->

# FP6 (E3M2 / E2M3) Intermediate-Precision Floats
- **Core Insight:** A 6-bit float bridges FP8 and FP4 with two natural splits — E3M2 (3 exp + 2 mantissa, range ~10^{2.4}) for activations, E2M3 (2 exp + 3 mantissa, range ~10^{1.2} with finer precision) for weights — providing a "soft landing" between FP8's range and FP4's coarseness; deployed via the OCP MX spec as MXFP6.
- **Guideline:** Treat FP6 as the *deployment* sweet spot when FP4 is too coarse for activations but FP8 is too wasteful — particularly for serving weight-only-quantized LLMs where activations stay BF16 but you want weights below 8 bits in a float-shaped code.
- **Authors:** Various; OCP Microscaling Working Group (E3M2/E2M3 standardized 2023); also explored in Liu 2023 LLM-FP4 lineage and Microsoft's MXFP6 papers
- **Year:** 2023 (OCP MX spec includes MXFP6)
- **URL:** https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf ; https://arxiv.org/abs/2310.16836 (FP6-LLM)
- **Relevant topics:** FP6, MXFP6, intermediate-precision, OCP MX

## Abstract
FP6 packs 6 bits into a floating-point representation with one of two standard splits: **E3M2** (1 sign + 3 exponent + 2 mantissa) provides ~10^{2.4} dynamic range and is the OCP-blessed default; **E2M3** (1 sign + 2 exponent + 3 mantissa) gives finer mantissa precision but only ~10^{1.2} dynamic range. Both forms are included in the OCP Microscaling spec as MXFP6 element formats (with an E8M0 block scale). Microsoft and academic groups (FP6-LLM, Liu 2023) demonstrated FP6 weight-only quantization for LLMs at close-to-FP16 quality with a 2.6× memory reduction over FP16.

## Key Contributions
- Two standardized FP6 splits (E3M2 and E2M3) covering the range-vs-precision tradeoff.
- Included as a first-class element format in OCP MX spec → MXFP6 with shared block scale.
- Microsoft's FP6-LLM kernel achieves near-FP16 quality with ~2.6× compression and matches AWQ INT4 throughput.
- Establishes that 6 bits is sufficient for both weight and (with per-block scale) activation quantization on most transformer layers.

## Key Figures/Tables to Study
- **E3M2 vs E2M3 bit layout side-by-side** — visualizes the range-vs-precision tradeoff at 6 bits.
- **Representable-value plot** on log axis: E3M2 reaches 28, E2M3 reaches 7.5 — but E2M3 has 2× more values per exponent bin.

## Technical Details

### E3M2 bit layout (FP6 default; MXFP6 element)
| Field | Bits |
|-------|------|
| sign s | 1 |
| exponent e | 3 |
| mantissa m | 2 |

- **Exponent bias:** 3. Unbiased E = e − 3.
- **Normal:** 1 ≤ e ≤ 7 → value = (−1)^s · (1 + m/4) · 2^{e−3}
- **Subnormal:** e = 0, m ≠ 0 → value = (−1)^s · (m/4) · 2^{−2}
- **Zero:** e = 0, m = 0 → ±0
- **OCP MX convention**: no ±∞, no NaN at element level (saturating arithmetic, NaN handled at block scale).
  - All 64 codes are finite; max-magnitude = (1 + 3/4) · 2^{7−3} = 1.75 · 16 = **28**.
- Smallest positive normal: 2^{−2} = 0.25
- Smallest positive subnormal: 2^{−4} = 0.0625
- Machine epsilon ε = 2^{−2} = 0.25

### E2M3 bit layout
| Field | Bits |
|-------|------|
| sign s | 1 |
| exponent e | 2 |
| mantissa m | 3 |

- **Exponent bias:** 1. Unbiased E = e − 1.
- **Normal:** 1 ≤ e ≤ 3 → value = (−1)^s · (1 + m/8) · 2^{e−1}
- **Subnormal:** e = 0, m ≠ 0 → value = (−1)^s · (m/8) · 2^{0}
- **Zero:** e = 0, m = 0 → ±0
- Max-magnitude = (1 + 7/8) · 2^{3−1} = 1.875 · 4 = **7.5**
- Smallest positive normal: 2^{0} = 1.0 (in the no-bias variant; tiny range)
- Machine epsilon ε = 2^{−3} = 0.125 (2× finer than E3M2)

### Comparison of FP6 splits
| | E3M2 | E2M3 |
|---|---|---|
| Exponent bits | 3 | 2 |
| Mantissa bits | 2 | 3 |
| Max finite (no scale) | 28 | 7.5 |
| Min positive normal | 0.25 | 1.0 |
| Dynamic range | ~112× | ~7.5× |
| Mantissa ε | 0.25 | 0.125 |
| Use | activations, KV | weights |

### MXFP6 block format
MX wraps either E3M2 or E2M3 elements with a shared E8M0 (8-bit power-of-2 exponent only, no mantissa) block scale per 32 elements. Effective bits/element = 6 + 8/32 = 6.25. See [[mx-formats]].

### FP6-LLM (Microsoft, 2023)
- Pure weight-only FP6 quantization (E3M2 elements, per-group scale).
- 2.6× compression over FP16; matches AWQ INT4 throughput in serving.
- Achieves ~99% of FP16 perplexity on LLaMA-70B.
- Crucial finding: FP6 is the *first* sub-8-bit format where weight quantization needs no rotation / smoothing / GPTQ-style Hessian update — naive per-group FP6 cast works.

### Hardware support
- **NVIDIA Blackwell (B100/B200, 2024–2025)**: native MXFP6 tensor-core support (E3M2 + E8M0 block scale).
- **AMD MI355X (2025)**: announced MXFP6 support.
- Pre-Blackwell: FP6 is software-only (pack 6-bit codes 4-into-3-byte groups; dequantize to BF16 for tensor-core matmul).

### Pack layout (software, pre-Blackwell)
Common pack: 4 × FP6 = 24 bits in 3 bytes:
```
byte0:  e3 e3 e3  e3 e3 e3  e3 e3      (low 6 bits of element 0 | high 2 bits of element 1)
byte1:  e3 e3 e3  e3 e3 e3  e3 e3
byte2:  e3 e3 e3  e3 e3 e3  e3 e3
```
Implementation varies; FP6-LLM uses a sliced load that fuses dequant + GEMM.

## Connections
- [[fp8-e4m3]] / [[fp8-e5m2]] — FP6 sits between FP8 and FP4 on the bit-vs-quality axis.
- [[fp4-e2m1]] — the next-lower step; FP4 needs rotation/scaling, FP6 generally doesn't.
- [[mx-formats]] — MXFP6 is the standardized FP6 deployment.
- [[ieee-754]] — FP6 inherits IEEE-style sign/exp/mantissa structure; no special encodings.
- [[llm-fp4]] — Liu 2023; sister work on sub-8-bit floats for LLMs.
