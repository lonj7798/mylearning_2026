<!-- scope: FP4 E2M1 — 4-bit float with 1 sign + 2 exp + 1 mantissa; Blackwell-era native format
     deps: [[ieee-754]], [[fp6]]
     see-also: [[nvfp4]], [[mxfp-training]], [[llm-fp4]]
-->

# FP4 E2M1 (OCP MX Element / NVIDIA Blackwell Native)
- **Core Insight:** With only 4 bits — 1 sign, 2 exponent, 1 mantissa — FP4 E2M1 has just 16 reconstruction levels (8 positive, including ±0), arranged log-spaced rather than uniform; this is the smallest "useful" float for deep learning and requires per-block scaling (MXFP4 or NVFP4) to be deployable.
- **Guideline:** Never use FP4 with a single per-tensor scale; always pair with a fine-grained per-block scale (MXFP4 = E8M0 scale per 32 elements; NVFP4 = FP8 scale per 16 elements + FP32 per-tensor scale) — naive per-tensor FP4 loses ~3+ perplexity points on any LLM.
- **Authors:** OCP Microscaling Working Group (2023); NVIDIA Blackwell team (2024)
- **Year:** 2023 (OCP MX); 2024 (Blackwell hardware)
- **URL:** https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf ; https://developer.nvidia.com/blog/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing/
- **Relevant topics:** FP4, E2M1, MXFP4, NVFP4, Blackwell, sub-8-bit floats

## Abstract
FP4 in the E2M1 layout — 1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit — gives 16 representable codes, of which 14 are distinct finite values (one each of ±0 collapse to one). The format is OCP-standardized as the element format for MXFP4 (with an E8M0 block scale per 32 elements) and is the basis for NVIDIA NVFP4 (with an FP8 block scale per 16 elements + an FP32 per-tensor scale). Hardware support arrived with NVIDIA Blackwell (B100/B200, 2024) giving ~9 PFLOPS of FP4 throughput per B100 GPU. FP4 is currently the bleeding-edge target for both inference and pretraining; sub-FP4 ([[bitnet-w158]]) requires non-FP integer/ternary formats.

## Key Contributions
- Smallest practical floating-point format; only 16 distinct codes.
- Standardized as OCP MXFP4 element (2023).
- Native Blackwell hardware support → ~9 PFLOPS FP4 per GPU (vs ~4.5 PFLOPS FP8).
- Enables 4-bit *training* (with NVFP4 + per-block scale, see [[nvfp4-training]]).
- Cuts memory and bandwidth ~2× over FP8 / ~4× over BF16.

## Key Figures/Tables to Study
- **Full 16-code table** (below) — small enough to memorize; useful for debugging.
- **MXFP4 vs NVFP4 block-scale comparison** — shows why finer-grained scaling matters at 4 bits.

## Technical Details

### Bit layout
| Field | Bits |
|-------|------|
| sign s | 1 |
| exponent e | 2 |
| mantissa m | 1 |

- **Exponent bias:** 1. Unbiased E = e − 1.
- **Normal:** 1 ≤ e ≤ 3 → value = (−1)^s · (1 + m/2) · 2^{e−1}
- **Subnormal:** e = 0, m = 1 → value = (−1)^s · (1/2) · 2^{0} = ±0.5
- **Zero:** e = 0, m = 0 → ±0
- **No ±∞, no NaN** (saturating arithmetic at element level; NaN handled by block scale).

### Full 16-code table (E2M1)
| s | e | m | Value |
|---|---|---|---|
| 0 | 00 | 0 | +0 |
| 0 | 00 | 1 | +0.5 (subnormal) |
| 0 | 01 | 0 | +1 |
| 0 | 01 | 1 | +1.5 |
| 0 | 10 | 0 | +2 |
| 0 | 10 | 1 | +3 |
| 0 | 11 | 0 | +4 |
| 0 | 11 | 1 | +6 |
| 1 | ... | ... | negatives of the above |

Reconstruction values: **{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}** (15 distinct after ±0 collapse).

### Key constants
- Max finite: **6.0** (= 1.5 · 2^{3−1})
- Min positive normal: 1.0
- Min positive subnormal: 0.5
- Machine epsilon ε = 2^{−1} = 0.5 (50% relative precision!)
- Dynamic range (normal): 6× (= 2^{2.585})
- Including subnormal: 12×

### Why FP4 needs block scaling
Raw FP4 covers [−6, +6] with only 16 levels — totally inadequate for LLM weights (typical range [−1, +1]) or activations (typical range [−10, +10]). Two block-scaled deployments:

**MXFP4**: shared E8M0 (power-of-2-only) scale per 32 elements; 4 + 8/32 = 4.25 effective bits/element. See [[mx-formats]].

**NVFP4** (Blackwell): two-level scaling — FP8 E4M3 scale per 16 elements + FP32 tensor scale; 4 + 8/16 + (negligible) = 4.5 effective bits/element. See [[nvfp4]].

### Comparison with INT4
| | FP4 E2M1 | INT4 |
|---|---|---|
| Codes | 16 (log-spaced) | 16 (uniform) |
| Max | 6 | 7 |
| Step | exponentially varying (0.5–2.0) | uniform (1.0) |
| Dynamic range / step | 12× | 14× |
| Best for | log-magnitude weights/activations | uniform-bounded data |

FP4 is better for sources with wide magnitude variation (LLM weights post-normalization, attention scores). INT4 is better for sources already in a bounded range (post-RMSNorm activations, normalized weights).

### Hardware
- **NVIDIA Blackwell B100/B200**: ~9 PFLOPS FP4 (dense), ~18 PFLOPS sparse.
- **AMD MI355X** (2025): announced FP4 support.
- Pre-Blackwell: software FP4 via dequant-to-BF16 + BF16 tensor-core matmul.

### Use cases
- **Inference**: weight-only FP4 (W4A16) is plug-and-play with negligible quality loss on Llama-class models.
- **W4A4 inference**: requires rotation ([[quarot]]) or smoothing ([[smoothquant]]) for activations.
- **FP4 training**: NVFP4 native pretraining demonstrated (see [[nvfp4-training]]); requires stochastic rounding, FP32 master weights.
- **FP4 KV-cache**: viable with per-block scale and per-channel K / per-token V partitioning ([[kvquant]]-style).

### Pack layout
2 × FP4 per byte (high nibble = element 0, low nibble = element 1). Pack/unpack is a single nibble shift.

## Connections
- [[ieee-754]] — E2M1 inherits sign/exp/mantissa structure (minus special values).
- [[fp6]] — the 2-bit-wider sibling; intermediate stop on the FP scaling ladder.
- [[mx-formats]] — MXFP4 = E2M1 element + E8M0 block scale per 32.
- [[nvfp4]] — NVIDIA's production FP4: E2M1 + FP8 block scale + FP32 tensor scale.
- [[llm-fp4]] — Liu 2023; early sub-8-bit FP studies.
- [[mxfp-training]] / [[nvfp4-training]] — production FP4 training recipes.
- [[int4]] — uniform 4-bit alternative.
