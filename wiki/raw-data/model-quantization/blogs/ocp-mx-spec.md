<!-- scope: Open Compute Project Microscaling (MX) format specification
     deps: [[mx-formats]]
     see-also: [[nvfp4]], [[microscaling-formats]], [[fp8-formats-paper]]
-->

# OCP Microscaling (MX) Format Specification
- **Core Insight:** The OCP MX spec ratified a single hardware-agnostic shared-block-scale format family (MXFP8 / MXFP6 / MXFP4 / MXINT8) so AMD, Intel, NVIDIA, Qualcomm, Microsoft, Arm, and Meta can interoperate on sub-8-bit AI compute.
- **Guideline:** Treat MX as the open / portable analogue of vendor-specific block-scaled formats (NVFP4, AMD BF8); pick MX when interoperability across cloud accelerators matters.
- **Authors:** Open Compute Project — Microscaling Formats subgroup (Microsoft + AMD + Intel + Meta + NVIDIA + Arm + Qualcomm)
- **Year:** 2023 (v1.0)
- **URL:** https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- **Relevant topics:** MX format family, block scaling, MXFP4, MXFP6, MXFP8, MXINT8, UE8M0

## Summary
The Open Compute Project Microscaling Formats (MX) v1.0 specification, published in September 2023, defines a family of block-scaled tensor representations targeted at deep-learning inference and training. Each MX type pairs a per-block scale (an 8-bit unsigned exponent, UE8M0) with a 32-element block of low-bit element values; the element type determines the format name. MX is the result of multi-vendor collaboration through OCP and is designed so that compliant hardware from different vendors produces bit-identical results. The spec is the open analogue of NVIDIA's NVFP4 (which uses a 16-element block + E4M3 scale instead) and is the format underpinning Microsoft's microscaling research line (Rouhani 2023). The 2024 follow-up OCP MX v1.1 added clarifications around saturation behaviour and special-value encoding.

## Key Points
- Block size: 32 elements.
- Block scale type: UE8M0 (8 bits unsigned, encodes 2^k for integer k ∈ [-127, 127]; 0xFF reserved for NaN).
- Element types defined: MXFP8 (E4M3, E5M2), MXFP6 (E3M2, E2M3), MXFP4 (E2M1), MXINT8.
- Effective overhead: 8 bits of scale per 32-element block = 0.25 bits/element.
- Industry adoption: AMD MI355 (planned), Intel Gaudi 3, NVIDIA Blackwell, Microsoft Cobalt.

## Technical Details

### Block scale encoding (UE8M0)
- 8 bits, unsigned, interpreted as a biased exponent.
- Value `e` ∈ [0, 254] → scale `2^(e − 127)`.
- `e = 255` (0xFF) → NaN (sentinel; entire block treated as NaN).
- `e = 0` → smallest scale 2^-127.

### Element formats
| Name | Bits | Layout | Range |
|------|------|--------|-------|
| MXFP8-E4M3 | 8 | 1s + 4e + 3m | ±448, no inf |
| MXFP8-E5M2 | 8 | 1s + 5e + 2m | ±57,344, IEEE inf/NaN |
| MXFP6-E3M2 | 6 | 1s + 3e + 2m | ±28 |
| MXFP6-E2M3 | 6 | 1s + 2e + 3m | ±7.5 |
| MXFP4-E2M1 | 4 | 1s + 2e + 1m | ±6 |
| MXINT8 | 8 | INT8 signed | ±127 |

### Encoding rule
For a 32-element block of FP32 values `x[0..31]`:
1. `amax = max(|x|)`
2. `e = floor(log2(amax / element_max))` — choose the largest power-of-two scale that keeps element_max within range.
3. Block scale = UE8M0(e + 127).
4. Each element = `round(x[i] / 2^e)` cast to the element format.

### Cross-format mixing
- A linear layer can use MXFP4 weights × MXFP8 activations.
- Accumulation always in FP32 (per the spec); the accumulator format is implementation-defined but must satisfy specified error bounds.

### Implementations
- Microsoft Research: software reference implementation in `microxcaling` (PyPI).
- NVIDIA Blackwell: hardware MXFP4 and MXFP8 path.
- AMD MI355 (announced): hardware MXFP4 / MXFP6 / MXFP8.
- Intel Gaudi 3: hardware MX path.

## Connections
- [[mx-formats]] — internal format-spec page mirroring the OCP definitions.
- [[microscaling-formats]] — Rouhani 2023 paper that motivated the spec.
- [[nvfp4]] — NVIDIA-proprietary cousin (16-element block, E4M3 scale).
- [[nvidia-blackwell-fp4]] — Blackwell tensor cores supporting both NVFP4 and MXFP4.
- [[fp8-formats-paper]] — earlier industry consortium effort for FP8 alone.
