<!-- scope: NVIDIA Blackwell B100/B200 5th-gen tensor cores and NVFP4
     deps: [[nvfp4]], [[mx-formats]]
     see-also: [[nvidia-h100-fp8]], [[nvidia-quantization]], [[ocp-mx-spec]]
-->

# NVIDIA Blackwell NVFP4 — 5th-Gen Tensor Cores
- **Core Insight:** Blackwell's 5th-generation tensor cores natively execute block-scaled FP4 (NVFP4 and MXFP4), pushing per-Watt throughput another 2× over Hopper FP8 by halving the operand width while keeping a small FP8 block scale to preserve dynamic range.
- **Guideline:** Treat NVFP4 as Blackwell's production 4-bit format — 16-element blocks with E4M3 block scale and per-tensor FP32 scale; use MXFP4 when interoperability with the OCP MX ecosystem matters.
- **Authors:** NVIDIA Blackwell architecture team (whitepaper + GTC 2024 announcements)
- **Year:** 2024
- **URL:** https://resources.nvidia.com/en-us-blackwell-architecture
- **Relevant topics:** NVFP4, MXFP4, Blackwell, 5th-gen tensor cores, block scaling

## Summary
The Blackwell B100/B200 GPUs introduce 5th-generation tensor cores supporting FP4 in two block-scaled variants: NVFP4 (NVIDIA-proprietary) and MXFP4 (OCP standard). Both encode each weight or activation as a 4-bit E2M1 float, but apply a shared block-level scaling factor that recovers most of the dynamic range lost when shrinking from FP8 to FP4. NVFP4 uses a 16-element block with an FP8 (E4M3) block scale and an additional per-tensor FP32 scale; MXFP4 uses a 32-element block with a UE8M0 (8-bit power-of-two) block scale. Peak dense FP4 throughput on B200 is ~9 PFLOPS (≈2× the Hopper FP8 number), and a single Blackwell die also keeps native FP8 / FP6 / INT8 paths. The 2nd-generation Transformer Engine adds automatic recipe selection between FP4, FP6, and FP8.

## Key Points
- Two FP4 formats: NVFP4 (16-element block, E4M3 scale) and MXFP4 (32-element block, UE8M0 scale).
- Element format E2M1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} representable values.
- Peak FP4 dense throughput on B200 ≈ 9 PFLOPS (2× over Hopper FP8).
- 2nd-gen Transformer Engine selects FP4/FP6/FP8 per-tensor automatically.
- NVFP4 reportedly recovers BF16-level accuracy on inference for Llama-scale models.

## Technical Details

### NVFP4 layout
- Element: 4-bit E2M1 (1 sign + 2 exponent + 1 mantissa).
- Block size: 16 elements.
- Block scale: FP8 E4M3 (8 bits per 16 elements → 0.5 bits/element scale overhead).
- Tensor scale: FP32 per-tensor amax-derived scale.
- Effective bits per weight ≈ 4 + 0.5 = 4.5.

### MXFP4 layout (OCP standard)
- Element: 4-bit E2M1 (same as NVFP4).
- Block size: 32 elements.
- Block scale: UE8M0 (8-bit unsigned, power-of-two scale, range 2^-127..2^127).
- No per-tensor scale required (block scale alone is sufficient).
- Effective bits per weight ≈ 4 + 0.25 = 4.25.

### B200 throughput (per-die, dense)
| Format | TFLOPS |
|--------|--------|
| FP4 | ~9,000 |
| FP6 | ~4,500 |
| FP8 | ~4,500 |
| FP16/BF16 | ~2,250 |
| TF32 | ~1,125 |
| FP32 | ~600 |

### Tensor-core instructions
- New `wgmma`-class FP4 variants with block-scale operands.
- Block-scale operand is loaded alongside the element tile via TMA.
- Accumulation still in FP32.

### When to use which FP4
- NVFP4: lower block-scale overhead per scale + finer granularity → better accuracy at the same average bit rate.
- MXFP4: portable across vendors that ratified the OCP MX spec (AMD, Intel, Qualcomm have signed on).

## Connections
- [[nvfp4]] — format spec for the NVIDIA-proprietary 4-bit format.
- [[mx-formats]] — OCP Microscaling parent of MXFP4.
- [[nvidia-h100-fp8]] — predecessor format that NVFP4 replaces in inference.
- [[transformer-engine-blog]] — software path for FP4/FP6/FP8 recipe selection.
- [[ocp-mx-spec]] — industry standard MXFP4 conforms to.
