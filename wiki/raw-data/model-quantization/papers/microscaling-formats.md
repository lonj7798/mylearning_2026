<!-- scope: Microscaling (MX) formats — shared block-exponent + low-bit element types, the OCP MX spec basis
     deps: [[fp8-formats-paper]]
     see-also: [[llm-fp4]], [[mxfp-training]]
-->

# Microscaling Data Formats for Deep Learning
- **Core Insight:** A shared 8-bit power-of-two exponent per *block* of 32 elements combined with narrow (4–8 bit) element representations gives near-FP32 dynamic range at a fraction of the bit budget — and the block size of 32 is small enough to fit on tensor-core hardware paths but large enough that the scale overhead amortises to <1 bit/element.
- **Guideline:** When designing or consuming a sub-8-bit numerical format, default to the OCP MX family: 32-element block with E8M0 shared scale and per-element FP4/FP6/FP8 or INT8 — supports both inference and training, matches FP32 accuracy on generative LLMs with sub-8-bit weights/activations/gradients.
- **Authors:** Bita Darvish Rouhani et al. (32 co-authors across Microsoft, NVIDIA, Intel, AMD, ARM, Qualcomm)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.10537
- **Relevant topics:** microscaling formats, block-scaled FP, MX spec, OCP, sub-8-bit training

## Abstract
Microscaling (MX) data formats combine a per-block scaling factor with narrow floating-point or integer element types to reduce computation cost while preserving accuracy. The MX family includes MXFP8 / MXFP6 / MXFP4 / MXINT8, each pairing a small (4–8 bit) per-element value with a shared 8-bit power-of-two exponent (E8M0) over a 32-element block. The paper presents the first successful generative LLM training with sub-8-bit weights, activations, *and* gradients at FP32-parity accuracy on user-facing benchmarks. Subsequently codified by the Open Compute Project (OCP) MX spec.

## Key Contributions
- Defines the MX format family: block size = 32, shared 8-bit exponent (E8M0) + per-element format.
- Element formats:
  - **E2M1** (4-bit FP, 2 exp + 1 mantissa): MXFP4.
  - **E2M3** (6-bit FP, 2 exp + 3 mantissa): MXFP6.
  - **E3M2** (6-bit FP, 3 exp + 2 mantissa): alternate MXFP6.
  - **E4M3** (8-bit FP): MXFP8.
  - **INT8**: MXINT8.
- Demonstrates that all four — weights, activations, and gradients — can simultaneously use MX sub-8-bit during generative LLM training with no accuracy loss on downstream evaluation.
- Establishes the bit-budget arithmetic: 32-element block × 8-bit scale = 0.25 bits/element overhead.

## Key Figures/Tables to Study
- **Table 1:** The MX format catalogue — block size, scale bits, element bits, dynamic range.
- **Figure 4:** Generative LLM training curves with MXFP8/MXFP6/MXFP4 — match FP32 within noise.
- **Section 5 (OCP spec):** the formalised MX block layout that became the production standard on Blackwell.

## Technical Details

### MX block structure
A block of 32 elements `(v_1, ..., v_32)` is stored as:
- `X` (shared scale, 8 bits): an E8M0 exponent representing 2^X.
- `(d_1, ..., d_32)` (per-element value, k bits each): each d_i is in the chosen element format (FP4/FP6/FP8/INT8).

Decoded value: `v_i = 2^X · d_i`.

Shared scale is power-of-two only (no mantissa), so dequantization is a shift, not a multiply — cheap in hardware.

### Element formats
- **E2M1 (FP4):** 1 sign + 2 exp + 1 mantissa. Representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. Dynamic range ~3 bits.
- **E2M3 (FP6):** 1 sign + 2 exp + 3 mantissa. 16 distinct positive values, dynamic range ~3 bits, finer resolution than E2M1.
- **E3M2 (FP6 alt):** 1 sign + 3 exp + 2 mantissa. Wider range, coarser resolution.
- **E4M3 (FP8):** 1 sign + 4 exp + 3 mantissa. Same as OFP8 E4M3.
- **INT8:** twos-complement integer.

### Bit accounting
For 32-element MXFP4 block: 8-bit scale + 32 × 4-bit values = 136 bits / 32 elements = 4.25 bits/element.
For MXFP6: 8 + 32 × 6 = 200 / 32 = 6.25 bits/element.
For MXFP8: 8 + 32 × 8 = 264 / 32 = 8.25 bits/element.

Scale overhead is always 0.25 bits/element — the "free" cost of microscaling.

### Why 32-element blocks
- Small enough to fit one tensor-core fragment (warp-level cooperation on H100/Blackwell).
- Small enough to track local outliers (per-channel variation captured by per-block scale).
- Large enough that the 8-bit scale is amortised to <1 bit/element.

### Hardware support
Blackwell (NVIDIA) and follow-on MI3xx (AMD) ship native MX support: tensor-core fragments load 32-element blocks with the shared E8M0 scale and produce FP32 accumulators directly. NVFP4 is a Blackwell extension with FP8 block scale + FP32 tensor scale (a 2-level hierarchy on top of MX).

### Training vs inference
MX supports both: forward (weights × activations in MX), backward (gradients in MX), optimizer state often kept FP32. Microsoft's MX paper trains generative LMs at MXFP6 / MXFP4 weights+activations+gradients at FP32-parity downstream accuracy.

## Connections
- FP8 ancestor: [[fp8-formats-paper]] (joint NVIDIA/ARM/Intel FP8 spec).
- NVFP4 successor (Blackwell, 2-level scale hierarchy): [[nvfp4]] (formats dir).
- LLM-FP4 inference cousin: [[llm-fp4]].
- MX training studies: [[mxfp-training]].
- DeepSeek V3 FP8 training is a "fine-grained scaling" cousin of MX with smaller block size: [[deepseek-v3-fp8]].
