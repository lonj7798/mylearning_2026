<!-- scope: OCP Microscaling 2023 spec — shared block exponent + element format (MXFP8/MXFP6/MXFP4/MXINT8)
     deps: [[fp8-e4m3]], [[fp4-e2m1]], [[ieee-754]]
     see-also: [[nvfp4]], [[microscaling-formats]], [[mxfp-training]]
-->

# OCP Microscaling (MX) Formats Specification (2023)
- **Core Insight:** Microscaling formats combine a small (4–8-bit) per-element format with a *shared* 8-bit power-of-2-only scale factor across each block of 32 elements — recovering the dynamic range that low-bit elements lose, at an amortized cost of 8/32 = 0.25 extra bits per element, while keeping all element-element math integer/log-integer and the scale a single shift.
- **Guideline:** When deploying sub-8-bit formats (FP4, FP6, INT4), prefer an MX-style block-scaled layout (block 32, E8M0 scale) over per-tensor scaling — the per-block scale absorbs outliers at sub-1% bit overhead, and hardware support (Blackwell, MI355) makes it free at runtime.
- **Authors:** OCP (Open Compute Project) Microscaling Working Group; original concept from Rouhani et al. 2023 (Microsoft); standardized by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm
- **Year:** 2023 (OCP MX v1.0 spec, Dec 2023)
- **URL:** https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf ; https://arxiv.org/abs/2310.10537 (Rouhani 2023)
- **Relevant topics:** MXFP8, MXFP6, MXFP4, MXINT8, block-scaled, shared exponent, E8M0

## Abstract
The OCP Microscaling (MX) specification standardizes a family of block-scaled low-precision number formats for AI accelerators. Each MX number is a (block-scale, element-data) pair where the block-scale X is an 8-bit power-of-2 exponent (E8M0 format: no sign, no mantissa, just a biased exponent) shared by 32 consecutive elements, and the element data uses one of four element formats: **MXFP8** (E4M3 or E5M2), **MXFP6** (E3M2 or E2M3), **MXFP4** (E2M1), or **MXINT8** (signed 8-bit integer). The shared scale is multiplied (= exponent-added) at GEMM time. The spec was jointly defined by all major accelerator vendors and is the basis for NVIDIA Blackwell, AMD MI355X, and Intel Gaudi 3.

## Key Contributions
- Industry-wide standard for sub-8-bit block-scaled formats.
- E8M0 (8-bit exponent-only) scale: cheap to implement, broad dynamic range (±127 powers of 2).
- Fixed block size of 32 → memory-aligned, GEMM-friendly tile dimensions.
- Six element formats covered: MXFP8-E4M3, MXFP8-E5M2, MXFP6-E3M2, MXFP6-E2M3, MXFP4-E2M1, MXINT8.
- Vendor-neutral standardization → hardware portability of quantized models.

## Key Figures/Tables to Study
- **MX format taxonomy table** (OCP spec §3): all six element formats with their bit budgets.
- **Block-scale dynamic range plot**: E8M0 covers ~10^{38} dynamic range, enough to absorb outliers in any deep-learning tensor.

## Technical Details

### MX number structure
```
MX vector = (X, P_0, P_1, …, P_31)
```
- **X**: 8-bit E8M0 block scale.
- **P_i**: low-precision element (FP8 / FP6 / FP4 / INT8) for i = 0…31.

### E8M0 (block scale format)
- **8 bits**: pure exponent, no sign, no mantissa.
- **Bias = 127** (same as FP32 exponent).
- Stored value x = 2^{e − 127} for e ∈ [0, 254]; e = 255 reserved for NaN.
- Range: 2^{−127} ≈ 5.88e−39 to 2^{127} ≈ 1.70e+38.
- Smallest positive scale: 2^{−127}; largest: 2^{127}.

### Effective decoded value of an MX element
```
value(P_i) = 2^{e_X − 127} · element_value(P_i)
```
where element_value depends on the element format. For MXFP4 (E2M1 element):
```
value = X · ((−1)^s · (1 + m/2) · 2^{e_P − 1})
      = 2^{e_X − 127} · (sign · (1 + m/2) · 2^{e_P − 1})
```

### Six MX format variants
| Name | Element format | Element bits | Effective bits/elem | Max element value |
|------|---------------|--------------|---------------------|---------------------|
| MXFP8-E4M3 | E4M3 (no ∞) | 8 | 8.25 | 448 |
| MXFP8-E5M2 | E5M2 (IEEE) | 8 | 8.25 | 57344 |
| MXFP6-E3M2 | E3M2 | 6 | 6.25 | 28 |
| MXFP6-E2M3 | E2M3 | 6 | 6.25 | 7.5 |
| MXFP4-E2M1 | E2M1 (no ∞, no NaN) | 4 | 4.25 | 6 |
| MXINT8 | signed INT8 | 8 | 8.25 | 127 |

Effective bits = element_bits + 8/32 = element_bits + 0.25.

### Block layout (memory)
```
Block 0:   [X_0][P_0,0][P_0,1]...[P_0,31]          → 33 bytes for MXFP8/INT8
                                                   → 25 bytes for MXFP6 (6×32 = 192 bits = 24 bytes + 1 X)
                                                   → 17 bytes for MXFP4 (4×32 = 128 bits = 16 bytes + 1 X)
Block 1:   [X_1][P_1,0]...
...
```

### Scale calibration (per-block, at quantization time)
For each 32-element block of raw FP32 values w_0, …, w_31:
```
abs_max = max(|w_i|)
e_target = floor(log2(abs_max)) − floor(log2(element_max))
X = clip(e_target + 127, 0, 254)        (E8M0 encoding)
P_i = round_to_element_format(w_i / 2^{X − 127})
```
where element_max is the max value of the element format (e.g. 6 for MXFP4).

### Block size of 32
Why 32 (and not 16 or 64)?
- **GEMM tile alignment**: 32 fits typical tensor-core K-dim subdivisions on H100/B100.
- **Outlier statistics**: 32 elements is short enough to be locally uniform, long enough to amortize the 8-bit scale to negligible overhead.
- **Hardware area**: per-block dequant unit width = 32 → matches tensor-core lane count.

### MXFP8 / MXFP6 / MXFP4 quality (vs naive per-tensor)
- MXFP8 (with E4M3 elements) ≈ BF16 quality on Llama-class inference; trivially deployable.
- MXFP6 ≈ FP16 with ~0.05 PPL gap; sub-1% accuracy loss.
- MXFP4 ≈ INT4-AWQ quality without any rotation; ~0.1–0.3 PPL gap from BF16.

### Hardware support
- **NVIDIA Blackwell (B100/B200)**: native MXFP8 / MXFP6 / MXFP4 tensor cores.
- **AMD MI355X (2025)**: announced MX format support.
- **Intel Gaudi 3**: FP8 + MXFP6 / MXFP4 support.
- Pre-Blackwell hardware: software emulation via dequant-to-BF16 → tensor core.

### Distinct from NVFP4
**NVFP4** (see [[nvfp4]]) is a *more aggressive* block-scaled FP4 with **finer block size (16) and finer-precision scale (FP8 E4M3 instead of E8M0)**, plus a coarse FP32 per-tensor scale on top. NVFP4 is NVIDIA-proprietary; MXFP4 is the OCP-standard. NVFP4 trades 0.25 → 0.5 extra bits/element for better quality at FP4.

## Connections
- [[fp8-e4m3]] / [[fp8-e5m2]] — element formats for MXFP8.
- [[fp6]] — element formats for MXFP6.
- [[fp4-e2m1]] — element format for MXFP4.
- [[nvfp4]] — NVIDIA's competing FP4 layout; finer block + FP8 scale.
- [[microscaling-formats]] — Rouhani 2023 originating paper.
- [[mxfp-training]] — MXFP4 / MXFP6 training experiments.
- [[ieee-754]] — element formats inherit IEEE-style structure.
