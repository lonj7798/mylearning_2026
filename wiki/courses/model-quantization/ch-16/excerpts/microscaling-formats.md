---
chapter: ch-16
course: model-quantization
phase: read
excerpt_of: "Microscaling Data Formats for Deep Learning (OCP MX)"
source_url: https://arxiv.org/abs/2310.10537
created_at: "2026-05-21"
---

# Excerpt: OCP MX formats — shared-exponent blocks for sub-8-bit numerics

**Authors:** Bita Darvish Rouhani et al. (32 co-authors: Microsoft, NVIDIA, Intel, AMD, ARM, Qualcomm)
**Year:** 2023
**URL:** https://arxiv.org/abs/2310.10537
**Raw-data source:** [[raw-data/microscaling-formats]]

---

## What MX is

A **format family**, not an algorithm. The MX spec combines:

- A shared **8-bit power-of-two exponent** (E8M0) per **32-element block**.
- A narrow (4–8 bit) **per-element value** (FP or INT).

Decoded value: `v_i = 2^X · d_i`.

Codified by the **Open Compute Project (OCP)** in 2023. Productionised on **Blackwell** (NVIDIA) and **MI3xx** (AMD) with native tensor-core support.

---

## The block structure

A block of 32 elements `(v_1, ..., v_32)` is stored as:

| Field | Bits | Meaning |
|-------|------|---------|
| X (shared scale) | 8 | E8M0 exponent → `2^X` |
| d_1 ... d_32 (elements) | 32 × k | per-element value (k = 4 / 6 / 8) |

Decoded: `v_i = 2^X · d_i`.

The shared scale is **power-of-two only** (no mantissa) → dequant is a **shift, not a multiply**. This is the architectural advantage: shifting by an integer is essentially free in hardware.

---

## Element formats

| Format | Bits | Composition | Representable (positive) | Dynamic range |
|--------|------|-------------|--------------------------|---------------|
| **E2M1 (MXFP4)** | 4 | 1 sign + 2 exp + 1 mant | 0, 0.5, 1, 1.5, 2, 3, 4, 6 | ~3 bits |
| **E2M3 (MXFP6)** | 6 | 1 sign + 2 exp + 3 mant | 16 values | ~3 bits, finer res |
| **E3M2 (MXFP6 alt)** | 6 | 1 sign + 3 exp + 2 mant | 16 values | wider range, coarser |
| **E4M3 (MXFP8)** | 8 | 1 sign + 4 exp + 3 mant | same as OFP8 E4M3 | wide |
| **MXINT8** | 8 | twos-complement integer | −127..127 | uniform |

E2M1 is the same as the FP4 spec ([[fp4-e2m1]] in formats dir). E4M3 is the same as the forward-pass FP8 spec ([[fp8-e4m3]]).

---

## Bit accounting

For 32-element block:

```
MXFP4:  8-bit scale + 32 × 4-bit values = 136 bits / 32 = 4.25 bits/element
MXFP6:  8 + 32 × 6 = 200 / 32 = 6.25 bits/element
MXFP8:  8 + 32 × 8 = 264 / 32 = 8.25 bits/element
```

**Scale overhead is always exactly 0.25 bits/element** — the "free" cost of microscaling.

---

## Why 32-element blocks (not 16, not 64)

- Small enough to fit one tensor-core fragment (warp-level cooperation on H100/Blackwell).
- Small enough to track local outliers (per-channel variation captured by per-block scale).
- Large enough that the 8-bit scale amortises to < 1 bit/element.

The 32 isn't arbitrary — it matches hardware tensor-core fragments, so dequant is essentially free in the GEMM pipeline.

---

## Hardware support

- **Blackwell** (NVIDIA) ships native MXFP4/MXFP6/MXFP8 tensor-core fragments. The hardware loads 32-element blocks with E8M0 scale and produces FP32 accumulators directly.
- **MI3xx** (AMD) ships parallel native MX support.
- **NVFP4** is a Blackwell **extension** with FP8 block scale + FP32 tensor scale (a 2-level hierarchy on top of MX) — closer to DeepSeek V3's fine-grained scaling philosophy.

On Hopper (H100/H200) MX is **emulated** — 2–3× slower than native FP8.

---

## Training vs inference

MX supports both:
- **Forward**: weights × activations in MX.
- **Backward**: gradients in MX (typically MXFP6 to preserve dynamic range).
- **Optimizer state**: often FP32 (master weights).

Microsoft's MX paper trains generative LMs at MXFP6 / MXFP4 weights+activations+gradients at FP32-parity downstream accuracy. This is the bridge to native sub-FP8 pretraining covered in ch-17.

---

## The cross-vendor consensus

The MX spec is the first **multi-vendor standard** for sub-8-bit numerics:

- 32 co-authors across Microsoft, NVIDIA, Intel, AMD, ARM, Qualcomm.
- Codified by OCP (Open Compute Project) for cross-cloud interoperability.
- A model trained / quantized to MXFP4 on Blackwell can be served on MI300X with bit-identical results.

This is in deliberate contrast to BitNet's ternary path (Microsoft-specific kernels, no native HW). MX is the consensus format; BitNet is the optimisation extreme.

---

## When to pick MX vs other low-bit options

| Goal | Pick |
|------|------|
| Cross-vendor inference deployment | MXFP4 (Blackwell + MI3xx) |
| Native sub-FP8 pretraining | MXFP6 / MXFP4 weights + gradients |
| Minimum energy (custom HW) | BitNet b1.58 ternary |
| 1-bit-from-FP fine-tuning | OneBit SVID |
| Block-scale FP4 with FP32 outer scale | NVFP4 (Blackwell-only extension) |
| Long-context weight-only PTQ | AWQ / GPTQ at 4-bit (ch-08/09) |

---

## Pitfalls

- **Block size 32 is fixed by the spec.** Don't experiment with 16 or 64 — your hardware path falls off the native MX support.
- **E8M0 has no mantissa.** Representable scales are `2^k` for integer k. For activations with non-power-of-2 dynamic range, you lose ~0.5 bits of effective precision; accepted as the cost of free shift dequant.
- **MX inference on Ampere needs emulation.** Native MX is Blackwell+ only; emulated MX on H100 is 2–3× slower than native FP8.
- **Element format choice matters.** E2M1 (MXFP4) has coarse 8-level resolution; for activations with smooth distributions prefer MXFP6 (16 levels) or MXINT8 (uniform). Verify per-layer before pushing to MXFP4 globally.
- **Stochastic rounding required for MXFP4 training.** Round-to-nearest produces gradient bias at FP4; stochastic rounding ([[stochastic-rounding]]) unbiased — mandatory for native MXFP4 pretraining (see ch-17).
- **Block alignment must match tile size.** If `d_in % 32 != 0` for some layer, pad to multiple of 32 — non-aligned blocks fall off the tensor-core path.

---

## Connections

- [[fp8-formats-paper]] / ch-02 — joint NVIDIA/ARM/Intel FP8 spec; MXFP8 element format is the same E4M3.
- [[nvfp4]] / ch-02 — Blackwell 2-level extension (FP8 block scale + FP32 tensor scale).
- [[llm-fp4]] / ch-09 — LLM-FP4 inference; MX is the format reference.
- [[mxfp-training]] / [[mxfp4-pretraining]] / ch-17 — MX in native pretraining; production formats.
- [[bitnet-b158]] — the orthogonal sub-2-bit path; MX is FP-family, BitNet is integer ternary.
- [[deepseek-v3-fp8]] / ch-17 — fine-grained FP8 scaling (1×128 act + 128×128 weight); structurally similar to MX with smaller block sizes.
