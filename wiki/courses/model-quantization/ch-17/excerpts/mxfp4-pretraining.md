---
chapter: ch-17
course: model-quantization
phase: read
excerpt_of: "Training LLMs with MXFP4 (Tseng, Yu, Park, AISTATS 2025) + Pretraining LLMs with MXFP4 on Native FP4 Hardware (Cim et al., 2026)"
source_url: https://arxiv.org/abs/2502.20586 + https://arxiv.org/abs/2605.09825
created_at: "2026-05-21"
---

# Excerpt: MXFP4 pretraining — RHT + SR, and the 2026 Wgrad plot twist

**Authors:** Albert Tseng, Tao Yu, Youngsuk Park (2025); Musa Cim, Poovaiah Palangappa, Miro Hodak, Ravi Dwivedula, Meena Arunachalam, Mahmut Taylan Kandemir (2026)
**Year:** 2025 (AISTATS); 2026-05 update
**URL:** https://arxiv.org/abs/2502.20586 + https://arxiv.org/abs/2605.09825
**Raw-data source:** [[raw-data/mxfp4-pretraining]] + [[raw-data/mxfp4-native-hardware-2026]]

---

## What MXFP4 is

OCP Microscaling FP4. Three differences from NVFP4:

| Aspect | MXFP4 | NVFP4 |
|--------|-------|-------|
| Element | FP4 E2M1 | FP4 E2M1 (same) |
| Block size | **32** | **16** |
| Block scale | **E8M0** (pure power-of-two, 8-bit unsigned exponent) | **E4M3** (fractional FP8) |
| Per-tensor scale | none | FP32 |
| Standardization | OCP (cross-vendor) | NVIDIA-native (Blackwell) |

MXFP4 is the format AMD's MI355X consumes natively; NVFP4 is Blackwell-only.

---

## The 2025 result (AISTATS)

Naïve MXFP4 pretraining diverges. Two fixes make it work:

### Fix 1: Random Hadamard Transform on GEMM inputs

Pre-multiply X by a random Hadamard matrix `H` (random ±1 entries), then quantize `X · H` to MXFP4. Fold `H` into the weight offline so inference is unchanged: `W' = W · Hᵀ`.

Theoretical bound: the SR variance for a fixed FP4 quantizer scales with the per-block max. RHT bounds the per-block max by `O(√(log d / d))` of the tensor's L2 norm — what keeps the E8M0 block scale tight enough for FP4 to resolve.

### Fix 2: Stochastic rounding on backward GEMMs

For each FP4 element: round up with probability proportional to fractional position, else down. `E[SR(x)] = x` — unbiased gradient estimator.

Forward stays RNE (deterministic inference).

### Result

| Model | MXFP4 final loss vs BF16 | Backward speedup vs BF16 |
|-------|--------------------------|--------------------------|
| GPT-1.3B | matches | ~1.7× |
| GPT-2.7B | matches | ~1.7× |
| GPT-6.7B | matches (within noise) | ~1.7× |

MXFP4 GEMMs are roughly 2× faster than FP8 on Blackwell, > 1.3× faster than FP8 backprop, > 1.7× faster than BF16 backprop.

---

## The 2026 plot twist — native hardware (MI355X, Cim et al.)

This paper revisited MXFP4 pretraining on AMD Instinct MI355X — *native* FP4, not emulation. Llama-3.1-8B pretraining on C4. FP4 enabled progressively across the three GEMMs:

| Path | Meaning | Diagnostic finding |
|------|---------|---------------------|
| Fprop | forward activation × weight | relatively stable under MXFP4 |
| Dgrad | activation-gradient GEMM | modest added token cost |
| **Wgrad** | **weight-gradient GEMM** | **main convergence degradation driver** |

Contrary to the 2025 recipe, *stochastic rounding and randomized Hadamard rotations were insufficient once Wgrad was quantized*. **Deterministic Hadamard rotations** restored stable optimization.

### Interpretation

The 2025 framing was "FP4 needs stochastic rounding to be unbiased + RHT to bound outliers." The 2026 framing sharpens this: **the FP4 error along sensitive gradient paths is structured, not random**. Stochasticity helps when the error is random scalar noise; deterministic Hadamard rotations help more when the per-block scale is poorly aligned with sensitive gradient directions.

The 2025 result still stands at its tested scope (GPT-6.7B with selective FP4 enablement). But full-pipeline MXFP4 training at the 8B / native-hardware boundary needs *deterministic* rotations and more careful Wgrad treatment.

---

## Why both papers matter for the course

The 2025 paper established that academic FP4 pretraining was viable; the 2026 paper sharpened the failure mode. Together they tell you:

1. **Forward FP4 stability does not imply Wgrad FP4 stability.** Test each GEMM separately.
2. **The right rotation may be deterministic, not random.** This contradicts the QuaRot / SpinQuant inference-time intuition (where randomization helps avoid pathological structure).
3. **Hardware-emulation results don't fully transfer to native FP4 hardware.** The accumulator behavior, scale-dispatch timing, and rounding hardware differ.

This is the active 2026 frontier — the "right" full-pipeline FP4 training recipe is not yet settled outside NVFP4 on Blackwell.

---

## Selective precision (both papers)

Same pattern as DSV3 and NVFP4:

- Embeddings, LM-head, normalizations: BF16.
- Optimizer state: FP32.
- Master weight: BF16.
- Everything else: MXFP4.

---

## Connections

- [[mx-formats]] / [[microscaling-formats]] / ch-16 — the OCP MX spec.
- [[nvfp4-training]] / [[excerpts/nvfp4-training]] — NVIDIA's parallel work; same recipe family at a different block size.
- [[deepseek-v3-fp8]] / [[excerpts/deepseek-v3-fp8]] — the FP8 elder cousin; per-block scaling generalizes one bit-width up.
- [[quarot]] / [[spinquant]] / ch-14 — the inference-time RHT lineage borrowed for training.
- [[stochastic-rounding]] / ch-01 — the underlying technique (Gupta 2015), applied to FP4 here.
- [[quartet-ii]] — 2026 NVFP4 follow-up; same theme of better gradient estimators.
- [[ch-17]] — parent synthesis.
