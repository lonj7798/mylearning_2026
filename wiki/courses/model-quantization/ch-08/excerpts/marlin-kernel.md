---
chapter: ch-08
course: model-quantization
phase: read
excerpt_of: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs (Frantar, Castro, Chen, Hoefler, Alistarh 2024)"
source_url: https://arxiv.org/abs/2408.11743
arxiv: 2408.11743
created_at: "2026-05-21"
---

# Excerpt: Marlin — the production W4A16 GEMM

**Authors:** Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh
**Year:** 2024
**Code:** https://github.com/IST-DASLab/marlin
**Raw-data source:** [[raw-data/papers/marlin-kernel]]

---

## What it solves

A W4A16 GEMM (INT4 weights × FP16 activations) that historically had a **critical batch size problem**: kernels crushed batch-1 latency (purely memory-bound) but stopped helping (and often hurt) as soon as the workload became compute-bound — typically around batch 8–16. Production LLM serving runs at batch 16–32. Pre-Marlin W4A16 kernels were useless in this range.

Marlin keeps near-peak speedup through batch 32 (the production sweet spot) by treating the kernel as an explicit memory + compute pipeline.

---

## The four engineering tricks

### 1. Async global-to-shared copies

Use Ampere's `cp.async` so weight loads happen in the **shadow of dequant + MMA**. The loader warps issue async copies into shared memory; compute warps read from shared memory two iterations later. Load latency is hidden by compute.

### 2. Warp specialization

Some warps do memory movement (loaders), others do dequant + MMA (compute). They cooperate via `mbarrier`-style sync in shared memory. This splits the kernel into two logical pipelines that overlap perfectly.

### 3. Double-buffered shared memory

While compute warps consume buffer 0, loader warps fill buffer 1. The swap is a flag-flip, no stall. This is the standard pipelining trick that lets the kernel hit peak utilization across the iteration boundary.

### 4. Static weight pre-shuffle

A one-time offline reordering of the 4-bit weights so that after the bit-shifts inside the dequant warp, **each tensor-core thread already has its operand in the right register slot**. Eliminates per-call permute instructions. This is what makes the dequant just a `shift + sign-extend` rather than a multi-instruction permute.

---

## Quantization format consumed

Marlin consumes the **GPTQ packed-int4** format with **`group_size = 128`**:

- Per-group scale and zero-point in FP16.
- Weights packed `8 × INT4` per 32-bit word.
- The static pre-shuffle changes in-memory layout but preserves bit content — **any GPTQ-quantized checkpoint can be re-packed for Marlin offline**.

This format compatibility is what makes Marlin the production kernel for both GPTQ and AWQ outputs in vLLM (`awq_marlin` runtime).

---

## The pipeline (per CTA)

```text
Stage A (loader warps):
  cp.async 4-bit weight tile (128K rows × N cols, packed)
  cp.async FP16 activation tile
  → into shared memory buffer i

Stage B (compute warps), in parallel with Stage A on iteration k+1:
  Read shared buffer (1-i)
  Dequant 4-bit weights via shift + sign-extend + per-group scale multiply
  Issue tensor-core MMA: FP16 weights × FP16 activations
  Accumulate

Stage C (barrier):
  mbarrier sync; swap buffers; repeat.
```

The double buffer means stage A on iteration k+1 runs in parallel with stage B on iteration k. Load latency is hidden by compute.

---

## Why the critical batch range matters

- **Batch 1.** Purely memory-bound: 4× weight compression → 4× speedup.
- **Batch grows.** FLOPs grow linearly but weight bytes stay constant.
- **At some batch, GEMM crosses into compute-bound.**

Pre-Marlin kernels (ExLlama, AutoGPTQ's CUDA kernel) couldn't keep tensor cores fed past the crossover; FLOPS/s collapsed to FP16-or-worse. Marlin's warp-specialized pipeline keeps both memory and compute units saturated through batch ~32 — the production sweet spot.

---

## End-to-end performance

| Batch | FP16 cuBLAS | AutoGPTQ kernel | ExLlama | **Marlin** |
|---|---|---|---|---|
| 1 | 1.0× | 3.2× | 3.4× | **3.6×** |
| 4 | 1.0× | 2.1× | 2.8× | **3.5×** |
| 16 | 1.0× | 1.2× | 1.4× | **3.2×** |
| 32 | 1.0× | 0.9× | 1.0× | **2.8×** |
| 64 | 1.0× | 0.7× | 0.7× | **1.6×** |

Marlin maintains > 2.5× over FP16 through batch 32; competitors collapse past batch 8. **vLLM-served Llama-70B on A100: end-to-end 2.8× speedup** over FP16.

---

## Hardware targets

- **Ampere (A100, A6000):** original target; uses `cp.async`.
- **Hopper (H100):** Marlin runs but doesn't exploit TMA/WGMMA — see [[machete-kernel]] for the Hopper-native successor.
- **Sparse Marlin:** extends to NVIDIA 2:4 structured sparsity, achieving another ~1.6× on top.

---

## Integration

- **vLLM:** `--quantization gptq_marlin` or `awq_marlin` automatically rewrites a GPTQ/AWQ checkpoint into the Marlin layout.
- **SparseML / NeuralMagic deepsparse:** Marlin is the default W4 GEMM.
- **TensorRT-LLM (community port):** Marlin available as an alternative to NVIDIA's W4A16 kernel.

---

## Common pitfalls

- **Forgetting the pre-shuffle.** Loading a GPTQ checkpoint directly into Marlin without `marlin_repack(...)` gives correct outputs but no speedup (dequant is slow). Always re-pack.
- **Using `group_size ≠ 128`.** Marlin's tile sizes assume 128. Other group sizes either fail or fall back to a slower path. Stick with 128 unless you have a strong reason.
- **Batch > 64.** Marlin's advantage shrinks past batch 64; for very large batches, plain FP16 cuBLAS can match or beat it. The crossover depends on tensor shapes.

---

## What Marlin meant for the field

Marlin transformed GPTQ from "research artifact" into "vLLM default". Before Marlin, W4 LLM inference was a curiosity used mainly for fitting models on consumer GPUs at batch 1. After Marlin, W4 LLM serving is the default production deployment for 7B–70B Llama-class models on Ampere/Hopper.

The lesson: **algorithm + kernel must co-design**. GPTQ's per-group packed INT4 format was specifically designed to consume well in a tensor-core kernel; Marlin's pre-shuffle exploits that format perfectly. Other quantization methods (e.g. [[squeezellm]]'s non-uniform codes) require different kernels and have struggled to match Marlin's deployment ubiquity.

---

## Connections

- [[excerpts/gptq]] — the quantization algorithm whose 4-bit checkpoint format Marlin consumes.
- [[ch-08]] — parent synthesis.
- [[ch-09]] — [[awq]] checkpoints also flow through Marlin via `awq_marlin`.
- [[ch-19]] — [[machete-kernel]] is the Hopper-native successor with TMA + WGMMA.
