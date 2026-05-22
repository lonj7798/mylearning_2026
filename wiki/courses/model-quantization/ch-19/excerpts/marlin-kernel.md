---
chapter: ch-19
course: model-quantization
phase: read
excerpt_of: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs (Frantar, Castro, Chen, Hoefler, Alistarh, 2024)"
source_url: https://arxiv.org/abs/2408.11743
created_at: "2026-05-21"
---

# Excerpt: Marlin — W4A16 GEMM that crossed the critical batch

**Authors:** Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh
**Year:** 2024
**URL:** https://arxiv.org/abs/2408.11743 • https://github.com/IST-DASLab/marlin
**Raw-data source:** [[raw-data/marlin-kernel]]

---

## The critical-batch-size problem

Pre-Marlin W4A16 kernels (ExLlama, AutoGPTQ's CUDA kernel, the original TinyChat) had a structural flaw:

- **Batch 1:** memory-bound. Any 4× weight compression → 4× speedup. Trivial win.
- **Batch ~4-8 onward:** compute-bound. The 4-bit weight load no longer dominates; the tensor cores need to stay fed. Pre-Marlin kernels couldn't do this and their FLOPS/s collapsed — often *slower* than FP16 past the crossover.

This made W4A16 useless for serving. Production workloads run batch 16-128.

**Marlin's contribution:** maintain near-FP16 throughput through batch 16-32, continue speedup up to batch ~64 on A100. ~4× FP16 throughput at batch 1-8, near-peak utilization at moderate batch.

---

## The pipeline (per CTA, double-buffered)

Three stages running concurrently:

1. **Stage A (loader warps):** `cp.async` 4-bit weight tile (128 K rows × N cols, packed) and FP16 activation tile from global memory into shared memory buffer *i*.
2. **Stage B (compute warps):** in parallel, dequant the FP16 weights from shared buffer *1−i* via shift + sign-extend + per-group scale multiply, then issue tensor-core MMA on FP16 weights × FP16 activations.
3. **Stage C (barrier):** `mbarrier` sync, swap buffers, repeat.

The double buffer means **stage A on iteration *k+1* runs in parallel with stage B on iteration *k*** — load latency is hidden by compute. This is the canonical W4A16 GEMM pattern; Machete follows the same skeleton with Hopper-native primitives.

### Warp specialization

Some warps do memory movement, others do dequant + MMA; warps cooperate via shared memory with `mbarrier`-style sync. The same producer/consumer pattern that DSV3 ([[deepseek-v3-fp8]]) uses for its FP8 GEMM and that Flash-Attention 3 uses for attention.

---

## The weight pre-shuffle (the "free" optimization)

A **static offline reordering** of the 4-bit weights so that **after the bit-shifts inside the dequant warp, each tensor-core thread already has its operand in the right register slot.** Eliminates per-call permute instructions.

The pre-shuffle changes the in-memory layout but **preserves the bit content** — i.e. you can take any GPTQ-quantized checkpoint and re-pack it for Marlin offline. vLLM does this automatically when you pass `--quantization gptq_marlin`. No re-quantization needed.

---

## Quantization format consumed

- **GPTQ packed-int4** with **group_size 128** (the de-facto standard since the GPTQ paper, ch-08).
- Per-group scale and zero-point in FP16.
- Weights packed 8 × INT4 per 32-bit word.
- AWQ checkpoints work too (`--quantization awq_marlin`) — same packed layout, different per-channel scale strategy.

---

## Why the critical batch range matters

At batch 1, the GEMM is purely memory-bound. As batch grows, FLOPs grow linearly but weight bytes stay constant — at some batch the GEMM crosses into compute-bound. Pre-Marlin kernels couldn't keep the tensor cores fed past that crossover; their FLOPS/s collapsed.

Marlin's warp-specialized pipeline keeps **both memory and compute units saturated through batch ~32**, which is the range that matters for production serving. The roofline figure in the paper shows the kernel staying on the memory-bandwidth-bound line through batch 16, then transitioning gracefully into the compute-bound region without a discontinuity.

---

## Performance

| Configuration | Result |
|---------------|--------|
| Batch 1-8 | ~4× FP16 throughput |
| Batch 16-32 | maintains near-4× |
| Batch ~64 (A100) | still positive speedup |
| End-to-end vLLM Llama-70B on A100 | **2.8× speedup** |

---

## Hardware targets and successors

- **Ampere (A100, A6000)**: original target; uses `cp.async`.
- **Hopper (H100)**: Marlin runs but doesn't use TMA/WGMMA — see [[machete-kernel]] / [[excerpts/machete-kernel]] for the Hopper-native successor.
- **Sparse Marlin**: extends to NVIDIA 2:4 sparsity, achieving another ~1.6× on top.

---

## Integration

- **vLLM**: `--quantization gptq_marlin` or `awq_marlin` automatically rewrites a GPTQ/AWQ checkpoint into the Marlin layout and uses the Marlin kernel.
- Also exposed through SparseML / NeuralMagic deepsparse / TensorRT-LLM (community port).

---

## Connections

- [[gptq]] / ch-08 — the algorithm whose 4-bit checkpoint format Marlin consumes.
- [[awq]] / ch-09 — same format with an AWQ-style activation-aware scale.
- [[machete-kernel]] / [[excerpts/machete-kernel]] — Hopper-native successor that uses TMA + WGMMA.
- [[vllm-quant]] — production integration where Marlin is the Ampere default backend.
- [[tensorrt-llm-quant]] — NVIDIA's competing W4A16 kernel; Marlin matches or beats it on Ampere.
- [[deepseek-v3-fp8]] — same warp-specialized producer/consumer skeleton.
- [[frantar-alistarh-ist-austria]] — the IST-Austria group that produced GPTQ, Marlin, AQLM, SparseGPT.
- [[ch-19]] — parent synthesis.
