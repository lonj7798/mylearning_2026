<!-- scope: NVIDIA H100 Hopper FP8 tensor-core hardware deep-dive
     deps: [[fp8-e4m3]], [[fp8-e5m2]]
     see-also: [[transformer-engine-blog]], [[fp8-formats-paper]], [[nvidia-quantization]]
-->

# NVIDIA H100 FP8 — Hopper Tensor Core Deep-Dive
- **Core Insight:** Hopper's 4th-generation tensor cores natively execute FP8 (E4M3 and E5M2) WGMMA matrix-multiply instructions at 2× the FP16 throughput, making FP8 the first sub-16-bit format to be a true first-class training citizen.
- **Guideline:** Use E4M3 for forward activations and weights, E5M2 for gradients in the backward pass; rely on Transformer Engine's automatic delayed scaling to choose the per-tensor scale factor.
- **Authors:** NVIDIA Architecture / Hopper team (whitepaper + developer blogs)
- **Year:** 2022 (Hopper launch), updated 2023-2024
- **URL:** https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- **Relevant topics:** FP8, WGMMA, Hopper, tensor cores, Transformer Engine

## Summary
The H100 SXM5 GPU introduces 4th-generation tensor cores that add native FP8 matmul on top of the existing FP16/BF16/TF32/INT8 paths. FP8 comes in two binary8 flavours from the IEEE-aligned NVIDIA/Arm/Intel joint spec: E4M3 (4 exponent + 3 mantissa, finite-only, range ±448) for forward activations and weights, and E5M2 (5 exponent + 2 mantissa, IEEE-style with inf/NaN, range ±57344) for backward gradients. The headline raw throughput is ~3,958 TFLOPS dense FP8 (and ~7,916 TFLOPS sparse) on H100 SXM5 — 2× the FP16 number and 4× FP32. The new WGMMA (Warp-Group Matrix Multiply Accumulate) instruction operates on entire warp-groups (128 threads), reads operands from shared memory or registers, and accumulates into FP32. The Transformer Engine library wraps these intrinsics with per-tensor delayed scaling and amax history to keep the dynamic range in bounds.

## Key Points
- Two FP8 formats supported in hardware: E4M3 (forward) and E5M2 (backward).
- Peak dense throughput: 1979 TFLOPS FP16 → 3958 TFLOPS FP8 (2× speedup) on H100 SXM5.
- WGMMA replaces Ampere's `mma.sync` for matrix tiles; operates over 128 threads.
- FP8 accumulation is in FP32 (loss of precision is bounded by the accumulator, not the operand format).
- Transformer Engine wraps `cuBLASLt` FP8 GEMM and `cuDNN` FP8 attention paths.
- Tensor Memory Accelerator (TMA) feeds FP8 operands asynchronously to keep WGMMA pipelines fed.

## Technical Details

### Hardware specs (H100 SXM5)
| Spec | Value |
|------|-------|
| FP8 dense TFLOPS | 3,958 |
| FP8 sparse TFLOPS | 7,916 |
| FP16 dense TFLOPS | 1,979 |
| INT8 dense TOPS | 3,958 |
| HBM3 bandwidth | 3.35 TB/s |
| L2 cache | 50 MB |

### Tensor-core instructions
- `wgmma.mma_async.sync.aligned.m64nNk32.f32.e4m3.e4m3` — FP8 WGMMA E4M3×E4M3 → FP32.
- `wgmma.mma_async.sync.aligned.m64nNk32.f32.e5m2.e5m2` — FP8 WGMMA E5M2×E5M2 → FP32.
- Mixed-format `e4m3.e5m2` variant also exists for backward pass mixing.
- Tile shapes: m=64, n∈{8,16,…,256}, k=32 for FP8.

### Scaling regime (Transformer Engine)
- Per-tensor scale `s`; the stored tensor is `x_fp8 = round(x_fp32 / s)`.
- Delayed scaling: scale for step t is computed from amax history over previous N=16 steps.
- Amax is tracked per tensor in an FP32 buffer; updated each iteration; `s = amax / fp8_max` with margin.

### FP8 attention
- cuDNN FP8 fused attention since v9: QKᵀ in FP8, softmax in FP32, AV in FP8.
- Output projection also FP8; per-head scales for Q, K, V independently.

## Connections
- [[fp8-e4m3]] — exponent/mantissa bits and dynamic range that drive Hopper's hardware.
- [[fp8-e5m2]] — the wider-range backward-pass format.
- [[transformer-engine-blog]] — software wrapper that schedules the WGMMA calls.
- [[fp8-formats-paper]] — joint NVIDIA/Arm/Intel spec ratified into Hopper silicon.
- [[nvidia-blackwell-fp4]] — successor format on the next-generation tensor cores.
