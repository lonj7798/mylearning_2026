<!-- scope: AMD Instinct MI300X FP8 matrix cores and ROCm quant stack
     deps: [[fp8-e4m3]], [[fp8-e5m2]]
     see-also: [[nvidia-h100-fp8]], [[transformer-engine-blog]]
-->

# AMD MI300X FP8 — CDNA3 Matrix Cores
- **Core Insight:** AMD's MI300X (CDNA3, 2023) brings native FP8 (E4M3 and E5M2) matrix-core support to the ROCm ecosystem, with 192GB of HBM3 and ~2,600 TFLOPS FP8 throughput putting it within striking distance of H100 on large-context LLM inference.
- **Guideline:** Use Composable Kernel (CK) or hipBLASLt FP8 GEMM kernels through the ROCm-fork of Transformer Engine; for inference, vLLM and SGLang both have ROCm FP8 paths.
- **Authors:** AMD CDNA3 architecture team (MI300 whitepaper + ROCm blog)
- **Year:** 2023 (MI300X launch), 2024 (FP8 SW stack maturation)
- **URL:** https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- **Relevant topics:** MI300X, CDNA3, FP8, ROCm, HBM3, hipBLASLt

## Summary
The MI300X is AMD's flagship CDNA3 GPU built for generative AI workloads, packing 192GB of HBM3 (highest single-package memory in 2023–2024) and 304 CUs delivering ~2,610 TFLOPS dense FP8 / ~5,220 TFLOPS sparse FP8. CDNA3 matrix cores execute FP8 (both E4M3 and E5M2), BF16, FP16, INT8, and TF32 through the `V_MFMA` (Matrix Fused Multiply-Add) instruction family. The 192GB capacity is the differentiating feature: a single MI300X holds a Llama-3-70B model at FP8 with hundreds of GB of headroom for KV cache, where H100 SXM (80GB) requires tensor parallelism. ROCm 6.x ships `hipBLASLt` with FP8 GEMM paths and the Composable Kernel library exposes templated FP8 attention. The ROCm fork of NVIDIA's Transformer Engine adapts the per-tensor delayed scaling recipe.

## Key Points
- 192GB HBM3 per package (vs H100 SXM's 80GB) — fits more model + KV cache per accelerator.
- Native FP8 E4M3 and E5M2 in CDNA3 matrix cores.
- ~2,610 TFLOPS dense FP8 / ~5,220 TFLOPS sparse FP8.
- HBM3 bandwidth 5.3 TB/s (vs H100 SXM's 3.35 TB/s) — bandwidth-bound LLM inference often favours MI300X.
- Software stack: hipBLASLt, Composable Kernel, ROCm Transformer Engine fork.

## Technical Details

### Hardware specs (MI300X)
| Spec | Value |
|------|-------|
| Compute units | 304 |
| FP8 dense TFLOPS | 2,610 |
| FP8 sparse TFLOPS | 5,220 |
| BF16/FP16 dense TFLOPS | 1,307 |
| INT8 dense TOPS | 2,610 |
| HBM3 capacity | 192 GB |
| HBM3 bandwidth | 5.3 TB/s |
| TDP | 750 W |

### Matrix-core instructions
- `V_MFMA_F32_*x*x*_F8` — FP8 matrix-fused multiply-add accumulating to FP32.
- Tile shapes (M×N×K): 16×16×32, 32×32×16 commonly used for FP8.
- Both `_FP8` and `_BF8` (AMD names for E4M3 / E5M2 variants) supported.

### Software entry points
- `hipBLASLt` — drop-in BLAS interface with FP8 GEMM since ROCm 6.0.
- `composable_kernel` — templated kernels for fused FP8 attention / GEMM.
- `Transformer Engine (ROCm fork)` — per-tensor delayed scaling, drop-in replacement for nn.Linear.
- vLLM ROCm backend — supports W8A8 FP8 weight + activation quant.
- SGLang ROCm backend — FP8 attention path.

### Quantization recipe on MI300X
- Per-tensor delayed scaling (TE-fork compatible).
- amax history depth: 16 (default).
- Forward: E4M3 weights and activations.
- Backward: E5M2 gradients.
- Accumulate: FP32.

### Differentiator
- KV cache for a 1M-context Llama-3-70B at FP8 fits in one MI300X (no TP needed).
- On H100 SXM, the same workload requires 4-way TP.

## Connections
- [[fp8-e4m3]] — AMD calls this "FP8"; identical bit layout to NVIDIA E4M3.
- [[fp8-e5m2]] — AMD calls this "BF8"; identical bit layout to NVIDIA E5M2.
- [[nvidia-h100-fp8]] — direct competitor; AMD wins on HBM capacity, NVIDIA on software maturity.
- [[transformer-engine-blog]] — upstream NVIDIA library that AMD forked for ROCm.
- [[vllm-quant]] — inference engine with ROCm FP8 path.
