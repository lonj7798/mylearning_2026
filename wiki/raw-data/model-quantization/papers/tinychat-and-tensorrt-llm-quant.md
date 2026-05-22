<!-- scope: production-system quantization integration notes (TinyChat + TensorRT-LLM)
     deps: [[awq]], [[smoothquant]], [[fp8-formats-paper]]
     see-also: [[marlin-kernel]], [[machete-kernel]], [[vllm-quant]]
-->

# Production Quantization Stacks: TinyChat & TensorRT-LLM
- **Core Insight:** Production W4A16 / W8A8 / FP8 inference is gated by the quality of the *kernel*, not the *algorithm* — the same AWQ or SmoothQuant checkpoint can give 1.5× or 4× speedup over FP16 depending on whether the serving stack uses a roofline-aware mixed-precision GEMM (TinyChat / Marlin / Machete / TRT-LLM) or a naïve dequant-then-FP16-GEMM.
- **Guideline:** Pick the quant *algorithm* by accuracy target (AWQ / GPTQ for W4A16, SmoothQuant for W8A8, FP8 per-tensor for serving on Hopper/Blackwell) and the *runtime* by hardware target (TinyChat for edge / single-GPU consumer, TRT-LLM for NVIDIA cloud, vLLM + Marlin/Machete for open-source serving).
- **Authors:** Han Lab (MIT — TinyChat); NVIDIA (TensorRT-LLM)
- **Year:** 2023-2025
- **URL:** https://github.com/mit-han-lab/llm-awq (TinyChat) • https://github.com/NVIDIA/TensorRT-LLM
- **Relevant topics:** W4A16 inference, FP8 serving, kernel fusion, in-place dequant, batched serving

## Abstract
TinyChat is MIT Han Lab's reference inference runtime for AWQ-quantized models; it ships the original AWQ W4A16 kernel (lookup-table-based dequant, fused FFN, in-place INT4) and was the first to demonstrate Llama-2 7B at ~30 tokens/s on a Jetson Orin Nano. TensorRT-LLM is NVIDIA's production serving framework — it implements W4A16 (AWQ/GPTQ format), W8A8 (SmoothQuant), and FP8 (per-tensor and per-channel) backed by hand-tuned CUTLASS / cuDNN kernels with continuous batching, paged KV cache, and tensor-parallel sharding. Both stacks treat quantization as a first-class deployment target rather than an afterthought: weight load layout, dequant fusion, activation casting, and KV-cache packing are co-designed with the GEMM kernel.

## Key Contributions

### TinyChat
- First W4A16 inference stack that matched the AWQ paper's theoretical 4× speedup on Jetson / consumer-GPU hardware — proved that 4-bit weight serving was practically deployable on the edge.
- Lookup-table-based dequant: each 4-bit weight indexes a per-group LUT in shared memory, avoiding per-element shift + scale on every MMA tile.
- In-place INT4 weight storage with on-the-fly cast directly into tensor-core registers.
- Fused FFN: combines gate, up, down projections + SwiGLU into one mega-kernel that holds activations in registers across the FFN.
- Reference implementation for the AWQ paper; later forked into vLLM's `awq` quantization backend.

### TensorRT-LLM
- Production NVIDIA serving stack with first-class support for:
  - **W4A16** AWQ/GPTQ via internal CUTLASS kernels (analogous to Marlin/Machete; closed-source but well-optimized).
  - **W8A8** SmoothQuant — fused per-channel weight scale + per-token activation scale into the GEMM.
  - **FP8** (per-tensor and per-row scaling) on Hopper / Blackwell — automatic scale calibration during the engine build step.
  - **NVFP4** on Blackwell — supported in TRT-LLM 0.13+.
- **KV cache quantization**: FP8 KV cache (per-channel K + per-token V scales) is a one-flag option, integrated with paged attention.
- **In-flight batching** (continuous batching) + **chunked prefill** so quant kernels run at high effective batch even with mixed-length requests.
- Engine compile step: the model is converted to a TRT engine where quantization decisions are baked in per layer (some layers can stay FP16/BF16 if calibration flagged them as sensitive).

## Key Figures/Tables to Study
- **TinyChat benchmark table** (AWQ paper appendix): tokens/s on RTX 4090, Jetson Orin, MacBook M2 across Llama-7B/13B at W4A16 vs FP16.
- **TensorRT-LLM perf guide tables**: throughput per GPU at 70B/405B across FP16/FP8/W4A16 with paged KV at INT8/FP8.
- The TRT-LLM "quant recipe" docs page: per-format calibration steps and scale-storage layout.

## Technical Details

### TinyChat kernel structure
- 4-bit weight stored packed; per-group (G=128) scale + zero in FP16.
- Forward: load tile of 4-bit weights → look up FP16 values from per-group LUT in shared memory → MMA against FP16 activations.
- Fused activation function (SiLU/SwiGLU) folded into the down-projection output cast.
- Targeted at single-batch (edge), batch ≤ 8 (consumer); for high-batch serving the Marlin/Machete pipeline dominates.

### TensorRT-LLM FP8 recipe
- Static per-tensor or per-row FP8 scales computed during a calibration pass over a small dataset.
- E4M3 weights + activations on forward; E5M2 on backward not relevant (inference-only).
- KV cache in FP8: per-channel scale on K (heads tend to have heterogeneous magnitudes), per-token scale on V.
- Selective layers (LM-head, last LayerNorm) remain BF16 — flagged automatically by TRT-LLM's sensitivity heuristic.

### TensorRT-LLM W4A16 recipe
- Consumes GPTQ or AWQ checkpoints; internal CUTLASS kernel similar in spirit to Marlin/Machete.
- Per-group scale (G=128 default); group size and zero-point handling configured at engine-build time.
- Speedup numbers in NVIDIA's perf guide are kernel-by-kernel comparable to vLLM + Machete on H100.

### TensorRT-LLM NVFP4
- Engine-build pass quantizes weights to NVFP4 (16-element FP4 blocks + E4M3 scale + FP32 tensor scale); activations cast per-block at runtime.
- Selective high-precision layers handled the same as the FP8 path.
- The Blackwell-only tensor-core code path consumes NVFP4 natively.

### Choosing a stack
| Hardware | Best W4A16 | Best W8A8 | Best FP8 |
|----------|-----------|-----------|----------|
| Edge / single consumer GPU | TinyChat (AWQ) | — | — |
| Open-source serving, Ampere | vLLM + Marlin | vLLM + SmoothQuant | — |
| Open-source serving, Hopper | vLLM + Machete | vLLM + SmoothQuant | vLLM FP8 / TRT-LLM |
| NVIDIA cloud production | TRT-LLM | TRT-LLM | TRT-LLM |
| Blackwell | TRT-LLM (NVFP4) | TRT-LLM | TRT-LLM |

## Connections
- [[awq]] — TinyChat is the reference runtime; TRT-LLM consumes AWQ checkpoints too.
- [[smoothquant]] — TRT-LLM's W8A8 path implements SmoothQuant scaling.
- [[fp8-formats-paper]] — TRT-LLM's FP8 path implements the joint E4M3 / E5M2 spec.
- [[marlin-kernel]] / [[machete-kernel]] — open-source counterparts to TRT-LLM's W4A16 kernel.
- [[vllm-quant]] — the OSS serving stack that integrates Marlin/Machete and is the primary open alternative to TRT-LLM.
- [[transformer-engine]] — NVIDIA library that TRT-LLM builds on for FP8 layer implementations.
