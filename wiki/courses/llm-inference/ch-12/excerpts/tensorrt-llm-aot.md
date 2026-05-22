---
chapter: ch-12
course: llm-inference
phase: read
excerpt_of: "TensorRT-LLM — ahead-of-time engine build"
source_url: https://nvidia.github.io/TensorRT-LLM/
created_at: "2026-05-21"
---

# Excerpt: TensorRT-LLM — AOT engine build, the extreme of graph capture

**Authors:** NVIDIA
**Year:** 2023–present
**URL:** https://nvidia.github.io/TensorRT-LLM/
**Raw-data source:** [[raw-data/tensorrt-llm]]

---

## The philosophy difference

vLLM captures CUDA graphs at engine init (~30 s). TensorRT-LLM builds an entire optimized engine **ahead of time** as a deployment artifact:

```bash
trtllm-build \
    --checkpoint_dir ./llama3-70b-converted \
    --output_dir ./llama3-70b-engine \
    --max_batch_size 64 \
    --max_input_len 8192 \
    --max_output_len 2048 \
    --tp_size 4 \
    --use_paged_kv_cache \
    --enable_kv_cache_reuse \
    --quantization fp8
```

Build time: 30–90 minutes for a 70B model. The output is a `.engine` file (typically 30–80 GiB after quantization) that contains:

- Pre-fused kernels selected per shape regime.
- Pre-selected attention backend (FA3 / FlashDecoding) per shape bucket.
- Pre-captured CUDA graphs.
- Pre-tuned NCCL communication.
- Quantization scales baked in (no runtime calibration).
- The model topology resolved into a fixed execution plan.

---

## What the engine optimizes that vLLM doesn't

1. **Kernel selection per shape**: TRT-LLM has many candidate kernels for matmul (cuBLAS, CUTLASS, custom). The builder benchmarks each at the target shapes and bakes in the winner.

2. **Fusion across layers**: vLLM fuses within a layer (e.g., RMSNorm + matmul). TRT-LLM fuses across layers (e.g., FFN output → residual → next RMSNorm).

3. **NCCL pattern selection**: which all-reduce algorithm (ring, tree, NVLS) is best for *this* topology *at this batch size* is pre-decided. vLLM uses NCCL default.

4. **Quantization-aware kernel selection**: FP8 matmuls dispatch to E4M3 tensor cores; the dequant + matmul + requant gets fused into one kernel.

---

## Reported numbers

TensorRT-LLM typically achieves:
- **5–15% lower TPOT vs vLLM** on the same model + hardware.
- **20–30% lower TTFT** because prefill kernels are also pre-fused.
- **FP8 path** is mature; INT4-W + FP8-A path competitive.

On Llama-3-70B at H100×4, batch=32:
- vLLM TPOT ≈ 17 ms.
- TensorRT-LLM TPOT ≈ 14 ms.

The gap is real but smaller than the build-time cost. vLLM is the better default; TRT-LLM is the right choice when latency is critical and the model + hardware are stable.

---

## The tradeoff

| Property | TensorRT-LLM | vLLM |
|---|---|---|
| Latency | best | very good |
| Build time | 30–90 min | ~30 s |
| Iteration speed | slow (rebuild for every change) | fast |
| Hardware portability | NVIDIA only | any (CPU, AMD, NVIDIA, ...) |
| Model coverage | NVIDIA-curated set | HF hub (1000s of models) |
| OSS contribution | NVIDIA-only | community PR-friendly |

For a startup with one model and one GPU SKU: TRT-LLM saves money on hardware. For a research lab or platform with many models: vLLM.

---

## When AOT compilation fails

- **Frequent model swaps**: rebuilds dominate operational cost.
- **Custom model architectures**: must implement TRT-LLM plugins (significant engineering).
- **Mixed-precision experiments**: every quantization config is a separate engine.
- **Multi-GPU SKU fleet**: each SKU needs its own engine build.

---

## Connections

- [[ch-12]] — parent chapter; TRT-LLM is the extreme version of graph capture.
- [[excerpts/cuda-graphs-inference]] — the mechanism TRT-LLM uses internally.
- [[excerpts/vllm-piecewise-graphs]] — the OSS alternative.
- [[ch-18]] — full framework comparison including TRT-LLM, TGI, LightLLM, llama.cpp.
- [[ch-20]] — production stacks that use TRT-LLM in deployment.
