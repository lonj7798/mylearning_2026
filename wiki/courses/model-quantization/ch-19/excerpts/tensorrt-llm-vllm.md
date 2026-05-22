---
chapter: ch-19
course: model-quantization
phase: read
excerpt_of: "TensorRT-LLM Quantization (NVIDIA, 2023-onward) + vLLM Quantization (Berkeley + Anyscale + NeuralMagic, 2023-onward)"
source_url: https://github.com/NVIDIA/TensorRT-LLM + https://github.com/vllm-project/vllm
created_at: "2026-05-21"
---

# Excerpt: TensorRT-LLM + vLLM — the two production serving stacks

**Authors:** NVIDIA TensorRT-LLM team; vLLM team
**Year:** 2023-onward
**URL:** https://github.com/NVIDIA/TensorRT-LLM + https://github.com/vllm-project/vllm
**Raw-data source:** [[raw-data/tensorrt-llm-quant]] + [[raw-data/vllm-quant]] + [[raw-data/tinychat-and-tensorrt-llm-quant]]

---

## The two production paths

TRT-LLM and vLLM are the two dominant LLM serving stacks with first-class quantization support. They cover the same quant matrix but with different deployment models:

| | TRT-LLM | vLLM |
|--|---------|------|
| **License** | NVIDIA closed-source (Apache-licensed bindings) | open-source (Apache 2.0) |
| **Engine model** | compiled TensorRT engine (GPU-SKU-specific) | runtime PyTorch with CUDA kernels |
| **Hardware** | NVIDIA only | NVIDIA + AMD (MI300X) + TPU |
| **Calibration tool** | ModelOpt (formerly AMMO) | external (AutoGPTQ, AutoAWQ, ModelOpt) |
| **Quant matrix** | FP8, INT8 SQ, INT4 GPTQ, INT4 AWQ, NVFP4 | same + bitsandbytes + gguf + compressed-tensors |
| **KV cache quant** | INT8 / FP8 | INT8 / FP8 (E4M3 / E5M2) |
| **Multi-tenant batching** | in-flight batching + chunked prefill | continuous batching + PagedAttention |
| **Best for** | NVIDIA cloud production | open-source serving, multi-tenant, multi-hardware |

---

## TensorRT-LLM — the build pipeline

```
ModelOpt (calibration) → quant_config.json + quantized weights
                        ↓
                trtllm-build → TensorRT engine (GPU-SKU-specific)
                        ↓
                ModelRunner / Triton backend → inference
```

### ModelOpt calibration

```python
import modelopt.torch.quantization as mtq

# Apply calibration
mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calib_loop)

# Or W4A16 GPTQ
mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop=calib_loop)
```

### Engine build

```bash
trtllm-build --checkpoint_dir ./calibrated \
             --output_dir ./engine \
             --use_fp8 --use_fp8_kv_cache \
             --max_batch_size 64 \
             --max_input_len 4096 \
             --max_output_len 1024
```

Engine is **GPU-SKU-specific** — separate engine build per H100 vs B100. The engine bakes in quantization decisions per layer (some layers can stay FP16/BF16 if calibration flagged them as sensitive).

### Kernel selection per format

| Format | Backend |
|--------|---------|
| FP8 | cuBLASLt FP8 GEMM (H100 WGMMA path) via Transformer Engine |
| INT8 SmoothQuant | custom CUDA plugin (`smoothQuant.cu`) |
| INT4 AWQ | AWQ GEMM kernel from `awq_ext` |
| INT4 GPTQ | Marlin-style internal kernel |
| NVFP4 | Blackwell tensor-core block-scaled GEMM |

### Fusion patterns

- GEMM + bias + activation fused as one TRT layer.
- LayerNorm + GEMM fused via `LayerNormGemm` plugin.
- Multi-head attention fused with KV cache quant via `gptAttentionPlugin`.

### NVFP4 path (Blackwell, TRT-LLM 0.13+)

Engine-build pass quantizes weights to NVFP4 (16-element FP4 blocks + E4M3 scale + FP32 tensor scale); activations cast per-block at runtime. Selective high-precision layers handled the same as the FP8 path. The Blackwell-only tensor-core code path consumes NVFP4 natively.

---

## vLLM — the unified registry

Located at `vllm/model_executor/layers/quantization/`. Auto-detects checkpoint format from `config.json`'s `quantization_config` field.

### The detection flow

```python
# vllm/config.py (simplified)
def _get_quantization_config(model_config, load_config):
    if quantization_arg is not None:
        return QuantConfig.from_args(quantization_arg)
    cfg = hf_config.get("quantization_config", {})
    method = cfg.get("quant_method")
    if method == "gptq":   return GPTQConfig.from_config(cfg)
    if method == "awq":    return AWQConfig.from_config(cfg)
    if method == "fp8":    return Fp8Config.from_config(cfg)
    if method == "compressed-tensors":
        return CompressedTensorsConfig.from_config(cfg)
    ...
```

### Supported methods

| Method | Source / format |
|--------|-----------------|
| `gptq` | GPTQ packed-int4 (legacy kernel) |
| `awq` | AWQ packed-int4 (legacy kernel) |
| `gptq_marlin` | GPTQ → Marlin (Ampere) / Machete (Hopper) |
| `awq_marlin` | AWQ → Marlin (Ampere) / Machete (Hopper) |
| `fp8` | FP8 (per-tensor / per-channel) |
| `bitsandbytes` | NF4 / INT8 from bitsandbytes |
| `gguf` | llama.cpp k-quant formats |
| `compressed-tensors` | NeuralMagic unified format |
| `nvfp4` | Blackwell NVFP4 |

### User-facing API

```python
from vllm import LLM

# Auto-detect from checkpoint
llm = LLM(model="TheBloke/Llama-2-70B-AWQ")

# Explicit, force Machete on Hopper
llm = LLM(model="meta-llama/Llama-3-8B",
          quantization="awq_marlin",
          kv_cache_dtype="fp8_e4m3")

# Full FP8 path
llm = LLM(model="meta-llama/Llama-3-8B-FP8",
          quantization="fp8",
          kv_cache_dtype="fp8")
```

### KV cache quantization (independent flag)

- **FP8 KV**: per-token scaling, stored as E4M3 (default) or E5M2.
- **INT8 KV**: per-token scaling, stored as INT8.

Both reduce KV memory ~50%; FP8 attention kernels avoid dequant before softmax.

### Linear method dispatch

Each `QuantConfig.get_quant_method(layer, prefix)` returns a `LinearMethodBase` that owns:

- `create_weights(layer, ...)` — instantiate quantized weight parameters (packed INT4 + scales + zeros).
- `apply(layer, x, bias)` — dispatch to the right CUDA kernel for the forward.
- `process_weights_after_loading(layer)` — re-pack into kernel-specific layout (e.g. Marlin re-tiles GPTQ weights).

---

## TinyChat — the AWQ edge runtime

[[tinychat-and-tensorrt-llm-quant]] documents MIT Han Lab's reference inference runtime for AWQ-quantized models. Distinctive contributions:

- First W4A16 stack to match AWQ's theoretical 4× speedup on **Jetson / consumer-GPU** hardware.
- **Lookup-table-based dequant**: each 4-bit weight indexes a per-group LUT in shared memory.
- **In-place INT4 weight storage** with on-the-fly cast directly into tensor-core registers.
- **Fused FFN**: combines gate, up, down projections + SwiGLU into one mega-kernel.
- Reference for the AWQ paper; later forked into vLLM's `awq` backend.

Use TinyChat for **single-batch consumer / edge** inference (Jetson Orin Nano runs Llama-2 7B at ~30 tokens/s). For high-batch serving the Marlin/Machete pipeline dominates.

---

## Stack-selection guide

| Hardware | Best W4A16 | Best W8A8 | Best FP8 |
|----------|-----------|-----------|----------|
| Edge / single consumer GPU | TinyChat (AWQ) | — | — |
| Open-source serving, Ampere | vLLM + Marlin | vLLM + SmoothQuant | — |
| Open-source serving, Hopper | vLLM + Machete | vLLM + SmoothQuant | vLLM FP8 / TRT-LLM |
| NVIDIA cloud production | TRT-LLM | TRT-LLM | TRT-LLM |
| Blackwell | TRT-LLM (NVFP4) | TRT-LLM | TRT-LLM |

---

## Connections

- [[marlin-kernel]] / [[machete-kernel]] / [[excerpts/marlin-kernel]] / [[excerpts/machete-kernel]] — the underlying high-throughput W4A16 GEMMs.
- [[gptq]] / [[awq]] / [[smoothquant]] / ch-08-09 — algorithms whose checkpoints these stacks consume.
- [[fp8-formats-paper]] / [[deepseek-v3-fp8]] / [[transformer-engine]] / ch-17 — FP8 deployment.
- [[nvfp4-training]] / [[nvfp4-qad]] / ch-17 — NVFP4 deployment endpoint is TRT-LLM's NVFP4 engine path.
- [[autogptq]] / [[autoawq]] — checkpoint producers vLLM is compatible with.
- [[transformer-engine-fp8]] — provides the underlying FP8 kernel library both use.
- [[ch-19]] — parent synthesis.
