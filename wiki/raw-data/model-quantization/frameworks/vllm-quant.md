<!-- scope: vLLM integration of GPTQ/AWQ/FP8/Marlin quantization
     deps: [[gptq]], [[awq]], [[marlin-kernel]]
     see-also: [[tensorrt-llm-quant]], [[autogptq]], [[autoawq]]
-->

# vLLM — Quantization Integration
- **Core Insight:** vLLM exposes a uniform `quantization=` flag that auto-detects checkpoint format (GPTQ, AWQ, FP8, Marlin, GGUF, BNB, NF4) and dispatches to the right CUDA kernel inside the PagedAttention engine, making quantized inference as easy as `LLM(model, quantization="awq")`.
- **Guideline:** Prefer Marlin (GPTQ-Marlin / AWQ-Marlin) on Ampere/Hopper for the best throughput; use FP8 (`quantization="fp8"`) on H100/MI300X for compute-bound workloads.
- **Authors:** vLLM team (UC Berkeley + Anyscale + NeuralMagic)
- **Year:** 2023 (initial), continuously expanding
- **URL:** https://github.com/vllm-project/vllm
- **Relevant topics:** PagedAttention, GPTQ-Marlin, AWQ-Marlin, FP8 KV cache, INT8 KV cache

## Summary
vLLM is the highest-throughput open-source LLM inference engine, built around PagedAttention for memory-efficient KV cache management. Its quantization story is a unified registry under `vllm/model_executor/layers/quantization/` that supports GPTQ, AWQ, GPTQ-Marlin, AWQ-Marlin, FP8 (per-tensor and per-channel), BitsAndBytes (NF4/INT8), GGUF (llama.cpp formats), Compressed Tensors (a unified format from NeuralMagic), TPU-INT8, and NVFP4. The runtime detects the checkpoint format via `config.json`'s `quantization_config` field and instantiates the correct `QuantizationConfig` subclass; this constructs the appropriate `LinearMethodBase` that intercepts every linear-layer call. KV cache quantization is independently configurable (INT8 or FP8). Marlin kernels (W4A16 GEMM at near-FP16 throughput) are the default fast path for both GPTQ and AWQ checkpoints.

## Key Points
- Unified `quantization=` flag covering ~10 different formats.
- Auto-detection from checkpoint `config.json` `quantization_config`.
- Marlin kernels for W4A16 at near-FP16 throughput.
- Separate KV cache quant (`kv_cache_dtype="fp8"` or `"int8"`).
- Speculative decoding works with quantized targets.
- PagedAttention manages KV cache in blocks; quantization integrated at block-write time.

## Technical Details

### Repository layout
- repo: `https://github.com/vllm-project/vllm`
- quant registry: `vllm/model_executor/layers/quantization/__init__.py` — maps name → config class.
- per-format configs: `vllm/model_executor/layers/quantization/gptq.py`, `awq.py`, `marlin.py`, `gptq_marlin.py`, `awq_marlin.py`, `fp8.py`, `bitsandbytes.py`, `gguf.py`, `compressed_tensors/`.
- Marlin kernel sources: `csrc/quantization/marlin/` (CUDA) and `csrc/quantization/awq/` (CUDA).
- FP8 path: `csrc/quantization/fp8/` and the cuBLASLt FP8 GEMM wrapper.
- KV cache quant: `vllm/attention/backends/` (FP8 KV via cuDNN FP8 attention path).

### Detection flow
```python
# vllm/config.py
def _get_quantization_config(model_config, load_config):
    # 1. CLI override
    if quantization_arg is not None:
        return QuantConfig.from_args(quantization_arg)
    # 2. Read from HF config.json
    cfg = hf_config.get("quantization_config", {})
    method = cfg.get("quant_method")
    if method == "gptq":   return GPTQConfig.from_config(cfg)
    if method == "awq":    return AWQConfig.from_config(cfg)
    if method == "fp8":    return Fp8Config.from_config(cfg)
    if method == "compressed-tensors":  return CompressedTensorsConfig.from_config(cfg)
    ...
```

### Linear method dispatch
Each `QuantConfig.get_quant_method(layer, prefix)` returns a `LinearMethodBase` that owns:
- `create_weights(layer, ...)` — instantiate quantized weight parameters (packed INT4 + scales + zeros).
- `apply(layer, x, bias)` — dispatch to the right CUDA kernel for the forward.
- `process_weights_after_loading(layer)` — re-pack into kernel-specific layout (e.g. Marlin re-tiles GPTQ weights).

### Key APIs (user-facing)
- `LLM(model, quantization="awq", dtype="auto")` — load AWQ checkpoint.
- `LLM(model, quantization="fp8", kv_cache_dtype="fp8")` — full FP8 path.
- `LLM(model, quantization="gptq_marlin")` — explicit Marlin backend for GPTQ.

### Config / hyperparameters
| CLI arg | Default | Notes |
|---------|---------|-------|
| `--quantization` | None | auto-detect; or `awq`, `gptq`, `gptq_marlin`, `awq_marlin`, `fp8`, `bitsandbytes`, `gguf`, … |
| `--kv-cache-dtype` | auto | `fp8`, `fp8_e5m2`, `fp8_e4m3`, `int8` |
| `--load-format` | auto | `safetensors`, `bitsandbytes`, `gguf`, `mistral`, `runai_streamer` |
| `--quantization-param-path` | None | static FP8 KV scales JSON |

### KV cache quantization
- FP8 KV: per-token scaling, stored as E4M3 (default) or E5M2.
- INT8 KV: per-token scaling, stored as INT8.
- Both reduce KV memory ~50%; FP8 attention kernels avoid dequant before softmax.

### Marlin kernel selection
- Default for GPTQ checkpoints with `bits=4, group_size ∈ {-1, 128}, sym=True`.
- Falls back to legacy `qlinear_cuda` if the checkpoint shape isn't Marlin-compatible.

## Connections
- [[gptq]] / [[awq]] — algorithms whose checkpoints vLLM consumes.
- [[marlin-kernel]] — default high-throughput W4A16 backend.
- [[autogptq]] / [[autoawq]] — checkpoint producers vLLM is compatible with.
- [[tensorrt-llm-quant]] — competitor with similar quant matrix coverage.
