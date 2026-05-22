<!-- scope: NVIDIA TensorRT-LLM quantization pipeline
     deps: [[fp8-e4m3]], [[gptq]], [[awq]]
     see-also: [[transformer-engine-fp8]], [[vllm-quant]]
-->

# TensorRT-LLM — Quantization Pipeline
- **Core Insight:** TensorRT-LLM consolidates GPTQ, AWQ, SmoothQuant, FP8, and NVFP4 into a single optimizer pass that fuses quantized GEMMs with attention, MLP, and norm into a compiled TensorRT engine targeted at a specific GPU SKU.
- **Guideline:** Use `ModelOpt` (formerly AMMO) for the calibration pass that emits TensorRT-LLM-ready quant configs; then `trtllm-build` compiles the engine with the right kernel selection.
- **Authors:** NVIDIA TensorRT-LLM team
- **Year:** 2023 (initial release); ongoing
- **URL:** https://github.com/NVIDIA/TensorRT-LLM
- **Relevant topics:** TensorRT engine compilation, FP8, INT8/INT4, kernel fusion, ModelOpt

## Summary
TensorRT-LLM is NVIDIA's production LLM inference stack built on top of TensorRT. Its quantization story spans the matrix of FP8 (E4M3, both per-tensor and per-channel SmoothQuant), INT8 SmoothQuant, INT4 GPTQ, INT4 AWQ, and NVFP4 (Blackwell). The calibration pass lives in `ModelOpt` (formerly AMMO, AlgorithMic Model Optimizer), which produces a `quant_config.json` and re-saves the model with quantized weights. `trtllm-build` then compiles a TensorRT engine targeting a specific GPU SKU (H100/B100/etc.), selecting kernels from the cuBLASLt + cuDNN + custom plugin pools. The engine is loaded via the `tensorrt_llm.runtime.ModelRunner` Python wrapper or the Triton Inference Server backend.

## Key Points
- Calibration in `ModelOpt`; engine compile via `trtllm-build`.
- Five main quant modes: FP8, INT8 SQ, INT4 GPTQ, INT4 AWQ, NVFP4 (Blackwell).
- Engine is GPU-SKU-specific (separate build per H100 vs B100).
- FP8 path uses Transformer Engine kernels via cuBLASLt.
- INT4 path uses Marlin-style W4A16 kernels.
- KV cache quantization separate: INT8 and FP8 KV cache.

## Technical Details

### Repository layout
- repo: `https://github.com/NVIDIA/TensorRT-LLM`
- ModelOpt (calibrator): separate repo `https://github.com/NVIDIA/TensorRT-Model-Optimizer`
- model definitions: `tensorrt_llm/models/<arch>/` — quantization-aware re-implementations
- quantization API: `tensorrt_llm/quantization/__init__.py`, `tensorrt_llm/quantization/quantize.py`
- Layer quant wrappers: `tensorrt_llm/quantization/layers.py`
- example pipelines: `examples/<model>/README.md` + `examples/<model>/quantize.py`

### Quant config schema
```python
from tensorrt_llm.quantization import QuantMode

quant_mode = QuantMode.from_description(
    quantize_weights=True,
    quantize_activations=True,
    per_token=True,
    per_channel=True,
    use_int4_weights=False,
    use_int8_kv_cache=False,
    use_fp8_kv_cache=True,
    use_fp8_qdq=True,
    use_fp8_rowwise=False,
)
```

### Key APIs
- `modelopt.torch.quantization.quantize(model, config, forward_loop)` — apply calibration.
- `trtllm.build(model_dir, output_dir, max_batch_size, max_input_len, max_output_len, use_fp8=True)` — engine build.
- `tensorrt_llm.runtime.ModelRunner.from_dir(engine_dir)` — load engine.
- `tensorrt_llm.runtime.ModelRunner.generate(input_ids, ...)` — inference.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `--use_fp8` | False | enable FP8 GEMM + KV cache |
| `--use_smooth_quant` | False | INT8 SmoothQuant; requires calibration |
| `--use_weight_only` | False | INT8 / INT4 weight-only (no activation quant) |
| `--weight_only_precision` | int8 | int4_awq, int4_gptq, int8 |
| `--per_channel`/`--per_token` | False | finer granularity for SmoothQuant |
| `--int8_kv_cache` / `--fp8_kv_cache` | False | KV cache quant |
| `--max_num_tokens` | 8192 | engine batching budget |

### Kernel selection
- FP8: cuBLASLt FP8 GEMM (H100 WGMMA path).
- INT8 SmoothQuant: custom CUDA plugin (`smoothQuant.cu`).
- INT4 AWQ: AWQ GEMM kernel from `awq_ext`.
- INT4 GPTQ: Marlin-style GEMM.
- NVFP4: Blackwell tensor-core block-scaled GEMM.

### Fusion patterns
- GEMM + bias + activation fused as a single TRT layer.
- LayerNorm + GEMM fused via `LayerNormGemm` plugin.
- Multi-head attention fused with KV cache quant via `gptAttentionPlugin`.

## Connections
- [[fp8-e4m3]] — main FP8 format used by the engine.
- [[transformer-engine-fp8]] — provides the underlying FP8 kernel library.
- [[gptq]] / [[awq]] — supported W4 algorithms.
- [[vllm-quant]] — competitor inference engine with overlapping quant support.
