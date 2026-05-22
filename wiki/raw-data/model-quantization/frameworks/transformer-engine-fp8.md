<!-- scope: NVIDIA Transformer Engine FP8 framework code-level reference
     deps: [[fp8-e4m3]], [[fp8-e5m2]]
     see-also: [[transformer-engine-blog]], [[megatron-fp8]]
-->

# Transformer Engine — FP8 Implementation Details
- **Core Insight:** Transformer Engine's FP8 path is built around a small core of three primitives — `Float8Tensor` (FP8-typed tensor with scale + amax history), `fp8_gemm` (cublasLt FP8 GEMM wrapper), and the `DelayedScaling` recipe — wired into a registry that maps high-level modules (Linear, LayerNorm, Attention) to FP8-aware implementations.
- **Guideline:** Read `transformer_engine/pytorch/module/linear.py` first; everything else (LayerNormLinear, LayerNormMLP, TransformerLayer) is a fusion built on top of the same Linear FP8 primitives.
- **Authors:** NVIDIA Transformer Engine team
- **Year:** 2022 (TE 0.x), 2024 (TE 1.x with MXFP)
- **URL:** https://github.com/NVIDIA/TransformerEngine
- **Relevant topics:** Float8Tensor, fp8_gemm, DelayedScaling, amax history, cuBLASLt

## Summary
Transformer Engine (TE) is the official NVIDIA library for FP8/FP6/FP4 training and inference on Hopper and Blackwell. It ships PyTorch, JAX, and PaddlePaddle frontends; the PyTorch path is the most mature. The library has two layers: (1) a C++/CUDA core under `transformer_engine/common/` exposing `Float8Tensor`, FP8 GEMM (`fp8_gemm`), FP8 activation kernels (gelu, swiglu), and FP8 attention bindings into cuDNN; (2) a Python module layer under `transformer_engine/pytorch/module/` that wraps the core into drop-in `nn.Module` replacements managing scale state through the `FP8GlobalStateManager`. The state manager is the load-bearing piece: it owns amax buffers, scale buffers, and the recipe (DelayedScaling, MXFP8BlockScaling, Float8CurrentScaling), and synchronizes them across data-parallel ranks.

## Key Points
- C++/CUDA core in `common/`, PyTorch wrappers in `pytorch/`.
- `Float8Tensor` stores `_data` (raw FP8 bytes), `_scale_inv` (FP32), `_fp8_dtype` (E4M3 or E5M2).
- `FP8GlobalStateManager` tracks amax history per tensor across all FP8 modules.
- DelayedScaling recipe drives scale derivation from amax history.
- TE 1.x adds `MXFP8BlockScaling` recipe and `MXFP8Tensor` for Blackwell.
- FP8 attention via cuDNN `cudnnFusedFlashAttnForward` since cuDNN 9.

## Technical Details

### Repository layout
- repo: `https://github.com/NVIDIA/TransformerEngine`
- C++/CUDA core: `transformer_engine/common/` (FP8 GEMM, casts, activations, layer norms)
- key kernel: `transformer_engine/common/gemm/cublaslt_gemm.cu` (FP8 GEMM via cuBLASLt)
- FP8 cast: `transformer_engine/common/util/cast.cu` (`cast_to_fp8`, `cast_from_fp8`)
- Python modules: `transformer_engine/pytorch/module/` (`linear.py`, `layernorm_linear.py`, `layernorm_mlp.py`, `transformer.py`)
- recipe: `transformer_engine/common/recipe.py` (`DelayedScaling`, `MXFP8BlockScaling`, `Float8CurrentScaling`)
- state manager: `transformer_engine/pytorch/fp8.py` (`FP8GlobalStateManager`)
- tensor subclass: `transformer_engine/pytorch/tensor/float8_tensor.py` (`Float8Tensor`)
- MX tensor: `transformer_engine/pytorch/tensor/mxfp8_tensor.py`

### Float8Tensor
```python
class Float8Tensor(torch.Tensor):
    _data: torch.Tensor          # uint8 underlying bytes
    _fp8_dtype: tex.DType        # kFloat8E4M3 or kFloat8E5M2
    _scale_inv: torch.Tensor     # FP32, shape ()
    _fp8_meta: dict              # links to amax / scale buffers
```

### FP8 GEMM wrapper
```python
# transformer_engine/pytorch/cpp_extensions/gemm.py
def fp8_gemm(A, A_scale_inv, A_fp8_tensor, A_fp8_dtype,
             B, B_scale_inv, B_fp8_tensor, B_fp8_dtype,
             out_dtype, workspace, ...):
    # Dispatches to cuBLASLt FP8 GEMM
    # Returns BF16/FP32 output tensor
```

### DelayedScaling recipe (algorithm)
```python
# Per training step, per tensor:
amax_history[t % H] = current_amax        # H = amax_history_len (default 16)
new_amax = max(amax_history)              # reduce
scale = fp8_max / (new_amax * (1 + margin))
scale_inv = 1 / scale
# Save scale_inv into the Float8Tensor metadata for next forward.
```

### Key APIs
- `te.Linear(in_features, out_features, bias=True, init_method=...)` — drop-in nn.Linear.
- `te.LayerNormLinear(in_features, out_features, eps)` — fused LN + Linear (one fused FP8 cast).
- `te.LayerNormMLP(hidden_size, ffn_hidden_size)` — fused LN + GEMM + GELU + GEMM.
- `te.TransformerLayer(...)` — full pre/post-LN transformer block.
- `te.MultiheadAttention(num_attention_heads, kv_channels, attention_dropout=0.0)` — FP8 attention with cuDNN.
- `te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(...))` — context manager.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `fp8_format` | HYBRID | E4M3 fwd / E5M2 bwd |
| `amax_history_len` | 1024 (1.x default) | larger = smoother scale |
| `amax_compute_algo` | "max" | "max" or "most_recent" |
| `margin` | 0 | extra safety factor in scale calc |
| `reduce_amax` | True | all-reduce amax across DP |
| `interval` | 1 | how often to update scale |

### Attention path
- cuDNN FP8 attention: Q/K/V cast to E4M3, softmax in FP32, AV in E4M3.
- Per-head FP8 scales for Q, K, V independently.
- Output projection also FP8.

## Connections
- [[transformer-engine-blog]] — higher-level practitioner overview.
- [[fp8-e4m3]] / [[fp8-e5m2]] — operand formats.
- [[megatron-fp8]] — Megatron-LM consumes TE's FP8 modules.
- [[nvidia-h100-fp8]] — hardware substrate.
