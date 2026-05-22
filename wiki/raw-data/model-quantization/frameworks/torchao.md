<!-- scope: PyTorch torchao quantization utilities
     deps: [[int8]], [[int4]]
     see-also: [[hqq-framework]], [[vllm-quant]]
-->

# torchao — PyTorch Native Quantization
- **Core Insight:** torchao is PyTorch's official native quantization library, exposing low-level dtype primitives (`int4`, `mx_fp4`, `fp8`) as `torch.Tensor` subclasses and composable APIs (`quantize_(model, config)`) that play cleanly with `torch.compile` for end-to-end inference + training quantization.
- **Guideline:** Use `int4_weight_only(group_size=128)` for inference, `float8_dynamic_activation_float8_weight()` for FP8 training; rely on `torch.compile` to fuse the quant + dequant + matmul into a single kernel.
- **Authors:** PyTorch core team (Meta) + community
- **Year:** 2024 (initial standalone release; formerly part of torchao prototypes)
- **URL:** https://github.com/pytorch/ao
- **Relevant topics:** AffineQuantizedTensor, int4 tinygemm, FP8 training, MX format, torch.compile

## Summary
torchao consolidates years of PyTorch quantization research (the prior `torch.ao.quantization` namespace, `prototype/quantization`, `executorch` quant) into a single composable library. The architectural centerpiece is `AffineQuantizedTensor`, a `torch.Tensor` subclass that carries quantized data + scale/zero_point + a layout policy; it transparently handles dispatch (e.g. `__torch_dispatch__` routes `torch.matmul` calls to the right INT4 kernel). The API surface is intentionally small: `quantize_(model, config)` walks the model and swaps every applicable `nn.Linear` weight with a quantized tensor subclass. Configurations include weight-only (int4, int8, NF4), dynamic activation (int8, fp8), and full training (FP8 with delayed scaling). The library is the underpinning for SGLang's INT4 path, Diffusers' FP8 inference, and HuggingFace `transformers` `--torch-dtype torchao_int4`.

## Key Points
- Tensor-subclass-based: `AffineQuantizedTensor`, `LinearActivationQuantizedTensor`, `NF4Tensor`.
- `quantize_(model, config)` walks model and swaps weights in-place.
- Configs: `int4_weight_only`, `int8_weight_only`, `int8_dynamic_activation_int8_weight`, `float8_dynamic_activation_float8_weight`, `float8_weight_only`, `fp6_e3m2_weight_only`, MX format prototypes.
- INT4 path uses CUTLASS `tinygemm` kernel.
- FP8 path uses scaled `_scaled_mm` op (CUDA).
- Designed to be `torch.compile`-friendly — quant ops are tracable and fusible.

## Technical Details

### Repository layout
- repo: `https://github.com/pytorch/ao`
- public API: `torchao/quantization/quant_api.py` — `quantize_()`, all `*_weight_only` configs.
- tensor subclasses: `torchao/dtypes/affine_quantized_tensor.py`, `torchao/dtypes/nf4tensor.py`.
- INT4 kernel binding: `torchao/quantization/quant_primitives.py` (`_choose_qparams_affine`, `_quantize_affine`).
- INT4 tinygemm kernel: `torchao/csrc/cuda/tensor_core_tiled_layout/` (CUTLASS-based).
- FP8 training: `torchao/float8/` (`Float8Linear`, `Float8DynamicLinear`, sync/async scaling).
- MX prototypes: `torchao/prototype/mx_formats/`.

### Quantization rule (`int4_weight_only`)
For weight `W ∈ R^{out×in}`, group size `G = 128`:
```
W_grouped = W.reshape(out, in/G, G)
scale = max(|W_grouped|, dim=-1) / 7              # symmetric INT4 → [-8, 7]
W_int4 = round(W_grouped / scale).clamp(-8, 7)    # INT4
# Pack: tinygemm-tiled layout (interleaved for tensor-core mma)
```

### Key APIs
- `torchao.quantization.quantize_(model, int4_weight_only(group_size=128))` — top-level entry.
- `class AffineQuantizedTensor(torch.Tensor)` — tensor subclass holding `int_data`, `scale`, `zero_point`, `_layout`.
- `int8_dynamic_activation_int8_weight()` — config: dynamic per-token INT8 act + INT8 weight, accumulate INT32.
- `float8_dynamic_activation_float8_weight(weight_dtype=torch.float8_e4m3fn)` — FP8 act + FP8 weight, accumulate FP32.
- `Float8Linear` — replacement nn.Linear for FP8 training (delayed or dynamic scaling).
- `autoquant(model, example_input)` — micro-benchmark each layer to pick the best quant per-layer.

### Config / hyperparameters
| Config | Bits (W/A) | Notes |
|--------|-----------|-------|
| `int4_weight_only(group_size=128)` | W4/A16 | inference; CUTLASS tinygemm |
| `int8_weight_only()` | W8/A16 | inference; simple `int8 @ bf16` |
| `int8_dynamic_activation_int8_weight()` | W8/A8 | dynamic per-token act quant |
| `float8_weight_only(weight_dtype=fn)` | W8/A16 | FP8 weight, BF16 act |
| `float8_dynamic_activation_float8_weight()` | W8/A8 | full FP8 inference |
| `gemlite_uintx_weight_only(bit_width=4)` | W4/A16 | GemLite kernel backend |

### FP8 training (Float8Linear)
- Replaces `nn.Linear` with `Float8Linear`.
- Tracks per-tensor scale via `Float8LinearConfig.cast_config_*`.
- Two scaling modes: `dynamic` (compute amax each step) and `delayed` (16-step history, like TE).
- Compatible with FSDP2 and DTensor.

## Connections
- [[hqq-framework]] — torchao is an HQQ inference backend (`int4` tinygemm).
- [[vllm-quant]] — vLLM has a torchao integration path.
- [[transformer-engine-fp8]] — sibling FP8 training stack from NVIDIA.
