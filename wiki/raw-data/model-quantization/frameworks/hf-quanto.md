<!-- scope: HuggingFace Quanto quantization library
     deps: [[int8]], [[int4]]
     see-also: [[torchao]], [[bitsandbytes-int8]]
-->

# HuggingFace Quanto / Optimum-Quanto
- **Core Insight:** Quanto is HuggingFace's PyTorch-native quantization library designed to work uniformly across CPU, CUDA, MPS, and XPU with a single API, prioritizing portability over peak-throughput specialization.
- **Guideline:** Use Quanto when you need quantization on heterogeneous hardware (Apple Silicon, Intel XPU, CPU); switch to bitsandbytes or torchao on CUDA-only deployments for better kernel performance.
- **Authors:** HuggingFace Optimum team (David Corvoysier and collaborators)
- **Year:** 2024
- **URL:** https://github.com/huggingface/optimum-quanto
- **Relevant topics:** cross-device quantization, QTensor, weight + activation quant

## Summary
Quanto (recently renamed `optimum-quanto` to fit under the HuggingFace Optimum umbrella) is a PyTorch-native quant library that emphasizes device-portability. Its `QTensor` subclass holds INT2/INT4/INT8 weights with per-axis scales, and `QLinear` / `QConv2d` replacements transparently dispatch to vectorized kernels on whichever backend PyTorch supports. The library exposes a small `quantize(model, weights, activations)` API; it supports FP8 (E4M3 / E5M2), INT8, INT4, and INT2 for weights, and FP8 / INT8 for activations. Quanto trades peak throughput (vs Marlin / bitsandbytes specialized kernels) for cross-device coverage — a quantized Quanto checkpoint runs unchanged on H100, M2 Max, Xeon SPR, or Intel Arc.

## Key Points
- `QTensor` subclass holds INT or FP8 data + per-axis scales.
- `QLinear` / `QConv2d` drop-in replacements.
- Works across CUDA, MPS (Apple Silicon), CPU, Intel XPU.
- Supports W2/W4/W8/FP8 weights, FP8/INT8 activations.
- Calibration loop via `Calibration` context manager.
- HF Transformers integration via `QuantoConfig`.

## Technical Details

### Repository layout
- repo: `https://github.com/huggingface/optimum-quanto`
- main API: `optimum/quanto/quantize.py` — `quantize(model, weights=qint4, activations=qint8)`.
- tensor subclass: `optimum/quanto/tensor/qtensor.py`, `optimum/quanto/tensor/qbits.py`.
- module replacements: `optimum/quanto/nn/qlinear.py`, `optimum/quanto/nn/qconv2d.py`.
- calibration: `optimum/quanto/calibrate.py` — `Calibration()` context manager.
- HF integration: `transformers/quantizers/quantizer_quanto.py` (in transformers repo).

### Quantization rule (weight)
For weight `W ∈ R^{out×in}`, per-output-channel scale:
```
s = max(|W|, dim=in) / int_max          # one scalar per output channel
W_int = round(W / s).clamp(int_min, int_max)
W_q = QTensor(W_int, scale=s, dtype=qint4 or qint8)
```

Forward:
```
y = (QTensor @ activation_qtensor).dequantize()
```
or in fused form depending on backend.

### Key APIs
- `quantize(model, weights=qint4, activations=qint8, exclude=["lm_head"])` — wrap modules with QLinear.
- `freeze(model)` — collapse `QTensor` representations and discard FP32 master weights.
- `Calibration(momentum=0.9)` — context manager during which observers record activation ranges.
- `class QTensor(torch.Tensor)` — INT-typed tensor with stored scale.
- `class QBitsTensor` — sub-byte INT2/INT4 tensor with packed layout.

### Config / hyperparameters
| Knob | Options | Notes |
|------|---------|-------|
| `weights` | `qint2` / `qint4` / `qint8` / `qfloat8_e4m3fn` / `qfloat8_e5m2` | weight dtype |
| `activations` | `None` / `qint8` / `qfloat8_e4m3fn` / `qfloat8_e5m2` | activation dtype; None = weight-only |
| `exclude` | list of module names | skip e.g. `lm_head` |
| `group_size` | None | per-axis (per-channel) only; no group quant currently |
| momentum (Calibration) | 0.9 | EMA over per-tensor max |

### HF Transformers entry
```python
from transformers import AutoModelForCausalLM, QuantoConfig

quant_config = QuantoConfig(weights="int4", activations=None)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=quant_config,
    device_map="auto",
)
```

### Backend dispatch
- CUDA: uses `torch._weight_int4pack_mm` (PyTorch native INT4 packed matmul) when available.
- MPS: falls back to dequantize + FP16 matmul (since Apple Silicon lacks INT4 matmul).
- CPU: AVX-512 / NEON SIMD path through PyTorch's `aten` ops.
- Intel XPU: routed through Intel Extension for PyTorch.

## Connections
- [[torchao]] — similar PyTorch-native approach but CUDA-optimized; Quanto is more portable.
- [[bitsandbytes-int8]] — older CUDA-only competitor.
- [[autogptq]] / [[autoawq]] — algorithmically richer alternatives (Hessian / activation-aware) but CUDA-bound.
