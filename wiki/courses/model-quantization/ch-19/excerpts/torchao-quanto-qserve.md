---
chapter: ch-19
course: model-quantization
phase: read
excerpt_of: "torchao (PyTorch, 2024) + HF Quanto / Optimum-Quanto (HuggingFace, 2024) + QServe / QoQ (Lin, Tang, Han et al., MIT 2024)"
source_url: https://github.com/pytorch/ao + https://github.com/huggingface/optimum-quanto + https://arxiv.org/abs/2405.04532
created_at: "2026-05-21"
---

# Excerpt: torchao + Quanto + QServe — PyTorch ecosystem and W4A8KV4

**Authors:** PyTorch core team + community (torchao); HuggingFace Optimum team (Quanto); Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han (QServe)
**Year:** 2024
**URL:** as above
**Raw-data source:** [[raw-data/torchao]] + [[raw-data/hf-quanto]] + [[raw-data/qserve]]

---

## torchao — PyTorch native quantization

[[torchao]] is PyTorch's official native quantization library. It consolidates years of PyTorch quantization work (`torch.ao.quantization`, `prototype/quantization`, `executorch` quant) into a single composable library.

### The centerpiece: AffineQuantizedTensor

A `torch.Tensor` subclass that carries quantized data + scale/zero-point + a layout policy. `__torch_dispatch__` transparently routes `torch.matmul` calls to the right INT4 kernel.

```python
class AffineQuantizedTensor(torch.Tensor):
    int_data: torch.Tensor       # packed INT data
    scale: torch.Tensor
    zero_point: torch.Tensor
    _layout: LayoutType          # which kernel layout to use
```

### The API surface

Intentionally small: `quantize_(model, config)` walks the model and swaps every applicable `nn.Linear` weight with a quantized tensor subclass.

```python
from torchao.quantization import (
    quantize_,
    int4_weight_only,
    float8_dynamic_activation_float8_weight,
)

# Inference: W4A16 with group_size=128
quantize_(model, int4_weight_only(group_size=128))

# FP8 dynamic act + FP8 weight (W8A8 in FP8 land)
quantize_(model, float8_dynamic_activation_float8_weight())

# FP8 training (drop-in nn.Linear replacement)
from torchao.float8 import Float8Linear
```

### Configs

| Config | Bits (W/A) | Notes |
|--------|-----------|-------|
| `int4_weight_only(group_size=128)` | W4/A16 | inference; CUTLASS tinygemm |
| `int8_weight_only()` | W8/A16 | inference; simple int8 @ bf16 |
| `int8_dynamic_activation_int8_weight()` | W8/A8 | dynamic per-token act quant |
| `float8_dynamic_activation_float8_weight()` | W8/A8 | full FP8 inference |
| `float8_weight_only(weight_dtype=fn)` | W8/A16 | FP8 weight, BF16 act |

### `torch.compile`-friendly

Quant ops are tracable and fusible — `torch.compile` can fuse quant + dequant + matmul into a single kernel.

### FP8 training (Float8Linear)

```python
from torchao.float8 import Float8Linear, Float8LinearConfig

# Two scaling modes: dynamic (compute amax each step) and delayed (16-step history)
config = Float8LinearConfig(...)
model = swap_linear_with_float8_linear(model, config)
```

Compatible with FSDP2 and DTensor.

### Where torchao fits

- **CUDA-optimized.** Uses CUTLASS `tinygemm` for INT4 and `_scaled_mm` for FP8.
- **PyTorch-first.** Designed to interoperate with `torch.compile`, FSDP2, DTensor.
- **SGLang / Diffusers / HF transformers backend.** `--torch-dtype torchao_int4` works in HF.

---

## HuggingFace Quanto — cross-device portable

[[hf-quanto]] (Optimum-Quanto) prioritizes **cross-device portability** over peak throughput.

### The QTensor subclass

Holds INT2/INT4/INT8 + FP8 (E4M3/E5M2) weights with per-axis scales. `QLinear` / `QConv2d` drop-in replacements dispatch to vectorized kernels on whichever backend PyTorch supports (CUDA, MPS, CPU, Intel XPU).

### HF Transformers integration

```python
from transformers import AutoModelForCausalLM, QuantoConfig

quant_config = QuantoConfig(weights="int4", activations=None)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=quant_config,
    device_map="auto",
)
```

### Configs

| Knob | Options |
|------|---------|
| `weights` | `qint2` / `qint4` / `qint8` / `qfloat8_e4m3fn` / `qfloat8_e5m2` |
| `activations` | `None` / `qint8` / `qfloat8_e4m3fn` / `qfloat8_e5m2` |
| `exclude` | list of module names — skip e.g. `lm_head` |

### Backend dispatch

- **CUDA**: uses `torch._weight_int4pack_mm` (PyTorch native INT4 packed matmul) when available.
- **MPS**: falls back to dequantize + FP16 matmul (Apple Silicon lacks INT4 matmul).
- **CPU**: AVX-512 / NEON SIMD path through PyTorch's `aten` ops.
- **Intel XPU**: routed through Intel Extension for PyTorch.

### Trade-off

Quanto trades peak throughput (vs Marlin / bitsandbytes specialized kernels) for **cross-device coverage**. A quantized Quanto checkpoint runs unchanged on H100, M2 Max, Xeon SPR, or Intel Arc.

**Use Quanto when:** you need heterogeneous hardware support. **Use torchao or bitsandbytes when:** CUDA-only and you want maximum throughput.

---

## QServe / QoQ — the W4A8KV4 production path

[[qserve]] co-designs the quantization scheme and the kernel for **Hopper W4A8KV4** serving. The diagnosis: naïve INT4 dequantization on GPUs incurs 20–90% runtime overhead because dequant happens at the wrong memory hierarchy level (HBM ↔ SMEM).

### Progressive group quantization (the key idea)

- **Stage 1 (per-channel INT8):** quantize each output channel of W to INT8 with a single per-channel scale `s_c ∈ FP16`. Store as INT8 `W_s`.
- **Stage 2 (per-group INT4):** within each group of g=128 weights along the input axis, quantize the INT8 values to INT4 with a per-group scale `s_g ∈ INT8` (note: **scale is itself an integer, not FP**). Store as INT4 `W_g`.
- **At inference:** dequantize INT4 → INT8 entirely **in registers** (multiply by `s_g`, an INT8×INT8 → INT16 op cheap on tensor cores); feed INT8 operand into the INT8 tensor-core GEMM with A8 activation. **No FP dequant in the critical path.**

### Why this matters

Prior W4A8 (e.g. Marlin) dequantizes INT4 → FP16 in registers and feeds FP16 into the tensor core. The FP dequant adds 2-3 instructions per element and pushes register pressure. **QoQ keeps everything integer until the GEMM accumulator.**

### SmoothAttention

Attention `softmax(QK^T/√d)V` is sensitive to KV4 because INT4 K introduces noise that gets amplified by softmax. QoQ applies a SmoothQuant-style learnable per-head scaling:

```
Q' = Q · s,  K' = K / s
```

`QK^T` is unchanged but `K'` has reduced dynamic range, making K4 quantization gentler. `s` is calibrated to minimize softmax KL divergence.

### Throughput

- Llama-3-8B: 1.2× over TensorRT-LLM W8A8 on H100, 2.4× on L40S.
- Qwen1.5-72B: 3.5× over Atom W4A4 on A100 (Atom is hurt by softmax instability from A4).

### Where it sits

- **W4A8 vs W4A4:** A8 keeps tensor-core utilization high; A4 introduces softmax instability that A8 avoids.
- **W4A8KV4:** the sweet spot for Hopper — W4 saves HBM bandwidth (the dominant decode cost), A8 keeps compute path stable, KV4 saves the KV-cache bandwidth.

### Why include it in the kernel chapter

QServe is the cleanest example of **algorithm-kernel co-design** in the 2024 quant literature. Progressive group quantization was *invented* to make register-level dequant work — the algorithm is a consequence of the kernel constraint, not the other way around. Read it as a model of how production quant should be designed in 2026.

---

## How the three relate

- **torchao**: PyTorch-native CUDA path. Best for new PyTorch projects.
- **Quanto**: cross-device PyTorch path. Best for heterogeneous deployment.
- **QServe**: research/system co-design exemplar. Best for understanding *how* to push to W4A8KV4 on Hopper.

vLLM / TRT-LLM consume all three: torchao as a backend, Quanto via HF integration, QServe as an inspiration for in-house W4A8KV4 paths.

---

## Connections

- [[torchao]] / [[hf-quanto]] / [[qserve]] / [[raw-data/torchao]] / [[raw-data/hf-quanto]] / [[raw-data/qserve]] — raw-data sources.
- [[smoothquant]] / ch-09 — SmoothAttention is a SmoothQuant variant for attention.
- [[gptq]] / [[awq]] / ch-08-09 — used as W4 baselines for QServe.
- [[marlin-kernel]] / [[excerpts/marlin-kernel]] — direct W4A8 ancestor for the dequant strategy.
- [[kivi]] / [[kvquant]] / ch-15 — KV-quant siblings.
- [[atom]] / ch-14 — sibling W4A4KV4 system (which QServe outperforms due to A4's softmax instability).
- [[transformer-engine-fp8]] / [[excerpts/transformer-engine]] (ch-17) — sibling FP8 training/inference stack from NVIDIA.
- [[ch-19]] — parent synthesis.
