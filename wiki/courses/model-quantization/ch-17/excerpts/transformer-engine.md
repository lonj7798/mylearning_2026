---
chapter: ch-17
course: model-quantization
phase: read
excerpt_of: "NVIDIA Transformer Engine (FP8 Mixed Precision Library, 2022-onward)"
source_url: https://github.com/NVIDIA/TransformerEngine
created_at: "2026-05-21"
---

# Excerpt: Transformer Engine — DelayedScaling and the FP8 recipe API

**Authors:** NVIDIA Transformer Engine team (Micikevicius, Casper, Korthikanti, et al.)
**Year:** 2022 onward (current: TE 2.x)
**URL:** https://github.com/NVIDIA/TransformerEngine
**Raw-data source:** [[raw-data/transformer-engine]] + [[raw-data/transformer-engine-fp8]]

---

## What TE is

Transformer Engine (TE) is NVIDIA's reference library for FP8 (and now NVFP4 / MXFP) training on Hopper / Blackwell. It ships drop-in `te.Linear`, `te.LayerNorm`, `te.Attention` modules that internally cast to FP8 via the H100/B200 Tensor Cores while keeping all model-facing math in BF16/FP16.

Two key abstractions:

1. The **FP8 recipe** — which tensors are E4M3 vs E5M2, how scales are computed, when amax history is updated.
2. The **scaling strategy** — per-tensor delayed scaling, per-tensor current scaling, or per-block scaling on Blackwell.

TE is the production baseline that [[deepseek-v3-fp8]] measured itself against in §3.3 (and beat by switching to per-block scales). TE 2.x has since absorbed per-block scaling as a first-class recipe.

---

## DelayedScaling — the algorithm

```python
# Per training step, per tensor:
amax_history[t % H] = current_amax        # H = amax_history_len (default 1024 in TE 1.x)
new_amax = max(amax_history)              # reduce over history window
scale = fp8_max / (new_amax * (1 + margin))
scale_inv = 1 / scale
# Save scale_inv into the Float8Tensor metadata for next forward.
```

The **delayed** trick: the scale is computed from *past* amax samples, so it's ready before the next step's GEMM launches. No synchronization between amax reduction and the GEMM stream.

Trade-off: if activations drift quickly, the scale lags and the FP8 tensor clips. `CurrentScaling` synchronizes and uses the current step's amax — slightly higher fidelity, slight overhead.

---

## Format split (HYBRID recipe)

```python
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID,
                        amax_history_len=1024, amax_compute_algo="max")
```

- **Forward (E4M3):** weights + activations cast to E4M3 before the GEMM; output dequantized to BF16/FP16 immediately after.
- **Backward (E5M2):** activation gradient `dY` cast to E5M2 (wider tail); weight gradient is BF16 output for safety.
- BF16/FP32 promotion happens inside the Tensor Core; the user-facing dtype stays BF16.

---

## Float8Tensor

```python
class Float8Tensor(torch.Tensor):
    _data: torch.Tensor          # uint8 underlying bytes
    _fp8_dtype: tex.DType        # kFloat8E4M3 or kFloat8E5M2
    _scale_inv: torch.Tensor     # FP32, shape ()
    _fp8_meta: dict              # links to amax / scale buffers
```

A `Float8Tensor` is a tensor subclass that owns its FP8 data + the scale needed to interpret it. `__torch_dispatch__` routes ops through `fp8_gemm`, which dispatches to cuBLASLt's FP8 GEMM.

---

## FP8GlobalStateManager

The load-bearing piece. It owns amax buffers, scale buffers, and the recipe across all FP8 modules in the model. Synchronizes amax across data-parallel ranks (via all-reduce, controlled by `reduce_amax`).

Config knobs:

| Knob | Default | Notes |
|------|---------|-------|
| `fp8_format` | HYBRID | E4M3 fwd / E5M2 bwd |
| `amax_history_len` | 1024 (TE 1.x) | larger = smoother scale |
| `amax_compute_algo` | "max" | "max" or "most_recent" |
| `margin` | 0 | extra safety factor in scale calc |
| `reduce_amax` | True | all-reduce amax across DP |
| `interval` | 1 | how often to update scale |

---

## Drop-in modules

```python
import transformer_engine.pytorch as te

# Drop-in nn.Linear
layer = te.Linear(in_features, out_features, bias=True)

# Fused LayerNorm + Linear (one fused FP8 cast)
layer = te.LayerNormLinear(in_features, out_features, eps=1e-5)

# Fused LayerNorm + GEMM + GELU + GEMM
layer = te.LayerNormMLP(hidden_size, ffn_hidden_size)

# Full transformer block
layer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, ...)

# FP8 attention via cuDNN (since cuDNN 9)
attn = te.MultiheadAttention(num_attention_heads, kv_channels, attention_dropout=0.0)

# Context manager that wraps the entire forward/backward
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    loss = layer(inputs).loss
loss.backward()
```

---

## TE 2.x — per-block scaling recipes (Blackwell)

The Blackwell-era successors to `DelayedScaling`:

- `MXFP8BlockScaling`: 32-element FP8 blocks with E8M0 shared scale (OCP MX).
- `MXFP4`: 32-element FP4 blocks with E8M0 shared scale.
- `NVFP4`: 16-element FP4 blocks + E4M3 shared scale + FP32 per-tensor scale — the Blackwell-native format.

These generalize the [[deepseek-v3-fp8]] per-block idea to first-class library support and to the FP4 element size.

`fp8_autocast()` extends to `nvfp4_autocast()`-style context managers.

---

## Performance

On H100 / GPT-175B: TE-based FP8 (via [[fp8-lm]]'s MS-AMP integration) reported **39 % memory reduction, 75 % wall-clock speedup vs BF16 Megatron-LM**, and 37 % vs the previous TE FP8 baseline.

---

## Connections

- [[fp8-formats-paper]] — the E4M3 / E5M2 spec TE implements.
- [[fp8-lm]] — the MS-AMP extension layer (FP8 gradients + comm + optimizer).
- [[deepseek-v3-fp8]] — DSV3 replaces TE's per-tensor delayed scaling with per-block online scaling; TE 2.x has since adopted similar recipes.
- [[nvfp4-training]] — TE 2.x's NVFP4 recipe is the production path.
- [[ch-17]] — parent synthesis.
