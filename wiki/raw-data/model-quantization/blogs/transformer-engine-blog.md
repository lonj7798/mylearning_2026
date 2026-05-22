<!-- scope: NVIDIA Transformer Engine FP8 usage patterns
     deps: [[fp8-e4m3]], [[fp8-e5m2]]
     see-also: [[transformer-engine-fp8]], [[nvidia-h100-fp8]], [[nvidia-blackwell-fp4]]
-->

# NVIDIA Transformer Engine — Practitioner Usage Patterns
- **Core Insight:** Transformer Engine (TE) hides the FP8 cast/scale/amax bookkeeping behind drop-in `nn.Linear` / `nn.LayerNorm` / attention replacements, automatically choosing per-tensor delayed scaling so that the user gets H100/B100 FP8 throughput with one autocast context manager.
- **Guideline:** Wrap your model in `te.Linear` and `te.LayerNorm`, enter `with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(margin=0, interval=1, amax_history_len=16))`, and let TE manage the scale state.
- **Authors:** NVIDIA Transformer Engine team (developer blogs + GTC talks)
- **Year:** 2022 (TE 0.x for H100), 2024 (TE 1.x with FP8 attention + MXFP4)
- **URL:** https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/
- **Relevant topics:** FP8 training, delayed scaling, amax history, drop-in modules

## Summary
Transformer Engine is NVIDIA's reference library for training and inference with FP8 on Hopper (H100) and FP8/FP6/FP4 on Blackwell (B100/B200). It exposes a small set of modules — `te.Linear`, `te.LayerNorm`, `te.LayerNormLinear`, `te.LayerNormMLP`, `te.TransformerLayer`, `te.MultiheadAttention` — that internally manage FP8 quantization state. The user wraps the forward pass in `te.fp8_autocast(enabled=True, fp8_recipe=...)` and the library handles per-tensor amax tracking, scale derivation, FP8 cast, FP32 accumulation, and master-weight maintenance. TE 1.x adds FP8 fused attention via cuDNN, MXFP4 support on Blackwell, and CPU-offload of master weights. Practitioner blogs from NVIDIA, MosaicML, and Mistral describe ~1.4-1.6× wall-clock speedup over BF16 for Llama-class models with negligible loss-curve divergence under the recommended recipe.

## Key Points
- Drop-in modules: `te.Linear`, `te.LayerNorm`, `te.TransformerLayer`, etc.
- Two recipes: `DelayedScaling` (default, FP8 training) and `MXFP8BlockScaling` (Blackwell, OCP MX).
- amax history depth typically 16; scale = amax / fp8_max with optional margin.
- Master weights in FP32 always; FP8 cast happens lazily before each GEMM.
- Inference path uses `Float8Tensor` with static (calibration-time) scales.

## Technical Details

### Core API surface
```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID,   # E4M3 forward, E5M2 backward
    amax_history_len=16,
    amax_compute_algo="max",
)

model = te.TransformerLayer(
    hidden_size=4096, ffn_hidden_size=11008,
    num_attention_heads=32, hidden_dropout=0.0,
)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    out = model(x)
```

### Delayed scaling algorithm
1. Track `amax_history` (deque of last 16 amax values per tensor).
2. At step t, compute `amax_t = max(amax_history)`.
3. Scale `s_t = fp8_max / amax_t` (with a margin if specified).
4. FP8 cast: `x_fp8 = round(x_fp32 * s_t)`.
5. After the GEMM, update history with this step's observed amax.

### Format selection
| Recipe option | Forward dtype | Backward dtype |
|---------------|---------------|-----------------|
| `Format.E4M3` | E4M3 | E4M3 |
| `Format.E5M2` | E5M2 | E5M2 |
| `Format.HYBRID` | E4M3 | E5M2 |

`HYBRID` is the recommended default — E4M3 has more precision for forward, E5M2 has more range for gradients.

### Blackwell MX recipe
```python
from transformer_engine.common.recipe import MXFP8BlockScaling
recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
```
Uses OCP MX 32-element blocks with UE8M0 scales; no amax history needed.

### Performance (Llama 70B, H100 SXM, MosaicML report)
| Recipe | Tokens/sec/GPU | Loss curve vs BF16 |
|--------|----------------|---------------------|
| BF16 | ~3000 | baseline |
| FP8 HYBRID delayed | ~4500 (1.5×) | indistinguishable |
| FP8 E4M3 forward + BF16 grad | ~4200 (1.4×) | indistinguishable |

### Common pitfalls (from blogs)
- LayerNorm in FP8: don't — keep LN in BF16 or use `te.LayerNormLinear` (fuses LN + Linear with internal cast).
- `amax_history_len` too small: scale chatters; too large: stale scale.
- Mixing `te.Linear` with vanilla `nn.Linear` inside `te.fp8_autocast`: only the `te.*` calls cast; `nn.Linear` stays BF16.

## Connections
- [[fp8-e4m3]] — forward-pass operand format.
- [[fp8-e5m2]] — backward-pass operand format.
- [[transformer-engine-fp8]] — frameworks-bucket sibling with code-level detail.
- [[nvidia-h100-fp8]] — hardware substrate.
- [[nvidia-blackwell-fp4]] — adds MXFP4/FP6 to the TE recipe registry.
