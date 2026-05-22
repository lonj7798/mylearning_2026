<!-- scope: Half-Quadratic Quantization reference implementation
     deps: [[hqq]]
     see-also: [[autoawq]], [[autogptq]]
-->

# HQQ — Half-Quadratic Quantization framework
- **Core Insight:** HQQ poses LLM weight quantization as a half-quadratic minimization over a sparsity-promoting loss in the quantization error, solves it in closed form per group without calibration data, and runs ~50× faster than GPTQ on a 70B model while matching its accuracy.
- **Guideline:** Use HQQ when you need data-free quantization with sub-minute wall-clock per layer; pair with `compute_dtype=torch.bfloat16` and `nbits=4, group_size=64`.
- **Authors:** Hicham Badri, Appu Shaji (Mobius Labs)
- **Year:** 2024
- **URL:** https://github.com/mobiusml/hqq
- **Relevant topics:** half-quadratic optimization, data-free PTQ, weight-only quantization

## Summary
HQQ (Half-Quadratic Quantization) is a fast, data-free PTQ framework from Mobius Labs. Instead of using a Gaussian (L2) quantization error model, HQQ minimizes a sparsity-promoting Lp loss (0 < p < 1) on the per-group quantization error, splitting the problem into the standard half-quadratic alternation between an auxiliary variable update (closed-form shrinkage) and a zero-point/scale update (closed-form least squares). The whole procedure converges in ~20 iterations per group with no calibration data and no Hessian, making it the fastest accurate LLM PTQ on the market. The library plugs into HuggingFace via `HQQLinear` (drop-in `nn.Linear`) and supports W2/W3/W4/W8 with optional double-quantization of the zero-point. Mobius also ships a torch.compile-compatible inference path (HQQ backends: `PYTORCH`, `ATEN`, `TORCHAO_INT4`).

## Key Points
- Data-free: no calibration tokens needed.
- Half-quadratic alternating optimization (~20 iters per group).
- Lp loss with p ∈ (0, 1) is more robust to outliers than L2.
- Supports W2/W3/W4/W8 with group sizes 32/64/128.
- Optional double-quantization: quantize the zero-points to INT8 with FP16 super-scale.
- Backends: pure PyTorch, ATen, torchao INT4 tinygemm.

## Technical Details

### Repository layout
- repo: `https://github.com/mobiusml/hqq`
- main quant module: `hqq/core/quantize.py` — `Quantizer.quantize()` and the `HQQLinear` module.
- optimizer: `hqq/core/optimize.py` — `optimize_weights_proximal()` implementing the half-quadratic loop.
- HF integration: `hqq/models/hf/base.py` — `AutoHQQHFModel` wrapper.
- backends: `hqq/backends/` — `bitblas.py`, `torchao_int4.py`, `aten.py`.

### Quantization formulation
For a weight group `w ∈ R^G`, target quantized representation `Q(w; s, z) = s · (round(w/s + z) − z)`, error `e = w − Q(w; s, z)`. Standard PTQ minimizes `||e||²`; HQQ minimizes:
```
min_{s, z, e}  ||e||_p^p   subject to   w = Q(w; s, z) + e
```
with `p ∈ (0, 1)` (default p = 0.7). Use the half-quadratic split with auxiliary variable `W_e`:
```
min_{s, z, W_e}  ||W_e||_p^p + (β/2) ||w − Q(w; s, z) − W_e||²
```
Alternate updates:
- `W_e ← shrinkage_p(w − Q(w; s, z), β)`   (closed-form proximal of Lp)
- `(s, z) ← argmin ||w − Q(w; s, z) − W_e||²`   (least squares on residual)

Increase β geometrically each iteration (continuation).

### Key APIs
- `Quantizer.quantize(W, nbits=4, group_size=64, axis=1)` — returns `(W_q, meta)` with packed bytes + scale + zero.
- `Quantizer.dequantize(W_q, meta)` — inverse.
- `HQQLinear(linear_layer, quant_config, compute_dtype, device)` — drop-in nn.Linear.
- `AutoHQQHFModel.quantize_model(model, quant_config)` — recurse over a HF model and replace Linears.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `nbits` | 4 | also 2, 3, 8 |
| `group_size` | 64 | 32 / 128 also common |
| `axis` | 1 | quantize along input dim (per-output-channel groups) |
| `quant_zero` | True | quantize zero-points to INT8 |
| `quant_scale` | False | quantize scales (extra savings, small accuracy hit) |
| `optimize_weights_proximal.iters` | 20 | half-quadratic iterations |
| `optimize_weights_proximal.lp_norm` | 0.7 | Lp loss exponent |

### Wall-clock comparison (Llama-3-70B, A100 80GB)
| Method | Quant time | PPL (wikitext) |
|--------|-----------|----------------|
| GPTQ (128 calib) | ~25 min | 4.45 |
| AWQ (128 calib) | ~12 min | 4.50 |
| HQQ (data-free) | ~30 sec | 4.55 |

## Connections
- [[hqq]] — algorithm paper.
- [[autogptq]] — Hessian-based competitor.
- [[autoawq]] — activation-aware competitor.
- [[torchao]] — torchao's INT4 tinygemm is an HQQ inference backend.
