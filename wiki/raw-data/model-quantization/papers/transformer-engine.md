<!-- scope: NVIDIA Transformer Engine FP8 mixed-precision library
     deps: [[fp8-formats-paper]], [[fp8-lm]]
     see-also: [[deepseek-v3-fp8]], [[nvfp4-training]]
-->

# NVIDIA Transformer Engine (FP8 Mixed Precision)
- **Core Insight:** A practical FP8 training recipe requires (a) per-tensor scales chosen with one-step delay using an amax history, (b) E4M3 on the forward pass for precision, (c) E5M2 on the backward pass for dynamic range, and (d) BF16 master weights — packaged behind a thin layer-level API so user models don't manage the FP8 plumbing themselves.
- **Guideline:** Use TE's `DelayedScaling` recipe as the baseline for new FP8 training projects; switch to `CurrentScaling` only if your activations have low-frequency drift. For per-block scaling (the DSV3 / NVFP4 style), use TE's block-scaling recipes on Hopper/Blackwell.
- **Authors:** NVIDIA (Micikevicius, Casper, Korthikanti, et al.)
- **Year:** 2022 onward (continuous; current version 2.x)
- **URL:** https://github.com/NVIDIA/TransformerEngine • https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/
- **Relevant topics:** FP8 mixed precision, delayed scaling, amax history, E4M3/E5M2 split, BF16 master weights

## Abstract
Transformer Engine (TE) is NVIDIA's reference library for FP8 training on Hopper (H100/H200) and Blackwell (B200/B300) GPUs. It ships drop-in `te.Linear`, `te.LayerNorm`, `te.Attention` modules that internally cast to FP8 via the H100/B200 Tensor Cores while keeping all model-facing math in BF16/FP16. The two key abstractions are (1) the FP8 *recipe* — which tensors are E4M3 vs E5M2, how scales are computed, when amax history is updated — and (2) the *scaling strategy* — per-tensor delayed scaling, per-tensor current scaling, or per-block scaling on Blackwell. TE is the production baseline that DeepSeek-V3 measured itself against in §3.3 of the V3 report (and beat by switching to per-block scales).

## Key Contributions
- First production-quality FP8 training library for Tensor Cores; integrates with PyTorch, JAX, and frameworks like Megatron-LM, NeMo, MaxText.
- Defines the **DelayedScaling** recipe: each FP8 tensor carries a rolling **amax history** (window ~1024 steps); the next step's scale is computed from the max of the history rather than the current amax — avoids needing a sync between scale computation and the GEMM.
- Defines the **E4M3 forward / E5M2 backward** split: forward activations + weights are E4M3 (4 exp / 3 mantissa, narrow range, high precision); gradients are E5M2 (5 exp / 2 mantissa, wider range for the long-tailed gradient distribution).
- Provides `fp8_autocast()` context manager — wraps a forward/backward pass so all `te.Linear` modules run in FP8 without changing user code.
- Recent versions (TE 2.x) added per-block scaling recipes (`BlockScaling`, `MXFP8`, `NVFP4`) for Blackwell, generalizing the DSV3-style approach to first-class library support.

## Key Figures/Tables to Study
- The recipe diagram from TE's user guide: which tensors are FP8 vs BF16 inside a `te.Linear`, where casts happen, where amax is updated.
- The amax history rolling-window figure — explains why DelayedScaling avoids a synchronization bubble in the GEMM stream.

## Technical Details

### Format split
- **Forward (E4M3):** weights + activations cast to E4M3 before the GEMM; output dequantized to BF16/FP16 immediately after.
- **Backward (E5M2):** activation gradient (dY) cast to E5M2 (gradients have a wider tail than activations); weight gradient is BF16 output to be safe.
- BF16/FP32 promotion happens inside the H100 Tensor Core; the user-facing dtype stays BF16.

### Delayed scaling
- Each FP8 tensor maintains an **amax history** of the last K (default 1024) per-step amax values.
- The scale for step *t+1* is computed from history[t-K:t] — typically `max(history) / FP8_MAX_REPRESENTABLE`.
- Because the scale is computed from past data, no synchronization is needed between the amax reduction and the GEMM launch — the scale is ready before the next step starts.
- Trade-off: if activations drift quickly, the scale lags and clips; the alternative `CurrentScaling` synchronizes and uses the current amax (higher fidelity, slight overhead).

### Master weights
- BF16 master weights kept in HBM; FP8 weights computed on the fly in the forward pass.
- Optimizer state in FP32 (m, v in Adam).

### Per-block scaling (Hopper/Blackwell, TE 2.x)
- `MXFP8`: 32-element FP8 blocks with E8M0 shared scale (OCP MX spec).
- `MXFP4`: 32-element FP4 blocks with E8M0 shared scale.
- `NVFP4`: 16-element FP4 blocks with E4M3 shared scale + FP32 per-tensor scale — the Blackwell-native format (see [[nvfp4-training]], [[blackwell-quantization]]).
- These replace `DelayedScaling` for the layers that benefit from finer-grained scales.

### Practical hyperparameters
| Knob | Default |
|------|---------|
| Recipe | `DelayedScaling` on Hopper, `MXFP8`/`NVFP4` on Blackwell |
| amax history length | 1024 |
| Forward format | E4M3 |
| Backward format | E5M2 (delayed) / E4M3 (block-scaled) |
| Margin | 0 (no extra headroom on scale) |
| FP8 fused TE-LN | yes (LayerNorm + FP8 cast fused) |

### Performance
- On H100 / GPT-175B: FP8-LM (built on TE) reported 39 % memory reduction, 75 % wall-clock speedup vs BF16 Megatron-LM, and 37 % vs the previous TE FP8 baseline.

## Connections
- [[fp8-formats-paper]] — the joint NVIDIA/Arm/Intel spec for E4M3 / E5M2 that TE implements.
- [[fp8-lm]] — Microsoft's MS-AMP layer that extends TE recipes to FP8 gradient comm + FP8 optimizer state.
- [[deepseek-v3-fp8]] — DSV3 replaces TE's per-tensor delayed scaling with per-block online scaling at frontier scale; TE 2.x has since adopted similar block-scaling recipes.
- [[nvfp4-training]] — TE 2.x's `NVFP4` recipe is the production path for the format used in NVIDIA's 12B / 10T-token NVFP4 pretrain.
- [[microscaling-formats]] — TE's `MXFP8` / `MXFP4` recipes implement the OCP MX spec.
