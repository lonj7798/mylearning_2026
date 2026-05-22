<!-- scope: AutoAWQ implementation of Activation-aware Weight Quantization
     deps: [[awq]]
     see-also: [[autogptq]], [[vllm-quant]], [[han-song-mit]]
-->

# AutoAWQ
- **Core Insight:** AutoAWQ implements Lin et al.'s AWQ by searching for per-channel scaling factors that protect the most activation-magnitude-salient 1% of weight channels, then quantizing the rest to W4A16 with group-wise scales.
- **Guideline:** Run AutoAWQ with `w_bit=4`, `q_group_size=128`, ~128 calibration samples, and `version="GEMM"` for prefill-heavy workloads or `"GEMV"` for batch=1 decode.
- **Authors:** Casper Hansen + AWQ paper contributors
- **Year:** 2023 (AWQ paper); AutoAWQ released later in 2023
- **URL:** https://github.com/casper-hansen/AutoAWQ
- **Relevant topics:** AWQ, activation-aware quantization, salient channel protection, W4A16

## Summary
AutoAWQ is the de-facto open implementation of AWQ (Activation-aware Weight Quantization). Unlike GPTQ which propagates rounding error via the Hessian, AWQ exploits the empirical observation that weight quantization difficulty is dominated by a small number of channels with large activation magnitudes. AWQ searches over a per-channel scaling factor `s_c = (mean|x_c|)^α` (with `α` swept on calibration data) such that `Y = (X/s) · (s·W) → quantize(s·W)` keeps the salient channels well-protected. The result is W4A16 quant matching FP16 perplexity within 0.1 on Llama-class models with no need for Hessian computation. AutoAWQ ships AWQ-GEMM (fast batched prefill) and AWQ-GEMV (fast batch=1 decode) kernels, both adapted from MIT's TinyChat.

## Key Points
- Selects salient weight channels by activation magnitude (top 1% are dominant).
- Per-channel scaling: `(X/s) · (s·W)` is mathematically equivalent to `XW` but quantizes more accurately.
- Grid search for scaling exponent α on calibration data (no gradient descent).
- W4A16 with group-wise scales (default group=128).
- Two inference kernel variants: GEMM (prefill) and GEMV (decode).
- Faster calibration than GPTQ (no Hessian inverse).

## Technical Details

### Repository layout
- repo: `https://github.com/casper-hansen/AutoAWQ`
- main quant module: `awq/quantize/quantizer.py` — `AwqQuantizer` class.
- scale search: `awq/quantize/scale.py` — `auto_scale_block()` and `apply_scale()`.
- GEMM/GEMV kernels: `awq/modules/linear/gemm.py`, `awq/modules/linear/gemv.py` (call into `awq_ext` CUDA extension).
- CUDA sources: `awq_ext/quantization/gemm_cuda_gen.cu`, `awq_ext/quantization/gemv_cuda.cu`.
- HF wrappers: `awq/models/*.py` (one file per architecture: llama, mistral, mixtral, qwen2, …).

### Quantization rule (AWQ)
For a weight matrix `W ∈ R^{out×in}` and calibration activation `X ∈ R^{batch×in}`:
```
# Per-channel activation magnitude
a = mean(|X|, dim=batch)            # [in]

# Search scaling exponent α ∈ [0, 1]
for α in linspace(0, 1, 20):
    s = a ** α                       # [in]
    W_scaled = W * s                 # equivalent transformation
    X_scaled = X / s
    W_q = quantize_group(W_scaled, group_size=128, bits=4)
    loss = ||W_scaled @ X_scaledᵀ − dequant(W_q) @ X_scaledᵀ||²
α* = argmin loss

# Final stored: W_q (4-bit) and absorb s into preceding LayerNorm/Linear bias
```

### Key APIs
- `AutoAWQForCausalLM.from_pretrained(model_id)` — load FP16 base.
- `model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128, "version": "GEMM"})` — run calibration + AWQ.
- `model.save_quantized(out_dir)` — pack and save in AWQ format.
- `AutoAWQForCausalLM.from_quantized(out_dir, device_map="auto")` — load quantized model with appropriate kernel.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `w_bit` | 4 | also supports 3 |
| `q_group_size` | 128 | per-128-element group scale |
| `version` | "GEMM" | "GEMM" (batched), "GEMV" (batch=1), "Marlin" (newer) |
| `zero_point` | True | use zero-point (asymmetric); False → symmetric |
| `n_samples` | 128 | calibration token count |

### Kernel layouts
- **GEMM**: weights packed as INT4 with FP16 scales; uses A100/H100 tensor cores via `mma.sync`; best for prefill / batch > 1.
- **GEMV**: weights dequantized into shared memory tile; FP16 vector-matrix product; best for autoregressive decode.
- **Marlin**: optional newer backend; faster prefill on Ampere/Hopper (same kernel family as GPTQ Marlin).

### Calibration absorption trick
The per-channel scale `s` would have to be applied at runtime, costing an extra multiply. AutoAWQ fuses it: `s` is divided into the preceding LayerNorm's weight (`γ_new = γ / s`) so no runtime overhead remains.

## Connections
- [[awq]] — paper this implements.
- [[autogptq]] — sibling weight-only PTQ framework (Hessian-based).
- [[han-song-mit]] — lab that invented AWQ + TinyChat.
- [[vllm-quant]] — vLLM directly consumes AWQ checkpoints.
- [[smoothquant]] — same group's activation-aware predecessor (W8A8 instead of W4A16).
