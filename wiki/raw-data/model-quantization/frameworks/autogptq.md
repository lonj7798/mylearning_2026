<!-- scope: AutoGPTQ implementation of GPTQ algorithm
     deps: [[gptq]]
     see-also: [[autoawq]], [[marlin-kernel]], [[vllm-quant]]
-->

# AutoGPTQ
- **Core Insight:** AutoGPTQ packages Frantar's GPTQ algorithm as a turnkey CLI + Python API that quantizes any HuggingFace causal-LM to 2/3/4/8-bit weights via Hessian-aware OBS updates, then registers triton + ExLlama + Marlin kernels for inference.
- **Guideline:** Use `group_size=128`, `desc_act=True`, `damp_percent=0.01`, and 128 calibration samples for best perplexity on Llama-class models; switch the inference backend to Marlin on Ampere/Hopper for 3-4× speedup.
- **Authors:** PanQiWei, qwopqwop200, and contributors
- **Year:** 2023 (initial release)
- **URL:** https://github.com/PanQiWei/AutoGPTQ
- **Relevant topics:** GPTQ, weight-only quantization, Hessian, group quantization

## Summary
AutoGPTQ is the most widely used implementation of Frantar et al.'s GPTQ algorithm for LLM weight-only PTQ. It wraps the OBS-style Hessian-weighted column-by-column updates into a simple per-layer `quantize()` API, then handles pack/unpack into 4-bit nibble layout, kernel registration, and HuggingFace model loading. Calibration runs once on ~128 samples of pile/c4 to estimate per-layer Hessians `H = 2 X Xᵀ + λI`, then the algorithm sweeps left-to-right through weight columns, quantizing each and redistributing the error to the remaining columns via Cholesky-factorized Hessian inverse. The 2024 successor `GPTQModel` is a fork by ModelCloud that ships faster CUDA kernels and broader model coverage; both share the same algorithm core.

## Key Points
- Implements GPTQ exactly: column-wise OBS update with Cholesky-factorized Hessian inverse.
- Supports W2/W3/W4/W8 weight-only quantization with group-wise scales.
- Three inference backends: cuda-old (naive), exllama (4-bit GEMV), Marlin (Ampere/Hopper W4A16 GEMM).
- Pack format: GPTQ-style 4-bit nibbles, group-wise FP16 scale + INT4 zero-point.
- Heavy fork ecosystem: GPTQModel (ModelCloud), AutoGPTQ-triton, AutoGPTQ-rocm.

## Technical Details

### Repository layout
- repo: `https://github.com/PanQiWei/AutoGPTQ`
- main quant module: `auto_gptq/quantization/gptq.py` — the GPTQ class with `add_batch()`, `quantize()`, `free()`.
- packer: `auto_gptq/quantization/quantizer.py` — `Quantizer` class with `configure()`, `find_params()`, `quantize()` for the codec.
- model wrappers: `auto_gptq/modeling/_base.py` — `BaseGPTQForCausalLM`.
- CUDA kernel: `auto_gptq/nn_modules/qlinear/qlinear_cuda.py` + `path/to/cuda/gemm.cu`.

### Quantization rule (per-column GPTQ update)
```python
# Per-layer Hessian
H = 2 * X @ X.T + lambda_damp * I              # X: [in_features, n_calib_tokens]
H_inv = cholesky_inverse(cholesky(H))          # upper-triangular

# Sweep columns left-to-right
for j in range(in_features):
    w_j = W[:, j]
    q_j = quantize(w_j, group_scale=s[j//G])   # nearest INT4 in the group
    err = (w_j - q_j) / H_inv[j, j]
    # Redistribute error to remaining columns
    W[:, j+1:] -= err.unsqueeze(1) * H_inv[j, j+1:].unsqueeze(0)
    W[:, j] = q_j
```

### Key APIs
- `BaseGPTQForCausalLM.from_pretrained(model_id, quantize_config)` — load FP16 base.
- `model.quantize(calibration_dataset)` — run the GPTQ algorithm layer-by-layer.
- `model.save_quantized(out_dir, use_safetensors=True)` — pack and save.
- `BaseGPTQForCausalLM.from_quantized(out_dir, device, use_marlin=True)` — load + register inference kernel.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `bits` | 4 | also supports 2, 3, 8 |
| `group_size` | 128 | -1 = per-column; larger group → smaller scale overhead, less accuracy |
| `desc_act` | False (prefer True) | sort columns by activation magnitude (act_order) — better accuracy |
| `damp_percent` | 0.01 | λ for Hessian damping |
| `sym` | True | symmetric INT quantization |
| `static_groups` | False | precompute group assignments before column sweep (needed for compile-time kernels) |

### Inference backends
- `qlinear_cuda` — naive dequant + FP16 GEMM (slowest).
- `qlinear_exllama` — 4-bit GEMV (batch=1 inference); fastest for autoregressive decode.
- `qlinear_marlin` — W4A16 GEMM with Ampere mma.sync (fastest for prefill / batch > 1).

## Connections
- [[gptq]] — paper this implements.
- [[autoawq]] — sibling weight-only PTQ framework (different calibration objective).
- [[marlin-kernel]] — the high-throughput inference backend.
- [[vllm-quant]] — vLLM consumes the GPTQ checkpoint format directly.
- [[frantar-alistarh-ist-austria]] — lab where GPTQ was invented.
