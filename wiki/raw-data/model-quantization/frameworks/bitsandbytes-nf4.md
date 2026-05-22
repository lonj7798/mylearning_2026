<!-- scope: bitsandbytes 4-bit NF4 / FP4 kernel implementation
     deps: [[nf4]], [[qlora]]
     see-also: [[bitsandbytes-int8]], [[dettmers-group]]
-->

# bitsandbytes — NF4 / FP4 4-bit Kernels
- **Core Insight:** bitsandbytes' 4-bit path stores weights as NF4 (Normal Float 4-bit, quantile-based code for Gaussian-distributed weights) or FP4, with a block-wise FP16 scale per 64 elements, and dequantizes on the fly inside a fused FP16 GEMV kernel — making QLoRA fine-tuning practical on consumer GPUs.
- **Guideline:** Use `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16` for QLoRA; switch to `"fp4"` only if you specifically need that codebook.
- **Authors:** Tim Dettmers et al. (QLoRA paper 2023)
- **Year:** 2023 (4-bit added to bitsandbytes; QLoRA paper)
- **URL:** https://github.com/bitsandbytes-foundation/bitsandbytes
- **Relevant topics:** NF4, FP4, double quantization, paged optimizer, QLoRA backbone

## Summary
The 4-bit half of bitsandbytes implements two codebooks — NF4 (Normal Float 4-bit, a 16-level non-uniform code derived from the quantiles of N(0, 1)) and FP4 (4-bit floating-point) — packed two-per-byte with block-wise FP16/FP32 scales every 64 elements. "Double quantization" further quantizes the block scales themselves to 8-bit, recovering an extra ~0.37 bits/parameter. The forward path is a fused GEMV kernel that dequantizes a tile of weights into shared memory, then performs an FP16 matmul; this avoids ever materializing the full FP16 weight matrix. The 4-bit module is the storage layer underneath QLoRA — frozen 4-bit base, FP16 LoRA adapters trained on top — and underpins most consumer-GPU LLM fine-tuning since 2023.

## Key Points
- Two codebooks: NF4 (16 quantiles of N(0,1)) and FP4 (4-bit float).
- Block size: 64 elements share one FP16 scale (default).
- Double quantization: block scales themselves quantized to INT8 with FP32 super-scale.
- Forward kernel fuses dequantize + GEMV; no FP16 weight ever materialized.
- Backward (for QLoRA): only LoRA adapters get gradients; base weights stay frozen.
- Paged optimizer: optimizer states paged to CPU via unified memory.

## Technical Details

### Repository layout
- repo: `https://github.com/bitsandbytes-foundation/bitsandbytes`
- 4-bit Python API: `bitsandbytes/functional.py` (`quantize_4bit`, `dequantize_4bit`)
- 4-bit module: `bitsandbytes/nn/modules.py` (`Linear4bit`, `Params4bit`)
- 4-bit CUDA kernels: `bitsandbytes/csrc/kernels.cu` (functions like `kQuantizeBlockwiseNF4`, `gemv_4bit_inference_naive_fp16`)
- HF integration: HF Transformers `BitsAndBytesConfig` with `load_in_4bit=True`.

### NF4 codebook (16 levels)
Derived as quantiles of a standard normal N(0, 1), then rescaled to [-1, 1]:
```
NF4_CODES = [
  -1.0, -0.6961928, -0.5250730, -0.39491748,
  -0.28444138, -0.18477343, -0.09105003, 0.0,
   0.07958029, 0.1609302, 0.24611232, 0.33791524,
   0.44070983, 0.5626170, 0.7229568, 1.0
]
```
Asymmetric: NF4 has 8 negative + 7 positive + zero. Optimal for weights ≈ N(0, σ²).

### Quantization rule
For a 64-element block of weights `w[0..63]`:
```
s = max(|w|) / 1.0                       # block scale (FP16)
w_norm = w / s                           # normalize to [-1, 1]
w_nf4 = argmin_{c ∈ NF4_CODES} |w_norm − c|   # nearest-code search
```
Two NF4 codes pack into one byte.

### Double quantization
The block scales `s_0, s_1, …` are themselves quantized to INT8 with their own per-256-block super-scale (FP32). Net savings: 32 bits per block scale → 8 + (32/256) ≈ 8.125 bits per block → effective bit rate drops from 4.5 bpw (with FP16 scale) to ~4.13 bpw.

### Key APIs
- `bnb.functional.quantize_4bit(A, blocksize=64, quant_type="nf4")` — quantize tensor → packed bytes + scales.
- `bnb.functional.dequantize_4bit(A, quant_state)` — inverse.
- `bnb.nn.Linear4bit(in_features, out_features, bias, compute_dtype, quant_type, quant_storage)` — drop-in 4-bit Linear.
- `class Params4bit(nn.Parameter)` — holds packed 4-bit weight + QuantState.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `bnb_4bit_quant_type` | "fp4" (legacy default; prefer "nf4") | codebook choice |
| `bnb_4bit_use_double_quant` | False (prefer True) | double-quantize scales |
| `bnb_4bit_compute_dtype` | float32 (prefer bfloat16) | accumulate dtype |
| `blocksize` | 64 | elements per block-scale |

### Fused GEMV kernel
- `gemv_4bit_inference_naive_fp16` — for batch=1 / small-batch inference.
- Dequantizes a tile of weights into shared memory, then performs FP16 GEMV.
- For batch > 1, the path falls back to `bnb.matmul_4bit` which calls `dequantize_4bit` + `torch.matmul`.

## Connections
- [[nf4]] — codebook spec.
- [[qlora]] — the fine-tuning recipe sitting on top of this kernel.
- [[bitsandbytes-int8]] — 8-bit cousin in the same library.
- [[dettmers-group]] — author lab.
