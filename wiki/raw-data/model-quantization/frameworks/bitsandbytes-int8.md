<!-- scope: bitsandbytes LLM.int8() mixed-precision INT8 GEMM implementation
     deps: [[llm-int8]]
     see-also: [[bitsandbytes-nf4]], [[dettmers-group]]
-->

# bitsandbytes — LLM.int8()
- **Core Insight:** bitsandbytes implements Dettmers' LLM.int8() by detecting outlier feature dimensions per forward pass, splitting them out as an FP16 matmul, and quantizing the remaining "regular" dimensions to row-wise INT8 with FP16 accumulation.
- **Guideline:** Use `load_in_8bit=True` in HuggingFace `from_pretrained` for any model > 6.7B; for < 6.7B prefer NF4 (more accuracy at lower bit rate).
- **Authors:** Tim Dettmers et al.
- **Year:** 2022 (LLM.int8 paper), continuously maintained
- **URL:** https://github.com/bitsandbytes-foundation/bitsandbytes
- **Relevant topics:** INT8 inference, outlier extraction, vector-wise quantization, mixed precision

## Summary
bitsandbytes (bnb) is the reference implementation of Dettmers' LLM.int8() algorithm. It targets HuggingFace Transformers integration: setting `load_in_8bit=True` replaces every `nn.Linear` with a `bnb.nn.Linear8bitLt` module that internally performs the mixed-precision GEMM. The library is CUDA-centric (with experimental ROCm and CPU paths) and has been the default 8-bit quantization choice for transformer models since 2022. Its 4-bit cousin (NF4 / FP4) is documented in [[bitsandbytes-nf4]]. The CUDA kernels are written in raw CUDA/C++ in `csrc/`; the Python wrapper handles autograd, state management, and the `Int8Params` parameter wrapper.

## Key Points
- Per-forward outlier detection: any feature dim with `max(|x|) > threshold` (default 6.0) goes to the FP16 path.
- Vector-wise (row-wise) INT8 quantization for both weights and activations on the INT8 path.
- FP16 accumulation; final dequantize merges INT8 and FP16 results.
- Memory savings: ~50% (FP16 → INT8 on the bulk of the matrix) with negligible accuracy loss.
- `Int8Params` wraps the INT8 weight + FP16 scales as a single Parameter.

## Technical Details

### Repository layout
- repo: `https://github.com/bitsandbytes-foundation/bitsandbytes`
- main quant module: `bitsandbytes/functional.py` (Python wrappers for quant/dequant ops)
- INT8 kernel entry: `bitsandbytes/csrc/ops.cu` and `bitsandbytes/csrc/kernels.cu`
- `nn.Linear8bitLt`: `bitsandbytes/nn/modules.py`
- HF integration shim: `bitsandbytes/optim/__init__.py` + HF's `BitsAndBytesConfig`

### Quantization rule (vector-wise / row-wise)
For weight matrix `W ∈ R^{out×in}` and input `X ∈ R^{batch×in}`:
```
# Per-row scale (one scalar per output channel)
s_W = max(|W|, dim=in) / 127
W_int8 = round(W / s_W).clamp(-128, 127)

# Per-row scale of the input (one scalar per token)
s_X = max(|X|, dim=in) / 127
X_int8 = round(X / s_X).clamp(-128, 127)

# Outlier mask: columns with any |X[i,j]| > threshold
mask = (|X|.max(dim=batch) > 6.0)

# Mixed-precision GEMM
Y = (s_X.outer(s_W)) * (X_int8[:, ~mask] @ W_int8[:, ~mask].T) \
  + X[:, mask] @ W[:, mask].T  # FP16 fallback for outlier columns
```

### Key APIs
- `bnb.matmul(A, B, out=None, state=MatmulLtState())` — top-level INT8 matmul with outlier extraction.
- `bnb.nn.Linear8bitLt(in_features, out_features, has_fp16_weights=False, threshold=6.0)` — drop-in nn.Linear.
- `bnb.functional.quantize_blockwise(A, code, blocksize=4096)` — quantize a tensor blockwise (for optimizer states).
- `bnb.optim.Adam8bit(...)` / `AdamW8bit(...)` — 8-bit optimizer states using block-wise quantization.

### Config / hyperparameters
| Knob | Default | Notes |
|------|---------|-------|
| `threshold` | 6.0 | outlier magnitude cutoff |
| `has_fp16_weights` | False | keep FP16 master weight in memory |
| `index` | None | per-tensor override for outlier index |
| `memory_efficient_backward` | False | re-quantize backward (slower, less memory) |

### Kernel set in `csrc/`
- `kQuantize_8bit_blockwise` — block-wise INT8 quantization (used for optimizer states).
- `gemm_mixed_8bit_lt` — the mixed-precision GEMM with outlier path.
- `dequantize_blockwise` — inverse for optimizer-state load.

## Connections
- [[llm-int8]] — paper this library implements.
- [[bitsandbytes-nf4]] — the 4-bit NF4 / FP4 cousin in the same repo.
- [[dettmers-group]] — author lab.
- [[qlora]] — QLoRA fine-tuning sits on top of the NF4 path in this library.
