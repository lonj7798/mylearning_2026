<!-- scope: llama.cpp ggml/gguf k-quant kernels
     deps: [[gguf-k-quants]]
     see-also: [[maxime-labonne-quant-guide]]
-->

# llama.cpp — ggml / gguf k-quant Kernels
- **Core Insight:** llama.cpp's k-quant family (`q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`) packs weights into 256-element super-blocks with two-level scaling (super-block FP16 scale + 16 per-sub-block 4/6-bit scales), yielding the gold-standard CPU-and-embedded inference at sub-5-bit precision.
- **Guideline:** Use `q4_k_m` (medium) for the best size/perplexity tradeoff on most workloads; jump to `q5_k_m` or `q6_k` if perplexity matters, drop to `q3_k_s` only for memory-constrained edge devices.
- **Authors:** Georgi Gerganov (ikawrakow + ggml-org community for k-quants)
- **Year:** 2023 (k-quants introduced); ongoing
- **URL:** https://github.com/ggml-org/llama.cpp
- **Relevant topics:** gguf, k-quants, super-blocks, CPU inference, Apple Silicon

## Summary
llama.cpp is the C/C++ inference engine that brought local LLM execution to CPUs, Apple Silicon, and consumer GPUs through the gguf model file format and the ggml tensor library. Its distinctive quantization contribution is the k-quant family — `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, plus the legacy `q4_0/q4_1/q5_0/q5_1/q8_0` round-to-nearest formats — implemented as SIMD-vectorized dequantize-and-dot-product kernels for AVX2/AVX-512/AVX-VNNI (x86), NEON (ARM), Apple Metal, and CUDA. K-quants use 256-element super-blocks with two-level scaling: an FP16 super-block scale plus sixteen 4-bit or 6-bit sub-block scales. The `_m` (medium) and `_s` (small) variants control which layers use the higher- vs lower-bit code (e.g. `q4_k_m` upgrades attention.wv and feed_forward.w2 to `q6_k`).

## Key Points
- File format: gguf (single-file model + metadata; replaced ggjt/ggml in 2023).
- Two-level scaling: FP16 super-block scale + 4/6-bit sub-block scales.
- Super-block size: 256 elements (sixteen 16-element sub-blocks).
- Variants: `_xs`, `_s`, `_m`, `_l` control per-layer code mix.
- SIMD kernels for AVX2/512/VNNI, NEON, Metal, CUDA, ROCm, Vulkan.
- 2024 additions: IQ-quants (`iq2_xs`, `iq3_s`, `iq4_xs`) — importance-aware with codebooks (akin to AWQ-style sensitivity).

## Technical Details

### Repository layout
- repo: `https://github.com/ggml-org/llama.cpp`
- core ggml: `ggml/src/ggml.c` (tensor + op dispatch)
- k-quant kernels: `ggml/src/ggml-quants.c` (scalar reference)
- SIMD variants: `ggml/src/ggml-cpu/` (per-arch SIMD)
- CUDA kernels: `ggml/src/ggml-cuda/` (`mul_mat_vec_q.cu`, `dequantize.cu`)
- Metal: `ggml/src/ggml-metal/ggml-metal.metal`
- model-loading + quantize tool: `examples/quantize/quantize.cpp`
- conversion: `convert_hf_to_gguf.py`

### q4_k super-block layout
For 256 weights, packed as 144 bytes:
| Field | Bytes | Notes |
|-------|-------|-------|
| FP16 super-block scale `d` | 2 | global scale |
| FP16 super-block min `dmin` | 2 | global min (for asymmetric) |
| 12 bytes of 6-bit packed sub-block scales + mins | 12 | 16 × (6 scale + 6 min) bits |
| 128 bytes packed INT4 weights | 128 | 256 weights × 4 bits |
| **Total** | **144 bytes / 256 weights = 4.5 bpw** |

### Dequantization rule
For sub-block `b ∈ {0..15}`, weight index `i ∈ {0..15}` within sub-block:
```
scale_b = d * unpack_6bit(sub_scales, b)        # FP16
min_b   = dmin * unpack_6bit(sub_mins, b)       # FP16
w[16*b + i] = scale_b * q[16*b + i] − min_b     # INT4 → FP16
```

### Quantization rule (search per super-block)
```
# For each super-block of 256 weights w:
for each candidate d in a small grid around max(|w|)/15:
    for each sub-block b:
        sub_scale, sub_min = best_per_sub_block(w[16b:16(b+1)], d)
    compute reconstruction error
choose d minimising error; encode sub-scales as 6-bit
```

### Key entry points
- `quantize_row_q4_K_reference(const float * x, void * y, int k)` — scalar quantize one row.
- `ggml_vec_dot_q4_K_q8_K(int n, float * s, const void * vx, const void * vy)` — fused dequant + dot product.
- `llama_model_quantize_internal(...)` — top-level model quantization in `src/llama.cpp`.

### Config / hyperparameters
| Variant | Effective bpw | Use case |
|---------|---------------|----------|
| `q2_k` | 2.6 | extreme size budget |
| `q3_k_s/m/l` | 3.4/3.6/3.9 | small models / edge |
| `q4_k_s` | 4.6 | balanced default |
| `q4_k_m` | 4.85 | recommended default for desktop |
| `q5_k_m` | 5.7 | accuracy-leaning |
| `q6_k` | 6.6 | near-FP16 quality |
| `q8_0` | 8.5 | safety baseline |
| `iq4_xs` | 4.25 | codebook-based, more accurate at the same size |

### IQ-quants (2024)
Importance-weighted quantization using an "imatrix" file computed from calibration text. Uses a learned codebook per super-block. `iq2_xs`/`iq3_xxs` are state-of-the-art at 2-3 bpw for llama.cpp.

## Connections
- [[gguf-k-quants]] — format spec page.
- [[maxime-labonne-quant-guide]] — practitioner guide to choosing quant variants.
- [[awq]] — algorithmic cousin of IQ-quants (importance-aware codebook).
