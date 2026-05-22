<!-- scope: llama.cpp gguf k-quant family (q2_K, q3_K, q4_K, q5_K, q6_K, q8_0); superblock layout + mixed scales
     deps: [[int4]], [[int8]]
     see-also: [[llama-cpp-ggml]], [[llama-cpp-gguf-releases]]
-->

# llama.cpp gguf k-quant Family (q2_K, q3_K, q4_K, q5_K, q6_K, q8_0)
- **Core Insight:** llama.cpp's k-quant family achieves CPU-deployable LLM inference at 2–8 bits by using **two-level superblock scaling** — quantizing weights to N bits within small blocks (16 or 32 elements), storing a per-block scale as a 6-bit integer, and quantizing the per-block scales themselves against a 16-bit superblock scale — and by mixing bit-widths within a layer (e.g. higher precision for attention output projections).
- **Guideline:** When deploying LLMs on CPU or low-VRAM GPU via llama.cpp, default to **q4_K_M** (medium quality 4-bit) for general use, **q5_K_M** for quality-sensitive workloads, **q8_0** for near-FP16 quality, and **q2_K / q3_K_S** only for memory-constrained deployment where some quality loss is acceptable.
- **Authors:** Georgi Gerganov + llama.cpp community (k-quants introduced June 2023)
- **Year:** 2023 (initial k-quant release); evolved continuously in llama.cpp
- **URL:** https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.h ; https://github.com/ggerganov/llama.cpp/pull/1684 (k-quants PR)
- **Relevant topics:** gguf, k-quants, superblock, llama.cpp, CPU LLM inference, mixed-precision layer assignment

## Abstract
The gguf k-quant family is the de facto LLM CPU inference quantization standard, used by llama.cpp, Ollama, LM Studio, and most local-deployment toolchains. Each k-quant tier (q2_K through q6_K, plus q8_0) packs weights into a tightly-engineered superblock layout: 256 weights are split into 16 sub-blocks of 16 elements each, with per-sub-block scales themselves quantized against a superblock-level FP16 scale. The "K" in the name denotes the superblock structure (introduced in 2023, replacing the older q4_0 / q4_1 / q5_0 / q5_1 "legacy" layouts). Suffix "_S/_M/_L" denotes "small / medium / large" variants that selectively bump precision on specific tensors (typically attention.wv, attention.wo, ffn_down — the most sensitive ones). Result: q4_K_M achieves Llama-7B perplexity within ~0.05 of FP16 at ~4.8 bits/weight effective.

## Key Contributions
- Two-level superblock scaling (per-sub-block 6-bit scale + FP16 superblock scale).
- Mixed-bit-width per-tensor assignment (_S/_M/_L variants).
- CPU SIMD-optimized dequant + GEMV kernels (AVX2 / AVX-512 / ARM NEON / SVE).
- gguf container format → tensor metadata + quant config + tokenizer + model weights in one file.
- Democratized LLM inference: runs Llama-70B on CPU + 32 GB RAM.

## Key Figures/Tables to Study
- **k-quant tier comparison table** (canonical llama.cpp PRs): bits/weight, perplexity gap from FP16, decode speed in tokens/sec on M2.
- **Superblock layout diagram**: 256 weights → 16 × 16-element sub-blocks → 1 superblock with FP16 master scale.

## Technical Details

### Superblock structure (k-quants)
All k-quants share the same outer layout:
```
1 superblock = 256 weights
            = 16 sub-blocks of 16 weights each
            + 1 FP16 superblock scale
            + 16 per-sub-block scales (each 6-bit, packed)
            + 16 per-sub-block mins (if asymmetric) (each 6-bit, packed)
```

### q4_K layout (group-128 INT4, asymmetric)
- 256 weights, asymmetric quantization
- Each 32-element sub-block has: 6-bit scale + 6-bit zero-point (min)
- Sub-block scales are themselves quantized vs an FP16 superblock scale
- Per-element: 4 bits
- Total bytes per 256-weight block: 256·4/8 (weights) + 8·6/8·2 (sub-block scales+mins) + 2 (FP16 superblock scale)
                                 = 128 + 12 + 2 = **144 bytes / 256 weights = 4.5 bits/weight**

### q5_K layout
- Per-element: 5 bits (split across two arrays: 1 high bit + 4 low bits)
- Same superblock structure as q4_K
- ~5.5 bits/weight effective

### q6_K layout
- Per-element: 6 bits (4 low bits + 2 high bits in a separate array)
- Per-sub-block 8-bit scale (no superblock scale needed at this precision)
- ~6.5 bits/weight effective

### q3_K layout
- Per-element: 3 bits
- Same superblock scale structure as q4_K but tighter
- ~3.5 bits/weight effective

### q2_K layout
- Per-element: 2 bits (4 levels)
- Aggressive mixing: per-sub-block 4-bit scale, FP16 superblock scale
- ~2.6 bits/weight effective

### q8_0 layout
- Per-element: 8 bits (signed INT8)
- Per-32-element block: FP16 scale
- ~8.5 bits/weight effective; near-FP16 quality

### _S / _M / _L variants — mixed precision per tensor
The suffix denotes how aggressively llama.cpp upgrades sensitive tensors:
- **_S (small)**: all tensors at the base bit-width.
- **_M (medium, default)**: bump attention.wv, attention.wo, ffn_down to q{N+1}_K.
- **_L (large)**: bump even more tensors.

Example: in **q4_K_M**, most weights are 4-bit, but attention output projections and FFN down-projections are q5_K → average bits/weight ≈ 4.85.

### Why mix bit-widths within a layer
Sensitivity analysis on Llama showed:
- attention.wv (value projection): most sensitive to quantization error → highest precision.
- attention.wo (output projection): high sensitivity → high precision.
- ffn_down (downward FFN projection): moderate sensitivity → +1 bit over baseline.
- attention.wq / attention.wk: low sensitivity → can use baseline.
- ffn_gate / ffn_up: low sensitivity → baseline.

This mirrors the "sensitivity-aware" insight in [[squeezellm]] and [[awq]].

### Per-row / per-tensor scale calibration
For each sub-block:
- **Symmetric (q*_K_S in some tiers)**: `s = max(|w|) / max_int`
- **Asymmetric (q4_K standard)**: `s = (max(w) − min(w)) / 15`, `z = round(−min(w)/s)`
- Sub-block scales are themselves quantized to 6 bits against the FP16 superblock master scale.

### Imatrix calibration
Modern llama.cpp k-quants support **imatrix** (importance matrix) calibration: a per-channel weighting derived from running the model on a calibration set and capturing activation magnitudes. The k-quant search then weights MSE by activation magnitude — closing most of the gap to AWQ at the same bit-width.

### Quality comparison (Llama-2-7B, perplexity on Wikitext)
| Quant | Bits/weight | PPL gap vs FP16 |
|-------|-------------|------------------|
| FP16 | 16 | 0 |
| q8_0 | 8.5 | +0.01 |
| q6_K | 6.5 | +0.04 |
| q5_K_M | 5.7 | +0.08 |
| q4_K_M | 4.8 | +0.16 |
| q4_K_S | 4.5 | +0.30 |
| q3_K_M | 3.9 | +0.60 |
| q3_K_S | 3.4 | +1.0 |
| q2_K | 2.6 | +1.5 |

### gguf container
gguf (GPT-Generated Unified Format) wraps quantized weights + tokenizer + model config in a single binary file. Replaces the older ggml format. Magic bytes `GGUF`; version + tensor metadata + tensor data.

### Hardware
- **CPU-first**: AVX2, AVX-512, AVX-512-VNNI, ARM NEON, SVE, Apple AMX.
- **GPU support**: CUDA, ROCm, Vulkan, Metal via dequant-to-FP16/BF16 + tensor-core matmul.
- **Apple Silicon**: dequant + AMX matmul; fastest local LLM platform per dollar.

## Connections
- [[int4]] — q4_K is essentially group-32 asymmetric INT4 with quantized scales.
- [[int8]] — q8_0 is per-32-block INT8 with FP16 scale.
- [[llama-cpp-ggml]] — the kernel/library implementation.
- [[llama-cpp-gguf-releases]] — community release patterns and naming conventions.
- [[awq]] — imatrix calibration captures similar information as activation-aware scaling.
- [[squeezellm]] — sensitivity-aware tensor-level precision mixing.
