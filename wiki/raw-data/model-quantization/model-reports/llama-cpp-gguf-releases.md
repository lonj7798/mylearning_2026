<!-- scope: the llama.cpp k-quant ladder — community gguf format and the q2_K..q8_0 quality/size tradeoff curve
     deps: [[gguf-k-quants]]
     see-also: [[llama-cpp-ggml]]
-->

# llama.cpp gguf Community Release Patterns — The K-Quant Ladder
- **Core Insight:** The community-standard way to ship a quantized LLM in 2024-2026 is via llama.cpp's gguf format using the **k-quant ladder** — q2_K, q3_K_S/M/L, q4_K_S/M, q5_K_S/M, q6_K, q8_0 — each level being a fully-specified block layout with super-block scales, and the M-variants spending extra bits on the perplexity-sensitive layers (attention K, attention output, FFN gate/down).
- **Guideline:** Default to `Q4_K_M` for a balanced quality/size point on consumer GPUs; jump to `Q5_K_M` if you have memory headroom; only drop to `Q3_K_M` or `Q2_K` if you're memory-constrained, and prefer the IQ-variants (IQ2_XS, IQ3_XS) over the K-variants for sub-4-bit since they use importance-aware codebooks.
- **Authors:** Georgi Gerganov + llama.cpp contributors (Iwan Kawrakow / "ikawrakow" — the k-quant author)
- **Year:** 2023 (q4_0 / q4_1 / q5_0 / q5_1), 2023-late (k-quants), 2024 (IQ-quants), continuous
- **URL:** https://github.com/ggerganov/llama.cpp; https://huggingface.co/TheBloke (early gguf hub); https://huggingface.co/bartowski (current dominant gguf curator)
- **Relevant topics:** gguf, k-quants, IQ-quants, llama.cpp, community deployment, super-block layout

## Abstract
llama.cpp's gguf format has become the **de facto community standard** for shipping quantized LLMs to consumer hardware (CPU + Apple Silicon + consumer GPU). The format encodes weights in fixed-layout blocks, each with a deterministic recipe of `(bits_per_weight, block_size, super-block_structure, scale_format)`. The two main families are the **K-quants** (q2_K through q6_K) which use super-blocks of 256 weights with 16 sub-blocks of 16 weights each and per-sub-block 6-bit scales; and the **IQ-quants** (iq1_S through iq4_NL) which add a small "importance matrix" calibration step and a non-uniform LUT, achieving GPTQ-class quality at extreme bit-widths. Curators like TheBloke (2023) and bartowski (2024+) publish dozens of bit-width variants per model release, letting users pick the largest variant that fits their VRAM.

## Key Contributions
- A standardized **block-layout family** spanning roughly 2.5 to 8 bits per weight, all readable by a single C++ runtime (llama.cpp / Ollama / LM Studio / koboldcpp).
- **K-quant** family (256-element super-block + 16 sub-blocks × 16 weights, 6-bit per-sub-block scale) — the canonical mid-tier format.
- **IQ-quant** family — importance-matrix-aware non-uniform codebooks; extends usable bit-width down to ~1.5–2 bits with calibration.
- **M / S / L sub-variants** — bit budget is shifted between layers; M = "mostly Q4_K but Q6_K on the attention output and FFN down projections", which preserves perplexity at small size cost.

## Key Figures/Tables to Study
- The community PPL-vs-size curves published with every llama.cpp k-quant PR.
- bartowski's HF gguf release listings — typical 10+ bit-width variants per model with file sizes.
- llama.cpp README "Quantization" table — bits/weight column for every quant type.

## Technical Details

### K-quant family bit budgets
| Quant | Effective bpw | Super-block | Notes |
|-------|---------------|-------------|-------|
| q2_K | 2.5625 | 256 weights | per-sub-block 4-bit scale + per-super-block 6-bit min |
| q3_K_S | 3.4375 | 256 | small variant |
| q3_K_M | 3.4375+ | 256 | M = upgrades attention output, FFN down to Q5_K |
| q3_K_L | 3.4375++ | 256 | L = upgrades more layers |
| q4_K_S | 4.5 | 256 | per-sub-block 6-bit scale + 6-bit min |
| q4_K_M | 4.5+ | 256 | the most popular variant — Q6_K on critical layers |
| q5_K_S | 5.5 | 256 | |
| q5_K_M | 5.5+ | 256 | |
| q6_K | 6.5625 | 256 | per-sub-block 8-bit scale |
| q8_0 | 8.5 | 32 | per-block FP16 scale; flat layout, no super-block |

The K-quant super-block: 256 weights → 16 sub-blocks of 16 weights → each sub-block has a 6-bit scale and a 6-bit min; super-block has a single FP16 scale-of-scales. This double-quantization is what gets q4_K to 4.5 bpw instead of 4 + (16 × 12)/256 = 4.75.

### IQ-quant family (importance-matrix calibration)
| Quant | Effective bpw | Method | Notes |
|-------|---------------|--------|-------|
| iq1_S | 1.5625 | super-block + LUT + IMatrix | extreme low-bit |
| iq1_M | 1.75 | + per-block weights | |
| iq2_XXS | 2.0625 | non-uniform codebook | with importance matrix |
| iq2_XS | 2.3125 | | |
| iq2_S | 2.5 | | |
| iq2_M | 2.7 | | |
| iq3_XXS | 3.0625 | | |
| iq3_XS | 3.3 | | |
| iq3_S / iq3_M | 3.4-3.7 | | |
| iq4_XS | 4.25 | | |
| iq4_NL | 4.5 | "non-linear" — LUT | |

The **importance matrix** (`--imatrix`) is generated from a calibration corpus (typically wiki.train.raw or a domain mix); it's a per-weight Fisher-information-proxy used to weight the quantization error. This is conceptually a lightweight cousin of GPTQ's Hessian: same idea, simpler implementation.

### M / S / L mixed-precision policy
The `_M` (medium) variants spend extra bits on:
- `attn.v_proj` (V projection) — output to attention values
- `attn.output` — attention output projection
- `ffn_down` — FFN down projection (the activation-heavy layer)

These three layer types empirically dominate the perplexity cost of low-bit quant. The M variant typically promotes them by 1-2 quant tiers (e.g. Q4_K_M = Q4_K + Q6_K on these three) at <5% size overhead.

### Hardware acceleration
- **CPU**: AVX2 / AVX-512 / NEON / SVE kernels per quant; gguf weights typically decode + matmul fused in a single tile.
- **Apple Silicon**: Metal kernels mirror the CUDA k-quant kernels; Apple's unified memory makes large quant variants easy.
- **CUDA**: every k-quant has a dedicated CUDA kernel; recent versions add Marlin/Machete-style W4A16 tensor-core paths.
- **Vulkan / OpenCL**: community-contributed; less optimized.

### Curator workflow (2024-2026)
Typical curator (e.g. `bartowski`) workflow for a new model release:
1. Download BF16/FP16 weights from the model HF org.
2. Convert to gguf via `convert_hf_to_gguf.py`.
3. Generate imatrix from a 1-2k token calibration corpus.
4. Quantize to 10-12 variants: Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, IQ4_XS, Q3_K_M, IQ3_M, IQ2_M, IQ1_M.
5. Upload all variants to a single HF repo with the imatrix file.
6. Publish PPL comparison table in the README.

### Quality vs size landscape (Llama-3-8B reference)
- BF16: 16 GB
- Q8_0: 8.5 GB (PPL ≈ baseline + 0.001)
- Q6_K: 6.6 GB (PPL ≈ baseline + 0.005)
- Q5_K_M: 5.7 GB (PPL ≈ baseline + 0.01)
- Q4_K_M: 4.9 GB (PPL ≈ baseline + 0.03) — **the consumer-default sweet spot**
- IQ4_XS: 4.4 GB (PPL ≈ baseline + 0.05)
- Q3_K_M: 4.0 GB (PPL ≈ baseline + 0.10)
- IQ3_M: 3.8 GB (PPL ≈ baseline + 0.15)
- IQ2_M: 2.7 GB (PPL ≈ baseline + 0.4)
- IQ1_M: 1.9 GB (PPL ≈ baseline + 1.0) — usable for chat but degraded

## Connections
- [[gguf-k-quants]] — the format spec these releases instantiate.
- [[llama-cpp-ggml]] — the runtime framework.
- [[gptq]] — IQ-quants' importance-matrix idea is conceptually a stripped-down GPTQ Hessian.
- [[bitnet-models]] — bitnet.cpp is a fork of this stack, extending the format with ternary-specific encoding.

## Notes
The K-quant + IQ-quant ladder is arguably the **most-deployed quantization stack in absolute model count** in the world — every llama.cpp / Ollama / LM Studio user is running one of these variants, and curator hubs like bartowski's HF org host literally tens of thousands of model-quant variants.
