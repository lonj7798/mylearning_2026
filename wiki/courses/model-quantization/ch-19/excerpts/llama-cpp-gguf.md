---
chapter: ch-19
course: model-quantization
phase: read
excerpt_of: "llama.cpp / ggml + gguf k-quants (Gerganov, ikawrakow, ggml-org community, 2023-onward)"
source_url: https://github.com/ggml-org/llama.cpp
created_at: "2026-05-21"
---

# Excerpt: llama.cpp + gguf k-quants — CPU / Apple Silicon deployment

**Authors:** Georgi Gerganov (llama.cpp); Iwan Kawrakow (k-quants); bartowski (current dominant gguf curator)
**Year:** 2023 (k-quants); 2024 (IQ-quants); continuous
**URL:** https://github.com/ggml-org/llama.cpp; https://huggingface.co/bartowski
**Raw-data source:** [[raw-data/llama-cpp-ggml]] + [[raw-data/gguf-k-quants]] + [[raw-data/llama-cpp-gguf-releases]]

---

## What this stack is

llama.cpp is the C/C++ inference engine that brought local LLM execution to CPUs, Apple Silicon, and consumer GPUs through the **gguf** model file format and the **ggml** tensor library. It is the **CPU / Apple Silicon deployment standard** — every Ollama, LM Studio, koboldcpp user is running this stack.

Distinctive quant contribution: the **k-quant family** (`q2_K` through `q6_K`, plus `q8_0`) + the **IQ-quant family** (importance-matrix-calibrated codebooks at 1.5–4 bpw).

---

## The k-quant ladder

| Variant | Effective bpw | Use case |
|---------|---------------|----------|
| `q2_K` | 2.6 | extreme size budget |
| `q3_K_S` / `_M` / `_L` | 3.4 / 3.6 / 3.9 | small / edge |
| `q4_K_S` | 4.6 | balanced |
| **`q4_K_M`** | **4.85** | **recommended consumer default** |
| `q5_K_M` | 5.7 | accuracy-leaning |
| `q6_K` | 6.6 | near-FP16 quality |
| `q8_0` | 8.5 | safety baseline |
| `iq4_xs` | 4.25 | codebook-based, more accurate at same size |
| `iq2_xs` | 2.3 | sub-3-bit IQ |
| `iq1_m` | 1.75 | extreme low-bit |

---

## Super-block layout (q4_K example, 256 weights = 144 bytes)

```
1 superblock = 256 weights = 16 sub-blocks of 16 weights each
            + 1 FP16 superblock scale (2 bytes)
            + 16 per-sub-block scales, 6-bit packed (12 bytes total for scales+mins)
            + 128 bytes packed INT4 weights
```

| Field | Bytes |
|-------|-------|
| FP16 superblock scale `d` | 2 |
| FP16 superblock min `dmin` | 2 |
| 12 bytes 6-bit packed sub-block scales + mins | 12 |
| 128 bytes packed INT4 weights | 128 |
| **Total** | **144 bytes / 256 weights = 4.5 bpw** |

### The two-level scaling trick (the load-bearing idea)

```
scale_b = d * unpack_6bit(sub_scales, b)        # FP16 × 6-bit → FP16 sub-scale
min_b   = dmin * unpack_6bit(sub_mins, b)
w[16*b + i] = scale_b * q[16*b + i] − min_b     # INT4 → FP16
```

The FP16 superblock master scale is multiplied by the 6-bit sub-block scale to recover the FP16 sub-block scale. **The scales themselves are quantized against a higher-level scale** — the same two-level idea NVFP4 (ch-17) uses with FP8 block scales + FP32 tensor scale.

This is the technique that gets q4_K to 4.5 bpw instead of 4 + (16 × 12)/256 = 4.75 — you double-quantize the scales, not just the weights.

---

## Quantization search (per super-block)

```
For each super-block of 256 weights w:
  for each candidate d in a small grid around max(|w|)/15:
    for each sub-block b:
      sub_scale, sub_min = best_per_sub_block(w[16b:16(b+1)], d)
    compute reconstruction error
  choose d minimising error; encode sub-scales as 6-bit
```

A grid search per super-block — fast, deterministic, runs entirely on CPU.

---

## The _S / _M / _L mixed-precision policy

The suffix denotes how aggressively llama.cpp upgrades sensitive tensors:

- **_S (small)**: all tensors at the base bit-width.
- **_M (medium, default)**: bump `attention.wv`, `attention.wo`, `ffn_down` to q{N+1}_K.
- **_L (large)**: bump even more tensors.

### Why these three layer types

Sensitivity analysis on Llama showed:

- **attention.wv** (value projection): most sensitive to quantization error → highest precision.
- **attention.wo** (output projection): high sensitivity → high precision.
- **ffn_down** (downward FFN projection): moderate sensitivity → +1 bit over baseline.
- attention.wq / attention.wk: low sensitivity → can use baseline.
- ffn_gate / ffn_up: low sensitivity → baseline.

A per-tensor-class echo of the [[squeezellm]] (ch-11) sensitivity-aware insight, but applied at deploy time without per-channel calibration.

---

## IQ-quants — importance-matrix calibration

Modern llama.cpp k-quants support **imatrix** (importance matrix) calibration: a per-channel weighting derived from running the model on a calibration set and capturing activation magnitudes. The k-quant search then weights MSE by activation magnitude — closing most of the gap to AWQ at the same bit-width.

| Quant | Effective bpw | Method |
|-------|---------------|--------|
| `iq1_S` | 1.5625 | super-block + LUT + IMatrix |
| `iq2_XXS` | 2.0625 | non-uniform codebook |
| `iq2_XS` | 2.3125 | |
| `iq3_XS` | 3.3 | |
| `iq4_XS` | 4.25 | |
| `iq4_NL` | 4.5 | "non-linear" — LUT |

The imatrix is conceptually a **lightweight cousin of GPTQ's Hessian** — same idea (weight quant error by activation importance), simpler implementation.

---

## Hardware acceleration

- **CPU**: AVX2, AVX-512, AVX-512-VNNI, ARM NEON, SVE, Apple AMX.
- **GPU support**: CUDA, ROCm, Vulkan, Metal via dequant-to-FP16/BF16 + tensor-core matmul.
- **Apple Silicon**: dequant + AMX matmul; fastest local LLM platform per dollar.

---

## Quality landscape (Llama-3-8B reference)

| Quant | Size | PPL gap vs BF16 |
|-------|------|-----------------|
| BF16 | 16 GB | 0 |
| Q8_0 | 8.5 GB | +0.001 |
| Q6_K | 6.6 GB | +0.005 |
| Q5_K_M | 5.7 GB | +0.01 |
| **Q4_K_M** | **4.9 GB** | **+0.03** ← consumer sweet spot |
| IQ4_XS | 4.4 GB | +0.05 |
| Q3_K_M | 4.0 GB | +0.10 |
| IQ2_M | 2.7 GB | +0.4 |
| IQ1_M | 1.9 GB | +1.0 (usable but degraded) |

---

## The community release pattern

Typical curator (e.g. `bartowski`, current dominant gguf curator) workflow:

1. Download BF16/FP16 weights from the model HF org.
2. Convert to gguf via `convert_hf_to_gguf.py`.
3. Generate imatrix from a 1-2k token calibration corpus.
4. Quantize to 10-12 variants: `Q8_0`, `Q6_K`, `Q5_K_M`, `Q5_K_S`, `Q4_K_M`, `Q4_K_S`, `IQ4_XS`, `Q3_K_M`, `IQ3_M`, `IQ2_M`, `IQ1_M`.
5. Upload all variants to a single HF repo with the imatrix file.
6. Publish PPL comparison table in the README.

The K-quant + IQ-quant ladder is arguably the **most-deployed quantization stack in absolute model count** in the world — every llama.cpp / Ollama / LM Studio user is running one of these variants, and curator hubs host literally tens of thousands of model-quant variants.

---

## Connections

- [[int4]] / ch-02 — q4_K is essentially group-32 asymmetric INT4 with double-quantized scales.
- [[int8]] / ch-02 — q8_0 is per-32-block INT8 with FP16 scale.
- [[gguf-k-quants]] / ch-02 — format spec.
- [[awq]] / ch-09 — imatrix calibration captures similar information as activation-aware scaling.
- [[squeezellm]] / ch-11 — sensitivity-aware tensor-level precision mixing; the _M variant idea.
- [[gptq]] / ch-08 — IQ-quants' importance-matrix idea is conceptually a stripped-down GPTQ Hessian.
- [[bitnet-models]] — bitnet.cpp is a fork of this stack, extending the format with ternary-specific encoding.
- [[ch-19]] — parent synthesis.
