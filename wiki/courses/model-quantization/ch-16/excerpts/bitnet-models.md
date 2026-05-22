---
chapter: ch-16
course: model-quantization
phase: read
excerpt_of: "BitNet Official Model Releases + bitnet.cpp Inference Framework"
source_url: https://github.com/microsoft/BitNet
created_at: "2026-05-21"
---

# Excerpt: BitNet model releases + bitnet.cpp inference framework

**Authors:** Microsoft BitNet team (Shuming Ma, Hongyu Wang, Furu Wei); 1bitLLM community; TII (Falcon team)
**Year:** 2024–2025
**URL:** https://github.com/microsoft/BitNet ; https://huggingface.co/microsoft/BitNet-b1.58-2B-4T
**Raw-data source:** [[raw-data/bitnet-models]]

---

## What this entry catalogues

The "Era of 1-bit LLMs" thesis ([[bitnet-b158]]) has produced **real model releases** starting late 2024. The headline release is Microsoft's official **BitNet-b1.58-2B-4T** (2B parameters, 4T training tokens, trained from scratch with the b1.58 recipe).

Plus the community follow-ups (1bitLLM, HF1BitLLM, TII Falcon-E) and the optimised `bitnet.cpp` inference framework — the moment the b1.58 paper becomes a deployable artefact.

---

## Released checkpoints

| Org / Repo | Model | Bits | Params | Training tokens | Notes |
|------------|-------|------|--------|-----------------|-------|
| `microsoft/BitNet-b1.58-2B-4T` | BitNet b1.58 | 1.58 | 2B | 4T | **Official reference**, scratch |
| `microsoft/BitNet-b1.58-2B-4T-gguf` | same | 1.58 | 2B | 4T | gguf packaging for bitnet.cpp |
| `1bitLLM/bitnet_b1_58-large` | BitNet b1.58 | 1.58 | 700M | ~100B | Early community proof |
| `1bitLLM/bitnet_b1_58-3B` | BitNet b1.58 | 1.58 | 3B | ~100B | Larger community release |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | Llama-3-8B → 1.58 | 1.58 | 8B | +100B continued | Post-hoc converted from BF16 |
| `tiiuae/Falcon-E-*` | Falcon-E | 1.58 | 1B–10B | varied | TII ternary family |

The `microsoft/BitNet-b1.58-2B-4T` checkpoint is the load-bearing artefact — first **officially trained from scratch** 2B-parameter 1.58-bit model with full training disclosure.

---

## bitnet.cpp framework

- **Source**: https://github.com/microsoft/BitNet (Apache-2.0).
- **Lineage**: forked from llama.cpp; reuses the ggml tensor library + gguf format, replaces the integer/float matmul kernels with **ternary-specialized** ones.

### Kernel idea

Weights stored packed (5 ternary weights → 1 byte using base-3 encoding, ~1.6 bits/weight); per-block scale stored as FP16/BF16. At matmul time, decode the byte → 5 ternary signs, then accumulate `±x_i` or skip-zero. **No multiply needed.**

```c
// pseudo-C for one ternary block (5 weights, base-3 encoded)
uint8_t b = packed_weights[byte_idx];
for (int k = 0; k < 5; k++) {
    int8_t w_k = base3_decode(b, k);  // {-1, 0, +1}
    if      (w_k == +1) acc += x[i + k];
    else if (w_k == -1) acc -= x[i + k];
    // w_k == 0: skip
}
```

### Lookup-table optimization

Precompute the `3^5 = 243`-entry LUT for 5-element ternary blocks → integer-only inner loop, no branches:

```c
// 5 activations + 1 packed-byte weight → table lookup gives the dot product
int32_t partial = lut_5elem[b][quantized_x[i:i+5]];
```

Faster on x86 (cache-resident table) than the branched ±x version.

### Platforms

| Platform | Speedup vs llama.cpp q2_K | Notes |
|----------|---------------------------|-------|
| x86 (AVX2 + AVX-VNNI) | **2.37–6.17×** | Intel Xeon, AMD EPYC |
| ARM (NEON) | **1.37–5.07×** | Apple Silicon, Cortex |
| GPU (May 2025) | varies | CUDA + Metal kernels |
| NPU | announced | Qualcomm + Intel NPU backends in development |

### Tiling + parallel kernels (late 2025)

Additional **1.15–2.1× speedup** through parallel-kernel + configurable-tiling work, embedding quantization, and per-tile LUT methodologies.

---

## Quality claims

BitNet-b1.58-2B-4T on the standard 0-shot harness:

| Benchmark | BitNet-2B-4T | Llama-3.2-1B (FP16) | Qwen2.5-1.5B (FP16) |
|-----------|--------------|---------------------|---------------------|
| MMLU | ~52 | ~46 | ~60 |
| HellaSwag | ~74 | ~63 | ~67 |
| Winogrande | ~70 | ~60 | ~65 |
| ARC-Challenge | ~50 | ~37 | ~50 |
| GSM8K | ~38 | ~30 | ~58 |

Within 1–2 points of Llama-3.2-1B-FP16 on most tasks; trailing Qwen2.5-1.5B on math-heavy GSM8K. At ~10× smaller memory footprint and ~5× lower energy than the FP16 baselines.

---

## Energy claims

bitnet.cpp measurements:
- **55–82% energy reduction** vs equivalent FP16 inference (measured joules-per-token across x86 + ARM).
- A **100B BitNet b1.58 model** projected to run at 5–7 tokens/sec on a single high-end CPU — the load-bearing claim for "1-bit LLMs as a path to CPU-only inference at frontier scale".

---

## Limitations

- **Real-world quality at > 3B not fully verified** — most quality claims are at the 2B scale.
- **Activation quantization in production releases is BF16** — full BitNet-a4.8 (4-bit activations) is still research, no public scaled checkpoint.
- **Training from scratch requires custom kernels in the trainer too** — not a trivial fine-tune-existing recipe.
- **Conversion to gguf-ternary is one-way** — cannot resume training from a converted artefact.
- **HF1BitLLM/Llama3-8B-1.58 is a post-hoc conversion** — uses the lambda-schedule fine-tune (era-of-1bit blog) and is not equivalent to a from-scratch BitNet b1.58 at the same scale.

---

## How to actually run it

```bash
# Clone bitnet.cpp
git clone https://github.com/microsoft/BitNet
cd BitNet
mkdir build && cd build
cmake .. -DBITNET_ENABLE_FP_BLAS=ON
make -j

# Download the 2B-4T gguf model
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf

# Run inference
./bin/main -m BitNet-b1.58-2B-4T-Q1_58.gguf -p "Tell me about LLM quantization." -n 256
```

Expect ~10–50 tokens/sec on a modern CPU (M2 Pro, Xeon Gold), ~100–300 tokens/sec on GPU (CUDA backend).

---

## Pitfalls

- **The 2B-4T checkpoint is the only frontier-quality reference.** Community 700M/3B trained on ~100B tokens are early proofs; expect lower quality.
- **Llama3-8B-1.58 is post-hoc converted** — don't compare it to from-scratch BitNet b1.58 of equivalent scale.
- **Activations are BF16 in released models** — the headline "1.58-bit LLM" refers to *weights*; ignore claims of "1.58-bit memory" that don't account for activation memory.
- **gguf packaging differs from native BitNet checkpoint format.** Convert before deploying with bitnet.cpp.
- **AVX-VNNI is required for max x86 speedup.** Older Xeons without VNNI get only ~2× over q2_K.

---

## Connections

- [[excerpts/bitnet-b158]] — the paper this release implements.
- [[excerpts/bitnet]] — original 1-bit paper; b1.58 is the ternary upgrade.
- [[microsoft-bitnet]] — lab summary; team is Furu Wei, Hongyu Wang, Shuming Ma.
- [[ch-19]] — [[llama-cpp-ggml]] is the upstream framework bitnet.cpp forks from; [[gguf-k-quants]] is the bit-packing format bitnet.cpp inherits and extends.
- [[era-of-1bit-llms]] — broader thesis the release operationalises.
