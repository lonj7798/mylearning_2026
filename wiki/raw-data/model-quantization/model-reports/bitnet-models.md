<!-- scope: official BitNet model releases (Microsoft + community) and the bitnet.cpp inference framework
     deps: [[bitnet]], [[bitnet-b158]], [[bitnet-a48]]
     see-also: [[microsoft-bitnet]], [[llama-cpp-ggml]]
-->

# BitNet Official Model Releases + bitnet.cpp Inference Framework
- **Core Insight:** Microsoft and the community have released several real 1.58-bit (ternary) and 1-bit checkpoints — BitNet-b1.58-2B-4T (Microsoft official), bitnet_b1_58-3B (1bitLLM community), Llama3-8B-1.58 (HF1BitLLM, post-hoc converted), Falcon-E (TII) — all runnable through `bitnet.cpp`, an optimized CPU/GPU/NPU inference framework purpose-built for the ternary regime.
- **Guideline:** To actually run a 1.58-bit LLM today: grab `microsoft/BitNet-b1.58-2B-4T-gguf` from HF and `bitnet.cpp` from GitHub. Expect 1.4-5× speedups over llama.cpp's q2_K on the same CPU/GPU, with 55-82% energy reduction.
- **Authors:** Microsoft BitNet team (Shuming Ma, Hongyu Wang, et al.); 1bitLLM community; TII (Falcon team)
- **Year:** 2024-2025
- **URL:** https://github.com/microsoft/BitNet; https://huggingface.co/microsoft/BitNet-b1.58-2B-4T; https://huggingface.co/1bitLLM
- **Relevant topics:** 1.58-bit LLM, ternary weights, bitnet.cpp, edge inference, energy efficiency

## Abstract
The "Era of 1-bit LLMs" claim from [[bitnet-b158]] has produced real model releases starting in late 2024. Microsoft released **BitNet-b1.58-2B-4T** (2B parameters, 4T training tokens) as the official reference — matches FP16 perplexity on standard benchmarks. The community (1bitLLM org) released earlier proof-of-concept checkpoints at 700M and 3B. HF1BitLLM released a Llama-3-8B converted to 1.58-bit. TII released the Falcon-E family (Falcon ternary). All of these are served via `bitnet.cpp`, a fork-style llama.cpp-derived inference framework with custom kernels for the ternary {-1, 0, +1} regime — kernels that exploit the fact that ternary multiplies reduce to add/subtract/no-op, bypassing FMA entirely.

## Key Contributions
- First **officially trained from scratch** 2B-parameter 1.58-bit model with full training disclosure (`microsoft/BitNet-b1.58-2B-4T`).
- `bitnet.cpp` framework with ternary-specialized CPU kernels (AVX2/NEON), GPU kernels (May 2025), and announced NPU support.
- Documented **1.37-5.07× speedups on ARM**, **2.37-6.17× on x86** vs the strongest llama.cpp baseline.
- **55-82% energy reduction** vs equivalent FP16 inference.
- Demonstration that a **100B BitNet b1.58 model** can run at 5-7 tokens/sec on a single high-end CPU.

## Key Figures/Tables to Study
- BitNet-b1.58-2B-4T HF model card: MMLU / Winogrande / HellaSwag / ARC table against FP16 baselines and against Llama-3.2-1B/3B.
- bitnet.cpp README: speedup table per platform (Apple M2, Intel Xeon, AMD EPYC, Cortex).
- Energy table: joules-per-token vs FP16.

## Technical Details

### Released checkpoints
| Org / Repo | Model | Bits | Params | Training tokens | Notes |
|------------|-------|------|--------|-----------------|-------|
| `microsoft/BitNet-b1.58-2B-4T` | BitNet b1.58 | 1.58 | 2B | 4T | Official reference; trained from scratch |
| `microsoft/BitNet-b1.58-2B-4T-gguf` | same | 1.58 | 2B | 4T | gguf packaging for bitnet.cpp |
| `1bitLLM/bitnet_b1_58-large` | BitNet b1.58 | 1.58 | 700M | ~100B | Early community proof |
| `1bitLLM/bitnet_b1_58-3B` | BitNet b1.58 | 1.58 | 3B | ~100B | Larger community release |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | Llama-3-8B → 1.58 | 1.58 | 8B | +100B continued | Post-hoc converted from BF16 |
| `tiiuae/Falcon-E-*` | Falcon-E | 1.58 | 1B–10B | varied | TII ternary family |

### bitnet.cpp framework
- **Source**: https://github.com/microsoft/BitNet (Apache-2.0)
- **Lineage**: forked from llama.cpp; reuses the ggml tensor library + gguf format, replaces the integer/float matmul kernels with ternary-specialized ones.
- **Kernel idea**: weights stored packed (5 ternary weights → 1 byte using base-3 encoding, ~1.6 bits/weight); per-block scale stored as FP16/BF16. At matmul time, decode the byte → 5 ternary signs, then accumulate `±x_i` or skip-zero. No multiply needed.
- **Platforms**:
  - x86 (AVX2 + AVX-VNNI): 2.37-6.17× speedup vs llama.cpp q2_K.
  - ARM (NEON): 1.37-5.07× speedup, optimized for Apple Silicon.
  - GPU: added May 2025; CUDA + Metal kernels.
  - NPU: announced; Qualcomm + Intel NPU backends in development.
- **Lookup-table optimization**: precompute the 3^5 = 243-entry LUT for 5-element ternary blocks → integer-only inner loop.

### Tiling and parallel kernels
The latest releases add an additional **1.15-2.1× speedup** through parallel-kernel + configurable-tiling work, embedding quantization, and per-tile LUT methodologies.

### Quality claims
On the standard 0-shot harness, BitNet-b1.58-2B-4T is **within 1-2 points** of Llama-3.2-1B and Qwen2.5-1.5B at FP16 across MMLU / HellaSwag / Winogrande / ARC / GSM8K — at the same parameter count but ~10× smaller memory footprint and ~5× lower energy.

### Limitations
- Real-world quality at >3B not fully verified — most quality claims are at the 2B scale.
- Activation quantization in production releases is BF16 — full BitNet-a4.8 (4-bit activations) is still research, no public scaled checkpoint.
- Training from scratch requires custom kernels in the trainer too — not a trivial fine-tune-existing recipe.

## Connections
- [[bitnet]] — the original 2023 1-bit transformer paper.
- [[bitnet-b158]] — the 2024 ternary-weight paper that established the recipe these releases use.
- [[bitnet-a48]] — research extension with 4-bit activations; not yet in officially released checkpoints.
- [[microsoft-bitnet]] — lab summary of the team behind these releases.
- [[llama-cpp-ggml]] — the upstream framework bitnet.cpp forks from.
- [[gguf-k-quants]] — the bit-packing format bitnet.cpp inherits and extends with ternary-specific encoding.

## Notes
The existence of these releases — particularly the Microsoft 2B-4T checkpoint — is what moved the "Era of 1-bit LLMs" claim from a paper into a working production reality. The energy claims (55-82% reduction) are the most significant load-bearing argument for ternary models in 2025.
