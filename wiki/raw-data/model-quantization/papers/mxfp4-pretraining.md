<!-- scope: MXFP4 LLM pretraining (Tseng et al., AISTATS 2025)
     deps: [[mx-formats]], [[microscaling-formats]]
     see-also: [[nvfp4-training]], [[mxfp4-native-hardware-2026]], [[deepseek-v3-fp8]]
-->

# Training LLMs with MXFP4
- **Core Insight:** 4-bit MXFP4 pretraining can match BF16 quality at GPT-scale (up to 6.7B) if a Random Hadamard Transform is applied before quantization and Stochastic Rounding is used on the backward GEMM — RHT keeps the per-block range tight enough for FP4 to resolve, SR keeps the gradient estimator unbiased.
- **Guideline:** When using the OCP MXFP4 format (E2M1 elements, E8M0 shared scale, 32-element blocks), apply a random Hadamard transform on the inputs to the linear layers (folded into the weight offline) and switch the backward GEMM to stochastic rounding; without both, MXFP4 diverges or underperforms.
- **Authors:** Albert Tseng, Tao Yu, Youngsuk Park
- **Year:** 2025 (submitted Feb 2025; AISTATS 2025)
- **URL:** https://arxiv.org/abs/2502.20586
- **Relevant topics:** MXFP4 pretraining, microscaling, random Hadamard transform, stochastic rounding, low-precision training

## Abstract
This is the first academic study (outside hardware vendors) of using OCP MXFP4 — 32-element blocks of FP4 E2M1 with an 8-bit shared exponent — as the GEMM compute format throughout LLM pretraining. Naïve MXFP4 pretraining diverges or stalls due to (a) outliers in activations that crush the 32-element block scale and (b) round-to-nearest bias that compounds over millions of steps. The authors fix both: a Random Hadamard Transform (RHT) on the GEMM inputs spreads outlier energy across the block; SR on the backward GEMM keeps the gradient estimator unbiased. With these two fixes, GPT-style models up to 6.7B parameters trained on MXFP4 GEMMs match a BF16-mixed-precision baseline on validation loss and downstream evals, while running > 1.3× faster than FP8 backprop and > 1.7× faster than BF16 backprop on supported hardware. MXFP4 GEMMs are roughly 2× faster than FP8 on Blackwell.

## Key Contributions
- First rigorous demonstration that MXFP4 — the OCP "microscaling" 4-bit float with 32-element block + E8M0 scale — is a viable *training* (not just inference) format for LLMs.
- Identifies two failure modes of vanilla MXFP4 pretraining: block-scale collapse from activation outliers, and biased gradients from RNE accumulating over the training run.
- Solution 1: pre-multiply the GEMM input by a Random Hadamard matrix (folded into the weight offline) — provably bounds the per-block variance and theoretically reduces SR variance as well.
- Solution 2: stochastic rounding on backward GEMMs — preserves the unbiased-gradient property even at FP4 element precision.
- Reports > 1/2 the training FLOPs of BF16 with comparable loss on GPT-1.3B / 2.7B / 6.7B.
- Empirically validates MXFP4 GEMM speedups: ≥ 1.3× over FP8, ≥ 1.7× over BF16 on the backward pass.

## Key Figures/Tables to Study
- **Loss curves figure:** MXFP4-with-RHT-and-SR overlaid on BF16 baseline through full training — flat gap.
- **Ablation table:** removing RHT alone (diverges on 6.7B) vs removing SR alone (slower convergence, plateau higher) — both are load-bearing.
- **Speedup table:** wall-clock per-step for BF16 / FP8 / MXFP4 backward on the same hardware.

## Technical Details

### Format
- **Element:** FP4 E2M1 (the OCP spec value) — same as NVFP4's element.
- **Block:** **32 elements** (vs NVFP4's 16) sharing **one E8M0 scale** (8-bit unsigned exponent only, no mantissa, no sign — a pure power-of-two scale).
- **No per-tensor scale** (vs NVFP4's FP32 outer scale) — relies entirely on the E8M0 block scale to absorb dynamic range.
- This is the **OCP Microscaling MXFP4** spec, not NVIDIA's NVFP4 variant.

### Random Hadamard transform (RHT)
- Applied to the input X of each linear layer before MXFP4 quantization.
- Hadamard matrix H is a random ±1 matrix of size matching the inner dimension; folded into the weight offline as W' = W · H^T, so inference cost is zero.
- Theoretical motivation: the SR variance for a fixed FP4 quantizer scales with the per-block max; RHT bounds the per-block max by O(√(log d / d)) of the tensor's L2 norm, which is what keeps the block scale tight.

### Stochastic rounding (SR) on backward
- For each FP4 element, round up with probability proportional to the fractional position, down otherwise — E[SR(x)] = x.
- Used only on the backward GEMMs (input-grad and weight-grad). Forward uses RNE so inference is deterministic.

### Selective precision
- Embeddings, LM-head, and normalizations kept in higher precision (BF16) — same pattern as DSV3-FP8 and NVFP4 work.
- Optimizer state in FP32; master weight in BF16.

### Empirical scales
| Model | MXFP4 final loss vs BF16 | Backward speedup vs BF16 |
|-------|--------------------------|--------------------------|
| GPT-1.3B | matches | ~ 1.7× |
| GPT-2.7B | matches | ~ 1.7× |
| GPT-6.7B | matches (within noise) | ~ 1.7× |

### Why MXFP4 vs NVFP4
- MXFP4 is OCP-standard, runs on any vendor that adopts the MX spec; NVFP4 is Blackwell-native.
- MXFP4's 32-element block is more memory-efficient than NVFP4's 16-element block but has less tight scaling — RHT is more load-bearing for MXFP4 than for NVFP4.
- NVFP4's extra FP32 per-tensor scale + FP8 block scale gives it more dynamic range; MXFP4 needs RHT to compensate.

## Connections
- [[mx-formats]] / [[microscaling-formats]] — the OCP MX spec that MXFP4 instantiates.
- [[nvfp4-training]] — NVIDIA's parallel work; same recipe family (RHT + SR + selective precision) at a different block size.
- [[mxfp4-native-hardware-2026]] — May 2026 native-hardware follow-up; separates Fprop/Dgrad/Wgrad and reports Wgrad as the main convergence failure point.
- [[deepseek-v3-fp8]] — the FP8 elder cousin; DSV3's per-block scaling generalizes the same intuition (block-scoped scales beat per-tensor) one bit-width down.
- [[stochastic-rounding]] — the underlying technique (Gupta 2015), here applied to FP4 elements.
- [[quarot]] / [[spinquant]] — the inference-time RHT lineage that this paper borrows from for training.

## Notes
May 2026 update: [[mxfp4-native-hardware-2026]] should be read after this paper. It does not invalidate this AISTATS 2025 result, but it narrows the claim: full-pipeline MXFP4 training needs separate treatment of Wgrad, and deterministic Hadamard rotations may matter more than stochastic rounding once Wgrad is quantized on native hardware.
