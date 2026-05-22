<!-- scope: SpQR — bilevel group-wise quantization of dense weights plus an FP16 sparse outlier matrix
     deps: [[gptq]], [[llm-int8]]
     see-also: [[squeezellm]], [[owq]], [[qlora]]
-->

# SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
- **Core Insight:** Within a transformer layer, the dominant quantization-error contributors are not "outlier channels" but a sparse set of **individual outlier weights** (≈1% of the matrix); store those weights in fp16 in a sparse sidecar and quantize the dense remainder with very small (G=8–32) two-level group-wise scales — net ~4 bits/weight at <1% perplexity loss across all model sizes.
- **Guideline:** When 4-bit GPTQ leaves a residual perplexity gap, switch to SpQR with `bilevel` quant (group_size_inner=16, group_size_outer=128, also quantise the inner scales to 3 bits) and sparse-outlier threshold τ such that ~1% of weights stay fp16; reuse Marlin/AutoGPTQ kernels with sparse-mm fusion.
- **Authors:** Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2306.03078
- **Relevant topics:** sparse-quantized representation, bilevel grouping, weight-level outliers, near-lossless 4-bit

## Abstract
SpQR is a near-lossless weight-only PTQ that targets the gap between 4-bit and FP16 LLM quality. Two innovations: (1) **bilevel group-wise quantization** — extremely small inner groups (G_in = 16) for tight per-group scales, with the inner scales themselves quantized to 3 bits inside outer groups (G_out = 128), keeping the average bits-per-weight near 4; (2) a per-layer **sparse outlier matrix** that keeps the ~1% of weights whose GPTQ-style sensitivity exceeds threshold τ in fp16. Across LLaMA, OPT, Falcon, SpQR achieves <1% perplexity gap to FP16 at ~3.4–4.0 effective bits/weight, with a 15% inference speedup on a 33B model fitting in a single 24 GB GPU.

## Key Contributions
- Bilevel (nested) group scales: solves the GPTQ trade-off between scale overhead and tight per-group range without exploding bits.
- Per-weight (not per-channel) outlier detection via GPTQ sensitivity score — captures isolated extreme weights that channel-wise schemes miss.
- ~1% sparse FP16 sidecar in CSR form, fused into the GEMM kernel.
- Demonstrates **near-lossless** 4-bit on LLaMA-65B (≤0.1 ppl) — the first to close the gap.

## Key Figures/Tables to Study
- **Figure 1:** the bilevel quant diagram — fp16 weight → INT3 inner scale → fp16 outer scale.
- **Figure 4:** distribution of weight sensitivities — long tail justifies sparse extraction.
- **Table 1:** LLaMA-65B 4-bit ppl vs GPTQ — SpQR closes the residual gap.

## Technical Details

### Bilevel group-wise quantization
For weight matrix `W ∈ R^{d_out × d_in}`, partition each row first into outer groups of size `G_out = 128`, then each outer group into inner groups of size `G_in = 16`.

For each inner group (G_in = 16 weights):
```
s_inner = max(|W_g|) / (2^{b−1} − 1)        # b = 3 weight bits
W_q = round(W_g / s_inner)                   # INT3
```
The inner scales `{s_inner}` (8 per outer group) are themselves quantized to 3 bits within the outer group:
```
s_outer = max(s_inner_group) / (2^3 − 1)
ŝ_inner = round(s_inner / s_outer)            # INT3 scale-of-scale
```
- Effective bits-per-weight ≈ `b + 3/G_in + 16/G_out + 16/G_out` ≈ 3 + 3/16 + 32/128 ≈ 3.44.
- Tight per-16-weight range without storing fp16 per group.

### Per-weight outlier extraction
Run a GPTQ-like sensitivity score per weight:
```
sens_{i,j} = (W_{i,j} − Q(W_{i,j}))² / [H⁻¹]_{jj}
```
Pick the top ~1% globally per layer (or by threshold τ). These go into a sparse CSR matrix `S` at FP16. The dense quantization above is re-run with those weights masked to zero.

### Inference
```
y = SpQR-decoded GEMV(W_dense, x)  +  CSR-SpMV(S, x)
```
- Dense kernel: 2-level dequant (INT3 → fp16 via inner-scale × outer-scale) fused into matmul.
- Sparse kernel: standard CSR SpMV in fp16, ~1% nnz → negligible compute.

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Inner group G_in | 16 |
| Outer group G_out | 128 |
| Weight bits | 3 |
| Inner-scale bits | 3 |
| Outer-scale bits | 16 |
| Outlier fraction | ~1% |
| Outlier sensitivity | GPTQ-style, threshold-tuned per layer |
| Average bits/weight | ~3.4–4.0 |
| Calibration | 128 sequences × 2048 tokens |

## Connections
- Direct ancestor: [[gptq]] (same Hessian-based sensitivity score reused for outlier picking).
- Outlier-FP16 lineage rivals: [[squeezellm]] (Fisher + k-means + sparse), [[owq]] (whole-column outliers), [[llm-int8]] (activation outliers, FP16 column path).
- Companion paper from Dettmers: [[qlora]] (NF4 weight quant + LoRA fine-tune).
- Successor that drops sparse storage via rotations: [[quarot]], [[quip-sharp]].
- Framework integrations: [[autogptq]] (SpQR back-end), [[bitsandbytes-nf4]] (Dettmers' kernel stack).
