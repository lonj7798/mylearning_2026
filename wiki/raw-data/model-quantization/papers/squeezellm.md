<!-- scope: SqueezeLLM — sensitivity-weighted non-uniform k-means quant + dense-and-sparse outlier split
     deps: [[gptq]], [[hawq]]
     see-also: [[awq]], [[spqr]], [[owq]], [[nuqmm]]
-->

# SqueezeLLM: Dense-and-Sparse Quantization
- **Core Insight:** A weight's quantization error contribution to model loss scales with its Fisher information; instead of uniform RTN, run k-means on the weights *weighted by Fisher diagonal* so quant levels concentrate where it matters, and store the top-0.5% sensitive/outlier weights in a tiny sparse FP16 sidecar.
- **Guideline:** For ≤3-bit weight-only deployment, use SqueezeLLM with Fisher-weighted k-means (16 centroids for 4-bit, 8 for 3-bit) plus a 0.45% sparse-outlier matrix in CSR; serve with the SqueezeLLM kernel that does LUT-decode and sparse-add in one pass.
- **Authors:** Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer
- **Year:** 2023 (ICML 2024)
- **URL:** https://arxiv.org/abs/2306.07629
- **Relevant topics:** non-uniform quant, Fisher-weighted k-means, dense-and-sparse decomposition, memory-bound LLM inference

## Abstract
SqueezeLLM observes that single-batch LLM inference is memory-bandwidth-bound (not compute-bound) and that aggressive weight-only quantization is the highest-leverage knob. It contributes two pieces. First, **sensitivity-based non-uniform quantization**: run k-means on the weight scalars but weight each weight by the diagonal of the empirical Fisher information `(∂L/∂w)²` so the codebook centroids concentrate near weights that matter for the loss. Second, **dense-and-sparse decomposition**: store the top ~0.45% of weights (sensitive + outlier) in a separate FP16 sparse matrix and quantize the rest. Achieves near-lossless 3-bit on LLaMA-7B/13B with a 2.3× decode speedup over FP16 on an A6000.

## Key Contributions
- Casts low-bit weight quant as a *loss-aware codebook design* problem rather than a uniform-grid problem.
- Fisher-information-weighted k-means: closed-form weighted Lloyd updates, ~minutes per layer.
- Dense-and-sparse split: a small CSR-format FP16 outlier table absorbs the long tail without inflating average bits/weight.
- LUT kernel that decodes 3/4-bit indices into FP16 weights, fuses sparse outlier add, achieves memory-bound speed.

## Key Figures/Tables to Study
- **Figure 3:** weight magnitude vs Fisher diagonal — shows they don't agree, motivating Fisher-weighting over magnitude-weighting.
- **Figure 4:** k-means centroids on a non-uniform weight distribution — codebook clusters where loss-mass is.
- **Table 3:** LLaMA-7B 3-bit perplexity — SqueezeLLM beats GPTQ by 0.7–1.0 ppl.

## Technical Details

### Sensitivity-weighted non-uniform quantization
Per output-row (or per-group) weight vector `w ∈ R^G`, learn a codebook `C = {c_1, …, c_K}` of K levels (K = 2^b, b ∈ {2, 3, 4}) by **weighted k-means**:
```
min_{C, assign}  Σ_i F_i · (w_i − c_{assign(i)})²
```
where `F_i ≈ (∂L/∂w_i)²` is the diagonal Fisher information estimated by squared gradients on calibration data (next-token loss on Pile / C4).

Weighted Lloyd update:
```
c_k = Σ_{i : assign(i)=k} F_i w_i / Σ_{i : assign(i)=k} F_i
```
- Cluster assignments use plain nearest-neighbour to `c_k`.
- Per-row codebook (K floats per row) + per-weight k-bit index → effective bits ≈ b + 16·K/G.

This is the "non-uniform" part: the codebook is not a uniform grid. Levels concentrate where Fisher-mass is.

### Dense-and-sparse decomposition
Identify weights with the largest Fisher · weight² product (top 0.4–0.5%); pull them out into a sparse FP16 matrix `S` stored in CSR. The remaining "dense" weights are quantized as above.
```
W = Q(W_dense)  +  S_sparse
y = Q-decoded GEMV(W_dense, x)  +  CSR-SpMV(S_sparse, x)
```
- Sparse matrix is ~0.5% nnz → CSR overhead trivial, FP16 outliers fully preserved.

### LUT inference kernel
At decode time, the per-row k-bit index decodes through a 16-or-8-entry LUT (the row's codebook) into FP16; the GEMV proceeds as a normal FP16 dot product. The sparse outlier add is fused into the same kernel.

### Hyperparameters
| Knob | Value |
|------|-------|
| Bits b | 3 or 4 |
| Codebook size K | 8 (3-bit), 16 (4-bit) |
| Sparse outlier % | 0.4–0.5% |
| Sensitivity | diag-Fisher = mean squared grad over calibration |
| Calibration | 128 sequences C4 |
| k-means iterations | ~30 |

## Connections
- Sensitivity-aware-bit-allocation ancestor: [[hawq]] (Hessian-aware mixed precision).
- Outlier-preserving cousins: [[spqr]] (similar idea, different threshold), [[owq]] (whole-column outliers).
- Uniform-grid + Hessian rival: [[gptq]].
- Activation-aware uniform-grid rival: [[awq]].
- LUT-GEMM kernel lineage: [[nuqmm]] (LUT-GEMM), [[gguf-k-quants]] (q3_k, q4_k, q5_k families).
- Sub-2-bit successor with similar codebook spirit: [[aqlm]].
