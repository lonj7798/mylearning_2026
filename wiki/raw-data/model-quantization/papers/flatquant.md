<!-- scope: FlatQuant — per-layer learnable affine (Kronecker) transformations that flatten weight/activation distributions for W4A4 PTQ
     deps: [[quarot]], [[spinquant]], [[duquant]]
     see-also: [[affinequant]], [[omniquant]]
-->

# FlatQuant: Flatness Matters for LLM Quantization
- **Core Insight:** Even after orthogonal rotation, transformed weights and activations are often still "steep and dispersed" — what actually matters for uniform-interval quantization is *flatness* of the distribution, which requires the freedom of non-orthogonal *affine* transforms (A·W + b) rather than rotation alone.
- **Guideline:** For each linear, learn a small Kronecker-decomposed affine transform that minimises post-quant reconstruction error; fuse the transform into a single CUDA kernel with the GEMM so the runtime overhead is negligible (<5% prefill).
- **Authors:** Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, Xin Jiang, Wulong Liu, Jun Yao
- **Year:** 2024 (ICML 2025)
- **URL:** https://arxiv.org/abs/2410.09426
- **Relevant topics:** affine transformation, flatness, Kronecker decomposition, W4A4 PTQ, kernel fusion

## Abstract
FlatQuant argues that rotation-based methods (QuaRot, SpinQuant, DuQuant) achieve outlier suppression but still leave activation distributions sharp and uneven — far from the uniform-density ideal that uniform-interval quantization is designed for. The paper learns per-linear *affine* transformations A·W + b that explicitly flatten the post-transform distribution, parametrizing A as a Kronecker product of small matrices for cheap storage and fast fused inference. Achieves <1% accuracy loss on LLaMA-3-70B W4A4 with up to 2.3× prefill / 1.7× decoding speedup vs FP16, beating SpinQuant by 7.5%.

## Key Contributions
- Identifies flatness — not just outlier elimination — as the right optimization target for uniform-interval quant.
- Drops the orthogonality constraint of QuaRot/SpinQuant: A is a general invertible affine map, parametrized as Kronecker product A = A_1 ⊗ A_2 to keep parameter count and runtime cost low.
- Learnable per-layer transforms calibrated by minimizing post-quant block reconstruction MSE.
- Single fused kernel for affine-transform + INT4 GEMM removes most of the runtime cost.

## Key Figures/Tables to Study
- **Figure 2:** Activation histograms before/after rotation vs after affine — affine produces visibly flatter distributions.
- **Figure 4:** Kronecker decomposition diagram for A.
- **Table 3:** LLaMA-3-70B W4A4 PPL/accuracy vs FlatQuant / SpinQuant / QuaRot / DuQuant.

## Technical Details

### Affine transformation
For a linear `y = Wx`, FlatQuant inserts an invertible affine A:
`y = (W A^{−1}) (A x) = W' x'`
where `W' = W A^{−1}` is folded into the weight (offline) and `x' = A x` is computed online before quantization. Quant operates on x' and W', which are flatter than x and W.

### Kronecker parametrization
For d = d_1 · d_2 (e.g. 4096 = 64 × 64), let
`A = A_1 ⊗ A_2`, with A_1 ∈ ℝ^{d_1×d_1}, A_2 ∈ ℝ^{d_2×d_2}.
- Storage: d_1² + d_2² ≪ d² (e.g. 2·64² = 8192 vs 4096² ≈ 16M).
- Online cost: `Ax = vec(A_2 · X · A_1ᵀ)` where X = reshape(x, d_2 × d_1) — two small matmuls per token.

### Calibration objective
Minimize per-block reconstruction MSE:
`L(A) = || f_FP(x) − f_quant(x; A) ||²`
with quant = INT4 round-to-nearest for both W' and x'. Train A_1, A_2 with AdamW for a few hundred steps on a small calibration set; no quantization-aware fine-tuning of W is required.

### Bias term (the +b)
Per-channel learned bias added before quantization to recentre the distribution; absorbed into the bias of the following linear at fold time.

### Why affine beats orthogonal
Orthogonal R preserves L² norm — so a heavy-tailed distribution stays heavy-tailed. An invertible affine can *re-scale* per coordinate, mapping a Gaussian-with-spike onto a near-uniform distribution. Uniform-interval INT4 round-to-nearest is optimal for a uniform source (the Bennett 1/12 noise model holds tightly), so flattening is the right thing to do.

### Kernel fusion
Forward pass for one linear:
1. `x' = A_2 · reshape(x) · A_1ᵀ` (fused into one CUDA kernel).
2. quantize x' to INT4 (dynamic per-token, symmetric).
3. INT4 GEMM with W' (Marlin / TRT-LLM).
Reported overhead: <5% of prefill, <10% of decode vs raw INT4.

## Connections
- Rotation lineage it generalizes: [[quarot]] → [[spinquant]] → [[duquant]] → [[flatquant]].
- Equivalent-transformation predecessor: [[omniquant]] (learnable scaling/shifting), [[affinequant]] (concurrent affine work).
- Theoretical motivation: Bennett uniform-noise model from [[uniform-quantization-noise]] — flatness ↔ uniform source assumption.
- Weight quantizer paired: [[gptq]] or RTN.
