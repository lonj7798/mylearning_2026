<!-- scope: 2025 refinements to sensitivity-based / non-uniform LLM quantization
     deps: [[squeezellm]], [[spqr]]
     see-also: [[awq]], [[gptq]], [[aqlm]]
-->

# 2025 Refinements to Sensitivity-Based Non-Uniform Quantization
- **Core Insight:** The 2025 follow-ups to SqueezeLLM and SpQR push sensitivity-based non-uniform quantization further by combining (a) Hessian-trace-weighted k-means codebooks with (b) explicit outlier sparse paths and (c) Hadamard-rotated baselines, narrowing the W4 / W3 / W2 quality gap to the rotation-based methods (QuaRot, SpinQuant) without paying their dense rotation cost at inference.
- **Guideline:** Use sensitivity-weighted non-uniform W4 (SqueezeLLM-family) when you can pay for a dense + sparse kernel path; switch to rotation-based W4 (QuaRot/SpinQuant + GPTQ) when you only have a pure dense INT4 GEMM path.
- **Authors:** various follow-ups (Kim et al., Park et al., MIT Han Lab, IST-Austria); SqueezeLLM original by Sehoon Kim et al.
- **Year:** 2024-2025
- **URL:** original SqueezeLLM: https://arxiv.org/abs/2306.07629 ; 2025 follow-ups under the SqueezeLLM and SpQR lineages
- **Relevant topics:** non-uniform quantization, sensitivity-weighted k-means, dense+sparse decomposition, outlier preservation

## Abstract
SqueezeLLM (2023) showed that non-uniform 4-bit quantization could beat uniform 4-bit by allocating codebook entries where the loss-Hessian-weighted weight distribution actually concentrates. SpQR (2023) showed that keeping a 1-2 % sparse outlier path at FP16 preserved most of the quality of W4. 2025 follow-ups combine the two ideas with more aggressive bit allocations (W3, W2) and with rotation-based pre-processing (apply a Hadamard rotation first, *then* fit the codebook to the rotated weights' Hessian-weighted distribution). The resulting recipes are competitive with rotation-only methods at W3 but rely on a sparse outlier path for the last 0.5-1 % of weights — a kernel cost that vLLM-style dense W4 GEMMs avoid.

## Key Contributions
- Combine **Hessian-trace-weighted k-means** (SqueezeLLM's core idea) with **explicit sparse outlier preservation** (SpQR's core idea) — neither alone matched rotation-based methods at W3, the combination does.
- Apply codebook fitting *after* a random Hadamard rotation: rotated weights are closer to Gaussian, so a small codebook (16 levels at 4-bit) fits the bulk well and only the outliers need the sparse path.
- Per-layer bit allocation: layers with higher Hessian trace get more bits; the budget can be expressed as average bpw rather than per-layer.
- Kernel: dense LUT-based GEMM for the in-codebook majority + sparse CSR FP16 GEMM for the outlier path; vLLM and SGLang have community kernels for this hybrid.

## Key Figures/Tables to Study
- The per-layer sensitivity (Hessian trace) bar chart; explains why attention layers want more bits than FFN.
- Codebook visualization after Hadamard rotation vs raw weights — the rotated distribution is visibly closer to a parametric Gaussian.
- Quality vs bits-per-weight curve at W2 / W3 / W4 for SqueezeLLM-follow-up vs QuaRot vs AQLM.

## Technical Details

### Sensitivity-weighted k-means
- For each weight w_i, compute s_i = (∂²L / ∂w_i²) — diagonal Hessian, approximated by Fisher information from a calibration set.
- Solve weighted k-means: minimize Σ_i s_i · (w_i - c_{k(i)})² over codebook {c_k} and assignments k(i).
- Result: codebook concentrates levels where the loss cares most, not where weight density is highest.

### Dense + sparse decomposition
- After k-means, identify the top 0.5-1 % of weights with the largest sensitivity-weighted quantization error.
- Keep those in a separate FP16 sparse tensor (CSR or per-row sparse list).
- The dense path serves the in-codebook majority; the sparse path is added in the epilogue.

### Hadamard pre-rotation
- Apply a random Hadamard transform H to the weight matrix W → W' = W · H^T; activations are rotated by H so the layer math is unchanged.
- After rotation, W' is closer to Gaussian; both the k-means codebook and the outlier mask are tighter.

### Kernel
- LUT-based: per-channel codebook of K levels (K=16 for 4-bit); main-loop dequant is an `ldmatrix` from per-CTA shared LUT + MMA.
- Sparse epilogue: standard CSR SpMV at FP16, added to the dense output.
- Throughput: roughly 0.7-0.9× of Marlin's pure-dense W4A16; the sparse path is the cost.

### Bit allocation
- Per-layer optimal bpw computed by Lagrangian over Hessian trace; expressed as an average target bpw.
- Common settings: average 3.0 bpw with some layers at 2 and others at 4; or pure 4 bpw with selective W2 in the late-FFN layers.

## Connections
- [[squeezellm]] — the parent paper this line extends.
- [[spqr]] — the sparse-outlier idea, combined here with sensitivity-weighted codebooks.
- [[awq]] — the activation-aware scale baseline; sensitivity-based methods use a similar calibration set but a different objective.
- [[gptq]] — Hessian-aware uniform quantization; sensitivity-based methods use the same Hessian but a non-uniform codebook.
- [[quarot]] / [[spinquant]] — rotation-based methods that compete in the same W3/W4 regime via a different mechanism.
- [[aqlm]] — additive-quantization-based sub-2-bit method; the W2 alternative.
