<!-- scope: BRECQ — block-wise PTQ reconstruction with cross-layer dependencies
     deps: adaround, obs-obd
     see-also: qdrop, gptq, omniquant
-->

# BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction
- **Core Insight:** Layer-wise PTQ (AdaRound) is suboptimal because it ignores cross-layer error coupling; reconstructing a whole **block** (a residual sub-graph, e.g. one ResNet block or one transformer layer) jointly via per-block MSE captures the dependency and unlocks sub-4-bit PTQ.
- **Guideline:** Group layers into blocks at residual boundaries; for each block, jointly optimise all weight-rounding parameters to minimise `‖f_block_fp(X) − f_block_q(X)‖²` on a calibration set; use Fisher information as the metric weighting to approximate the per-block Hessian without storing it.
- **Authors:** Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, Shi Gu
- **Year:** 2021 (ICLR)
- **URL:** https://arxiv.org/abs/2102.05426
- **Relevant topics:** block-wise PTQ, cross-layer reconstruction, Fisher-information, sub-4-bit

## Abstract
BRECQ argues that AdaRound's per-layer reconstruction loses accuracy below 4-bit because it ignores the error a layer's quantization induces on downstream layers. The paper introduces a hierarchy — layer / block / stage / network — and shows empirically that **block** is the sweet spot: jointly optimising a small contiguous sub-graph captures cross-layer dependencies without exploding compute. The block-level Hessian is approximated by per-output Fisher information (squared gradients), which is cheap. BRECQ achieves 4-bit PTQ matching FP32 on ResNet, and is the first PTQ method to reach feasible 2-bit accuracy.

## Key Contributions
- Hierarchy of reconstruction granularities: layer ⊂ block ⊂ stage ⊂ network.
- Empirical demonstration block is the optimal grain (best accuracy/compute tradeoff).
- Fisher-information approximation to the block Hessian for tractable second-order weighting.
- Mixed-precision extension: sensitivity-based per-block bit allocation.
- First viable PTQ method at 2-bit for CNNs.

## Key Figures/Tables to Study
- **Figure 2** — accuracy vs reconstruction granularity (layer/block/stage/network); block wins.
- **Table 4** — sub-4-bit PTQ ResNet-50: BRECQ vs AdaRound vs ZeroQ vs LSQ-QAT.

## Technical Details

### Block-wise objective (the load-bearing formula)
For a block B_k with FP function f_k and quantized version f̂_k:
`min_{W_k}  E_X ‖f_k(X) − f̂_k(X; W_k)‖²_F`
where ‖·‖_F is the Fisher-information-weighted Frobenius norm:
`‖y‖²_F = Σ_i diag(F)_i · y_i²,  diag(F) = E[(∂L/∂y)²]`
≈ Gauss-Newton diagonal approximation to the task Hessian w.r.t. block output.

Compared with AdaRound's per-weight `‖W·X − Ŵ·X‖²` (a single linear layer), BRECQ's objective acts on the **block output** (an entire residual sub-graph including non-linearities), capturing all internal couplings.

### Why per-weight isn't enough
AdaRound minimises `ΔWᵀ (X Xᵀ) ΔW` per layer. If layer L_k is quantized first, layer L_{k+1} sees a perturbed input X' ≠ X; AdaRound for L_{k+1} optimises against X' but ignores how its own ΔW_{k+1} amplifies the upstream perturbation. Block reconstruction folds both errors into one loss.

### Soft assignment (inherited from AdaRound)
Per-weight rectified-sigmoid `h(V)` parameterisation, optimised by Adam, with annealed sparsity regularisation:
`L_total = L_recon + λ·Σ (1 − |2h(V) − 1|^β)`

### Mixed-precision bit allocation
For each block, compute the second-order sensitivity:
`Ω_k = trace(H_k)·‖ΔW_k(b)‖²`
and allocate higher bit-widths to blocks with larger Ω_k subject to a global memory budget. This is essentially HAWQ-style allocation operating on BRECQ's block granularity.

### Calibration data
1024 samples typical; gradient w.r.t. block output computed once during a forward+backward of the FP model to estimate Fisher.

## Connections
- [[adaround]] — direct predecessor; BRECQ generalises layer → block.
- [[obs-obd]] — second-order foundations (same Taylor argument).
- [[qdrop]] — orthogonal: randomly drops quantization during BRECQ optimisation to regularise.
- [[hawq]] — same mixed-precision bit-allocation philosophy.
- [[gptq]] — LLM-era descendant of the AdaRound/BRECQ line; operates per-layer with an exact sequential OBS update.
- [[omniquant]] — block-wise LLM PTQ that follows BRECQ's grain choice but learns equivalent transformations instead of per-weight rounding.
