<!-- scope: AdaRound — per-weight learned rounding direction via Hessian objective
     deps: quantization-mapping, obs-obd
     see-also: brecq, gptq, omniquant
-->

# Up or Down? Adaptive Rounding for Post-Training Quantization (AdaRound)
- **Core Insight:** Rounding each weight to the nearest grid point is provably suboptimal for the layer's downstream loss; the optimal per-weight rounding direction (up vs down) can be learned by minimising the local Hessian-weighted reconstruction error `‖W·X − Ŵ·X‖²`, which is the parent objective for GPTQ.
- **Guideline:** For each layer, freeze all others, collect input activations X on a small calibration set (~1024 samples), parameterise each weight's rounding decision via a rectified-sigmoid `h(V)`, and optimise V by SGD to minimise the layer-wise MSE `‖W·X − (W̃ + h(V)·Δ)·X‖²` plus an annealed sparsity regulariser pushing h(V) toward {0,1}.
- **Authors:** Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort
- **Year:** 2020 (ICML)
- **URL:** https://arxiv.org/abs/2004.10568
- **Relevant topics:** PTQ, Hessian objective, learned rounding, layer-wise reconstruction, GPTQ ancestor

## Abstract
AdaRound challenges the universal assumption that "round-to-nearest is optimal" for PTQ. A Taylor expansion of the task loss around the FP weights shows the optimal per-weight rounding direction is determined by the per-layer Hessian `H = X Xᵀ`, not by which grid point is closer. Solving the discrete problem exactly is NP-hard, so AdaRound relaxes it with a continuous, rectified-sigmoid soft assignment annealed to {0,1} during a brief optimisation. With ~10k SGD steps per layer on a calibration set of 1024 images, AdaRound delivers state-of-the-art PTQ at 4-bit — matching QAT on ResNet and InceptionV3, beating it on MobileNet.

## Key Contributions
- Proves nearest rounding is suboptimal for any non-trivial Hessian via a 2nd-order Taylor argument.
- Reduces the per-layer problem to a quadratic in the rounding variables with Hessian H = X Xᵀ.
- Parameterises rounding direction by a rectified-sigmoid soft assignment + annealing schedule.
- Establishes layer-wise reconstruction with calibration activations as the PTQ paradigm.
- Direct ancestor of GPTQ (which solves the same objective with a sequential, exact OBS update).

## Key Figures/Tables to Study
- **Figure 2** — accuracy vs rounding-perturbation distance: nearest is not optimal.
- **Equation 14** — the rectified-sigmoid relaxation `h(V) = clip(σ(V)(ζ−γ) + γ, 0, 1)`.
- **Table 2** — ResNet-18/50 4-bit PTQ: AdaRound vs naive rounding.

## Technical Details

### Layer-wise objective (the load-bearing formula)
For a single layer with weights W, calibration inputs X, quantized weights Ŵ:
`min_{Ŵ}  E_X ‖W·X − Ŵ·X‖²  =  min_{ΔW}  ΔWᵀ · (X Xᵀ) · ΔW`
where ΔW = W − Ŵ. This is the **Hessian-weighted reconstruction objective**; H = X Xᵀ.

### Quantizer and the rounding decision
Standard uniform quantizer with step Δ:
`Ŵ = Δ · (⌊W/Δ⌋ + b)`
where b ∈ {0,1} is the per-weight rounding direction (down = 0, up = 1). Each weight has its own b.

### Relaxation (rectified sigmoid)
Replace b with a continuous soft assignment h(V) ∈ [0,1]:
`h(V) = clip(σ(V)·(ζ−γ) + γ, 0, 1),  γ=−0.1, ζ=1.1`
The "stretched" σ saturates exactly at {0,1} so the final discrete decision is hard.

### Full objective
`min_V  ‖W·X − Δ·(⌊W/Δ⌋ + h(V))·X‖² + λ·f_reg(V)`
where `f_reg(V) = Σ (1 − |2h(V) − 1|^β)` pushes h toward {0,1}; β is annealed from 20 down to 2 across iterations to encourage exploration then commitment.

### Training loop
- ~10k Adam steps per layer; ~1024 calibration samples.
- Quantize layer by layer in forward order (each subsequent layer sees the previous layer's quantized output, capturing error propagation).
- After convergence, set b = round(h(V)) and freeze.

### Why nearest is wrong (2nd-order argument)
Task loss around W:
`L(W + ΔW) ≈ L(W) + gᵀ ΔW + (1/2) ΔWᵀ H ΔW`
At a converged FP model g ≈ 0 so the loss is dominated by ΔWᵀ H ΔW. Off-diagonal H means coupled weights — flipping one weight's rounding can cancel another's error.

## Connections
- [[quantization-mapping]] — sits inside the PTQ branch; replaces nearest rounding.
- [[obs-obd]] — uses the same 2nd-order argument applied to weight removal (pruning).
- [[brecq]] — extends AdaRound from layer-wise to block-wise reconstruction.
- [[gptq]] — direct LLM-era descendant: solves the same Hessian objective with a sequential OBS update instead of soft relaxation; lives in `papers/gptq.md` (bucket 6).
- [[omniquant]] — modern heir for LLMs: learns clip + equivalent-transforms instead of per-weight rounding.
