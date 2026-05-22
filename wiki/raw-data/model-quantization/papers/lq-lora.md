<!-- scope: LQ-LoRA — low-rank + quantized matrix decomposition for memory-efficient LLM fine-tuning at sub-3-bit
     deps: [[qlora]], [[loftq]]
     see-also: [[gptq]], [[efficientqat]]
-->

# LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning
- **Core Insight:** QLoRA quantizes the frozen base and trains a separate low-rank adapter; the cleaner formulation is to *decompose* each pretrained weight matrix into a quantized component + a high-precision low-rank component as a single joint optimization, where only the low-rank part trains and the bit-width per matrix is allocated to fit a global memory budget via integer linear programming.
- **Guideline:** For memory-budgeted fine-tuning of large LLMs, use LQ-LoRA: decompose W ≈ Q + LR via Fisher-weighted objective, solve an ILP to pick per-layer bit-width given a target memory cap, fine-tune only the low-rank L, R; achieves 2.75-bit LLaMA-2-70B fitting in 27GB.
- **Authors:** Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim
- **Year:** 2023 (revised 2024)
- **URL:** https://arxiv.org/abs/2311.12023
- **Relevant topics:** low-rank + quantized decomposition, Fisher-weighted PTQ, ILP bit allocation, sub-3-bit fine-tuning

## Abstract
LQ-LoRA decomposes each pretrained weight matrix as the sum of a high-precision low-rank component (the LoRA part) and a memory-efficient quantized component. During fine-tuning only the low-rank part updates while the quantized part remains static. The decomposition uses a Fisher-information-weighted reconstruction objective, and an integer linear program (ILP) allocates per-layer bit-widths to satisfy a global memory budget. Outperforms QLoRA and GPTQ-LoRA on RoBERTa and LLaMA-2 7B/70B; a 2.75-bit LLaMA-2-70B requires 27 GB GPU memory.

## Key Contributions
- Decomposition objective: `min_{Q, L, R} || (W − Q − LR^T) ||_F (Fisher-weighted)` with Q an integer-quantized matrix and L, R low-rank.
- Replaces NF4 + LoRA (QLoRA) with a *jointly* optimal decomposition rather than treating the LoRA adapter as a residual after quantization.
- ILP-based dynamic bit-width allocation across layers given a global memory budget.
- Achieves 2.75-bit average LLaMA-2-70B in 27 GB, enabling fine-tuning on a single 32 GB GPU.

## Key Figures/Tables to Study
- **Figure 1:** Decomposition diagram — W into Q + LR.
- **Table 4:** LLaMA-2-70B fine-tuning under different memory budgets — LQ-LoRA Pareto frontier vs QLoRA.
- **Figure 4:** Per-layer bit allocation chosen by the ILP — non-uniform across the transformer stack.

## Technical Details

### The decomposition
For each weight W ∈ ℝ^{m×n}:
`W ≈ Q + L R^T`
- Q ∈ {INT-b grid}^{m×n}, with b-bit per element + per-group scale.
- L ∈ ℝ^{m×r}, R ∈ ℝ^{n×r}, low rank r (typically 8 or 16).

Q is frozen after decomposition; L, R are trained during fine-tuning.

### Fisher-weighted reconstruction
`L_decomp = || diag(F)^{1/2} · (W − Q − LR^T) ||_F^2`
where F = diag of empirical Fisher information `E[(∂L/∂W)^2]`. Down-weights weights that downstream loss doesn't depend strongly on; up-weights critical weights.

### Alternating minimization
1. Fix Q, solve for L, R: weighted truncated SVD of (W − Q).
2. Fix L, R, solve for Q: per-group quantization of (W − LR^T).
3. Iterate.

### ILP bit-width allocation
Goal: minimize total reconstruction loss subject to memory constraint.
Variables: bit-width b_i ∈ {2, 3, 4, 8} per layer i.
Constraint: Σ_i bits_i ≤ B_total.
Solve as ILP; gives non-uniform allocation — early / attention layers often get more bits than late FFN layers.

### Memory math for LLaMA-2-70B
- Naïve FP16: 140 GB.
- Uniform 4-bit (QLoRA): 35 GB.
- LQ-LoRA 2.75-bit average: 24 GB + low-rank adapters (1 GB) + activations (2 GB) = 27 GB.

### Comparison to QLoRA
QLoRA: W_FP ≈ NF4(W); train separate LoRA L, R added to outputs. Two separate decompositions never aligned.
LQ-LoRA: W = Q + LR^T in one optimization; L, R explicitly cover what Q can't.

## Connections
- Direct predecessor: [[qlora]] (NF4 + LoRA, residual not joint).
- Adapter-aware quant initialization: [[loftq]].
- QAT alternatives: [[efficientqat]], [[bitdistiller]].
- Fisher-information weighting also used in: [[hawq]] (mixed-precision allocation).
- ILP allocation lineage: [[hawq]] sensitivity-based allocation.
