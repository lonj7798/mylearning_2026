<!-- scope: 2025 learnable rotations beyond fixed Hadamard for LLM quantization
     deps: [[spinquant]], [[quarot]]
     see-also: [[rotation-unification-2025]], [[orthogonal-finetuning-quant]], [[flatquant]]
-->

# Learnable Rotations Beyond Hadamard (2025)
- **Core Insight:** Random Hadamard rotations (QuaRot) are universal but suboptimal — the per-layer activation/weight distribution has structure a random matrix cannot exploit; parameterizing the rotation on the Stiefel manifold (SpinQuant) and training it against a quantization-loss objective beats random Hadamard by 0.5-1 % at W4A4, and 2025 work pushes this further with structured / sparse / fused-with-adjacent-layer rotations that are cheaper at inference.
- **Guideline:** Start with QuaRot (no training, zero inference cost) as the floor; if you need 0.5-1 % more, train SpinQuant rotations; if you need zero inference cost *and* the trained quality, fuse the trained rotation into a structured (Walsh-Hadamard-like) factorization or merge it into the adjacent layer's weights offline.
- **Authors:** various — Meta (SpinQuant lineage), Microsoft, IST-Austria, MIT Han Lab
- **Year:** 2024 (SpinQuant) and 2025 successors
- **URL:** SpinQuant: https://arxiv.org/abs/2405.16406; 2025 successor papers in arxiv cs.LG late 2025
- **Relevant topics:** learned orthogonal rotation, Stiefel manifold, Cayley parameterization, structured rotation, rotation fusion

## Abstract
QuaRot established the rotation-based quantization template using random Walsh-Hadamard matrices — universal, zero-cost, but not adapted to any specific layer's outlier structure. SpinQuant (Liu et al. 2024) showed the rotation can be *trained* on the Stiefel manifold via Cayley parameterization, picking up roughly 0.5-1 % quality at W4A4 by exploiting per-layer outlier structure. 2025 follow-ups extend the SpinQuant idea along three axes: (1) **structured** rotations (block-diagonal, Householder products, FFT-like factorizations) that have a fast multiply at inference; (2) **fused** rotations that merge with the adjacent layer's weight at training time so inference cost is zero; (3) **per-block** rotations that change within a single layer (e.g. one rotation per attention head, one per FFN expert). The 2025 takeaway: with the right factorization, you can have trained-rotation quality at fixed-Hadamard inference cost.

## Key Contributions
- **Stiefel manifold parameterization (SpinQuant)**: R = exp(skew-symmetric A) or R via Cayley R = (I - A)(I + A)^{-1}; train A by gradient descent against the quant-loss objective.
- **Structured factorizations (2025)**: parameterize R as a product of Householder reflectors or as a block-diagonal × permutation × block-diagonal — recovers ~ 80-90 % of the SpinQuant quality at ~ 10-30 % of the inference cost.
- **Fused rotations (2025)**: train R, then absorb R into the weight of the *next* (or previous) linear layer, leaving zero rotation-specific operations at inference. Requires careful chain analysis so the rotation can actually be folded (e.g. into the previous down-proj output, then it becomes the next layer's gate-proj input rotation).
- **Per-head / per-expert rotations**: one trained rotation per attention head and one per MoE expert, rather than one per layer — picks up another ~ 0.2-0.4 % at the cost of more parameters in the rotation table.
- **Joint training with codebook / scale**: rotation, per-group quant codebook, and per-channel scale optimized jointly via a single layer-output-MSE loss on a calibration set.

## Key Figures/Tables to Study
- Quality vs inference cost scatter plot at W4A4 for: random Hadamard, SpinQuant, structured-SpinQuant, fused-SpinQuant — the Pareto frontier.
- The per-block rotation visualization: heatmap of rotation magnitudes per attention head per layer — shows which layers benefit most from trained rotation.
- Ablation table: training the rotation alone vs jointly with codebook vs jointly with scale.

## Technical Details

### Cayley parameterization
- Skew-symmetric A ∈ R^{d×d}, A = -A^T.
- R = (I - A)(I + A)^{-1} is orthogonal for any A.
- Gradient flows back to A; constraint A = -A^T enforced by taking the skew-symmetric part of every update.

### Householder factorization
- R = H_1 · H_2 · … · H_k where H_i = I - 2 v_i v_i^T (rank-1 reflection).
- k controls expressivity; k = d gives full SO(d), k << d gives a low-cost subset.
- Multiply cost: O(k · d) per vector — fast for k << d.

### Block-diagonal structured rotation
- R = blkdiag(R_1, R_2, …, R_m) where each R_j is a small dense orthogonal block (e.g. 64×64).
- Multiply cost: O(d · b) where b is the block size.
- Composable with permutations: P · blkdiag(R_j) · P^T can express block-permuted patterns.

### Fusion into adjacent layer
- For a linear chain W_2 · σ(W_1 · A), if σ is identity (skip connection within a block), then a rotation R between W_1 and W_2 can be folded:
  - Train R such that W_1 · A becomes R · W_1 · A (rotate W_1's output);
  - At inference, replace W_2 with W_2 · R, no extra ops.
- For chains separated by non-linearities (most cases), need rotation-commutativity tricks: e.g. RMSNorm + LayerScale can be made to commute with diagonal scale changes, allowing fold.

### Per-head / per-expert
- Attention heads have different outlier statistics → per-head R_h.
- MoE experts route different token populations → per-expert R_e.
- Each adds (h or e) × O(d²) parameters in the rotation tables; small in absolute terms.

### Empirical quality at W4A4 on Llama-3 8B
| Method | MMLU drop vs FP16 |
|--------|--------------------|
| RTN (round-to-nearest) | ~ 12 % |
| GPTQ | ~ 5 % |
| QuaRot (random Hadamard) | ~ 2 % |
| SpinQuant | ~ 1 % |
| Structured-fused SpinQuant 2025 | ~ 1 % at zero inference cost |
| Per-head SpinQuant | ~ 0.7 % |

## Connections
- [[quarot]] — random Hadamard baseline.
- [[spinquant]] — the learned-rotation parent.
- [[rotation-unification-2025]] — meta-framework that this and SpinQuant fit inside.
- [[orthogonal-finetuning-quant]] — OFT/BOFT applied as adapters in the rotated frame.
- [[flatquant]] — affine extension; loses orthogonality but gains expressivity.
- [[duquant]] — dual-rotation; structured form of two stacked rotations.
