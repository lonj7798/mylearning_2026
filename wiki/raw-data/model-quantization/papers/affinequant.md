<!-- scope: AffineQuant — learnable invertible affine transformations for LLM PTQ, generalizing the equivalent-transformation idea beyond per-channel scaling
     deps: [[omniquant]], [[smoothquant]]
     see-also: [[flatquant]], [[awq]], [[quarot]]
-->

# AffineQuant: Affine Transformation Quantization for Large Language Models
- **Core Insight:** OmniQuant's "equivalent transformations" restrict themselves to diagonal scaling (a per-channel affine); the optimization space can be widened to *general invertible affine maps* A·W with A square invertible, dramatically increasing freedom to reshape the distribution before quantization — provided a gradual mask schedule prioritizing diagonal elements is used to preserve numerical stability.
- **Guideline:** For PTQ where activation outliers block W4A4, use AffineQuant: insert a learned invertible A between activations and the linear (folded into the weight), train A with a gradual mask that first fits diagonal entries (SmoothQuant-equivalent) then expands off-diagonal entries; standard W4A4 RTN on the transformed weight.
- **Authors:** Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, Rongrong Ji
- **Year:** 2024 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2403.12544
- **Relevant topics:** equivalent transformation, invertible affine, gradual mask, W4A4 PTQ

## Abstract
AffineQuant generalises OmniQuant's diagonal equivalent transformations to general invertible affine matrices. The expanded optimization space includes far better quant-friendly solutions but introduces invertibility / conditioning concerns; AffineQuant addresses this with a gradual mask optimization scheme that first learns diagonal entries (Levitt-style identity initialization) and then progressively unmasks off-diagonal entries, guaranteed by the Levy-Desplanques theorem to keep A diagonally dominant and hence invertible. Achieves C4 PPL 15.76 on LLaMA-2-7B W4A4 (vs 18.02 prior SOTA) and SOTA W4A4 across LLaMA-30B/65B.

## Key Contributions
- Widens the equivalent-transformation search space from diagonal (SmoothQuant / OmniQuant) to general affine A ∈ ℝ^{d×d}.
- Gradual mask schedule M_t that controls which entries of A are trainable at step t, expanding from diagonal-only at t=0 to full at t=T.
- Levy-Desplanques (diagonal dominance) condition: training preserves invertibility throughout optimization.
- Significant PPL gains over OmniQuant at W4A4 on LLaMA family.

## Key Figures/Tables to Study
- **Figure 2:** The gradual-mask schedule — diagonal first, then expanding bands.
- **Figure 3:** Activation distribution before / after AffineQuant transformation — much flatter than after SmoothQuant diagonal scaling.
- **Table 2:** LLaMA-7B/13B/30B/65B W4A4 PPL — AffineQuant vs OmniQuant vs SmoothQuant.

## Technical Details

### The affine transformation
For linear `y = W x`, insert invertible A:
`y = (W A^{−1}) (A x) = W' x'`
where W' = W A^{−1} is folded into the weight offline, and x' = A x is computed online (or A is also folded into the previous linear when possible).

OmniQuant restricted A = diag(s) (per-channel scaling). AffineQuant lifts to full A.

### Gradual mask schedule
Let M_t ∈ {0, 1}^{d×d} be a binary mask. At training step t:
`A_t = I + M_t ⊙ (A − I)`
- t = 0: M_0 = I (only diagonal entries trainable; A_0 = diag).
- t = T/2: M = banded around diagonal.
- t = T: M = all-ones (full A trainable).
This schedule prevents the early optimization from driving A into singular regions.

### Invertibility guarantee (Levy-Desplanques)
A matrix is invertible if it is strictly diagonally dominant: `|A_{ii}| > Σ_{j≠i} |A_{ij}|`. The gradual mask + diagonal init keeps |A_{ii}| ≈ 1 while off-diagonal entries grow slowly, so the dominance is preserved across training.

### Loss
Per-block reconstruction MSE:
`L(A) = || f_FP(x) − f_quant(x; A) ||²`
optimised by AdamW on a small calibration set for a few hundred steps.

### Quantization on top
After A is learned and folded, apply standard per-channel weight RTN at 4-bit and per-token activation RTN at 4-bit. No GPTQ needed (though it can be added).

### Cost
Same as OmniQuant: per-block training on calibration data. Inference: A absorbed into W' offline (when possible) → zero added cost; or one extra small matmul per linear (Kronecker decomposition optional).

## Connections
- Direct predecessor: [[omniquant]] (diagonal equivalent transforms).
- Sibling / generalization: [[flatquant]] (Kronecker affine, flatness-targeted).
- Diagonal-only predecessor: [[smoothquant]], [[awq]].
- Rotation-only sibling: [[quarot]], [[spinquant]].
- Theoretical foundation for invertibility: Levy-Desplanques (1881) diagonal dominance theorem.
