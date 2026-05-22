<!-- scope: SpinQuant — learned orthogonal rotations (Cayley-parametrized SGD on the Stiefel manifold) replace random Hadamards for W4A4KV4 LLM PTQ
     deps: [[quarot]]
     see-also: [[duquant]], [[flatquant]], [[quip-sharp]]
-->

# SpinQuant: LLM Quantization with Learned Rotations
- **Core Insight:** Among the infinite family of computationally-invariant rotations, different choices differ by up to ~13 points in downstream accuracy — so the rotation should be learned (via Cayley-parametrized SGD on the Stiefel manifold of orthogonal matrices), not random.
- **Guideline:** Replace the random Hadamards in a QuaRot-style pipeline with rotations R1–R4 trained for ~hundreds of steps with Cayley SGD on a small calibration set; the same inference graph as QuaRot, but better accuracy especially at W4A4KV4.
- **Authors:** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort
- **Year:** 2024 (ICLR 2025)
- **URL:** https://arxiv.org/abs/2405.16406
- **Relevant topics:** learned rotation, Stiefel manifold, Cayley parametrization, W4A4 PTQ, KV-cache quantization

## Abstract
SpinQuant generalizes QuaRot's random-Hadamard rotation to *learnable* rotation matrices that retain full-precision-equivalent outputs while specifically optimizing post-quantization accuracy. Different choices of rotation matrix yield up to 13-point swings on downstream reasoning, so SpinQuant treats them as trainable parameters constrained to the Stiefel manifold via Cayley parametrization. On LLaMA-2 7B at W4A4KV4, SpinQuant narrows the gap to FP to 2.9 points, beating LLM-QAT by 19.1, SmoothQuant by 25.0, and improving over QuaRot by up to 45.1% on LLaMA-3 8B.

## Key Contributions
- Empirical result: among rotation choices preserving computational invariance, downstream accuracy varies by >13 points — random Hadamard is not optimal.
- Casts rotation selection as constrained optimization on the Stiefel manifold St(d, d) = {R ∈ ℝ^{d×d} : RᵀR = I}.
- Trains rotations R1–R4 (same insertion points as QuaRot) end-to-end with a quantization-aware loss on a small calibration corpus.
- Demonstrates SOTA W4A4KV4 PTQ on LLaMA-2 (7B/13B/70B) and LLaMA-3 (8B/70B).

## Key Figures/Tables to Study
- **Figure 1 / Table 1:** Accuracy spread across random rotations vs learned rotation — the empirical motivation.
- **Section 3:** Cayley parametrization schematic for R1/R2/R3/R4.
- **Table 4:** LLaMA-2 7B W4A4KV4 ARC/HellaSwag/PIQA — SpinQuant vs QuaRot vs baselines.

## Technical Details

### Rotation indices (same insertion graph as QuaRot)
- **R1:** residual-stream rotation, folded offline into embedding, W_q/W_k/W_v/W_{up}/W_{gate}, W_o, W_{down}, and LM-head.
- **R2:** rotation between V and W_o (and on K after RoPE, mirrored), applied online.
- **R3:** rotation between SwiGLU output and W_{down}, applied online.
- **R4:** rotation on K cache after RoPE.
R1 and R2 are dense d×d learnable; R3 and R4 are block-diagonal Hadamards for inference speed.

### Stiefel manifold + Cayley parametrization
Each learnable rotation R is parametrized as
`R = (I − A)(I + A)^{−1}`
where A is a learnable *skew-symmetric* matrix (A = −Aᵀ). This Cayley map is a diffeomorphism from skew-symmetric matrices onto a dense subset of SO(d), so any gradient step on A produces a valid orthogonal R automatically — no projection or QR retraction needed.

Equivalent view: optimize R on the Stiefel manifold St(d, d) using Riemannian SGD; the Cayley map is the closed-form retraction.

### Loss
Per-block reconstruction loss between FP output and quantized output of the same block:
`L(R) = Σ_blocks || f_FP(x) − f_quant(x; R) ||²`
optimized over a few hundred steps with a small calibration corpus (typically WikiText-2 segments). Weights are GPTQ-quantized after R is frozen; activations are dynamic per-token round-to-nearest.

### Why learning beats random
Random Hadamard uniformly spreads any single outlier coordinate over all d dimensions. But real LLM activations have *structured* outliers (specific channels, head-correlated). Learned R can match this structure — concentrating rotation mass where outliers actually live — recovering several extra points vs the uniform Hadamard prior.

### Inference cost
Same as QuaRot. R1 is folded offline (zero runtime cost). R2/R3/R4 are block-diagonal Hadamards (~O(d log d) per token), implemented with fused CUDA kernels.

## Connections
- Direct predecessor: [[quarot]] — same insertion graph, random Hadamard.
- Sibling adding permutation: [[duquant]].
- Sibling with non-orthogonal affine: [[flatquant]].
- Codebook side: [[quip-sharp]] uses learned rotations with lattice codebooks.
- Optimization geometry: relates to orthogonal fine-tuning lineage (BOFT, OFT).
- Weight quantizer used after rotation: [[gptq]].
