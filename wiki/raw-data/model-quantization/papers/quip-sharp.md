<!-- scope: QuIP# — E8 lattice codebook + randomized Hadamard incoherence + fine-tuning compensation
     deps: [[quip]], [[product-quantization]], [[vector-quantization]]
     see-also: [[aqlm]], [[quarot]], [[bitnet-b158]]
-->

# QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- **Core Insight:** Once incoherence processing has Gaussianised the weight distribution ([[quip]]), the optimal scalar quantizer is no longer best — you can do vector quantization in 8 dimensions using the E₈ lattice (the densest 8-dim sphere packing), getting strictly better rate-distortion than per-element rounding; combine with the faster randomised Hadamard transform for incoherence and a final fine-tuning pass and 2-bit LLM quality matches 3-bit GPTQ.
- **Guideline:** For 2-bit LLM PTQ in late 2023–2024, use QuIP#: randomized Hadamard for incoherence, E₈P (E₈ + half-integer shift) codebook with 256 codewords for 2 bits/weight, then 1–2 epochs of fine-tuning on the codebook assignments to compensate for residual error.
- **Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
- **Year:** 2024 (ICML 2024)
- **URL:** https://arxiv.org/abs/2402.04396
- **Relevant topics:** lattice codebook, E8 lattice, randomized Hadamard transform, vector quantization, sub-2-bit fine-tuning

## Abstract
QuIP# upgrades [[quip]] on three axes. (1) The random orthogonal incoherence matrix is replaced by a **randomised Hadamard transform** (RHT) `H = (1/√d) S H_d` where H_d is the Sylvester Hadamard and S is a random ±1 sign vector — same incoherence guarantees, O(d log d) instead of O(d²), and trivially block-diagonalisable. (2) Scalar rounding is replaced by **vector quantization on the E₈ lattice** — once weights are Gaussianised, the optimal 8-dim codebook is the densest sphere packing in 8 dims, the E₈ lattice; QuIP# uses E₈P, a half-integer-shifted E₈ that fits 256 codewords in a small ball (2 bits per weight in groups of 8). (3) A short **fine-tuning pass** updates codeword assignments to compensate for residual error. Achieves SOTA 2-bit LLaMA-2 70B PTQ, beating QuIP, GPTQ, AWQ, and OmniQuant by clear margins.

## Key Contributions
- **Randomized Hadamard transform** as incoherence processing — O(d log d) cost, fuses with adjacent ops.
- **E₈ lattice codebook (E₈P variant)** for 2-bit vector quantization — exploits Gaussianisation post-RHT.
- **Fine-tuning compensation** — short ~1-epoch refinement of the assignments and per-layer scales.
- First PTQ where **2-bit LLaMA-2-70B** is within 0.5 ppl of FP, beating prior 3-bit results.

## Key Figures/Tables to Study
- **Figure 1:** packing density vs dimension — E₈ peaks at 8-dim, justifying the lattice choice.
- **Figure 3:** post-RHT weight distribution — near-Gaussian and isotropic, the hypothesis VQ relies on.
- **Table 2:** LLaMA-2 70B 2-bit ppl across QuIP#, AQLM, QuIP, GPTQ — QuIP# leads.

## Technical Details

### Randomized Hadamard Transform (RHT)
Let `S ∈ {±1}^d` be a uniformly random sign vector and `H_d` the d×d Sylvester Hadamard (d power of 2). Define
```
R = (1/√d) · diag(S) · H_d
```
R is orthogonal, achieves the same µ-incoherence guarantees as a uniform random orthogonal, but the matmul `R x` costs O(d log d) via fast Walsh–Hadamard transform (FWHT). Pre/post-process W as `R^⊤ W R'` with independent S, S' on the two sides.

### E₈ lattice + E₈P codebook
The E₈ lattice in R⁸ achieves the densest sphere packing in 8 dimensions (kissing number 240). For a target rate of `b · 8` bits per 8-vector (e.g. 2 bits × 8 = 16 bits per vec → 2¹⁶ codewords, but QuIP# uses 256 in practice for 2 bits/weight after symmetries):

- Group rotated weights into 8-vectors.
- E₈P codebook: E₈ lattice points within a half-integer-shifted ball of radius chosen to give 256 codewords with high symmetry → 8 bits per 8-vec = 1 bit/weight nominal; combined with sign + scale gives 2 effective bits/weight.
- Each 8-vector is mapped to its nearest codeword (closest-point on E₈, fast via known algorithms).

Lookup at inference: 8-bit index → 8-dim FP16 vec via shared codebook.

### Fine-tuning compensation
After PTQ:
- Treat codeword assignments as latent variables, scale and codebook itself as learnable.
- Run ~1 epoch of cross-entropy distillation from FP teacher on C4 (50–500 sequences) updating only the scales and (optionally) reassigning codewords for worst-error groups.
- Recovers ~0.2–0.5 ppl on 70B at 2-bit.

### Inference
```
x_rot = R^⊤ x                        # FWHT, O(d log d)
y_rot = VQ-GEMV(W_E8P, x_rot)        # 8-vec dot-products via LUT
y     = R' y_rot                     # FWHT inverse
```
RHT operations fuse into the residual path / LayerNorm; net runtime cost is dominated by the LUT GEMV.

### Hyperparameters
| Knob | Value |
|------|-------|
| Bits | 2 (also 3, 4) |
| Codebook | E₈P, 256 entries |
| Group | 8 weights → 1 codeword |
| Incoherence | randomized Hadamard, both sides |
| Fine-tune | 1 epoch C4, AdamW lr 1e-5 |
| Calibration | 256 sequences for FT |

## Connections
- Parent: [[quip]] (random orthogonal + LDLQ; no lattice).
- Lattice / VQ ancestor: [[vector-quantization]] (LBG), [[product-quantization]] (PQ).
- Additive-quant rival at sub-2-bit: [[aqlm]].
- Rotation idea extended to activations too: [[quarot]], [[spinquant]].
- 1-bit/1.58-bit lineage as an alternative path to extreme compression: [[bitnet]], [[bitnet-b158]].
