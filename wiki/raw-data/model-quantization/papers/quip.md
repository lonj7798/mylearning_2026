<!-- scope: QuIP — incoherence processing via random orthogonal pre/post-rotation + LDLQ adaptive rounding
     deps: [[gptq]], [[adaround]], [[obs-obd]]
     see-also: [[quip-sharp]], [[quarot]], [[spinquant]], [[aqlm]]
-->

# QuIP: 2-Bit Quantization of Large Language Models With Guarantees
- **Core Insight:** Adaptive rounding (GPTQ/AdaRound) is provably optimal precisely when the weight matrix and the calibration Hessian are *incoherent* — their entries are spread evenly with no single dominating direction; you can *enforce* incoherence by sandwiching W between random orthogonal matrices `U^⊤ W V`, run a Hessian-aware rounding (LDLQ) in the rotated frame, then undo the rotation at inference, unlocking the first viable 2-bit LLM PTQ.
- **Guideline:** For 2-bit weight-only LLM PTQ, apply random orthogonal U (left) and V (right) to W (and to the Hessian), then run LDLQ block-wise on the rotated weight; at inference, replace the per-layer matmul with `V (Q (U^⊤ x))` (the rotations cost two extra matmuls but kill outliers).
- **Authors:** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa
- **Year:** 2023 (NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2307.13304
- **Relevant topics:** incoherence processing, random orthogonal rotation, LDLQ adaptive rounding, 2-bit LLM PTQ, first theoretical PTQ guarantees

## Abstract
QuIP is the first PTQ to produce viable 2-bit LLM weights and the first to come with theoretical guarantees on the quantization error. The paper formalises *incoherence* of a weight–Hessian pair `(W, H)` — the property that no single entry of W or eigenvector of H is disproportionately large — and proves that adaptive rounding methods (GPTQ is a special case) achieve near-optimal error precisely when (W, H) are µ-incoherent. To turn this into an algorithm, QuIP pre-processes the layer by left- and right-multiplying W by uniformly random orthogonal matrices U, V; the rotated weight `U^⊤ W V` is provably µ-incoherent with high probability. A novel adaptive-rounding procedure (LDLQ) then quantizes the rotated weight in optimal order derived from the LDL decomposition of H. At inference, the rotations are absorbed into surrounding ops (or kept as fused matmuls). On OPT, LLaMA, and Falcon, QuIP holds 2-bit quality where GPTQ collapses.

## Key Contributions
- **Theory**: defines the incoherence assumption and proves O(1/2^b) error scaling for adaptive rounding under it — the first PTQ theorem at LLM scale.
- **Incoherence processing**: explicit, cheap preprocessing (random orthogonal rotations) that makes the assumption hold.
- **LDLQ**: a new adaptive-rounding algorithm that processes weights in the order given by the LDL decomposition of H, generalising GPTQ.
- First **2-bit** LLM PTQ that holds zero-shot accuracy (LLaMA-65B at 2 bits within 1 ppl of FP).

## Key Figures/Tables to Study
- **Figure 2:** distribution of weight magnitudes before vs after random rotation — heavy-tailed → near-Gaussian.
- **Figure 4:** OPT/LLaMA 2-bit and 3-bit perplexity vs GPTQ — QuIP wins by 5–30 ppl at 2-bit.
- **Theorem 1/2:** the incoherence-conditional error bound — the rare PTQ paper with proofs.

## Technical Details

### µ-incoherence
A pair (W, H) is µ-incoherent if for all i, j:
```
|W_{i,j}|  ≤  μ · ||W||_F / sqrt(d_out · d_in)
|⟨v_i, e_j⟩|² ≤ μ / d         (v_i = i-th eigvec of H)
```
The maximum-entry of W is bounded by a constant times the average, and the eigenvectors of H are not axis-aligned. Heavy-tailed LLM weights and Hessians violate this; rotated versions satisfy it w.h.p.

### Incoherence processing (the rotation step)
Draw uniformly random orthogonal matrices `U ∈ O(d_out)`, `V ∈ O(d_in)`. Replace
```
W ← U^⊤ W V
H ← V^⊤ H V
```
Standard random matrix theory: `U^⊤ W V` is µ-incoherent with µ = O(log d) with high probability.

To make U, V cheap at inference, QuIP factorises them as `U = U_1 U_2` with `U_1` a permutation and `U_2` a structured (e.g. Hadamard) matrix. Successor [[quip-sharp]] takes this further with randomised Hadamard transforms.

### LDLQ adaptive rounding
Compute the LDL decomposition `H = L D L^⊤` (lower triangular L, diagonal D). Process weight columns in the order implied by L (later columns can be corrected for the error of earlier ones via L's off-diagonal entries).

For each column j of W (rotated):
```
δ_j = quant(w_j) − w_j
w_{k > j} ← w_{k > j} − L_{k,j} · δ_j     (error propagation)
```
GPTQ is the special case where the propagation uses `H⁻¹` directly; LDLQ uses the cleaner LDL factor and provably matches the optimal adaptive-rounding bound under incoherence.

### Inference
For layer matmul `y = W x`:
```
y_rot = (U^⊤ W V) (V^⊤ x)        # quantized GEMV in the rotated frame
y     = U y_rot                  # un-rotate
```
The right-rotation `V^⊤ x` fuses into the previous layer's output projection; the left-rotation `U y_rot` fuses into the next layer's input. Hadamard-structured U, V mean these "fused" ops are O(d log d).

### Hyperparameters
| Knob | Value |
|------|-------|
| Bits | 2 (also 3, 4) |
| Rotations U, V | random orthogonal, structured Hadamard variant |
| Adaptive rounding | LDLQ |
| Calibration | 128 sequences C4 |
| Group size | per-row (or G=128 + scale) |

## Connections
- Parent / special case: [[gptq]] (LDLQ with identity rotations).
- Pre-LLM adaptive-rounding lineage: [[adaround]], [[obs-obd]].
- Successor with structured Hadamard + E8 lattice: [[quip-sharp]].
- Same-rotation-idea descendants for *activations* too: [[quarot]], [[spinquant]], [[duquant]], [[flatquant]].
- Sub-2-bit VQ alternative: [[aqlm]].
