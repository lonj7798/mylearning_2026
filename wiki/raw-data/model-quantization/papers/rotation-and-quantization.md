<!-- scope: theory paper consolidating rotation-based LLM quantization (QuaRot / SpinQuant / DuQuant family)
     deps: [[quarot]], [[spinquant]], [[duquant]]
     see-also: [[flatquant]], [[quip]], [[quip-sharp]]
-->

# Rotation and Quantization: A Unified View of Rotation-based LLM PTQ
- **Core Insight:** All rotation-based LLM quantization methods (QuaRot, SpinQuant, DuQuant, QuIP/QuIP#) instantiate the same template — insert orthogonal transforms preserving computational invariance, then quantize in the rotated basis — so they can be analyzed jointly via the *incoherence* condition `max_{i,j} |(UWVᵀ)_{ij}| ≤ μ / √(mn)`.
- **Guideline:** When choosing a rotation method, treat it as choosing a point in the design space {random Hadamard, learned orthogonal, block-orthogonal + permutation, lattice} along three axes — incoherence guarantee, optimization difficulty, inference cost.
- **Authors:** (placeholder — covered by surveys / blog consolidations including the [[duquant]] background section, the SpinQuant ablations, and 2024–2025 surveys)
- **Year:** 2024
- **URL:** consolidated view; see [[quarot]], [[spinquant]], [[duquant]], [[quip]], [[quip-sharp]]
- **Relevant topics:** rotation theory, incoherence processing, Stiefel manifold, lattice quantization

## Abstract
This entry is a synthesis page (not a single arxiv paper) tying together the rotation-based LLM PTQ literature. The thesis: every method here uses the *same* algebraic trick — pre/post multiplication by orthogonal Q in computationally-invariant positions — and differs only in how Q is chosen and how the rotated distribution is subsequently quantized. The page is included so the planner can sequence a single "rotation theory" chapter before diving into the individual papers.

## Key Contributions (consolidated)
- Uniform definition of "computationally invariant rotation" applicable to QuaRot/SpinQuant/DuQuant/QuIP.
- The incoherence condition as the unifying objective: post-rotation matrix entries should be near-uniformly bounded.
- Random Hadamard (QuaRot) achieves incoherence in expectation; learned R (SpinQuant) achieves it adaptively; block-rotation + permutation (DuQuant) achieves it locally; lattice + Hadamard (QuIP#) achieves it with structured codebooks.
- The trade-off triangle: incoherence quality vs calibration cost vs inference cost.

## Key Figures/Tables to Study
- A unified block diagram showing rotation insertion in residual / attention / FFN paths (mirrors QuaRot Fig 2, SpinQuant Fig 1, DuQuant Fig 3).
- Comparison table summarising rotation type, calibration data, online cost, and target bit-width for QuaRot / SpinQuant / DuQuant / FlatQuant / QuIP#.

## Technical Details

### The shared algebraic skeleton
For any orthogonal Q (QᵀQ = I), the residual block `y = x + f(W_2 σ(W_1 x))` is invariant under
- `x ← Qx` (rotate residual stream),
- `W_1 ← W_1 Qᵀ`, `W_2 ← Q W_2` (fold rotations into weights).
LM head absorbs Qᵀ at the output. All rotation-based methods plug into this skeleton.

### Incoherence as the optimization target
A matrix W is μ-incoherent if `max_{i,j} |(UWVᵀ)_{ij}| ≤ μ √(1/(mn))` for some random orthogonal U, V. Quantization error after RTN on an incoherent matrix is ~O(Δ²/12) (Bennett model). Hadamard rotations achieve this in expectation; learned rotations achieve it conditionally; DuQuant's block rotations achieve it locally.

### Design-space axes
| Method | Q type | Calibration | Inference cost |
|--------|--------|-------------|----------------|
| QuaRot | random Hadamard | none for R | online Hadamard for R3/R4 |
| SpinQuant | learned dense Stiefel | ~500 SGD steps | same as QuaRot |
| DuQuant | block-diag rotation + permutation | outlier indices from data | block-rotation matmul |
| FlatQuant | learned affine (not orthogonal) | block-MSE optimization | small Kronecker matmul |
| QuIP / QuIP# | random Hadamard + lattice codebook | E₈ codebook tabulated | lattice-decode kernel |

### Why orthogonal is the right constraint (for most)
Orthogonal Q preserves L², so the spectrum of W is unchanged → no spectral-norm degradation. Non-orthogonal A can flatten more aggressively but introduces a condition number that must be controlled — the trade-off FlatQuant accepts.

## Connections
- All members of the family: [[quarot]], [[spinquant]], [[duquant]], [[flatquant]], [[quip]], [[quip-sharp]].
- Predecessor in classical compressed sensing: incoherence-based recovery (Candès–Romberg–Tao).
- Theory backbone: [[uniform-quantization-noise]] (Bennett), [[information-theoretic-bounds]] (Gish–Pierce).
- Downstream applications: KV-cache quantization papers ([[kivi]], [[kvquant]]) often adopt rotation as a preprocessing step.
