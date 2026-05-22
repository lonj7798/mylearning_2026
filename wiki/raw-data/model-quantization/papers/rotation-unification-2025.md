<!-- scope: 2025 unified frameworks consolidating rotation-based quantization (QuaRot/SpinQuant/DuQuant)
     deps: [[quarot]], [[spinquant]], [[duquant]]
     see-also: [[learnable-rotation-2025]], [[flatquant]], [[awq]]
-->

# Unified Rotation Frameworks for LLM Quantization (2025)
- **Core Insight:** QuaRot, SpinQuant, DuQuant, and FlatQuant are all instances of a single template — apply a (rotation, permutation, scaling) transformation that leaves the layer math invariant while making the activation/weight distribution friendlier to quantization — and 2025 work shows the optimal choice of transformation can be learned end-to-end rather than picked from a fixed catalog.
- **Guideline:** Don't pick a rotation method by lineage (Hadamard vs learned vs dual); pick by deployment constraint: if you need zero inference cost, fold the rotation into weights offline (QuaRot / SpinQuant); if you can pay a fused rotation kernel, the learned-rotation methods give ~0.5-1 % loss back at W4A4.
- **Authors:** various — IST-Austria (Frantar, Alistarh), Meta (Liu et al.), Microsoft, Han Lab
- **Year:** 2024-2025
- **URL:** survey-style; representative 2025 paper: https://arxiv.org/abs/2402.06825 (QuaRot) + 2025 unifications via OmniQuant successors
- **Relevant topics:** Hadamard rotation, learnable rotation, dual rotation, permutation, equivalent transformation

## Abstract
The 2024 rotation-based quantization wave (QuaRot, SpinQuant, DuQuant, FlatQuant) produced an embarrassment of similar-looking methods. 2025 unification work — both academic surveys and consolidated open-source toolkits — recast them as instances of a single "invariant transformation" template: find an orthogonal R (and possibly a permutation P and per-channel scale S) such that activations and weights, after applying R, P, S, have lower per-block max / lower kurtosis / fewer outliers, then quantize the transformed tensors. Hadamard rotations are the random-fixed special case; SpinQuant's Cayley-parameterized rotations are the learned-orthogonal case; DuQuant's dual rotation is two stacked R matrices; FlatQuant is the affine generalization. The unification clarifies which knob matters most for which bit-budget (rotation for W4A4, scale for W4A16, permutation for W2 outliers).

## Key Contributions
- **Single template**: (R, P, S) ↦ Y = (R · A · P · S) · (S^-1 · P^-1 · W · R^-1) — invariant; quantize the transformed activation and weight independently.
- **Hadamard rotation (QuaRot)** = R fixed random Hadamard, P = I, S = I; zero inference cost (fold into weights offline), modest quality.
- **Learned orthogonal (SpinQuant)** = R parameterized on the Stiefel manifold and trained to minimize quantization loss; small inference cost (one matvec) or fold offline; ~0.5 % loss recovery at W4A4 vs QuaRot.
- **Dual rotation (DuQuant)** = two stacked R₁ · R₂ with a permutation in between; addresses the worst remaining outliers QuaRot leaves.
- **Affine (FlatQuant)** = R + additive shift; further loss reduction but breaks the orthogonal invariance and needs more care to preserve layer math.
- **2025 takeaway** (consolidation): a single trainable transformation per block, jointly optimized over the SmoothQuant-style channel scale, the rotation, and the per-group quant codebook, dominates fixed Hadamard at any bit-budget ≤ 4.
- Practical guidance: at W4A16 the rotation buys almost nothing (per-group scale already absorbs outliers); at W4A8 / W4A4 / W3 / W2 the rotation is load-bearing.

## Key Figures/Tables to Study
- The "outlier reduction" pre/post histograms for activations of FFN-gate input — the standard demonstration shared across all rotation papers.
- The W4A4 quality table comparing QuaRot / SpinQuant / DuQuant / FlatQuant / learned-affine on Llama-2/3 — small differences in the right column tell the story.
- The "rotation matters when" plot: x-axis bit width, y-axis quality recovery from rotation; rotation effect is essentially zero at ≥ W4A16, large at ≤ W4A4.

## Technical Details

### The invariant transformation template
For a linear layer Y = W · A, choose orthogonal R such that R^T R = I:
- A' = R · A (rotated activation)
- W' = W · R^T (counter-rotated weight)
- Y = W' · A' = W · R^T · R · A = W · A — exact.

Adding a permutation P (P^T P = I, P is a 0/1 matrix) and a diagonal scale S:
- A' = R · P · S · A
- W' = W · S^-1 · P^-1 · R^T
- Y = W' · A' = W · A — exact.

Quantize A' and W' independently; the round-trip preserves the inner product.

### Where each method sits in the template
| Method | R | P | S |
|--------|---|---|---|
| SmoothQuant | I | I | learned per-channel |
| AWQ | I | I | activation-aware per-channel |
| QuaRot | random Hadamard | I | I |
| SpinQuant | learned (Cayley/Stiefel) | I | I |
| DuQuant | R₁ · R₂ (with permute) | dim-reorder | I |
| FlatQuant | learned + additive | I | learned |

### Where rotation helps (and where it doesn't)
- **W4A16, group ≤ 128:** per-group scale already absorbs activation outliers within the group; rotation adds ~0.1-0.3 % at best.
- **W4A8 / W4A4:** activation quantization has no per-group scale (would be too many scales); rotation is the only way to keep block max bounded, ~1-3 % quality recovery.
- **W3 / W2 weight-only:** weight outliers dominate; rotation + sparse outlier path is the winning recipe.

### Inference cost of rotation
- Hadamard: R is structured (Walsh-Hadamard), so R · A is O(d log d), not O(d²); but typically folded into weights offline → zero runtime cost.
- Learned orthogonal: R is dense d×d → O(d²) per layer; can be folded into the *previous* layer's output weight if the chain of layers is compatible (typical: fold into the down-proj output and the next gate-proj input).
- DuQuant: two stacked rotations + permutation, ~ 2× the rotation cost; usually folded.

### 2025 trend: end-to-end joint optimization
- Treat (R, P, S, weight codebook, activation scale) as one big optimization problem; minimize layer output reconstruction MSE on a calibration set with all knobs trainable.
- Released as toolkits (e.g. the AutoQuant / NeuralCompressor 2025 rotations module) that ship a single config knob and pick the best (R, P, S) per layer.

## Connections
- [[quarot]] — fixed random Hadamard rotation, zero inference cost.
- [[spinquant]] — learned orthogonal rotations via Cayley parameterization.
- [[duquant]] — dual rotation + dim-reorder.
- [[flatquant]] — affine extension beyond orthogonal.
- [[learnable-rotation-2025]] — 2025 papers extending learned rotations beyond Hadamard.
- [[awq]] / [[smoothquant]] — per-channel scaling baselines that the rotation methods extend.
- [[orthogonal-finetuning-quant]] — sibling line on orthogonal fine-tuning interacting with rotated quant.
