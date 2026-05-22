<!-- scope: GPTQ — Hessian-based one-shot PTQ; the canonical W4 weight-only algorithm
     deps: [[adaround]], [[obs-obd]], [[obc]]
     see-also: [[awq]], [[spqr]], [[quip]], [[autogptq]]
-->

# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **Core Insight:** Quantizing a weight matrix can be reduced to a sequence of Optimal-Brain-Surgeon-style closed-form updates on a layer-local quadratic objective `||W X − Ŵ X||²`; using the Cholesky factor of the inverse calibration Hessian, one can quantize a 175B model column-by-column to 3–4 bits in 4 GPU-hours without retraining.
- **Guideline:** Use GPTQ with `group_size=128`, `actorder=True`, `percdamp=0.01` and ~128 calibration sequences (≥2k tokens each) for any W4 weight-only deployment; fall back to `group_size=−1` (per-channel) only for ≥6-bit, and prefer [[awq]] for activation-aware targets.
- **Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Year:** 2022 (ICLR 2023)
- **URL:** https://arxiv.org/abs/2210.17323
- **Relevant topics:** Hessian-based PTQ, OBS lineage, weight-only quantization, Cholesky update, group-wise quant

## Abstract
GPTQ is a one-shot weight-only PTQ that scales the second-order Optimal Brain Surgeon idea ([[obs-obd]]) up to LLMs. For each linear layer, GPTQ minimises the layer-local squared error `||W X − Ŵ X||²` where X is a calibration activation batch. Rather than per-weight gradient descent (as in [[adaround]]), it processes weights column-by-column, propagating the rounding error of column j into the remaining columns via the inverse Hessian `H⁻¹ = (2 X Xᵀ + λ I)⁻¹`. A Cholesky-based reformulation and a "lazy" block update make the algorithm numerically stable and roughly compute-bound rather than memory-bound. GPTQ quantizes OPT-175B / BLOOM-176B to 3–4 bits in ~4 GPU-hours with <1 ppl loss, and enables single-GPU 175B inference with a 3.25–4.5× speedup.

## Key Contributions
- Scales the OBS / [[obc]] Hessian-based pruning–quantization framework from BERT-scale to 175B-class LLMs.
- Reformulates the per-column update via the Cholesky factor `L` of `H⁻¹`, eliminating repeated matrix inversions and stabilising the recursion.
- **Lazy block update** with block size B=128: defers Hessian updates until a block is finished, turning the inner loop into a dense matmul.
- Demonstrates the first 3-bit LLM PTQ that holds zero-shot accuracy.
- Becomes the basis of the ubiquitous `AutoGPTQ` / `Marlin` / `gptq-for-llama` stacks (see [[autogptq]], [[marlin-kernel]]).

## Key Figures/Tables to Study
- **Algorithm 1:** the column-by-column quantization loop with Cholesky-driven error propagation — the single most-implemented quant algorithm.
- **Figure 3:** perplexity vs bits for OPT-175B/BLOOM-176B — 4-bit GPTQ is within 0.1 ppl of FP16; 3-bit is within 0.5.
- **Table 1:** wall-clock — ~4 GPU-hours for 175B; ~1 hr for 13B.

## Technical Details

### Layer-local objective
For one linear layer with weights `W ∈ R^{d_out × d_in}` and calibration activations `X ∈ R^{d_in × N}`:
```
Ŵ* = argmin_{Ŵ ∈ Q} ||W X − Ŵ X||_F²
```
where Q is the set of representable quantized weights (per-row/group scaled INT-k).

### Optimal column update (OBS-style)
Define the per-row Hessian `H = 2 X Xᵀ` (same for every output row, so amortise). For column q of weight row w, the optimal rounding direction and the compensation for the remaining columns is:
```
w_q  := quant(w_q)
δ_F  := − (w_q − quant(w_q)) / [H⁻¹]_{qq}  · [H⁻¹]_{q, F}    (F = remaining columns)
w_F  := w_F + δ_F
```
i.e. after rounding column q, add a closed-form correction to all yet-to-be-quantized columns to compensate for the rounding error in the L2 objective.

### Cholesky reformulation
Instead of recomputing `H⁻¹` after each column, GPTQ pre-computes the upper-triangular Cholesky factor `R` of `H⁻¹` once. The needed quantities `[H⁻¹]_{qq}` and `[H⁻¹]_{q,F}` are read directly from `R`. This is numerically more stable and turns the inner loop into a triangular solve.

### Lazy block update
Process columns in blocks of `B = 128`:
- Within a block: standard per-column update against the local block.
- At block boundary: apply the accumulated correction to all remaining columns in one dense GEMM.

This makes GPTQ compute-bound and ~10× faster than per-column updates.

### Damping (`percdamp`)
Numerical stability requires regularising H before Cholesky:
```
H ← H + λ I,   λ = percdamp · mean(diag(H))   (typical percdamp = 0.01)
```

### Activation ordering (`actorder`)
Quantize columns in descending `diag(H)` order — high-activation-energy columns get the cleanest rounding (the most slack from `[H⁻¹]_{qq}`). Empirically gives ~0.1–0.3 ppl improvement at 4-bit.

### Grouping
Per-output-channel scale is too aggressive at low bits. **`group_size = 128`** stores one (scale, zero_point) per 128 consecutive input dims per output row → ~0.05 bit overhead at INT4, big accuracy win.

### Calibration data
~128 sequences of length 2048 from C4 / WikiText / domain text. Larger / longer is rarely worth it.

### Hyperparameters (standard recipe)
| Knob | Value |
|------|-------|
| Bits | 4 (also 3, 2 with [[quip]]-style preprocessing) |
| `group_size` | 128 |
| `actorder` | True |
| `percdamp` | 0.01 |
| Block size B | 128 |
| Calibration samples | 128 sequences × 2048 tokens |
| Symmetric / asymmetric | asymmetric (zero point) for INT4 |

## Connections
- Parent: [[adaround]] (per-weight learned rounding) and [[obc]] (Hessian PTQ for BERT).
- Grandparent: [[obs-obd]] (Optimal Brain Surgeon / Damage).
- Activation-aware sibling that often beats GPTQ at W4: [[awq]].
- 2-bit successor using rotation preprocessing: [[quip]], [[quip-sharp]].
- Outlier-aware extension: [[spqr]], [[owq]].
- Inference kernels: [[marlin-kernel]], [[machete-kernel]], [[autogptq]].
- Same authors' production framework: [[autogptq]].
