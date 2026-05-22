<!-- scope: GPTVQ — interleaves vector-quantization column updates with GPTQ-style Hessian weight updates
     deps: [[gptq]], [[product-quantization]]
     see-also: [[aqlm]], [[vptq]], [[quip-sharp]]
-->

# GPTVQ: The Blessing of Dimensionality for LLM Quantization
- **Core Insight:** Quantization dimensionality (1-D scalar → 2-D pairs → higher-D vector) is itself a knob that improves the size-vs-accuracy frontier, and the GPTQ Hessian update generalizes cleanly to vectors — interleave column-wise VQ with Hessian-weighted residual propagation and you get monotonic gains from increasing d.
- **Guideline:** When choosing between scalar GPTQ (d=1) and vector GPTQ (d=2 to 8), prefer higher d at the same bits/weight budget; GPTVQ adds initialization via data-aware EM-style k-means + SVD-compressed codebook storage to keep overhead small.
- **Authors:** Mart van Baalen, Andrey Kuzmin, Ivan Koryakovskiy, Markus Nagel, Peter Couperus, Cedric Bastoul, Eric Mahurin, Tijmen Blankevoort, Paul Whatmough (Qualcomm AI Research)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.15319
- **Relevant topics:** vector quantization, GPTQ extension, dimensionality, SVD codebook compression

## Abstract
GPTVQ shows that the quantization size-vs-accuracy trade-off is significantly improved by increasing quantization dimensionality. It interleaves the quantization of weight columns with weight updates using Hessian information from per-layer output reconstruction (the GPTQ trick, generalized to vector codes). Codebooks are initialized via a data-aware EM algorithm and further compressed via SVD + INT quantization to keep storage small. Demonstrates SOTA size-accuracy on Llama-2 and Mistral with practical 3–11 hour calibration for Llama-2-70B on a single H100.

## Key Contributions
- Generalizes GPTQ's column-by-column Hessian update from scalars to vectors (d = 2, 4, 8).
- Data-aware EM-style codebook initialization seeded by k-means on calibration-weighted weight samples.
- Codebook compression: each codebook is itself stored as SVD-compressed + INT-quantized to keep amortised cost low.
- Demonstrates that even moving from d=1 to d=2 improves the Pareto frontier — supporting the "blessing of dimensionality" thesis.

## Key Figures/Tables to Study
- **Figure 1:** Pareto frontier showing monotonic improvement as d increases from 1 to 8 at fixed bits/weight.
- **Figure 4:** Interleaved column-VQ + Hessian-residual-update algorithm diagram.
- **Table 3:** Llama-2 70B PPL at 2/2.5/3 bits — GPTVQ vs GPTQ vs AQLM.

## Technical Details

### Vector GPTQ update
GPTQ minimises `||W X − ŴX||² ≈ Σ_j (w_j − ŵ_j)ᵀ H (w_j − ŵ_j)` with H = X X^T. Quantize columns one by one; after quantizing column j, propagate the residual via the Cholesky factor of H_remaining to compensate later columns.

GPTVQ replaces the scalar quantization of column j with a *vector* quantization: pair up d consecutive entries into a length-d group and snap to the nearest codeword in the codebook:
`ŵ_g = argmin_{c ∈ C} (c − w_g)ᵀ Σ_g (c − w_g)`
where Σ_g is the appropriate sub-block of H. Then propagate the Cholesky residual to all remaining groups.

### Data-aware EM initialization
Codebook C ∈ ℝ^{K × d} initialized by running EM on calibration-weighted samples of w_g:
- E-step: assign each w_g to nearest c_k under Σ_g-weighted distance.
- M-step: update c_k as weighted mean of assigned w_g, weights = effective Hessian magnitude.
Better than random init or vanilla k-means because it accounts for downstream-impact weighting.

### Codebook compression
Each codebook C is decomposed via SVD into low-rank factors `C ≈ U Σ V^T`, then U, Σ, V are themselves quantized to INT8/INT4. Reduces codebook overhead from O(K · d · 16 bits) to O((K + d) · r · 4 bits) where r is the SVD rank kept.

### Bits/weight budget
- K = 256, d = 4 → 8 bits / 4 weights = 2.0 bits/weight + codebook overhead.
- K = 4096, d = 8 → 12 bits / 8 weights = 1.5 bits/weight.

### Wall clock
Llama-2-70B: 3–11 hours on a single H100 (depending on d, K). Cheaper than AQLM at similar quality.

## Connections
- Direct ancestor: [[gptq]] (scalar Hessian update); GPTVQ = GPTQ with d-dim quant.
- Concurrent / siblings: [[aqlm]] (additive multi-codebook), [[vptq]] (channel-independent Hessian).
- Lattice alternative: [[quip-sharp]].
- Classical ancestor: [[vector-quantization]], [[product-quantization]].
- Qualcomm-lab lineage: [[adaround]], [[brecq]] all use Hessian-aware PTQ.
