<!-- scope: k-means-based codebook design in ℝ^d; closes the scalar-quantizer space-filling gap
     deps: [[lloyd-max-quantizer]], [[rate-distortion-theory]]
     see-also: [[product-quantization]], [[aqlm]], [[squeezellm]]
-->

# Linde-Buzo-Gray Vector Quantization (LBG, 1980)
- **Core Insight:** By quantizing blocks of d source samples jointly to a codebook of N vectors in ℝ^d, vector quantization closes the 1.53 dB space-filling gap that any scalar quantizer pays — and approaches the Shannon R(D) bound as d → ∞ — at the cost of a codebook of size N = 2^{Rd}.
- **Guideline:** When you need sub-2-bit weight quantization, switch from scalar codes to VQ on small weight blocks (e.g. d=8 with N=256 codes = 1 bit/element); use LBG / k-means to fit the codebook to the empirical block distribution.
- **Authors:** Yoseph Linde, Andrés Buzo, Robert M. Gray
- **Year:** 1980
- **URL:** https://ieeexplore.ieee.org/document/1094577 (IEEE Trans. Communications, Jan 1980)
- **Relevant topics:** vector quantization, codebook design, k-means generalization, LBG splitting, AQLM precursor

## Abstract
LBG generalizes Lloyd's 1-D iteration to ℝ^d. Given training data {x_1, …, x_T} ⊂ ℝ^d, design an N-vector codebook C = {c_1, …, c_N} minimizing average distortion D = (1/T) Σ_t min_k ||x_t − c_k||². Two ingredients: (a) Lloyd-style alternation between assignment (nearest-neighbour) and update (centroid), now in d dimensions — i.e. exactly k-means; (b) the LBG splitting initialization that grows the codebook from 1 → 2 → 4 → … → N by perturb-and-relloyd, avoiding poor local minima from random init. Achieves rate-distortion performance asymptotic to R(D) as d → ∞.

## Key Contributions
- Generalizes Lloyd iteration to ℝ^d (= k-means with squared-error distortion).
- Provides the splitting initialization: double the codebook by perturbing each centroid by ±ε, then re-Lloyd.
- Empirically demonstrates that d = 4–16 closes most of the space-filling gap for image / speech sources.
- Foundation for product quantization, residual VQ, and additive quantization → directly the lineage to [[aqlm]] / [[vptq]] / [[gptvq]] / [[quip-sharp]].

## Key Figures/Tables to Study
- **SNR vs dimension d** plot (typical): for fixed bits/sample, SNR rises rapidly from d=1 to d=8 and asymptotes near the Shannon bound by d=16.
- **Voronoi partition** of ℝ^d induced by the codebook: each cell is the set of points nearest to one centroid; in 2D this is the canonical Voronoi diagram.

## Technical Details

### Codebook design problem
Given training vectors {x_t} ⊂ ℝ^d and target codebook size N:
```
minimize  D(C) = (1/T) Σ_t min_{k} ||x_t − c_k||²
subject to  C = {c_1, …, c_N} ⊂ ℝ^d
```

### Two necessary conditions (generalized Lloyd)
**1. Nearest-neighbour partition:**
```
S_k = { x : ||x − c_k|| ≤ ||x − c_j|| for all j }     (Voronoi cell)
```
**2. Centroid condition:**
```
c_k = (1/|S_k ∩ X|) Σ_{x_t ∈ S_k} x_t                  (mean within cell)
```

### LBG splitting algorithm
```
C ← {(1/T) Σ x_t}                                       # single centroid
while |C| < N:
    C ← C ∪ {c + ε · u : c ∈ C}                         # double via perturb
    repeat:
        assign each x_t to nearest c_k                  # NN step
        update each c_k = mean(assigned x_t)            # centroid step
    until D change < threshold
```
The splitting init avoids the local minima that plague random k-means init.

### Rate-distortion at dimension d
For dimension-d VQ with codebook size N = 2^{Rd}, the achievable distortion satisfies (high-rate, Gaussian source):
```
D_VQ(R, d) ≈ σ² · G_d · 2^{−2R}
```
where G_d → 1 as d → ∞ (Shannon bound) and G_1 = π√3/2 ≈ 2.72 (scalar Lloyd-Max).
**G_d values: G_1 ≈ 2.72 (1.53 dB loss), G_2 ≈ 1.16, G_4 ≈ 1.07, G_∞ = 1 (= R(D) bound).**

### Cost
- Encoding: O(N · d) per vector (brute-force nearest neighbour).
- Storage: N · d floats for codebook.
- Bit-rate: log₂ N bits per d-dim vector = (log₂ N)/d bits per scalar.

### Reducing encoding cost — the route to PQ
Brute-force encoding becomes infeasible for large N · d. Tree-structured VQ, multi-stage VQ (residual), and **product quantization** (split vector into sub-vectors, quantize each independently with a small sub-codebook) trade some distortion for log/linear-cost encoding. PQ is the direct ancestor of [[aqlm]] / [[vptq]].

### Operational use in LLM quant
- Group d=8 weights together → 256-vector codebook → 1 bit/weight.
- Run LBG on the actual layer weight distribution (or on rotated weights post-[[quarot]]) to learn the codebook.
- Encoding cost amortized over all vectors in the layer.
- AQLM uses *additive* VQ — sum of M codewords from M small codebooks — to get further compression with manageable encoding cost.

## Connections
- [[lloyd-max-quantizer]] — LBG = Lloyd in d dimensions.
- [[rate-distortion-theory]] — VQ asymptotically achieves the R(D) bound.
- [[product-quantization]] — Jégou's PQ scales VQ to high-d feature vectors via product structure.
- [[squeezellm]] — uses non-uniform LUT (essentially scalar VQ) weighted by sensitivity.
- [[aqlm]] — additive VQ for sub-2-bit LLM weights, direct LBG descendant.
- [[gptvq]] — combines GPTQ Hessian update with VQ codebooks.
