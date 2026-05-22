<!-- scope: optimal scalar quantizer design via fixed-point iteration on centroid + nearest-neighbour conditions
     deps: [[rate-distortion-theory]]
     see-also: [[uniform-quantization-noise]], [[vector-quantization]], [[nf4]]
-->

# Lloyd-Max Quantizer (Lloyd 1957/1982; Max 1960)
- **Core Insight:** For a given source distribution p(x), the MSE-optimal N-level scalar quantizer is characterized by two coupled conditions — decision boundaries are midpoints between adjacent reconstruction levels (nearest-neighbour), and reconstruction levels are conditional centroids of their cells — and the Lloyd iteration alternates between them.
- **Guideline:** When designing a non-uniform low-bit code (e.g. NF4-style), run Lloyd's algorithm on the empirical weight/activation distribution to get the MSE-optimal codebook; expect ~1–2 dB improvement over uniform at low bit-widths.
- **Authors:** Stuart P. Lloyd (Bell Labs 1957, published IEEE Trans. IT 1982); Joel Max (1960)
- **Year:** 1957 (Lloyd internal report); 1960 (Max); 1982 (Lloyd republished)
- **URL:** https://ieeexplore.ieee.org/document/1056489 (Lloyd 1982); https://ieeexplore.ieee.org/document/1057548 (Max 1960)
- **Relevant topics:** scalar quantization, optimal codebook design, k-means precursor, MSE minimization

## Abstract
Given source X with pdf p(x), find N reconstruction levels {y_1, …, y_N} and N+1 decision boundaries {b_0=−∞, b_1, …, b_{N−1}, b_N=+∞} minimizing E[(X − Q(X))²]. The two necessary conditions are derived by setting partial derivatives to zero, then solved iteratively: fix levels → update boundaries to midpoints; fix boundaries → update levels to centroids. Lloyd's algorithm is the 1-D ancestor of k-means and the foundation of Lloyd-Buzo-Gray vector quantization.

## Key Contributions
- Two necessary conditions for an MSE-optimal scalar quantizer (centroid + nearest-neighbour).
- A converging iterative algorithm to solve them (Lloyd iteration).
- Closed-form numerical tables for Gaussian / Laplacian / uniform sources at common N (Max 1960).
- Establishes that uniform quantization is optimal only for the uniform distribution.

## Key Figures/Tables to Study
- **Max 1960 Table I**: optimal Gaussian quantizer boundaries and levels for N = 2…36; reference for any "optimal symmetric N-level code for Gaussian weights."
- **Decision-cell diagram**: each cell's boundary is the midpoint of its two neighbouring reconstruction levels — the defining picture of nearest-neighbour quantization.

## Technical Details

### Two necessary conditions
For decision boundaries {b_k} and reconstruction levels {y_k} that minimize MSE D = Σ_k ∫_{b_{k−1}}^{b_k} (x − y_k)² p(x) dx:

**1. Nearest-neighbour condition (boundary update):**
```
b_k = (y_k + y_{k+1}) / 2          for k = 1, …, N−1
```
i.e. decision boundaries are midpoints between adjacent reconstruction levels.

**2. Centroid condition (level update):**
```
y_k = ∫_{b_{k−1}}^{b_k} x p(x) dx  /  ∫_{b_{k−1}}^{b_k} p(x) dx
```
i.e. reconstruction levels are conditional means within each cell.

### Lloyd iteration
```
initialize y_1, …, y_N (e.g. uniform on quantile spacing)
repeat:
    b_k ← (y_k + y_{k+1}) / 2           for all k
    y_k ← E[X | X ∈ (b_{k−1}, b_k]]      for all k
until D stops decreasing
```
Converges to a local MSE minimum. For log-concave densities (Gaussian, Laplacian), the local minimum is the global minimum.

### Performance vs Shannon bound
For Gaussian source, Lloyd-Max N-level quantizer achieves
```
D_Lloyd(N) ≈ σ² · c_N · 2^{−2 log₂ N}
```
with c_N → π√3/2 ≈ 2.72 (≈ 1.53 dB) as N → ∞. That 1.53 dB is the **space-filling loss** — the unavoidable penalty of 1-D scalar quantization vs the Shannon R(D) bound. Vector quantization in dimension d reduces this loss as d increases.

### Connection to k-means
Lloyd's iteration *is* the 1-D version of k-means. The LBG algorithm (see [[vector-quantization]]) generalizes it to ℝ^d for VQ codebook design.

### Connection to NF4
[[nf4]] (NormalFloat-4) is essentially a 16-level (4-bit) Lloyd-Max quantizer tuned for a unit Gaussian; Dettmers tabulates the 16 quantile-based reconstruction levels rather than running Lloyd online.

## Connections
- [[rate-distortion-theory]] — Lloyd-Max is the scalar-quantizer realization of the rate-distortion theorem at finite rate.
- [[uniform-quantization-noise]] — uniform quantization is the Lloyd-Max solution only when p(x) is uniform.
- [[vector-quantization]] — LBG generalizes Lloyd iteration to ℝ^d.
- [[nf4]] — practical 4-bit Lloyd-Max code for Gaussian-distributed LLM weights.
- [[squeezellm]] — sensitivity-weighted k-means → non-uniform LUT quantization.
