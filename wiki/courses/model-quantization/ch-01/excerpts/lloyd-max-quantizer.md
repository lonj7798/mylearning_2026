---
chapter: ch-01
course: model-quantization
phase: read
excerpt_of: "Lloyd-Max Quantizer (Lloyd 1957/1982; Max 1960)"
source_url: https://ieeexplore.ieee.org/document/1056489
created_at: "2026-05-21"
raw_data_source: [[raw-data/lloyd-max-quantizer]]
---

# Excerpt: Lloyd-Max — the optimal scalar quantizer at finite rate

**Authors:** Stuart P. Lloyd (Bell Labs 1957, published IEEE Trans. IT 1982); Joel Max (IEEE Trans. IT 1960).
**Year:** 1957 (Lloyd internal report); 1960 (Max); 1982 (Lloyd republished).
**Venue:** IEEE Transactions on Information Theory.
**URLs:** Lloyd 1982 — see source_url; Max 1960 — https://ieeexplore.ieee.org/document/1057548

---

## The two necessary conditions

For an `N`-level scalar quantizer with reconstruction levels `{y_1, …, y_N}` and decision boundaries `{b_0 = −∞, b_1, …, b_{N−1}, b_N = +∞}` minimizing

```math
D = \sum_{k=1}^{N} \int_{b_{k-1}}^{b_k} (x - y_k)^2 \, p(x) \, dx,
```

setting `∂D/∂b_k = 0` and `∂D/∂y_k = 0` gives:

```math
\boxed{\;b_k \,=\, \tfrac{1}{2} (y_k + y_{k+1})\;}  \qquad\text{(nearest-neighbour: midpoints)}
```

```math
\boxed{\;y_k \,=\, \frac{\int_{b_{k-1}}^{b_k} x \, p(x)\, dx}{\int_{b_{k-1}}^{b_k} p(x)\, dx}\;}  \qquad\text{(centroid: conditional mean)}
```

---

## The Lloyd iteration

```
initialize y_1, ..., y_N    (e.g. uniform-quantile spacing)
repeat:
    b_k ← (y_k + y_{k+1}) / 2          for k = 1..N-1     # nearest-neighbour
    y_k ← E[X | X ∈ (b_{k-1}, b_k]]    for k = 1..N       # centroid
until D stops decreasing
```

Converges to a local MSE minimum. For log-concave densities (Gaussian, Laplacian), the local minimum is the global minimum. This is exactly **`k`-means in 1-D**; the LBG extension (see [[excerpts/vector-quantization]] in ch-03) lifts the same iteration to `ℝ^d`.

---

## Performance gap to Shannon

For the Gaussian source, the `N`-level Lloyd-Max quantizer achieves

```math
D_{\text{Lloyd}}(N) \;\approx\; \sigma^2 \cdot c_N \cdot 2^{-2 \log_2 N},
\quad c_N \to \frac{\pi\sqrt{3}}{2} \approx 2.72 \text{ as } N \to \infty
```

The constant `c_∞ ≈ 2.72` is **1.53 dB above** Shannon's `R(D)` for Gaussian — the **space-filling loss**, the unavoidable price of being a 1-D scalar quantizer. Vector quantization in dimension `d` shrinks this loss as the dimension-`d` cell-shape constant `G_d → 1`.

---

## NF4 as quantile-spaced Lloyd-Max

[[nf4]] (NormalFloat-4) places 16 reconstruction levels at the 16-quantiles of the symmetric `N(0,1)` CDF, normalized so `|max| = 1`. This is not literally the Lloyd-Max output for 16-level `N(0,1)` — quantile spacing is a quick analytic surrogate that differs by ~0.05 dB. Dettmers chose the quantile form because the values can be computed once from `erf⁻¹`; the Lloyd version requires solving the centroid/NN fixed point numerically.

---

## Max 1960 reference tables

Max's original paper tabulates the optimal Gaussian quantizer boundaries and reconstruction levels for `N = 2, …, 36`. These tables remain the cited reference for any "optimal symmetric `N`-level code for Gaussian weights" claim. Modern LLM quantizers (NF4, SqueezeLLM LUT) implicitly approximate Max's `N = 16` entry.

---

## Connections

- [[excerpts/rate-distortion-theory]] — Lloyd-Max is the *scalar-quantizer realization* of `R(D)` at finite rate.
- [[excerpts/uniform-quantization-noise]] — uniform quantization is Lloyd-Max only when `p(x)` is uniform.
- [[excerpts/information-theoretic-bounds]] — Gish-Pierce derives the asymptotic optimal density `p^{1/3}` that Lloyd-Max approximates as `N → ∞`.
- [[ch-01]] — parent synthesis.
