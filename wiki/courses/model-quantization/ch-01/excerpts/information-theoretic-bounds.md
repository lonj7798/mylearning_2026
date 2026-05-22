---
chapter: ch-01
course: model-quantization
phase: read
excerpt_of: "Gish-Pierce High-Rate Quantization Bounds (Gish & Pierce 1968)"
source_url: https://ieeexplore.ieee.org/document/1054193
created_at: "2026-05-21"
raw_data_source: [[raw-data/information-theoretic-bounds]]
---

# Excerpt: Gish-Pierce 1968 — optimal point density `p(x)^{1/3}`

**Authors:** Herbert Gish, John N. Pierce.
**Year:** 1968.
**Venue:** "Asymptotically Efficient Quantizing", IEEE Transactions on Information Theory, September 1968.
**URL:** see source_url.

---

## The one-box result

For a smooth source `p(x)` and squared-error distortion `d(x, x̂) = (x − x̂)²`, the asymptotically optimal scalar quantizer has reconstruction-level density (= point density `λ(x)`, normalized `∫ λ = 1`)

```math
\boxed{\;\lambda^*(x) \,\propto\, p(x)^{1/3}\;}
```

and achieves high-rate distortion

```math
\boxed{\;D^*(R) \,\approx\, \frac{1}{12} \,\|p\|_{1/3}^{\,3}\, 2^{-2R},
\qquad \|p\|_{1/3} = \int p(u)^{1/3}\, du\;}
```

---

## Bennett's integral (general non-uniform quantizer)

For any quantizer with point density `λ(x)` and `N` levels, the high-rate distortion is

```math
D(N)  \approx  \frac{1}{12 N^2} \int p(x) \, \lambda(x)^{-2}\, dx
```

This is the variational object Gish-Pierce minimize. Setting `∂D/∂λ = 0` subject to `∫ λ = 1` (Hölder / Lagrange-multiplier argument) yields the `p^{1/3}` solution above.

---

## Specific source values

- **Gaussian** `p(x) = (1/√(2πσ²)) exp(−x²/(2σ²))`: `||p||_{1/3}³ = σ² · π√3 ≈ 5.44 σ²`, giving

  ```math
  D^*_{\text{Gauss}}(R) \approx 0.453 \,\sigma^2\, 2^{-2R}.
  ```

  Gap to Shannon `σ² · 2^{−2R}` is the **1.53 dB space-filling loss** — unavoidable in 1-D scalar quantization.

- **Laplacian** `p(x) = (1/(2b)) exp(−|x|/b)`: `||p||_{1/3}³ ≈ 6 b²`.

- **Uniform** `p(x) = 1/(2A)`: `||p||_{1/3}³ = (2A)²`, giving `D = A²/3 · 2^{−2R}` — **Bennett's `Δ²/12` recovered exactly**.

---

## Companding interpretation

A monotone compressor `F: ℝ → [0, 1]` followed by uniform quantization is equivalent to non-uniform quantization with point density `λ(x) = F'(x)`. Setting `F'(x) ∝ p(x)^{1/3}` yields the optimal compander:

```math
F^*(x) \,=\, \frac{\int_{-\infty}^{x} p(u)^{1/3}\, du}{\int_{-\infty}^{+\infty} p(u)^{1/3}\, du}
```

This is the principled basis for:

- [[companding-mu-law]] — fixed `F'(x) ∝ 1/(1 + µ|x|)` tuned for log-distributed audio signals.
- [[nf4]] — tabulated `F*` for the unit-Gaussian weight prior; ~0.5 PPL better than INT4 on Llama-class models at 4-bit.
- Every floating-point format is a *discrete piecewise* compander on the log axis.

---

## Space-filling loss is unavoidable in 1-D

The Gaussian `1.53 dB = 10 log₁₀ (0.453 · 12 / 1) ≈ 10 log₁₀ 5.44` gap between Gish-Pierce optimum and Shannon `R(D)` cannot be closed by any 1-D scalar code. Closing it requires vector quantization in dimension `d ≥ 2`, where the cell-shape constant `G_d → 1` as `d → ∞` (`G_1 ≈ 2.72`, `G_2 ≈ 1.16`, `G_4 ≈ 1.07`).

---

## Operational use in LLM quant

- **Non-uniform 4-bit weight code (NF4, SqueezeLLM LUT):** set reconstruction levels at the 1/16 quantiles of `p(w)^{1/3}` — *not* at uniform spacing nor at equiprobable spacing (the equiprobable code allocates too many levels to the tails and is suboptimal).
- **Activation outliers:** the `L_{1/3}` norm is dominated by the heavy tail → before applying any scalar code, either *rotate* to flatten ([[quarot]]) or *split* the outlier channels ([[spqr]]).

---

## Connections

- [[excerpts/rate-distortion-theory]] — `R(D)` is the absolute floor; Gish-Pierce gives the best scalar achievable.
- [[excerpts/uniform-quantization-noise]] — Bennett's `Δ²/12` is the special case `λ = const`, optimal only for uniform `p`.
- [[excerpts/lloyd-max-quantizer]] — finite-rate algorithm whose `N → ∞` limit hits the Gish-Pierce density.
- [[nf4]] — practical 4-bit tabulation of `F*` for `N(0,1)`.
- [[ch-01]] — parent synthesis.
