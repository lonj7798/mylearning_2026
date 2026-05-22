---
chapter: ch-01
course: model-quantization
phase: read
excerpt_of: "Shannon Rate-Distortion Theory (Shannon 1948 / 1959; Cover & Thomas Ch. 10)"
source_url: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
created_at: "2026-05-21"
raw_data_source: [[raw-data/rate-distortion-theory]]
---

# Excerpt: Shannon Rate-Distortion — the floor every quantizer fights

**Authors:** Claude E. Shannon (1948 "A Mathematical Theory of Communication"; 1959 "Coding Theorems for a Discrete Source with a Fidelity Criterion"); Cover & Thomas textbook treatment.
**Year:** 1948 / 1959 (Shannon); 1991/2006 (Cover & Thomas).
**Venue:** Bell System Technical Journal; IRE Convention Record; Wiley textbook.
**URLs:** Shannon 1948 PDF — see source_url; Cover & Thomas, *Elements of Information Theory*, Ch. 10.

---

## The one-box theorem

For source `X ~ p(x)` and distortion measure `d(x, x̂)`:

```math
R(D)  =  \min_{p(\hat{x}|x) \,:\, \mathbb{E}[d(X,\hat{X})] \le D}  I(X; \hat{X})
```

`R(D)` is non-increasing and convex. The achievability theorem (random codebooks, long-block-length) and converse theorem (no scheme beats `R(D)` asymptotically) together pin this as the **operational minimum bit-rate** to reproduce `X` within average distortion `D`.

---

## Gaussian source, squared error — the canonical closed form

For `X ~ N(0, σ²)` with `d(x, x̂) = (x − x̂)²`:

```math
\boxed{\;R(D) \,=\, \tfrac{1}{2} \log_2 \!\left(\frac{\sigma^2}{D}\right) \quad\text{for } 0 \le D \le \sigma^2\;}
\qquad
\Leftrightarrow
\qquad
\boxed{\;D(R) \,=\, \sigma^2 \cdot 2^{-2R}\;}
```

`R(D) = 0` for `D > σ²` (zero rate, output the mean). **The "−6 dB per bit" rule: every additional bit reduces MSE by a factor of 4** (= 6.02 dB SNR). This is the canonical benchmark — every scalar quantizer in subsequent chapters is judged against this slope.

---

## Reverse water-filling for parallel Gaussian sources

For independent `X_i ~ N(0, σ_i²)`, the rate-distortion-optimal distortion allocation is

```math
D_i  =  \min(\lambda, \sigma_i^{\,2})
```

where the water level `λ` is chosen so `Σ D_i = D`. Channels with `σ_i² < λ` are not coded at all (`R_i = 0`, `D_i = σ_i²`).

**Operational consequence.** Per-channel, per-group, and per-token scales in modern LLM quantization (group-128 INT4, per-channel SmoothQuant, per-token KV-cache) are direct instantiations of reverse water-filling — give bits to the high-variance directions, starve the low-variance ones.

---

## High-rate distortion (Gish-Pierce regime)

For a smooth source `p(x)` and the *optimal* non-uniform scalar quantizer at high rate `R`:

```math
D(R)  \approx  \frac{1}{12} \,\|p\|_{1/3}^{\,3}\, 2^{-2R}, \quad
\|p\|_{1/3} = \left(\int p(x)^{1/3} dx\right)
```

The exponent matches `R(D)`'s `2^{−2R}` exactly. The constant prefactor lies above the Shannon constant `σ²` by the **space-filling loss** — about 1.53 dB for any Gaussian source under any 1-D scalar code. Closing that gap requires vector quantization in dimension `d ≥ 2` (see [[excerpts/vector-quantization]] in ch-03 once that chapter exists).

---

## Operational meaning for LLM quantization

- An `N`-bit per-weight quantizer has rate `R = N`. Compute the Gaussian floor `σ²·2^{−2N}` for the tensor's measured variance; compare to your achieved MSE. The ratio is the **rate-distortion gap** and is the only quantizer-efficiency metric that is hardware-independent.
- Vector quantization (`k`-means, PQ, AQLM) closes the 1.53 dB gap by going to higher dimension. Scalar quantizers cannot.
- For activation tensors with outliers, the *effective* `σ²` is dominated by a tiny fraction of channels → reverse water-filling says you should isolate them. This is exactly the [[llm-int8]] / [[smoothquant]] motivation.

---

## Connections

- [[excerpts/uniform-quantization-noise]] — Bennett's `Δ²/12` is the scalar-uniform realization of `D(R)`, suboptimal by the space-filling loss.
- [[excerpts/lloyd-max-quantizer]] — Lloyd-Max achieves the scalar-quantizer optimum at finite rate.
- [[excerpts/information-theoretic-bounds]] — Gish-Pierce derives the `p^{1/3}` density that minimizes the prefactor.
- [[ch-01]] — parent synthesis chapter.
