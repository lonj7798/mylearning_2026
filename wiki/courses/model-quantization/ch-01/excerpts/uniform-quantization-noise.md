---
chapter: ch-01
course: model-quantization
phase: read
excerpt_of: "Bennett's Uniform Quantization Noise Model (Bennett 1948)"
source_url: https://ieeexplore.ieee.org/document/6773383
created_at: "2026-05-21"
raw_data_source: [[raw-data/uniform-quantization-noise]]
---

# Excerpt: Bennett 1948 — `σ_q² = Δ²/12` and 6 dB per bit

**Author:** W. R. Bennett (Bell Labs).
**Year:** 1948.
**Venue:** "Spectra of Quantized Signals", Bell System Technical Journal 27.
**URL:** see source_url.

---

## The one-box result

Under the **high-resolution assumption** — the input pdf `p_X(x)` is approximately constant within any one quantization cell of width `Δ` — the rounding error `e = Q(x) − x` is

```math
p_E(e) = \tfrac{1}{\Delta} \text{ for } e \in (-\Delta/2, +\Delta/2], \quad 0 \text{ otherwise}
\qquad\Rightarrow\qquad
\mathbb{E}[e] = 0, \quad
\boxed{\;\sigma_q^{\,2} \,=\, \frac{\Delta^2}{12}\;}
```

This single formula underlies every "6 dB per bit" claim in DSP and quantization analysis.

---

## Step size for a B-bit uniform quantizer

For full-scale range `[−A, +A]` with `2^B` levels:

```math
\Delta \,=\, \frac{2A}{2^{B}}
```

---

## "6 dB per bit" SNR formulas

For a full-loaded sinusoid of amplitude `A`:

```math
\text{SNR}_{\sin} \,=\, 10 \log_{10} \!\left( \frac{A^2/2}{\Delta^2/12} \right)
       \,=\, 10 \log_{10}\!\left( \tfrac{3}{2} \cdot 2^{2B} \right)
       \,=\, 6.02\, B + 1.76 \text{ dB}
```

For a 4σ-loaded Gaussian (typical assumption for weight tensors after normalization, `σ_X = A/4`):

```math
\text{SNR}_{\text{Gaussian}}  \approx  6.02\, B - 7.27 \text{ dB}
```

Same `6.02 B` slope — that is, **same `2^{−2B}` MSE exponent as Shannon's `D(R) = σ² · 2^{−2R}`** — but the prefactor lies above Shannon by the 1.53 dB scalar space-filling penalty.

---

## When the model breaks (three regimes)

1. **Low resolution (`B ≤ 3`).** pdf is not flat at scale `Δ`; quantization error becomes signal-dependent and Bennett under-estimates `σ_q²`. → Use non-uniform / block-scaled code.
2. **Periodic / low-amplitude inputs.** Error correlates with input → limit cycles, idle tones. (Rare in LLMs; common in audio DSP.)
3. **Outlier-heavy distributions (LLM activations).** Clipping and the heavy tail dominate `σ_q²`, not `Δ²/12`. **This is the precise SmoothQuant / AWQ / QuaRot problem.**

---

## Operational use in LLM PTQ calibration

For a per-tensor symmetric INT8 quantizer with clip `α`:

```
Δ = 2α / 255
predicted unclipped MSE = Δ²/12 = α² / (3 · 255²)
```

If measured MSE `> 3×` predicted, clipping is dominant → either widen `α` (more headroom, coarser bulk) or switch to per-channel / non-uniform code. This is the cheapest possible PTQ diagnostic.

---

## Connection to rate-distortion

Bennett's distortion `D ≈ Δ²/12 = (2A)² / (12 · 2^{2B}) = const · 2^{−2B}` has the **same `2^{−2B}` decay** as Shannon's `R(D)`. The prefactor differs — Bennett is *uniform* density, suboptimal off-uniform sources. Gish-Pierce ([[excerpts/information-theoretic-bounds]]) shows the optimal non-uniform prefactor is `(1/12) ||p||_{1/3}³`, which equals Bennett's `(2A)²/12` exactly when `p` is uniform.

---

## Connections

- [[excerpts/rate-distortion-theory]] — Bennett is the scalar-uniform realization of `D(R)`, up to space-filling loss.
- [[excerpts/lloyd-max-quantizer]] — non-uniform optimum that beats Bennett's `Δ²/12` for non-uniform pdfs.
- [[excerpts/stochastic-rounding]] — SR has `Var = Δ²/4`, 3× larger than RNE's Bennett `Δ²/12`; the cost of unbiasedness.
- [[llm-int8]] — empirical breakdown of Bennett's model at 6.7B+ scale.
- [[ch-01]] — parent synthesis.
