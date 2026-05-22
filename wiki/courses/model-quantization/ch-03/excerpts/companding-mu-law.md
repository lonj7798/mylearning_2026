---
chapter: ch-03
course: model-quantization
phase: read
excerpt_of: "µ-law / A-law Companding (Smith 1957; ITU-T G.711)"
source_url: https://www.itu.int/rec/T-REC-G.711
created_at: "2026-05-21"
raw_data_source: [[raw-data/companding-mu-law]]
---

# Excerpt: µ-law / A-law companding — non-uniform quantization via log compression

**Authors:** Bernard Smith (Bell Labs, µ-law); A-law from European telecom; standardized as ITU-T G.711.
**Year:** 1957 (Smith); 1972 (G.711 standardization).
**URLs:** see source_url; Smith 1957 "Instantaneous Companding of Quantized Signals", Bell System Technical Journal.

---

## The one-box compressor

For signal `x ∈ [−1, +1]` with parameter `µ` (standard `µ = 255`):

```math
\boxed{\;F_\mu(x) \,=\, \text{sign}(x) \cdot \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}\;}
```

Decoder:

```math
F_\mu^{-1}(y) \,=\, \text{sign}(y) \cdot \frac{1}{\mu} \cdot \left( (1 + \mu)^{|y|} - 1 \right)
```

A-law analogue (parameter `A`, standard `A = 87.6`):

```math
F_A(x) = \text{sign}(x) \cdot \frac{A |x|}{1 + \ln A}  \quad \text{for } |x| \le 1/A,
\qquad
F_A(x) = \text{sign}(x) \cdot \frac{1 + \ln(A|x|)}{1 + \ln A}  \quad \text{for } 1/A < |x| \le 1
```

---

## Effective quantization step in original domain

After uniform `B`-bit quantization of `F(x)` with step `Δ`, the back-mapped step at signal level `x` is approximately

```math
\Delta_{\text{eff}}(x) \,\approx\, \Delta / F'(x)
```

For µ-law: `F'(x) = µ / [(1 + µ|x|) · ln(1 + µ)]`, so `Δ_eff(x) ∝ (1/µ + |x|)` — **linear in `|x|`**. ⇒ relative error `|e/x| ≈ constant` for `|x| ≫ 1/µ`.

---

## SNR comparison

For full-loaded sinusoid, µ-law 8-bit telephony achieves **SNR ≈ 38 dB over a 30 dB dynamic range**, vs uniform 8-bit which gives 38 dB only at full scale and drops 6 dB per halving of amplitude. Equivalently µ-law 8-bit ≈ uniform 13-bit in **worst-case** SNR.

---

## Floating-point as companding

A floating-point number `x = (−1)^s · 1.m · 2^e` is exactly a piecewise-uniform quantizer on the log axis: within each exponent bin `[2^e, 2^{e+1})`, the `2^M` mantissa values are uniformly spaced; across bins the step doubles. So **FP = piecewise-linear approximation to logarithmic companding**. FP4 / FP8 / FP6 are low-resolution companders.

---

## Connection to NF4

[[nf4]] (NormalFloat-4) is the **Lloyd-Max-optimal companding code for unit-Gaussian inputs** — companding tuned to the source distribution rather than a fixed log curve. The Gish-Pierce condition ([[excerpts/information-theoretic-bounds]]) prescribes `F'(x) ∝ p(x)^{1/3}` as the asymptotically optimal compressor; µ-law approximates this for log-distributed audio; NF4 tabulates it for `N(0,1)` weights.

---

## Limitations for LLM quant

- Companding gives constant *relative* error; but LLM downstream loss is often more sensitive to absolute large-magnitude error in outlier channels → motivates [[smoothquant]]-style equalization rather than pure companding.
- µ-law is fixed-form; better to learn the optimal compander per-tensor (which is what NF4 + per-channel-scale does).

---

## Connections

- [[excerpts/uniform-quantization-noise]] — companding turns into uniform quantization on the transformed axis, where Bennett's `Δ²/12` applies.
- [[excerpts/lloyd-max-quantizer]] — Lloyd-Max with log-concave pdf converges to roughly log-spaced levels; µ-law is the fixed-form approximation.
- [[excerpts/information-theoretic-bounds]] — Gish-Pierce `p^{1/3}` formalizes the "more levels where mass is" idea companding implements heuristically.
- [[nf4]] — Lloyd-Max-optimal companding code for Gaussian weights (ch-02 §6).
- [[ieee-754]] / [[fp8-e4m3]] — FP formats are discrete companders.
- [[ch-03]] — parent synthesis.
