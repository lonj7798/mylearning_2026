<!-- scope: non-uniform quantization via logarithmic compression for signals with heavy-tailed amplitude distributions
     deps: [[uniform-quantization-noise]]
     see-also: [[lloyd-max-quantizer]], [[nf4]], [[information-theoretic-bounds]]
-->

# µ-law / A-law Companding (Smith 1957; ITU-T G.711)
- **Core Insight:** For sources with heavy-tailed amplitude distributions, applying a logarithmic compressor before a uniform quantizer (and an expander after) yields an effective non-uniform code that allocates more reconstruction levels to small-amplitude values — equalizing relative error and dramatically improving low-amplitude SNR.
- **Guideline:** When weights or activations span many orders of magnitude (log-normal, Laplacian-with-outliers), prefer a log-scale or quantile-based code (companding, NF4, FP formats) over a linear INT code at the same bit-width.
- **Authors:** Bernard Smith (Bell Labs, µ-law); A-law from European telecom; standardized as ITU-T G.711
- **Year:** 1957 (Smith); 1972 (G.711 standardization)
- **URL:** https://www.itu.int/rec/T-REC-G.711 ; Smith 1957 "Instantaneous Companding of Quantized Signals", Bell System Tech. J.
- **Relevant topics:** non-uniform quantization, log-scale code, signal-dependent SNR, FP-as-companding

## Abstract
A companding quantizer applies a non-linear compressor F to the input, uniformly quantizes the compressed signal, then applies F⁻¹ at decode. For a logarithmic compressor, the effective quantization step Δ(x) scales linearly with |x|, so the *relative* error |e/x| is approximately constant across the dynamic range — instead of constant *absolute* error as in uniform quantization. µ-law (North America, µ=255) and A-law (Europe, A=87.6) are the two standardized telephony-grade compressors. The same idea underlies every floating-point format: FP is a piecewise-linear approximation to log-quantization.

## Key Contributions
- Demonstrates that pre-distortion + uniform quantize + post-distortion can approximate the optimal non-uniform quantizer for heavy-tailed sources.
- Establishes the µ-law formula (US/Japan) and A-law formula (Europe) as international telephony standards.
- Provides the intuition that floating-point representation = block-wise companding with log-scale exponent.
- Foundational reference for FP4 / FP8 / log-quantization for LLMs.

## Key Figures/Tables to Study
- **µ-law compressor curve**: characteristic "S-shape" piecewise-log → uniform quantization on this transformed axis → exponentially-spaced reconstruction levels in original axis.
- **SNR vs input amplitude** plot: uniform quantizer has SNR ∝ x²; µ-law quantizer has SNR ≈ constant over 30+ dB of dynamic range.

## Technical Details

### µ-law compressor
For signal x ∈ [−1, +1] and parameter µ (standard µ = 255):
```
F_µ(x) = sign(x) · ln(1 + µ|x|) / ln(1 + µ)
```
Decoder:
```
F_µ⁻¹(y) = sign(y) · (1/µ) · ((1+µ)^{|y|} − 1)
```

### A-law compressor
For parameter A (standard A = 87.6):
```
F_A(x) = sign(x) · A|x| / (1 + ln A)              for |x| ≤ 1/A
F_A(x) = sign(x) · (1 + ln(A|x|)) / (1 + ln A)    for 1/A < |x| ≤ 1
```

### Effective quantization step in original domain
After uniform B-bit quantization of F(x) with step Δ, the back-mapped step at signal level x is approximately
```
Δ_eff(x) ≈ Δ / F'(x)
```
For µ-law: F'(x) = µ / [(1 + µ|x|) · ln(1+µ)], so Δ_eff(x) ∝ (1/µ + |x|) — linear in |x|. ⇒ relative error |e/x| ≈ constant for |x| ≫ 1/µ.

### SNR
For full-loaded sinusoid, µ-law 8-bit telephony achieves SNR ≈ 38 dB over a 30 dB dynamic range, vs uniform 8-bit which would give 38 dB only at full scale and drop 6 dB per halving of amplitude. Equivalently µ-law 8-bit ≈ uniform 13-bit in *worst-case* SNR.

### Connection to floating-point as companding
A floating-point number x = (−1)^s · 1.m · 2^e is exactly a piecewise-uniform quantizer on the log axis: within each exponent bin [2^e, 2^{e+1}), the (1+m)·2^e values are uniformly spaced; across bins the step doubles. So **FP = piecewise-linear approximation to logarithmic companding**. FP4 / FP8 are effectively low-resolution companders.

### Connection to NF4
[[nf4]] (NormalFloat-4) is the *Lloyd-Max-optimal* companding code for unit-Gaussian inputs — companding tuned to the source distribution rather than a fixed log curve.

### Limitations for LLM quant
- Companding gives constant *relative* error; but LLM downstream loss is often more sensitive to absolute large-magnitude error in outlier channels → motivates [[smoothquant]]-style equalization rather than pure companding.
- µ-law is fixed; better to learn the optimal compander per-tensor (which is what NF4 + per-channel-scale does).

## Connections
- [[uniform-quantization-noise]] — companding turns into uniform quantization on the transformed axis, where Bennett's Δ²/12 applies.
- [[lloyd-max-quantizer]] — Lloyd-Max with log-concave pdf converges to roughly log-spaced levels; µ-law is the fixed-form approximation.
- [[nf4]] — NF4 = Lloyd-Max companding for Gaussian.
- [[information-theoretic-bounds]] — Gish-Pierce optimal density p(x)^{1/3} formalizes the "more levels where mass is" idea that companding implements heuristically.
- [[ieee-754]] / [[fp8-e4m3]] — FP formats *are* discrete companders.
