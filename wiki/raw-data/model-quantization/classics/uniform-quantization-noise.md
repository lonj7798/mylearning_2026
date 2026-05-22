<!-- scope: high-resolution model treating quantization error as uniform additive white noise with variance Δ²/12
     deps: [[rate-distortion-theory]]
     see-also: [[lloyd-max-quantizer]], [[round-to-nearest-even]], [[stochastic-rounding]]
-->

# Bennett's Uniform Quantization Noise Model (Bennett 1948)
- **Core Insight:** Under the high-resolution assumption (input pdf approximately flat over each quantization step Δ), quantization error is well-modelled as a zero-mean, uniformly distributed white noise process with variance σ_q² = Δ²/12, independent of the input signal.
- **Guideline:** Use Δ²/12 as a quick analytic prediction of per-tensor quantization MSE during calibration; if measured MSE is much larger than Δ²/12, the high-resolution assumption is violated (too few levels, outlier-dominated, or clipping) and you need a non-uniform quantizer or rotation.
- **Authors:** W.R. Bennett (Bell Labs)
- **Year:** 1948
- **URL:** https://ieeexplore.ieee.org/document/6773383 ("Spectra of Quantized Signals", Bell System Tech. J. 27, 1948)
- **Relevant topics:** quantization noise, additive noise model, SNR per bit, signal processing

## Abstract
Bennett analyses uniform quantization of a continuous signal by treating the rounding error as an additive noise source. Under the high-resolution assumption — the input pdf is approximately constant within any one quantization cell of width Δ — the quantization error e = Q(x) − x is uniformly distributed on (−Δ/2, +Δ/2], has zero mean, variance Δ²/12, and is uncorrelated with the input. This yields the canonical "6 dB per bit" SNR formula and underlies essentially all signal-processing-style analysis of fixed-point arithmetic.

## Key Contributions
- The uniform-noise model: quantization error ≈ U(−Δ/2, +Δ/2) when the input pdf is smooth at scale Δ.
- Variance formula σ_q² = Δ²/12 — the single most-quoted equation in DSP quantization analysis.
- "6 dB per bit" SNR scaling: each additional bit gains 20 log₁₀ 2 ≈ 6.02 dB SNR.
- Bennett conditions for when the white-noise approximation breaks down (low-amplitude periodic inputs, low-resolution regimes).

## Key Figures/Tables to Study
- **Sawtooth error function e(x) = Q(x) − x**: triangular wave of period Δ; visualizing it makes the "uniform on (−Δ/2, Δ/2]" model intuitive.
- **SNR vs bit-width plot**: linear in bits with slope 6.02 dB/bit + 1.76 dB sinewave constant.

## Technical Details

### Step size
For a uniform B-bit quantizer covering full-scale range [−A, +A]:
```
Δ = 2A / 2^B
```

### Error distribution under high-resolution assumption
Quantization error e = Q(x) − x. If p_X(x) ≈ const over any width-Δ interval:
```
p_E(e) = 1/Δ      for e ∈ (−Δ/2, +Δ/2]
       = 0        otherwise
E[e] = 0
σ_q² = Var(e) = ∫_{−Δ/2}^{+Δ/2} e² (1/Δ) de = Δ²/12
```

### "6 dB per bit" SNR formula
For a sinusoidal input of amplitude A driving a B-bit quantizer with full-scale ±A:
```
SNR = 10 log₁₀ (P_signal / σ_q²)
    = 10 log₁₀ ((A²/2) / (Δ²/12))
    = 10 log₁₀ (3/2 · 2^{2B})
    = 6.02 B + 1.76    [dB]
```
For a full-loaded Gaussian source with σ_X = A/4 (4σ loading):
```
SNR ≈ 6.02 B − 7.27  [dB]
```

### Equivalence with rate-distortion at high rate
Distortion D ≈ Δ²/12 = (2A)² / (12 · 2^{2B}) = (const) · 2^{−2B} — same exponent −2B as the Gaussian R(D) bound, but the prefactor is suboptimal by the space-filling loss (1.53 dB at high rate).

### When the model breaks
Three regimes where Bennett's uniform-noise model fails:
1. **Low resolution** (B ≤ 3): pdf is *not* flat at scale Δ; error becomes signal-dependent.
2. **Periodic / low-amplitude inputs**: error correlates with input → limit cycles, idle tones.
3. **Outlier-heavy distributions** (LLM activations!): clipping and the heavy tail dominate σ_q², not the uniform-noise term — this is precisely the SmoothQuant / AWQ / QuaRot problem.

### Operational use in LLM quant calibration
For a per-tensor symmetric INT8 quantizer with clip range [−α, +α]:
- Δ = 2α / 255
- predicted unclipped MSE = Δ²/12 = α² / (3 · 255²)
- measured MSE > 3× predicted ⇒ clipping is dominant ⇒ either widen α (more clipping headroom but coarser steps) or switch to per-channel / non-uniform code.

## Connections
- [[rate-distortion-theory]] — Bennett's Δ²/12 is the scalar-uniform realization of the rate-distortion bound up to space-filling loss.
- [[lloyd-max-quantizer]] — Lloyd-Max is the *non-uniform* optimum that beats Bennett's uniform Δ²/12 for non-uniform pdfs.
- [[round-to-nearest-even]] — RNE bias is zero, consistent with Bennett's E[e] = 0 assumption.
- [[stochastic-rounding]] — SR enforces zero-mean error even at low rate (B ≤ 3) where Bennett's assumption breaks.
- [[llm-int8]] — outlier breakdown of Bennett's model is the motivating story for LLM.int8().
