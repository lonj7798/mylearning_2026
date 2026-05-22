<!-- scope: high-rate optimal density formula p(x)^{1/3}; the Gish-Pierce theorem
     deps: [[rate-distortion-theory]], [[lloyd-max-quantizer]]
     see-also: [[uniform-quantization-noise]], [[companding-mu-law]], [[nf4]]
-->

# Gish-Pierce High-Rate Quantization Bounds (Gish & Pierce 1968)
- **Core Insight:** For a smooth source with pdf p(x) and squared-error distortion, the asymptotically optimal scalar quantizer has reconstruction-level density λ(x) ∝ p(x)^{1/3}, giving high-rate distortion D ≈ (1/12) ||p||_{1/3}³ · 2^{−2R} — the constant prefactor of any scalar quantizer is determined by an L_{1/3} norm of the source density.
- **Guideline:** Don't allocate quantizer levels uniformly in *value* nor uniformly in *probability* — allocate them with density ∝ p(x)^{1/3}; this gives roughly equal contribution to distortion across the dynamic range and is what NF4 / SqueezeLLM / non-uniform LUT codes approximate.
- **Authors:** Herbert Gish, John N. Pierce
- **Year:** 1968
- **URL:** https://ieeexplore.ieee.org/document/1054193 ("Asymptotically Efficient Quantizing", IEEE Trans. IT, Sept 1968)
- **Relevant topics:** high-resolution quantization, optimal density, asymptotic distortion, L_p norm

## Abstract
Gish and Pierce derive the asymptotic (high-rate, large-N) distortion of the optimal N-level scalar quantizer for a continuous source with smooth density p(x). They show that the optimal *point density* λ(x) — the number of reconstruction levels per unit interval at x — is proportional to p(x)^{1/(r+1)} for L_r distortion, which is p(x)^{1/3} for the standard squared-error case. The resulting distortion has the closed form D = (1/12) (∫ p(x)^{1/3} dx)³ · 2^{−2R}, identifying the L_{1/3} "norm" of the source density as the sole determinant of constant-factor performance among scalar quantizers.

## Key Contributions
- Optimal point density λ*(x) ∝ p(x)^{1/(r+1)} for general L_r distortion.
- Closed-form asymptotic distortion D = (1/12) ||p||_{1/3}³ · 2^{−2R} for squared error.
- Identifies the **space-filling loss** of scalar quantization vs Shannon R(D): π·e/6 ≈ 1.42 (1.53 dB) for Gaussian.
- Foundation for companding theory: the optimal compressor F satisfies F'(x) ∝ p(x)^{1/3}.

## Key Figures/Tables to Study
- **Optimal density curve** for Gaussian: λ*(x) ∝ exp(−x²/6) — broader than the source itself (∝ exp(−x²/2)) because density is p^{1/3}.
- **Comparison**: optimal-density quantizer vs uniform vs equiprobable — equiprobable is *not* optimal; allocates too many levels to the tails.

## Technical Details

### Setup
Smooth source pdf p(x), distortion measure d(x, x̂) = |x − x̂|^r (typically r=2). Define point density λ(x) such that the number of reconstruction levels in (x, x+dx) is approximately N·λ(x) dx, with ∫ λ(x) dx = 1.

### Bennett's integral (1948 generalization)
For a quantizer with point density λ(x) and N levels, the high-rate distortion is
```
D(N) ≈ (1/[(r+1) · 2^r · N^r]) · ∫ p(x) λ(x)^{−r} dx
```
For r=2 (squared error):
```
D(N) ≈ (1/(12 N²)) · ∫ p(x) λ(x)^{−2} dx
```

### Optimal point density (Gish-Pierce)
Minimize Bennett's integral over λ subject to ∫ λ = 1. By Hölder / variational argument:
```
λ*(x) ∝ p(x)^{1/(r+1)}
```
For r=2:
```
λ*(x) = p(x)^{1/3} / ∫ p(u)^{1/3} du
```

### Optimal distortion
Plug λ* back in:
```
D*(N) = (1/(12 N²)) · (∫ p(x)^{1/3} dx)³
     = (1/12) · ||p||_{1/3}³ · 2^{−2R}      (with N = 2^R)
```
where ||p||_{1/3} := (∫ p(x)^{1/3} dx)³ (the standard L_{1/3} "norm" notation; technically a quasi-norm).

### Specific source values
- **Gaussian** p(x) = (1/√(2πσ²)) exp(−x²/(2σ²)): ||p||_{1/3}³ = σ² · π√3 ≈ 5.44·σ², giving D* ≈ 0.453 · σ² · 2^{−2R}. Compared to Shannon σ²·2^{−2R}: space-filling loss = 0.453·12 ≈ 5.44, i.e. **1.53 dB** above R(D).
- **Laplacian** p(x) = (1/(2b)) exp(−|x|/b): ||p||_{1/3}³ = 8·b²·54 / 27 ≈ 6·b² (approx).
- **Uniform** p(x) = 1/(2A): ||p||_{1/3}³ = (2A)², giving D = A²·2^{−2R}/3 = exactly Bennett's Δ²/12.

### Optimal compressor (companding interpretation)
A monotone compressor F: ℝ → [0,1] followed by uniform quantization is equivalent to a non-uniform quantizer with point density λ(x) = F'(x). Setting F'(x) ∝ p(x)^{1/3} gives the optimal compressor:
```
F*(x) = (∫_{−∞}^{x} p(u)^{1/3} du) / (∫_{−∞}^{+∞} p(u)^{1/3} du)
```
This is the **principled basis** for [[companding-mu-law]] (which approximates this for log-distributed sources) and for [[nf4]] (which tabulates F* for the unit Gaussian).

### Space-filling loss is unavoidable in 1-D
The 1.53 dB gap to R(D) for any scalar source cannot be closed by any 1-D quantizer; it requires vector quantization in higher dimension (G_d → 1 as d → ∞; see [[vector-quantization]]).

### Operational use in LLM quant
- For non-uniform 4-bit weight code (e.g. NF4, SqueezeLLM LUT): set reconstruction levels at the 1/16 quantiles of p(w)^{1/3}, *not* at uniform spacing nor at uniform probability.
- For activation outliers: the L_{1/3} norm is dominated by the heavy tail → either rotate to flatten (QuaRot) or split outlier channels (SpQR) before applying any scalar code.

## Connections
- [[rate-distortion-theory]] — R(D) is the absolute bound; Gish-Pierce gives the best scalar achievable.
- [[uniform-quantization-noise]] — Bennett's Δ²/12 is the special case λ = const, only optimal for uniform p.
- [[lloyd-max-quantizer]] — Lloyd-Max converges to point density ≈ p^{1/3} at large N.
- [[companding-mu-law]] — practical compressors approximating p^{1/3} for log-distributed sources.
- [[nf4]] — 4-bit tabulation of the optimal compressor for the unit Gaussian.
- [[squeezellm]] — non-uniform LUT weighted by Fisher sensitivity (close cousin of p(x)^{1/3} allocation).
