<!-- scope: information-theoretic lower bound on distortion at a given bit-rate; the floor every quantizer fights
     deps:
     see-also: [[lloyd-max-quantizer]], [[uniform-quantization-noise]], [[information-theoretic-bounds]]
-->

# Rate-Distortion Theory (Shannon 1948 / 1959; Cover & Thomas Ch. 10)
- **Core Insight:** For any source X and distortion measure d, there is a sharp lower bound R(D) = min_{p(x̂|x): E[d(X,X̂)]≤D} I(X; X̂) on the number of bits per symbol required to reproduce X within average distortion D — quantization is fundamentally a rate-distortion problem, not just a rounding problem.
- **Guideline:** When designing a quantizer, first compute or estimate R(D) for the actual weight/activation distribution; use the gap between your achieved (rate, distortion) point and the R(D) curve as the only meaningful efficiency metric.
- **Authors:** Claude E. Shannon (1948 "A Mathematical Theory of Communication"; 1959 "Coding Theorems for a Discrete Source with a Fidelity Criterion"); Cover & Thomas textbook treatment
- **Year:** 1948 / 1959 (Shannon); 1991/2006 (Cover & Thomas)
- **URL:** https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf ; Cover & Thomas, Elements of Information Theory, Ch. 10
- **Relevant topics:** information theory, lossy source coding, quantization lower bound, Gaussian source

## Abstract
Rate-distortion theory establishes the minimum rate R(D) at which a source can be encoded such that the expected distortion does not exceed D. For lossless coding D=0 and R = H(X). For lossy coding, R(D) is the infimum of mutual information I(X; X̂) over all conditional distributions p(x̂|x) satisfying the distortion constraint. The function is non-increasing and convex. Shannon's 1959 source-coding theorem shows R(D) is operationally achievable in the long-block-length limit.

## Key Contributions
- Defines R(D) as a single-letter mutual-information optimization, making it computable.
- Proves the converse: no code can beat R(D) asymptotically.
- Proves achievability: random codebooks achieve R(D) as block length → ∞.
- Closed-form for the Gaussian source under squared-error distortion (see below).
- Foundation for every subsequent quantization-theoretic result.

## Key Figures/Tables to Study
- **R(D) curve for Gaussian source** (convex, decreasing from H(X) to 0) — sets the absolute lower bound any FP/INT format is trying to approach.
- **Reverse water-filling** diagram for parallel Gaussian sources — explains why bit-budget should be allocated per-channel by variance.

## Technical Details

### Definition
For source X with distribution p(x) and distortion d(x, x̂):
```
R(D) = min_{p(x̂|x) : E[d(X,X̂)] ≤ D} I(X; X̂)
```

### Gaussian source under squared-error distortion
For X ~ N(0, σ²) and d(x, x̂) = (x − x̂)²:
```
R(D) = (1/2) log₂ (σ² / D)    for 0 ≤ D ≤ σ²
R(D) = 0                       for D > σ²
```
Equivalently `D(R) = σ² · 2^{−2R}`. **The "−6 dB per bit" rule: every additional bit reduces MSE by a factor of 4** (= 6.02 dB). This is the canonical benchmark; all scalar quantizers fall short of it by a "space-filling loss" factor.

### Parallel Gaussian sources (reverse water-filling)
For independent Gaussians X_i ~ N(0, σ_i²), the rate-distortion-optimal allocation gives distortion D_i = min(λ, σ_i²) for some water level λ chosen so that Σ D_i = D. Channels with σ_i² < λ are not coded at all (D_i = σ_i², R_i = 0). **This is the theoretical justification for per-channel / per-group bit allocation in modern LLM quantization.**

### Distortion at high rate (Gish-Pierce regime)
For smooth source p(x) and optimal (non-uniform) scalar quantizer, the high-rate distortion behaves as
```
D(R) ≈ (1/12) ||p||_{1/3}³ · 2^{−2R}
```
where ||p||_{1/3} = (∫ p(x)^{1/3} dx)³. See [[information-theoretic-bounds]] for the derivation.

### Operational meaning for LLM quantization
- An N-bit per-weight quantizer has R = N. Compare its measured MSE against σ_W² · 2^{−2N} (the Gaussian floor) — the ratio is the quantizer's "rate-distortion gap."
- Vector quantization (k-means, PQ) can close the gap by exploiting dimension; scalar quantizers always pay a ≥ 1.53 dB space-filling penalty.

## Connections
- [[lloyd-max-quantizer]] — the optimal *scalar* quantizer for a fixed distribution; achieves R(D) only asymptotically.
- [[uniform-quantization-noise]] — high-resolution noise model that gives the Δ²/12 distortion formula.
- [[information-theoretic-bounds]] — Gish-Pierce high-rate density derivation.
- [[vector-quantization]] — VQ closes the gap to R(D) by going to higher dimension.
- [[nf4]] — Dettmers' NF4 is essentially a Lloyd-Max approximation tuned for the Gaussian weight prior, exploiting this same theory.
