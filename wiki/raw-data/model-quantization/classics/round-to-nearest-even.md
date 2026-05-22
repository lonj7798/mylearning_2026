<!-- scope: IEEE-754 default rounding mode; ties-to-even (banker's rounding); interaction with quantization bias
     deps: [[ieee-754]]
     see-also: [[stochastic-rounding]], [[uniform-quantization-noise]]
-->

# Round-to-Nearest-Even (IEEE-754 "Banker's Rounding")
- **Core Insight:** Among the four IEEE-754 rounding modes, ties-to-even is the only one whose long-run rounding error has zero mean — rounding x.5 to the nearest *even* representable value (rather than always up) prevents the systematic +Δ/2 drift that "ties-up" rounding introduces over many operations.
- **Guideline:** Use RNE as the default rounding mode for *one-shot* quantization (PTQ) and within deterministic forward passes; switch to stochastic rounding for *accumulated* operations (training updates) where Δ-bias compounds.
- **Authors:** IEEE-754 standard (1985, revised 2008, 2019); concept predates IEEE in actuarial/banking practice
- **Year:** 1985 (IEEE-754); 2008/2019 (revisions)
- **URL:** https://ieeexplore.ieee.org/document/8766229 (IEEE 754-2019); summary at https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules
- **Relevant topics:** IEEE-754, rounding modes, quantization bias, deterministic arithmetic

## Abstract
IEEE-754 defines five rounding modes: roundTiesToEven (RNE, default), roundTiesToAway, roundTowardPositive (+∞), roundTowardNegative (−∞), and roundTowardZero (truncation). RNE rounds any value to its nearest representable neighbour, breaking exact-half ties by selecting the neighbour whose LSB is zero (even). This eliminates the bias of "round half up" arithmetic, which would shift accumulated sums by Δ/2 per tied operation. RNE is the de facto industry standard for floating-point and is the default rounding rule cited in every quantization paper unless explicitly using SR.

## Key Contributions
- Defines RNE as the IEEE-754 default rounding mode.
- Establishes ties-to-even (vs ties-up / ties-away-from-zero) for bias elimination.
- Provides the formal correctly-rounded basic arithmetic semantics that hardware vendors must conform to.
- Sets the baseline against which stochastic rounding's bias-variance tradeoff is measured.

## Key Figures/Tables to Study
- **Comparison table: RNE vs ties-up vs truncate** — mean error, variance, max error per operation; RNE is the only mode with E[err] = 0 in the limit of many tied inputs.
- **Cumulative error plot** of summing N RNE-rounded operations vs N ties-up — RNE error grows as O(√N), ties-up as O(N).

## Technical Details

### RNE rule
For input x with adjacent representable neighbours a and b (a ≤ x ≤ b) at distance Δ apart:
```
RNE(x) = a               if x − a < Δ/2
RNE(x) = b               if x − a > Δ/2
RNE(x) = the one of {a, b} with LSB = 0    if x − a = Δ/2  (tie)
```

### Why ties-to-even
Consider summing many values of the form k.5 (Δ=1, ties). Modes:
- **Ties-up** ("schoolbook"): always round k.5 → k+1. Mean error +1/2 per tied op.
- **Ties-down**: always round k.5 → k. Mean error −1/2 per tied op.
- **Ties-away-from-zero**: round k.5 → k+1 if k≥0, k → k if k<0. Symmetric but mean error sign-correlated with input.
- **Ties-to-even (RNE)**: round half of the time to k, half to k+1 (since exactly one of {k, k+1} is even). **Mean error 0** as long as ties are equally distributed across even/odd LSBs.

### Bias under high-resolution assumption
For inputs sampled from a smooth (non-tie-concentrated) distribution at scale Δ:
```
E[RNE(x) − x] = 0
Var[RNE(x) − x] = Δ²/12              (Bennett's formula)
```
Identical first-order statistics to Bennett's uniform-noise model; this is why DSP analysis assumes RNE rounding implicitly.

### Bias when ties dominate
If the input is highly quantized at scale Δ/2 (e.g. a previously rounded signal being re-rounded to a coarser grid), ties happen often. RNE's bias-zero property kicks in here — ties-up rounding would accumulate a Δ/2 DC offset, which is catastrophic in repeated weight updates.

### When RNE fails for training
Even with zero mean error, RNE's variance Δ²/12 plus its deterministic-truncation behaviour for small updates kills sub-FP32 training:
- Updates Δw < Δ_w/2 are deterministically rounded to 0 ⇒ **no update**.
- Over many minibatches the systematic "swallowing" of small updates dominates over Var = Δ²/12.
- This is precisely what stochastic rounding fixes; see [[stochastic-rounding]].

### Hardware modes
- x86: `MXCSR.RC` bits set rounding mode; default RNE.
- ARM: `FPCR.RMode`; default RNE.
- NVIDIA Hopper/Blackwell tensor cores: RNE for FP8/FP16 multiplications; SR available as instruction-level option for the accumulator cast.
- Per-op rounding override is mostly compiler/intrinsic-level; setting a global rounding mode is expensive.

### Other IEEE-754 modes (when they appear in quant)
- **roundTowardZero (truncate)**: cheapest in hardware; max error Δ instead of Δ/2; biased toward 0. Used in some integer-only quant kernels for speed.
- **roundTowardPositive / Negative**: directed rounding; required for interval arithmetic and gradient bounds.

## Connections
- [[ieee-754]] — RNE is defined as part of IEEE-754; subnormal and overflow handling interact with rounding.
- [[stochastic-rounding]] — bias-zero alternative for accumulated operations; SR is to RNE what dithering is to truncation.
- [[uniform-quantization-noise]] — Bennett's Δ²/12 implicitly assumes RNE-like unbiased rounding.
- [[fp8-e4m3]] / [[fp8-e5m2]] — hardware FP8 cast uses RNE by default; SR is an optional override.
- [[mxfp-training]] — recipe explicitly switches RNE↔SR between forward and backward passes.
