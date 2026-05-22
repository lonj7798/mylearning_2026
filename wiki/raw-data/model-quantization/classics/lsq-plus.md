<!-- scope: LSQ+ — asymmetric extension of LSQ with learnable zero-point
     deps: lsq, pact
     see-also: quantization-mapping, adaround
-->

# LSQ+: Improving Low-bit Quantization Through Learnable Offsets and Better Initialization
- **Core Insight:** Activations after Swish/h-swish/PReLU have negative-side mass that symmetric quantizers waste; adding a learnable zero-point β (offset) and learning both (s, β) jointly recovers the lost resolution and re-enables sub-4-bit QAT for modern architectures.
- **Guideline:** For any non-ReLU activation, use the asymmetric quantizer `q = s·round(clamp((x−β)/s, Qmin, Qmax)) + β`; init s from per-channel percentile, init β = min(x); grid-search (s₀, β₀) over a small calibration set before joint SGD.
- **Authors:** Yash Bhalgat, Jinwon Lee, Markus Nagel, Tijmen Blankevoort, Nojun Kwak
- **Year:** 2020 (CVPR Workshop)
- **URL:** https://arxiv.org/abs/2004.09576
- **Relevant topics:** asymmetric QAT, learnable zero-point, EfficientNet quantization, Swish activations

## Abstract
LSQ+ extends Learned Step Size Quantization (LSQ) with a learnable per-tensor zero-point β to handle modern non-ReLU activations (Swish in EfficientNet, h-swish in MobileNetV3) whose distributions have non-negligible negative-tail mass. The paper also introduces a calibration-driven initialisation of (s, β) instead of LSQ's heuristic, which sharply improves convergence at 3-bit. On EfficientNet-B0 / MobileNetV3, LSQ+ matches FP32 at 4-bit and closes most of the gap at 3-bit, where original LSQ fails.

## Key Contributions
- Adds a learnable offset β alongside step s — making LSQ asymmetric.
- Gradient ∂q/∂β derived analytically through STE.
- Robust initialisation: per-channel statistics → grid search over (s, β).
- Demonstrates that learnable β is critical for Swish / h-swish networks at ≤4-bit.

## Key Figures/Tables to Study
- **Figure 1** — Swish activation histogram with negative mass that symmetric LSQ wastes.
- **Table 2** — EfficientNet-B0 4-bit / 3-bit: LSQ+ vs LSQ vs PACT.

## Technical Details

### Asymmetric quantizer
For integer range [Qmin, Qmax]:
`q(x; s, β) = s · round(clamp((x − β)/s, Qmin, Qmax)) + β`

### Forward
1. `v = (x − β)/s`
2. `v_clip = clamp(v, Qmin, Qmax)`
3. `v_int = round(v_clip)`
4. `q = s·v_int + β`

### Backward w.r.t. s (LSQ formula, unchanged)
Inside active range: `∂q/∂s = −round((x−β)/s) + clamp((x−β)/s, Qmin, Qmax)`
Outside: ∂q/∂s = Qmin or Qmax (boundary).

### Backward w.r.t. β (new)
Inside active range: `∂q/∂β = 0` (the s·round term moves with β and cancels the +β).
Outside: `∂q/∂β = 1` (saturated → β moves the saturation level 1:1).
Equivalently: `∂q/∂β = 1 − 1[Qmin ≤ (x−β)/s ≤ Qmax]`.

### Backward w.r.t. x (STE)
`∂q/∂x = 1[Qmin ≤ (x−β)/s ≤ Qmax]`

### Initialisation
1. Calibration pass: collect per-tensor min and max.
2. β₀ = min(x); s₀ = (max(x) − min(x))/(Qmax − Qmin).
3. Optional small grid search ±20% on (s₀, β₀) minimising MSE on the calibration batch.

## Connections
- [[lsq]] — symmetric predecessor; LSQ+ adds β.
- [[pact]] — PACT learns only the positive clip α; LSQ+ generalises to a signed clip on both sides.
- [[quantization-mapping]] — sits inside the symmetric-vs-asymmetric taxonomy.
- [[adaround]] — orthogonal: LSQ+ learns (s, β) with nearest rounding; AdaRound learns per-weight rounding direction with fixed (s, β).
- [[omniquant]] — modern LLM heir: learns clip thresholds + equivalent transforms via gradient descent on a frozen model.
