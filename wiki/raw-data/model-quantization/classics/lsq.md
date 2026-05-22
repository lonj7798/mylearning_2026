<!-- scope: LSQ — learn the quantizer step size itself by SGD
     deps: straight-through-estimator, pact
     see-also: lsq-plus, adaround, omniquant
-->

# Learned Step Size Quantization (LSQ)
- **Core Insight:** Treat the quantizer step size s as a learnable parameter, and derive its gradient analytically through the round() operation — the per-sample gradient ∂q/∂s = clamp(x/s) − ⌊x/s⌋ inside the quantizer's active range, allowing the scale to adapt jointly with the weights.
- **Guideline:** Use LSQ as the default QAT baseline at ≥2 bits; init s = 2·⟨|x|⟩/√Qmax; scale the s gradient by `1/√(N·Qmax)` to balance the per-step update magnitude against the weights themselves.
- **Authors:** Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha
- **Year:** 2020 (ICLR)
- **URL:** https://arxiv.org/abs/1902.08153
- **Relevant topics:** QAT, learned step size, uniform quantizer, gradient through round, SOTA baseline

## Abstract
LSQ frames the quantizer step size s as a learnable parameter trained by SGD. A standard symmetric uniform quantizer maps `x ↦ s · round(clamp(x/s, Qmin, Qmax))`; the paper derives the gradient ∂q/∂s in closed form by treating round() with STE, giving `∂q/∂s = −⌊x/s⌋ + clamp(x/s, Qmin, Qmax)` inside the active range and `±Qmax` at the saturation boundary. With this single change LSQ matches or surpasses every prior QAT scheme on ImageNet down to 2-bit, becoming the default benchmark every later LLM QAT method (OmniQuant, LLM-QAT, EfficientQAT) compares against.

## Key Contributions
- Analytic gradient of the quantizer w.r.t. its step size s (the load-bearing formula).
- A `1/√(N·Qmax)` gradient-scale rule that stabilises joint W + s optimisation.
- Demonstrates state-of-the-art ImageNet QAT at 2/3/4-bit on ResNet, VGG, SqueezeNext.
- Layer-wise per-tensor s (separate scales for weights vs activations) is enough — no per-channel needed for activations.

## Key Figures/Tables to Study
- **Equation 4 & Figure 2** — the ∂q/∂s gradient and the staircase plot illustrating it.
- **Table 2** — ImageNet ResNet-18 at 2/3/4-bit: LSQ beats QIL, PACT, DoReFa across the board.

## Technical Details

### Quantizer (symmetric uniform)
For integer range [Qmin, Qmax] (e.g. Qmax = 2^{k−1} − 1 for signed k-bit):
`q(x, s) = s · round(clamp(x/s, Qmin, Qmax))`

### Forward
1. `v = x/s`
2. `v_clip = clamp(v, Qmin, Qmax)`
3. `v_int = round(v_clip)`
4. `q = s · v_int`

### Backward w.r.t. x (STE)
`∂q/∂x = 1[Qmin ≤ x/s ≤ Qmax],  0 otherwise`

### Backward w.r.t. s (the LSQ formula)
Treating round() with STE:
- If `x/s < Qmin`:  ∂q/∂s = Qmin
- If `x/s > Qmax`:  ∂q/∂s = Qmax
- Otherwise:         ∂q/∂s = −⌊x/s⌉ + (x/s)
  ≡ in clipped notation: `∂q/∂s = −round(x/s) + clamp(x/s, Qmin, Qmax)`

Equivalently the local approximation:
`∂q/∂s = clamp(x/s, Qmin, Qmax) − round(clamp(x/s, Qmin, Qmax))`
which is the quantization residual divided by s — pushes s toward values that minimize residual.

### Gradient scale
Per-tensor gradient scale `g = 1/√(N·Qmax)` where N is the number of elements. Applied as
`∂L/∂s ← g · ∂L/∂s`
so the per-step update to s remains comparable to the per-element weight updates.

### Initialisation
`s₀ = 2·⟨|x|⟩ / √Qmax`
from calibration statistics on the first batch.

## Connections
- [[straight-through-estimator]] — round() backward used throughout.
- [[pact]] — PACT learns the clip α, which is mathematically equivalent to LSQ's s · Qmax for ReLU activations; LSQ generalises to signed/symmetric weights.
- [[lsq-plus]] — adds learnable zero-point for asymmetric activations (post-ReLU+).
- [[dorefa-net]] — fixed-step baseline LSQ supersedes.
- [[adaround]] — orthogonal: AdaRound learns per-weight rounding direction with fixed step; LSQ learns step with fixed (nearest) rounding.
- [[omniquant]] — modern LLM heir: learns both clip and equivalent transformations on a frozen LLM.
