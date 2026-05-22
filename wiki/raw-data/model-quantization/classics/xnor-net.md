<!-- scope: XNOR-Net — binary CNNs with per-filter scale recovery
     deps: bnn, straight-through-estimator
     see-also: dorefa-net, bitnet
-->

# XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
- **Core Insight:** Pure-binary convolutions lose too much accuracy on ImageNet; restoring a single per-filter scaling factor α = ‖W‖₁/n recovers most of the loss while keeping the inner product as XNOR + popcount.
- **Guideline:** When binarizing a conv layer, compute α = mean(|W|) per output channel and multiply the popcount result by α (and by per-pixel activation scale β for binary-activation variants); never use plain sign(W) without rescaling.
- **Authors:** Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
- **Year:** 2016
- **URL:** https://arxiv.org/abs/1603.05279
- **Relevant topics:** binary CNN, ImageNet, XNOR-popcount, per-channel scale, mobile inference

## Abstract
XNOR-Net binarizes both weights and inputs of convolutional layers, replacing 32-bit floating-point operations with XNOR and popcount. To close the accuracy gap on ImageNet, the paper introduces two variants — Binary-Weight-Network (weights binary, activations float, ~32× memory) and XNOR-Net (both binary, ~58× speedup). The key trick is a closed-form per-filter (and per-pixel) scaling factor that minimises ‖W − αB‖² in `argmin_{α,B} ‖W − αB‖² s.t. B ∈ {−1,+1}^n`. On ImageNet AlexNet, BWN matches FP and XNOR-Net loses ~12% top-1 — large but viable.

## Key Contributions
- Closed-form derivation of optimal binary approximation: `B* = sign(W), α* = (1/n)‖W‖₁`.
- Two-variant taxonomy (BWN vs XNOR-Net) that became the standard comparison axes.
- ImageNet-scale binary CNN feasibility study (first at this scale).
- Architectural fix: BN-before-binarization, ReLU-after-binarization reordering.

## Key Figures/Tables to Study
- **Figure 2** — block reordering for binary convs (BN → BinAct → BinConv → Pool).
- **Table 2** — ImageNet top-1 / top-5 for BWN vs XNOR-Net vs FP AlexNet/ResNet.
- **Equation 6** — the closed-form scale α* derivation.

## Technical Details

### Weight binarization with optimal scale
For W ∈ ℝⁿ:
`min_{α≥0, B∈{−1,+1}ⁿ} ‖W − αB‖²`
Expanding gives `B* = sign(W)` and `α* = (1/n) Σ |Wᵢ| = ‖W‖₁/n`.
Used per output channel (per filter) in conv layers.

### Activation binarization (XNOR-Net only)
For an input patch I_{:,:,k}:
`β_k = (1/n) ‖I_{:,:,k}‖₁`
Binarized patch H = sign(I); convolution becomes
`I * W ≈ (sign(I) ⊛ sign(W)) ⊙ (K * α)`
where K is a 2-D average of per-pixel β scales, computed cheaply.

### Binary inner product
Same XNOR + popcount kernel as [[bnn]]:
`⟨a,b⟩ = N − 2·popcount(a XOR b)`
multiplied by α·β afterwards.

### Training (STE)
- Forward uses binarized W and (optionally) binarized inputs.
- Backward uses STE through sign(); gradient w.r.t. real-valued shadow weights.
- Block order matters: pooling before binarization wastes precision; BN before binarization centres the distribution at zero so sign() is well-conditioned.

## Connections
- [[bnn]] — the predecessor without per-filter scaling.
- [[straight-through-estimator]] — gradient mechanism.
- [[dorefa-net]] — generalises α to k-bit quantization.
- [[bitnet]] — modern transformer reincarnation of binary weights with similar per-tensor scaling.
- [[awq]] — distant descendant: per-channel scaling factors absorb quantization difficulty (different math, same instinct).
