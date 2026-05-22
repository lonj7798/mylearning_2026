<!-- scope: STE — biased-but-useful gradient through non-differentiable quantizers
     deps: uniform-quantization-noise
     see-also: bnn, dorefa-net, pact, lsq
-->

# Estimating or Propagating Gradients Through Stochastic Neurons (Straight-Through Estimator)
- **Core Insight:** Treat a non-differentiable quantizer / threshold as the identity on the backward pass, so gradients flow through as if the discretization step were not there — biased but empirically convergent.
- **Guideline:** Whenever you put a rounder, sign(), or clip-and-round in the forward graph, register a custom autograd op whose backward is `dL/dx = dL/dq` (optionally masked to the clipped range); never call `.detach()` on the quantizer output.
- **Authors:** Yoshua Bengio, Nicholas Léonard, Aaron Courville
- **Year:** 2013
- **URL:** https://arxiv.org/abs/1308.3432
- **Relevant topics:** QAT, gradient estimation, discretization, binary neurons, surrogate gradient

## Abstract
The paper studies how to train neural networks containing stochastic or hard-threshold neurons that block gradient flow. It compares four estimators (REINFORCE, decomposed sampling, noise-injection, and the straight-through estimator) on small classification tasks. The straight-through estimator — backpropagate the upstream gradient as if the non-differentiable step were the identity — is shown to work surprisingly well in practice despite being biased. STE became the universal trick that made every later QAT method (BNN, DoReFa, PACT, LSQ) possible.

## Key Contributions
- Formalizes the family of gradient estimators for stochastic / hard-threshold neurons.
- Introduces the straight-through estimator: bypass the non-differentiable op on the backward pass.
- Empirically shows STE outperforms REINFORCE-style unbiased estimators despite its bias.
- Sets up the slope-annealed variant (slope ≥ 1 on the surrogate sigmoid) used by later binary nets.

## Key Figures/Tables to Study
- **Figure 1** — diagram of the four estimators; clarifies what "passing the gradient through" actually means.
- **Table 1/2** — MNIST training curves: STE converges where REINFORCE stalls.

## Technical Details

### Forward (uniform quantizer)
With step size Δ and clip range [−α, α]:
`q(x) = clamp(round(x/Δ)·Δ, −α, α)`

### Backward (STE)
`∂L/∂x = ∂L/∂q · 1[|x| ≤ α]`
i.e. identity inside the clip, zero outside. The `round(·)` derivative (a Dirac comb of zeros almost everywhere) is replaced by 1.

### Why it works (intuition)
The expected gradient under symmetric uniform noise of width Δ is `∂L/∂x` to first order; STE is its zero-noise approximation. The bias is bounded by the local curvature of L times Δ², which vanishes as training progresses and Δ shrinks (or as the network learns to keep activations near quantizer levels).

### Variants used later
- **Slope-annealed STE** (BNN): backward uses `d/dx tanh(βx)` with β raised across training.
- **Clipped STE** (PACT, LSQ): backward zero outside learned clip α.
- **Soft-staircase STE**: replace round() backward with the derivative of a sum of shifted sigmoids.

## Connections
- [[bnn]] — first large-scale use of STE for 1-bit weights and activations.
- [[xnor-net]] — uses STE with scale factor α = ‖W‖₁/n.
- [[dorefa-net]] — generalises STE to k-bit weights, activations, gradients.
- [[pact]] — adds a learned clip α inside the STE.
- [[lsq]] — learns the step size Δ itself, gradient derived via STE.
- [[wage]] — STE applied also to gradients/errors during training.
