---
chapter: ch-04
course: model-quantization
phase: read
excerpt_of: "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation (Bengio, Léonard, Courville 2013)"
source_url: https://arxiv.org/abs/1308.3432
created_at: "2026-05-21"
raw_data_source: [[raw-data/straight-through-estimator]]
---

# Excerpt: STE — the biased-but-useful gradient through round()

**Authors:** Yoshua Bengio, Nicholas Léonard, Aaron Courville.
**Year:** 2013.
**Venue:** arXiv 1308.3432.
**URL:** see source_url.

---

## The one-box rule

```
Forward (uniform quantizer):
    q(x) = clamp(round(x / Δ) · Δ,  −α,  +α)

Backward (STE):
    ∂L/∂x = ∂L/∂q · 1[|x| ≤ α]              # identity inside clip, zero outside
```

The `round(·)` derivative — a Dirac comb of zeros almost everywhere — is replaced by 1. Outside the clip range, the gradient is killed because saturated values cannot meaningfully respond.

---

## Why it works (intuition)

The expected gradient under symmetric uniform noise of width `Δ` is `∂L/∂x` to first order; STE is its zero-noise approximation. The bias is bounded by the local curvature of `L` times `Δ²`, which vanishes as training progresses and either `Δ` shrinks or the network learns to keep activations near quantizer levels.

The 2013 paper compares four gradient estimators (REINFORCE, decomposed sampling, noise-injection, and STE) on small classification tasks. **STE outperforms every unbiased estimator** despite the bias — REINFORCE-style variance dominates and stalls training.

---

## Variants used by later QAT methods

- **Slope-annealed STE** (BNN): backward uses `d/dx tanh(βx)` with `β` raised across training to harden the transition.
- **Clipped STE** (PACT, LSQ): backward zero outside learned clip `α`.
- **Soft-staircase STE**: replace `round()` backward with the derivative of a sum of shifted sigmoids.
- **Per-parameter STE** (AdaRound): backward through a rectified sigmoid `h(V)` that anneals to `{0, 1}`.

---

## Implementation idiom (PyTorch)

```python
class QuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        ctx.save_for_backward(x, scale)
        ctx.qmin, ctx.qmax = qmin, qmax
        return scale * torch.round(torch.clamp(x / scale, qmin, qmax))

    @staticmethod
    def backward(ctx, grad_out):
        x, scale = ctx.saved_tensors
        mask = (x / scale >= ctx.qmin) & (x / scale <= ctx.qmax)
        return grad_out * mask, None, None, None
```

Never call `.detach()` on the quantizer output — that kills the gradient entirely instead of routing it through STE.

---

## Connections

- [[bnn]] — first large-scale use of STE for 1-bit weights and activations.
- [[xnor-net]] — uses STE with scale factor `α = ‖W‖₁ / n`.
- [[dorefa-net]] — generalizes STE to `k`-bit weights, activations, and gradients.
- [[pact]] — adds a learned clip `α` inside the STE.
- [[lsq]] — learns the step size `Δ` itself; gradient derived via STE.
- [[adaround]] — applies STE through a rectified-sigmoid soft assignment for per-weight rounding direction.
- [[ch-04]] — parent synthesis.
