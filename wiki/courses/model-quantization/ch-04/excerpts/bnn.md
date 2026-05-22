---
chapter: ch-04
course: model-quantization
phase: read
excerpt_of: "Binarized Neural Networks (Courbariaux, Hubara, Soudry, El-Yaniv, Bengio 2016)"
source_url: https://arxiv.org/abs/1602.02830
created_at: "2026-05-21"
raw_data_source: [[raw-data/bnn]]
---

# Excerpt: BNN — first end-to-end 1-bit weight + activation training

**Authors:** Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio.
**Year:** 2016.
**Venue:** arXiv 1602.02830.
**URL:** see source_url.

---

## The one-box training loop (Algorithm 1)

```
maintain real-valued shadow weights  W_r
for each minibatch:
    W_b = sign(W_r) ∈ {−1, +1}                       # forward weight binarization
    a_b = sign(a_r)                                   # forward activation binarization
    forward(W_b, a_b)                                 # binary matmul = XNOR + popcount
    backward via STE:  ∂L/∂a_r = ∂L/∂a_b · 1[|a_r| ≤ 1]
                       ∂L/∂W_r = ∂L/∂W_b  (identity through sign)
    W_r ← clip(W_r − η · g,  −1,  +1)                 # SGD on shadow buffer
```

The shadow real-valued buffer `W_r` is what SGD updates; the binary `W_b` is what enters the matmul.

---

## The binary inner product

For `x_b, w_b ∈ {−1, +1}^n`:

```math
\boxed{\; x_b^T w_b \,=\, n - 2 \cdot \text{popcount}(x_b \,\text{XOR}\, w_b) \;}
```

Implementable on a CPU with one `xnor` + one `popcnt` instruction per 64-bit word. **32× memory reduction** vs FP32 weights.

---

## Activation STE — clipped form

`a_b = sign(a_r)` with the *clipped* STE backward (gradient passes inside `[−1, 1]`, dies outside):

```math
\frac{\partial L}{\partial a_r} \,=\, \frac{\partial L}{\partial a_b} \cdot \mathbf{1}[|a_r| \le 1]
```

This is the canonical "clipped STE" used by every subsequent QAT method.

---

## Empirical headline numbers (paper)

- MNIST: 0.96% error (vs 0.94% FP32) — essentially identical.
- CIFAR-10: 10.15% error (vs 9.94% FP32).
- SVHN: 2.80% error (vs 2.64% FP32).

Proof-of-concept that 1-bit training works at moderate scale.

---

## BatchNorm shift trick

After binarization, BN constants `α, β, σ, γ` fold into a single shift + sign threshold `τ`:

```math
a_b = \text{sign}(a - \tau), \quad \tau = \beta \cdot \sigma / \gamma - \mu
```

→ **pure integer at inference.** No FP32 left in the hot path.

---

## Why shadow weights matter

Without `W_r`, SGD would round `W_b` to `±1` after every update; small gradient signals never accumulate to a sign flip. The shadow buffer accumulates millions of small updates between flips — analogous to FP32 master weights in mixed-precision training.

---

## Connections

- [[excerpts/straight-through-estimator]] — the gradient backbone BNN depends on.
- [[xnor-net]] — adds per-filter scale `α = ‖W‖₁ / n` for both weights and activations.
- [[dorefa-net]] — extends to arbitrary `k`-bit weights / activations / gradients.
- [[bitnet]] / [[bitnet-w158]] — modern LLM-era reincarnation of binary weights; ternary `{−1, 0, +1}` extension (ch-16).
- [[ch-04]] — parent synthesis.
