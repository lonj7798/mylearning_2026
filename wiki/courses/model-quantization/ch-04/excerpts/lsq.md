---
chapter: ch-04
course: model-quantization
phase: read
excerpt_of: "Learned Step Size Quantization (Esser, McKinstry, Bablani, Appuswamy, Modha 2020)"
source_url: https://arxiv.org/abs/1902.08153
created_at: "2026-05-21"
raw_data_source: [[raw-data/lsq]]
---

# Excerpt: LSQ — analytic gradient through the quantizer's step size

**Authors:** Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha (IBM Research).
**Year:** 2020 (ICLR).
**URL:** see source_url.

---

## The one-box quantizer (symmetric uniform)

For integer range `[Q_min, Q_max]` (e.g. `Q_max = 2^{k−1} − 1` for signed `k`-bit):

```math
q(x, s) \,=\, s \cdot \text{round}\big(\text{clamp}(x / s, \, Q_{\min}, \, Q_{\max})\big)
```

`s` is **learnable**.

---

## Backward w.r.t. `x` (standard STE)

```math
\frac{\partial q}{\partial x} \,=\, \mathbf{1}[Q_{\min} \le x/s \le Q_{\max}], \quad 0 \text{ otherwise}
```

---

## Backward w.r.t. `s` — the load-bearing LSQ formula

```math
\boxed{\;
\frac{\partial q}{\partial s} \,=\,
\begin{cases}
Q_{\min} & x/s < Q_{\min} \\
Q_{\max} & x/s > Q_{\max} \\
- \text{round}(x/s) + (x/s) & \text{otherwise}
\end{cases}\;}
```

Equivalently, inside the active range:

```math
\frac{\partial q}{\partial s} \,=\, \text{clamp}(x/s, \, Q_{\min}, \, Q_{\max}) - \text{round}(\text{clamp}(x/s, \, Q_{\min}, \, Q_{\max}))
```

This is the **quantization residual divided by `s`** — pushes `s` toward values that minimize the residual. Too-small `s` ⇒ many saturated samples ⇒ boundary term `Q_max` drives `s` upward. Too-large `s` ⇒ coarse residuals dominate ⇒ residual term drives `s` downward.

---

## Gradient-scale heuristic (the second load-bearing trick)

Per-tensor gradient scale `g = 1 / √(N · Q_max)`, where `N` is the number of elements, applied to `∂L/∂s`:

```math
\frac{\partial L}{\partial s} \,\leftarrow\, g \cdot \frac{\partial L}{\partial s}
```

This ensures the per-step update to `s` is comparable in magnitude to per-element weight updates. Without it, joint W + s optimization is unstable.

---

## Initialization

```math
s_0 \,=\, \frac{2 \cdot \langle |x| \rangle}{\sqrt{Q_{\max}}}
```

from calibration statistics on the first batch.

---

## Forward pseudocode

```
v = x / s
v_clip = clamp(v, Qmin, Qmax)
v_int = round(v_clip)
q = s · v_int
```

---

## Empirical headline (paper, ImageNet ResNet-18)

| Bits (W = A) | LSQ | DoReFa | PACT | QIL |
|---|---|---|---|---|
| 4 | matches FP | gap | matches FP | matches FP |
| 3 | matches FP | gap | gap | small gap |
| **2** | **small gap** | **collapses** | **collapses** | **gap** |

**LSQ is the default QAT baseline every later LLM QAT method (OmniQuant, LLM-QAT, EfficientQAT) compares against.**

---

## Connections

- [[excerpts/straight-through-estimator]] — `round()` backward used throughout.
- [[pact]] — learns the clip `α`, mathematically equivalent to LSQ's `s · Q_max` for ReLU activations; LSQ generalizes to signed/symmetric weights.
- [[lsq-plus]] — adds learnable zero-point `β` for asymmetric activations.
- [[dorefa-net]] — fixed-step baseline LSQ supersedes.
- [[excerpts/adaround]] — orthogonal: LSQ learns step with fixed (nearest) rounding; AdaRound learns per-weight rounding with fixed step.
- [[omniquant]] — modern LLM heir: learns clip + equivalent transformations on a frozen model (ch-10).
- [[ch-04]] — parent synthesis.
