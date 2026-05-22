<!-- chapter: ch-04
     track: math-foundations
     title: STE + QAT Lineage — BNN → LSQ → AdaRound
     sources: [[straight-through-estimator]], [[bnn]], [[xnor-net]], [[dorefa-net]], [[pact]], [[lsq]], [[lsq-plus]], [[adaround]], [[brecq]]
     figures: figures/ste-backward.html
-->

# Chapter 4 — STE + QAT Lineage: BNN → LSQ → AdaRound

> **Core insight.** Quantization-aware training reduces to one trick (the straight-through estimator: "pretend the rounder is the identity in backward") wrapped around a sequence of progressively richer parameterizations. **BNN** learned binary weights with shadow real-valued copies; **XNOR-Net** added a per-filter scale; **DoReFa** generalized to `k`-bit; **PACT** learned the activation clip; **LSQ** learned the quantizer step size itself; **LSQ+** added a learnable zero-point; **AdaRound** learned the per-weight rounding *direction* via a Hessian objective; **BRECQ** extended the Hessian objective from layer to block. Every later LLM QAT method (OmniQuant, LLM-QAT, EfficientQAT) and every PTQ method (GPTQ, GPTVQ) descends from this chain.
>
> **Guideline.** When you see a new "learnable quantizer" or "learned rounding" paper, decompose it into (a) the parameterization (what is learnable: scale, zero-point, rounding bit, codebook?), (b) the reconstruction objective (layer-wise MSE, block-wise MSE, Hessian-weighted, KL?), and (c) the optimization scaffold (STE, soft-relaxation + anneal, closed-form?). Almost every method in the wild is one of three parameterizations crossed with one of two objectives.

---

## Why this chapter exists

PTQ tells you how to fit a quantizer to a frozen model using only calibration data. QAT tells you how to *train* a model that lives at low precision. The two literatures share one mathematical object — the straight-through estimator — and most modern PTQ methods (GPTQ, AdaRound, BRECQ) are best understood as PTQ-flavored QAT: they fit *something* by gradient descent or closed-form on a small calibration set, using STE-style derivatives through the rounding step.

Three things you should walk away with:

1. The STE itself: forward `q(x) = round(x/Δ)·Δ`, backward `∂q/∂x = 1` inside the clip and 0 outside; the bias the trick injects and why it doesn't break SGD.
2. The QAT lineage: BNN → XNOR-Net → DoReFa → PACT → LSQ → LSQ+, with the load-bearing formula at each step.
3. The PTQ-Hessian lineage: AdaRound → BRECQ → (next chapter) GPTQ; the layer-wise `‖W·X − Ŵ·X‖²` objective is the parent of every Hessian-aware LLM PTQ method shipping in 2026.

---

## 1. The straight-through estimator

From [[straight-through-estimator]] (Bengio, Léonard, Courville 2013), the problem: a forward pass that includes `sign(x)`, `round(x/Δ)·Δ`, or any other non-differentiable threshold blocks gradient flow on the backward pass. The straight-through estimator solves this by **treating the non-differentiable op as the identity on the backward pass**:

```
Forward:    q(x) = clamp(round(x / Δ) · Δ, −α, α)
Backward:   ∂L/∂x = ∂L/∂q · 1[|x| ≤ α]                 # identity inside clip, zero outside
```

The `round(·)` derivative — a Dirac comb of zeros almost everywhere — is replaced by 1. Outside the clip range, the gradient is killed because saturated values cannot meaningfully respond.

### Why a biased gradient still works

The expected gradient under symmetric uniform noise of width `Δ` is `∂L/∂x` to first order; STE is its zero-noise approximation. The bias is bounded by the local curvature of `L` times `Δ²`, which vanishes as training progresses and either `Δ` shrinks or the network learns to keep activations near quantizer levels. Empirically, STE beats every formally-unbiased estimator (REINFORCE, decomposed sampling, noise-injection) on the toy tasks in the 2013 paper and on every QAT benchmark since.

### Variants

- **Slope-annealed STE** (BNN): backward uses `d/dx tanh(βx)` with `β` ramped up across training.
- **Clipped STE** (PACT, LSQ): backward zero outside learned clip `α`.
- **Soft-staircase STE**: replace `round()` backward with the derivative of a sum of shifted sigmoids.

> **Practical pitfall.** Never call `.detach()` on the quantizer output. Wrap the quantizer in a custom autograd op whose `backward` implements the STE rule. PyTorch idiom: a `torch.autograd.Function` subclass with `forward(ctx, x): return q(x)` and `backward(ctx, grad_out): return grad_out * mask` where `mask` is `1[|x| ≤ α]`.

---

## 2. BNN and XNOR-Net — binary weights and the first per-filter scale

From [[bnn]] (Courbariaux, Hubara, Soudry, El-Yaniv, Bengio 2016), networks can be trained from scratch with both weights and activations constrained to `{−1, +1}` if:

- A **real-valued shadow buffer** `W_r` is updated by SGD between steps,
- The forward pass uses `W_b = sign(W_r)`,
- The backward pass uses STE.

```
Weight binarization:        W_b = sign(W_r) ∈ {−1, +1}
Activation binarization:    a_b = sign(a_r)        with STE backward:  ∂L/∂a_r = ∂L/∂a_b · 1[|a_r| ≤ 1]
Post-update:                W_r ← clip(W_r − η · g, −1, +1)
Binary inner product:       xᵀw = N − 2·popcount(x_b XOR w_b)
```

The binary multiply collapses to XNOR + popcount — one xnor + one popcnt per 64-bit word. Empirical: BNN reaches MNIST 0.96% error and CIFAR-10 10.15% error with 1-bit weights and activations.

### XNOR-Net — the closed-form per-filter scale

From [[xnor-net]] (Rastegari, Ordonez, Redmon, Farhadi 2016), pure-binary convolution loses too much on ImageNet. The fix is to restore a **per-filter scaling factor**:

```
For W ∈ ℝⁿ:    min_{α ≥ 0, B ∈ {−1, +1}ⁿ}  ‖W − α·B‖²
                  ⇒   B* = sign(W),   α* = (1/n) · Σ |W_i| = ‖W‖₁ / n
```

The conv approximates as `(sign(I) ⊛ sign(W)) · α · β`, with `β = (1/n)‖I‖₁` per input patch. BWN (binary weights, FP activations) matches FP AlexNet on ImageNet; XNOR-Net (both binary) loses ~12% top-1 — large but viable. The instinct "the per-filter scale absorbs quantization difficulty" recurs at every later step of this lineage and resurfaces in [[awq]] as "per-channel scaling factors absorb activation difficulty" (ch-09).

---

## 3. DoReFa and PACT — arbitrary-bit + learnable clip

From [[dorefa-net]] (Zhou et al. 2016), the single quantizer family

```math
q_k(x) \,=\, \frac{1}{2^k - 1} \cdot \text{round}\!\big((2^k - 1) \cdot x\big), \qquad x \in [0, 1]
```

quantizes weights, activations, and **gradients** independently. The novel piece is **gradient quantization with stochastic rounding** — `q_k(g)` is added to a uniform noise draw before rounding so that `E[g_q] = g`. Without SR, gradient quantization biases SGD and training diverges; with SR, on ImageNet AlexNet with `(W=1, A=2, G=6)` bits, DoReFa reaches 46.1% top-1 (vs 55.9% FP).

The `(Wbits, Abits, Gbits)` notation comes from this paper and is the canonical way to describe a QAT recipe.

### PACT — learn the activation clip

From [[pact]] (Choi et al. 2018), post-ReLU activations are heavy-tailed and unbounded; a fixed clip threshold either wastes resolution (clip too high) or saturates aggressively (clip too low). PACT replaces ReLU with a learnable clip:

```
y = PACT(x; α) = clip(x, 0, α) = 0.5 · (|x| − |x − α| + α)
y_q = round(y · (2^k − 1) / α) · α / (2^k − 1)
```

The **gradient w.r.t. learned `α`** — the load-bearing formula — is

```math
\frac{\partial L}{\partial \alpha} \,=\, \frac{\partial L}{\partial y} \cdot \mathbf{1}[x \ge \alpha]
```

`α` receives gradient *only from saturated samples*. Intuition: if too many samples clip, `∂L/∂α` pushes `α` up; if `α` is so high that quantization noise dominates, the indirect path through `y_q` pushes it down. With a small `L2` penalty (`λ ≈ 10⁻⁴`) on `α²` to prevent unbounded growth, PACT delivered the first 4-bit ResNet-50 within ~1% of FP on ImageNet.

---

## 4. LSQ — learn the step size itself

From [[lsq]] (Esser, McKinstry, Bablani, Appuswamy, Modha 2020), instead of fixing the step `Δ` from a per-tensor calibration, treat `s` as a learnable parameter of the quantizer and derive its gradient analytically.

### Quantizer (symmetric uniform)

```math
q(x, s) \,=\, s \cdot \text{round}(\text{clamp}(x / s, Q_{\min}, Q_{\max}))
```

### Backward w.r.t. `x` (standard STE)

```
∂q/∂x  =  1[Qmin ≤ x/s ≤ Qmax],   0 otherwise
```

### Backward w.r.t. `s` — the LSQ formula

```math
\boxed{\;
\frac{\partial q}{\partial s} \,=\,
\begin{cases}
Q_{\min} & x/s < Q_{\min} \\
Q_{\max} & x/s > Q_{\max} \\
- \text{round}(x/s) + (x/s) & \text{otherwise}
\end{cases}\;}
```

Equivalently, inside the active range, `∂q/∂s = clamp(x/s, Qmin, Qmax) − round(clamp(x/s, Qmin, Qmax))` — the quantization residual divided by `s`. **This pushes `s` toward values that minimize the residual**: too-small `s` → many saturated samples → `s` increases via the boundary term; too-large `s` → coarse residuals → `s` decreases.

### Gradient-scale rule

A per-tensor scale `g = 1 / √(N · Qmax)`, where `N` is the number of elements, is applied to `∂L/∂s` so that the per-step update to `s` remains comparable to the per-element weight updates. Init: `s₀ = 2·⟨|x|⟩ / √Qmax`. With these two heuristics, LSQ matches or surpasses every prior QAT scheme on ImageNet down to 2-bit and is the default QAT baseline every later LLM QAT method (OmniQuant, LLM-QAT, EfficientQAT) compares against.

### LSQ+ — asymmetric extension

From [[lsq-plus]] (Bhalgat et al. 2020), modern non-ReLU activations (Swish, h-swish) have non-negligible negative-tail mass that symmetric LSQ wastes. LSQ+ adds a learnable zero-point `β`:

```math
q(x; s, \beta) \,=\, s \cdot \text{round}(\text{clamp}((x - \beta)/s, Q_{\min}, Q_{\max})) + \beta
```

Gradient w.r.t. `β`: inside active range `∂q/∂β = 0`; outside, `∂q/∂β = 1` — `β` shifts the saturation level. With percentile-based init and a small grid search on `(s, β)` before joint SGD, LSQ+ closes the 3-bit gap on EfficientNet that broke LSQ.

> **Practical pitfall.** Quantization parameters live in a fundamentally different scale than weights. Without the LSQ `1/√(N·Qmax)` gradient scaling, the joint W + s optimization is unstable — `s` either flies off or barely moves. This single heuristic is what makes LSQ trainable in practice.

---

## 5. AdaRound — learn the per-weight rounding direction

The QAT methods above optimize the *parameterization* of the quantizer (clip, step, zero-point). AdaRound ([[adaround]], Nagel et al. 2020) optimizes the *rounding decision itself*: for each weight, learn whether to round up or round down, in a per-layer reconstruction objective that uses the local Hessian.

### Why nearest is suboptimal — the Taylor argument

For task loss `L` around the FP weights `W`:

```math
L(W + \Delta W) \,\approx\, L(W) + g^T \Delta W + \tfrac{1}{2} \Delta W^T H \Delta W
```

At a converged FP model `g ≈ 0`, so the loss is dominated by `ΔWᵀ H ΔW`. **Off-diagonal `H` means coupled weights — flipping one weight's rounding can cancel another's error.** Round-to-nearest minimizes `‖ΔW‖²` (assumes `H = I`); the true objective is the Hessian-weighted norm.

### The load-bearing layer-wise objective

```math
\boxed{\;\min_{\hat W} \; \mathbb{E}_X \, \| W \cdot X - \hat W \cdot X \|^2 \,=\, \min_{\Delta W} \Delta W^T (X X^T) \Delta W\;}
```

`H = X Xᵀ` is the **Gauss-Newton Hessian** for the squared-error reconstruction (one calibration-batch evaluation supplies it). This is the parent objective of **every later Hessian-aware PTQ method**: GPTQ solves the same thing with a sequential OBS update; OmniQuant learns equivalent transformations against the same loss; QuIP applies a random orthogonal rotation first.

### The relaxation (rectified sigmoid)

The discrete rounding direction `b_ij ∈ {0, 1}` is NP-hard to optimize directly. AdaRound parameterizes it as a continuous soft assignment annealed to `{0, 1}`:

```math
h(V) \,=\, \text{clip}\!\big(\sigma(V) \cdot (\zeta - \gamma) + \gamma, \, 0, \, 1\big), \quad \gamma = -0.1, \, \zeta = 1.1
```

The "stretched" sigmoid saturates *exactly* at `{0, 1}` so the final discrete decision is hard.

### Full objective + annealing

```math
\min_V \; \big\| W \cdot X - \Delta \cdot (\lfloor W/\Delta \rfloor + h(V)) \cdot X \big\|^2 \,+\, \lambda \cdot \sum (1 - |2 h(V) - 1|^\beta)
```

`β` is annealed from 20 → 2 across iterations to encourage exploration then commitment. ~10k Adam steps per layer, ~1024 calibration samples. After convergence, `b = round(h(V))` and freeze.

### Production impact

AdaRound delivered state-of-the-art PTQ at 4-bit on ResNet and InceptionV3 — matching QAT on most networks and beating it on MobileNet. **It is the direct spiritual parent of GPTQ** (ch-08), which solves the same Hessian-weighted layer-wise objective with a sequential, exact OBS-style update instead of soft relaxation. Every modern Hessian-aware LLM PTQ method runs against this objective.

---

## 6. BRECQ — extend the Hessian objective from layer to block

From [[brecq]] (Li et al. 2021), AdaRound's per-layer reconstruction is suboptimal below 4-bit because it ignores cross-layer error coupling. When layer `L_k` is quantized first, layer `L_{k+1}` sees a perturbed input `X' ≠ X`; per-layer AdaRound optimizes `L_{k+1}` against `X'` but ignores how its own `ΔW_{k+1}` amplifies the upstream perturbation.

**Fix:** reconstruct a whole **block** (a residual sub-graph, e.g. one ResNet block or one transformer layer) jointly:

```math
\min_{W_k} \; \mathbb{E}_X \, \| f_k(X) - \hat f_k(X; W_k) \|_F^2
```

where `‖·‖_F` is **Fisher-information-weighted** Frobenius norm:

```math
\| y \|_F^2 \,=\, \sum_i \text{diag}(F)_i \cdot y_i^2, \qquad \text{diag}(F) \,=\, \mathbb{E}\!\big[(\partial L / \partial y)^2\big]
```

(Gauss-Newton diagonal approximation to the task Hessian w.r.t. the block output.) Computed once during a forward+backward of the FP model on the calibration set.

### Why block is the right grain

The paper sweeps the granularity hierarchy `layer ⊂ block ⊂ stage ⊂ network` and shows block is the empirical sweet spot — small enough to optimize tractably, large enough to capture the dominant cross-layer dependencies. **First viable PTQ method at 2-bit for CNNs.** Mixed-precision extension: per-block sensitivity `Ω_k = trace(H_k) · ‖ΔW_k‖²` allocates higher bit-widths to blocks with larger `Ω_k` subject to a global memory budget — same philosophy as HAWQ, applied at BRECQ's granularity.

---

## 7. The lineage map and what carries forward to LLMs

```
Bengio 2013 STE
        │
        ▼
   BNN (Courbariaux 2016)        ── shadow real-valued buffer + sign() + STE
        │
        ▼
   XNOR-Net (Rastegari 2016)      ── per-filter scale  α = ‖W‖₁ / n   (→ AWQ instinct, ch-09)
        │
        ▼
   DoReFa (Zhou 2016)             ── arbitrary k-bit + SR-on-gradient  (→ FP8 / MXFP4 training, ch-17)
        │
        ▼
   PACT (Choi 2018)               ── learnable activation clip α
        │
        ▼
   LSQ (Esser 2020)               ── learnable step size s, ∂q/∂s closed form
        │
        ▼
   LSQ+ (Bhalgat 2020)            ── + learnable zero-point β
        │
        ╳ (PTQ branch)
        ▼
   AdaRound (Nagel 2020)          ── learn per-weight rounding direction; ‖WX − ŴX‖² objective
        │
        ▼
   BRECQ (Li 2021)                ── block-wise reconstruction with Fisher weights
        │
        ▼
   GPTQ (Frantar 2022, ch-08)     ── sequential OBS-style exact update on the same objective
                                     ── LLM-scale; the production W4A16 PTQ default
```

Three threads carry forward:

- **Per-channel / per-filter scaling absorbs quantization difficulty.** From XNOR-Net's per-filter `α` to [[awq]]'s per-channel activation scale to [[smoothquant]]'s migration of outliers from activations to weights. Same instinct, different math.
- **Stochastic rounding on gradient casts.** DoReFa's "gradient quantization with SR" is exactly the trick FP8 / MXFP4 / NVFP4 training revives at scale (ch-17).
- **Learn the quantizer parameters jointly with the model.** PACT → LSQ → LSQ+ → OmniQuant ([[omniquant]], ch-10): the modern LLM heir of LSQ adds equivalent-transformation parameters and learns them on a frozen model via block-wise reconstruction.

---

## 8. Cheat-sheet — pick the QAT/PTQ scaffold

```
Pretrain from scratch at low precision:
  Binary:        BNN / XNOR-Net (CNN era);  BitNet b1.58 (LLM, ch-16)
  k-bit:         DoReFa + per-layer LSQ;    LLM-QAT (ch-12)
  FP8 / MX:      Transformer Engine + SR on weight update (ch-17)

QAT fine-tune from a pretrained model:
  Activation clip: PACT  →  LSQ for symmetric weights  →  LSQ+ for non-ReLU activations
  Modern LLM:    OmniQuant block-wise gradient-based PTQ (ch-10)

PTQ from calibration data only:
  Per-layer Hessian reconstruction:   AdaRound (ch-04)
  Block-wise reconstruction:           BRECQ (ch-04)
  LLM-scale exact OBS update:          GPTQ (ch-08)
  Activation-aware per-channel scale:  AWQ (ch-09)
  Equivalent transformation:           SmoothQuant (ch-09)

Choosing between QAT and PTQ:
  Have a pretrained model + calibration data only:        PTQ (GPTQ / AWQ / AdaRound)
  Can afford a few epochs of fine-tune:                    QAT (LSQ / LSQ+ / OmniQuant)
  Building a model from scratch at exotic precision:       Pretrain QAT (BNN / BitNet / FP8-native)
```

---

## Connections and what's next

- **[[straight-through-estimator]] / every chapter from here on** — the gradient device every QAT method depends on.
- **[[adaround]] / ch-08** — the Hessian-weighted `‖WX − ŴX‖²` objective is exactly what GPTQ solves at LLM scale with a sequential OBS-style update.
- **[[brecq]] / ch-05 + ch-10** — block-wise reconstruction is the granularity that GPTVQ, OmniQuant, and EfficientQAT all inherit.
- **[[lsq]] / ch-10** — OmniQuant's "Learnable Equivalent Transformation" generalises LSQ-style learned-step to learned-rotation-and-clip on a frozen LLM.
- **[[xnor-net]] / ch-09** — per-filter scale `α = ‖W‖₁ / n` is the conceptual seed of AWQ's per-channel activation scaling.
- **[[dorefa-net]] / ch-17** — DoReFa's gradient-SR is the trick FP8 / MXFP4 / NVFP4 training revives at scale.
- **[[bnn]] / ch-16** — BitNet b1.58's BitLinear is the modern LLM revival of BNN, with ternary `{−1, 0, +1}` instead of binary.

## Further reading

- [[straight-through-estimator]] — Bengio, Léonard, Courville 2013.
- [[bnn]] — Courbariaux et al. 2016 (binary weights and activations).
- [[xnor-net]] — Rastegari et al. 2016 (per-filter scale).
- [[dorefa-net]] — Zhou et al. 2016 (arbitrary k-bit + gradient SR).
- [[pact]] — Choi et al. 2018 (learnable activation clip).
- [[lsq]] — Esser et al. 2020 (learnable step size).
- [[lsq-plus]] — Bhalgat et al. 2020 (asymmetric LSQ with learnable zero-point).
- [[adaround]] — Nagel et al. 2020 (per-weight learned rounding direction, Hessian objective).
- [[brecq]] — Li et al. 2021 (block-wise reconstruction, Fisher weighting).

## Companion visualization

**[figures/ste-backward.html](figures/ste-backward.html)** — interactive plot of `q(x) = round(x/Δ)·Δ` and its STE backward `∂q/∂x = 1[|x| ≤ α]`, with sliders for `Δ` and `α`. Hover any point to see forward value and gradient that flows through. *(Optional — skip on first read.)*
