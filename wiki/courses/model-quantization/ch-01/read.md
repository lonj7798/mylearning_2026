<!-- chapter: ch-01
     track: math-foundations
     title: Rate-Distortion + Math Foundations of Quantization
     sources: [[rate-distortion-theory]], [[lloyd-max-quantizer]], [[uniform-quantization-noise]], [[information-theoretic-bounds]], [[stochastic-rounding]]
     figures: figures/rate-distortion-curve.html
-->

# Chapter 1 — Rate-Distortion + Math Foundations of Quantization

> **Core insight.** Every quantizer is fighting one fixed enemy: Shannon's rate-distortion bound `R(D)`. For a Gaussian source under squared error, that bound is `D(R) = σ²·2^{−2R}` — six dB of MSE reduction per bit, no exceptions. The entire quantization literature, from Bennett's 1948 noise model through Lloyd-Max, Gish-Pierce, NF4, and NVFP4, is a series of progressively cleverer attempts to close the gap between an achievable scalar code and this floor.
>
> **Guideline.** Before you design a new quantizer, compute the rate-distortion benchmark `σ²·2^{−2N}` for your tensor's actual variance. Compare your achieved MSE to that floor — the *ratio* is the only meaningful efficiency metric. A scalar 1-D quantizer can never close more than 1.53 dB of the remaining gap; if you need to go further, you must either change the source distribution (rotate, smooth, clip) or move to vector quantization.

---

## Why this chapter exists

Quantization gets taught as a list of tricks — INT8 here, group-128 there, NF4 for Gaussians, stochastic rounding for FP4 training. The tricks make sense only against the theoretical backdrop they are approximating. Without the backdrop, every empirical decision (group size, clip percentile, symmetric vs asymmetric, FP vs INT) reduces to taste. With it, each one collapses to a single question: *how close does this design sit to Gish-Pierce optimal, and what is the structural reason for the residual gap?*

This chapter installs the four pieces of theory that every later chapter assumes:

1. **Shannon rate-distortion** `R(D)` — the unbeatable lower bound; Gaussian closed form `D = σ²·2^{−2R}`; "6 dB per bit."
2. **Bennett's uniform-noise model** `σ_q² = Δ²/12` — the high-resolution analytic prediction every INT/FP scalar quantizer obeys when its input is locally smooth.
3. **Gish-Pierce optimal density** `λ*(x) ∝ p(x)^{1/3}` — the asymptotic prescription for *non-uniform* scalar codes; the theoretical reason NF4, mu-law, and the entire FP family beat uniform INT on heavy-tailed data.
4. **Stochastic rounding** — the unbiased rounding rule that makes accumulated low-precision arithmetic (FP8 training, MXFP4 pretraining) converge where round-to-nearest-even silently drops every update below one ULP.

Three things you should walk away with:

1. Why "6 dB per bit" is universal and where the 1.53 dB scalar-quantizer space-filling penalty comes from.
2. When Bennett's `Δ²/12` is a reliable cost estimate and when it isn't (low resolution, outliers, clipping).
3. Why FP8 native training breaks at the *accumulator* without stochastic rounding even when individual round-trips look fine.

---

## 1. Shannon rate-distortion: the floor every quantizer fights

From [[rate-distortion-theory]], the rate-distortion function for a source `X ~ p(x)` under distortion measure `d(x, x̂)` is

```
R(D) = min_{p(x̂|x) : E[d(X,X̂)] ≤ D}  I(X; X̂)
```

In words: the minimum mutual-information rate needed to encode `X` so that average distortion is at most `D`. For lossless coding (`D = 0`) this reduces to `H(X)`. For lossy coding, `R(D)` is non-increasing and convex; the converse theorem says **no code, however clever, can achieve `(R, D)` strictly below this curve asymptotically.**

For the Gaussian source `X ~ N(0, σ²)` with squared-error distortion the closed form is famous:

```
R(D) = (1/2) log₂ (σ² / D),   0 ≤ D ≤ σ²
D(R) = σ² · 2^{−2R}
```

The second form is the one to memorize. **Every additional bit cuts MSE by a factor of 4** — equivalently 6.02 dB SNR per bit. This is the canonical benchmark. When you read a paper reporting "INT8 achieves 0.3% perplexity gap" you are implicitly asking: is the empirical SNR above or below `σ²·2^{−2·8}`?

### Reverse water-filling for parallel sources

For `N` independent Gaussian channels `X_i ~ N(0, σ_i²)`, the optimal distortion allocation gives

```
D_i = min(λ, σ_i²)
```

where the "water level" `λ` is chosen so `Σ D_i = D`. Channels with `σ_i² < λ` get no rate at all. **This is the theoretical justification for every per-channel, per-group, and per-token scale in modern LLM quantization** — bits flow to the high-variance directions; cheap directions get coarser quantizers or none.

> **Practical pitfall.** R(D) is achievable only in the long-block-length limit with optimal codes. A real scalar quantizer (Lloyd-Max, uniform INT, FP) gives up at least the **space-filling loss** — for Gaussian, 1.53 dB above R(D) at high rate. No 1-D quantizer beats that gap. Closing it requires vector quantization in dimension `d ≥ 2` (see [[vector-quantization]] in ch-03).

---

## 2. Bennett's uniform-noise model: `σ_q² = Δ²/12`

From [[uniform-quantization-noise]], if the input pdf `p(x)` is approximately flat over each width-Δ quantization cell (the **high-resolution assumption**), then the rounding error `e = Q(x) − x` is well-modelled as

- zero-mean,
- uniformly distributed on `(−Δ/2, +Δ/2]`,
- uncorrelated with the input.

The variance is one line of integration:

```
σ_q² = ∫_{−Δ/2}^{+Δ/2}  e² · (1/Δ) de  =  Δ² / 12
```

That single number is the foundation of every "6 dB per bit" analysis. For a `B`-bit uniform quantizer over `[−A, +A]`, `Δ = 2A / 2^B` and so

```
SNR(sinusoid) = 10 log₁₀ ( (A²/2) / (Δ²/12) ) = 6.02 · B + 1.76  dB
SNR(4σ-loaded Gaussian) ≈ 6.02 · B − 7.27  dB
```

The exponent `2^{−2B}` is *the same exponent* as Shannon's `R(D)`. The prefactor is suboptimal by a constant (the space-filling penalty); the slope is exactly right.

### Operational use during INT8 calibration

For a per-tensor symmetric INT8 quantizer with clip `α`, the per-element noise prediction is `Δ²/12 = α² / (3 · 255²)`. After calibration you measure actual per-tensor MSE. If

- `measured ≈ predicted` → Bennett holds, your quantizer is operating in its sweet spot.
- `measured ≫ predicted` → either clipping is dominant (widen `α`), or the input is not smooth at scale `Δ` (need per-channel / non-uniform code), or outliers blow up the tails.

This is the cheapest possible diagnostic. Run it before you reach for SmoothQuant.

### Three regimes where Bennett breaks

From [[uniform-quantization-noise]]:

1. **Low resolution (`B ≤ 3`).** pdf is not flat over `Δ`; error becomes signal-dependent; FP4 / 2-bit codes need non-uniform or block-scaled treatment.
2. **Periodic / low-amplitude inputs.** Error correlates with input → limit cycles, idle tones. Rare in LLMs but appears in clean audio DSP.
3. **Outlier-heavy distributions (the LLM activation case).** A handful of 100× channels dominate `σ_q²` through clipping, not through `Δ²/12`. This is exactly the SmoothQuant / AWQ / QuaRot motivation. We will return to it in ch-09 and ch-14.

---

## 3. Gish-Pierce: the optimal non-uniform code is `p(x)^{1/3}`

Uniform quantization is optimal only when `p(x)` is itself uniform. For any other distribution, Bennett's `Δ²/12` is leaving SNR on the table because some cells contain little probability mass and contribute nothing while others are overcrowded. The 1968 Gish-Pierce theorem ([[information-theoretic-bounds]]) makes this precise.

Define the **point density** `λ(x)`: the number of reconstruction levels per unit interval at `x`, normalized so `∫ λ = 1`. Bennett's high-rate distortion formula generalizes to

```
D(N) ≈ (1 / (12 N²)) · ∫ p(x) · λ(x)^{−2} dx
```

Minimizing this over `λ` subject to the normalization constraint (Hölder inequality) yields the **Gish-Pierce optimal density**:

```
λ*(x)  =  p(x)^{1/3} / ∫ p(u)^{1/3} du
```

Plugging back gives the optimal distortion in closed form:

```
D*(R)  =  (1/12) · ||p||_{1/3}³ · 2^{−2R}     with  ||p||_{1/3} = (∫ p^{1/3} dx)
```

Two consequences worth burning in:

- **For Gaussian `N(0, σ²)`**, the `L_{1/3}` norm evaluates to `||p||_{1/3}³ = σ² · π√3 ≈ 5.44 σ²`, giving `D* ≈ 0.453 σ² · 2^{−2R}`. The gap to Shannon's `σ² · 2^{−2R}` is exactly the **1.53 dB space-filling loss** that scalar 1-D quantization cannot close.
- **For uniform `U(−A, A)`**, `||p||_{1/3}³ = (2A)²` and `D* = A²/3 · 2^{−2R} = Δ²/12`. Bennett is recovered as the special case `p = const`.

### Why `p(x)^{1/3}` and not `p(x)`?

A common wrong intuition is "put more levels where there's more mass" — i.e. `λ ∝ p(x)`, the equiprobable code. This is *suboptimal*. The `1/3` exponent compromises between two pressures: more density reduces step `Δ_eff` (good for MSE locally) but spreads density too thin elsewhere (bad globally). The variational calculation balances them at `p^{1/3}`.

### Companding interpretation

A monotone compressor `F: ℝ → [0,1]` followed by uniform quantization is equivalent to non-uniform quantization with point density `λ(x) = F'(x)`. Setting `F'(x) ∝ p(x)^{1/3}` gives the **optimal compander**:

```
F*(x)  =  (∫_{−∞}^{x} p(u)^{1/3} du) / (∫ p(u)^{1/3} du)
```

This is the principled basis for [[companding-mu-law]] (an `F'(x) ∝ 1/(1+µ|x|)` approximation tuned for log-distributed audio) and for [[nf4]] (which simply tabulates `F*` for the unit-Gaussian weight prior; see ch-02 §5).

> **Practical pitfall.** Gish-Pierce optimality is asymptotic (high rate, `N → ∞`). At 4 bits (`N = 16`) the asymptotic argument is shaky; that's why NF4 uses 16-quantile spacing — a convenient surrogate — rather than the exact Lloyd-Max levels for `N(0,1)`. The two differ by ~0.05 dB; nobody has bothered to retune.

---

## 4. Lloyd-Max: optimal scalar codes at finite rate

From [[lloyd-max-quantizer]], the MSE-optimal `N`-level scalar quantizer for a given pdf `p(x)` is characterized by two coupled necessary conditions:

```
1.  Nearest-neighbour:  b_k = (y_k + y_{k+1}) / 2          (boundary = midpoint of adjacent levels)
2.  Centroid:           y_k = E[X | X ∈ (b_{k−1}, b_k]]    (level = conditional mean of cell)
```

These do not have a closed form for arbitrary `p`, but they admit a fixed-point iteration that converges to a local minimum (and to the *global* minimum for log-concave densities like Gaussian and Laplacian):

```
initialize y_1, ..., y_N (e.g. uniform quantile spacing)
repeat:
    b_k ← (y_k + y_{k+1}) / 2           for all k
    y_k ← E[X | X ∈ (b_{k−1}, b_k]]      for all k
until D stops decreasing
```

This is exactly **1-D `k`-means**. The vector generalization (LBG; see [[vector-quantization]] in ch-03) is `k`-means in `ℝ^d`. NF4's reconstruction values are a quantile-spaced approximation of this iteration applied to `N(0,1)` — Dettmers tabulated quantiles rather than running Lloyd because the difference is `< 0.1 dB` and the quantile form is easier to derive analytically.

---

## 5. Stochastic rounding: the only rounding that survives accumulation

Everything above is about static distortion: how badly does a single rounding step corrupt a tensor? Once you start training in low precision — where the same weight is incremented thousands of times per epoch — *bias* in the rounding rule matters far more than per-step variance. This is where [[stochastic-rounding]] enters.

### The rule

For a real value `x` between adjacent representable values `⌊x⌋` and `⌈x⌉` separated by `Δ`:

```
SR(x)  =  ⌈x⌉  with probability  p = (x − ⌊x⌋) / Δ
SR(x)  =  ⌊x⌋  with probability  1 − p = (⌈x⌉ − x) / Δ
```

### Unbiasedness in one line

```
E[SR(x)]  =  p · ⌈x⌉ + (1 − p) · ⌊x⌋
          =  ((x − ⌊x⌋)/Δ) · (⌊x⌋ + Δ) + ((⌈x⌉ − x)/Δ) · ⌊x⌋
          =  x                                                    ✓
```

Variance is `p(1 − p) · Δ² ≤ Δ²/4`. Compare round-to-nearest-even (RNE): variance zero, but bias up to `Δ/2` per step under adversarial accumulation.

### Why RNE silently fails low-precision training

Consider `w ← w + Δw` with `|Δw| ≪ Δ_w` (the ULP of `w` in the low-precision representation). Under RNE:

```
RNE(w + Δw)  =  w   whenever  |Δw| < Δ_w / 2
```

For 99%+ of typical SGD/Adam updates this condition holds → **no update ever happens; the loss curve flat-lines.** This is the [[stochastic-rounding]] paper's central empirical finding (Gupta et al. 2015): 16-bit fixed-point training with RNE catastrophically stagnates, while the same setup with SR matches FP32.

Under SR the *expected* increment is exactly `Δw` even when `|Δw| < Δ_w`; the noise it injects has variance `O(Δ²)` and averages out across many minibatches.

### Bias-variance tradeoff summary

| Mode | Bias | Variance | Use case |
|------|-----:|---------:|----------|
| RNE  | `O(Δ)` per step | 0 | One-shot PTQ; deterministic forward |
| SR   | 0 | `O(Δ²)` per step | Accumulated low-precision arithmetic |

For tensor reductions across `N` terms: RNE error scales as `O(Δ · N)` worst case, SR error as `O(Δ · √N)`. The square-root saving is the entire reason FP8/MXFP4/NVFP4 native training works.

### Where SR shows up in modern training

- **FP8 master-weight update** ([[fp8-e5m2]], DeepSeek V3 FP8): SR on the cast from FP32 master to FP8/BF16 storage; preserves the expectation of `η · g`.
- **MXFP4 / NVFP4 pretraining** ([[nvfp4]], MXFP-training literature): SR on activations during the *gradient cast* — the single load-bearing trick making sub-8-bit pretraining stable on Blackwell.
- **Reduction trees** in attention `QKᵀ` accumulators at low precision.

By contrast, *forward* activations are usually rounded with RNE — you want determinism within a forward pass for reproducibility, and per-forward noise hurts more than it helps. We'll see this asymmetry again in ch-17 when we dissect the NVFP4-QAD training recipe.

> **Practical pitfall.** Stochastic rounding needs one uniform random sample per round. On Hopper/Blackwell this is a single instruction; on older accelerators it requires a software PRNG, which is expensive enough to dominate the kernel. Check your hardware before specifying SR in a training recipe.

---

## 6. Putting the bounds onto LLM tensors

What does this theory predict for the tensors you actually quantize?

**LLM weights post-normalization** are approximately `N(0, σ²)` with `σ ≈ 0.02` for typical attention projections (see any open weight checkpoint). Per-block absmax normalization rescales to roughly `N(0, c²)` with `c ≈ 0.3`. For a 4-bit quantizer:

- Shannon floor: `σ² · 2^{−8} ≈ 3.5 · 10^{−3} · c²`
- Gish-Pierce optimal scalar: `0.453 · σ² · 2^{−8}` (1.53 dB above Shannon)
- Bennett uniform INT4: prefactor `≈ 1.0 · σ² · 2^{−8}` (about 3.5 dB worse than Gish-Pierce, because the Gaussian tails are starved)
- **This is why NF4 beats INT4 by ~0.5 PPL** on Llama-class models at 4-bit — it implements `p^{1/3}` density, capturing the Gish-Pierce optimum.

**LLM activations** are *not* Gaussian. Post-GELU/SiLU activations are heavy-tailed positive; residual-stream activations have outlier channels with 100× the bulk magnitude. Bennett's uniform-noise model fails immediately — the `Δ²/12` prediction is off by orders of magnitude because clipping dominates. This is exactly why activation quantization needs rotation ([[quarot]], ch-14), smoothing ([[smoothquant]], ch-09), or per-channel / per-token granularity.

**Attention KV cache** has a structural asymmetry: K has channel-aligned RoPE-induced outliers; V is roughly Gaussian per token. Rate-distortion / reverse-water-filling logic predicts they need different bit allocations — exactly what KIVI / KVQuant exploit at the production level (ch-15).

---

## 7. Cheat-sheet

```
Shannon (Gaussian, MSE):     D = σ² · 2^{−2R}                 # absolute floor
Bennett (uniform quantizer): σ_q² = Δ²/12                     # high-resolution MSE
SNR per bit (uniform):       6.02 B + 1.76 dB (sinusoid)
                             6.02 B − 7.27 dB (4σ Gaussian)
Gish-Pierce (optimal scalar): D = (1/12) · ||p||_{1/3}³ · 2^{−2R}
                              ≈ 0.453 σ² · 2^{−2R}  for Gaussian (= 1.53 dB above R(D))
Optimal point density:       λ*(x) ∝ p(x)^{1/3}
Optimal compressor:          F'(x) ∝ p(x)^{1/3}

Stochastic rounding:         SR(x) = ⌈x⌉ w.p. (x−⌊x⌋)/Δ ; ⌊x⌋ otherwise
                             E[SR(x)] = x exactly
                             Var[SR(x)] ≤ Δ²/4
                             use whenever accumulating many low-precision values
```

---

## Connections and what's next

- **[[uniform-quantization-noise]] / ch-02** — Bennett's `Δ²/12` becomes the per-element MSE prediction for every INT/FP format in the reference table.
- **[[information-theoretic-bounds]] / ch-02 §5** — `p(x)^{1/3}` is the principle behind NF4's quantile spacing.
- **[[lloyd-max-quantizer]] / ch-03** — the centroid/NN iteration becomes `k`-means and LBG vector quantization.
- **[[stochastic-rounding]] / ch-17** — SR on the gradient cast is the load-bearing trick of NVFP4 pretraining.
- **[[rate-distortion-theory]] / ch-18** — TurboQuant's data-oblivious KV claims "rate-distortion bound up to constant"; you need this chapter to evaluate that claim.

## Further reading

- [[rate-distortion-theory]] — Shannon 1948/1959; Cover & Thomas Ch. 10.
- [[uniform-quantization-noise]] — Bennett 1948, the original analytical treatment.
- [[information-theoretic-bounds]] — Gish & Pierce 1968 high-rate optimal density.
- [[lloyd-max-quantizer]] — Lloyd 1957/1982; Max 1960.
- [[stochastic-rounding]] — Gupta et al. 2015 IBM, the deep-learning-era SR foundation.

## Companion visualization

**[figures/rate-distortion-curve.html](figures/rate-distortion-curve.html)** — interactive plot of `D(R) = σ²·2^{−2R}` for the Gaussian source, overlaid with achievable `(R, D)` points for INT4, NF4, INT8, FP8 E4M3, and FP16. Use it to internalize how each format sits relative to the Shannon floor and the 1.53 dB scalar-quantizer ceiling. *(Optional — skip on first read.)*
