<!-- chapter: ch-03
     track: math-foundations
     title: Quantizer Design — Uniform vs Non-Uniform + Calibration Objectives
     sources: [[lloyd-max-quantizer]], [[vector-quantization]], [[product-quantization]], [[companding-mu-law]], [[mse-vs-kl-calibration]], [[percentile-clipping]], [[quantization-error-propagation]]
-->

# Chapter 3 — Quantizer Design: Uniform vs Non-Uniform + Calibration Objectives

> **Core insight.** Once you pick a format (ch-02), three independent design decisions remain: (1) **shape** — uniform, non-uniform scalar (Lloyd-Max / companding), or vector quantizer (LBG / PQ); (2) **calibration objective** — what metric you minimize when fitting the scale (MSE, KL, cosine); (3) **clipping rule** — how you handle the outlier tail (min/max, percentile, learned clip). Each axis can move independently; bad combinations dominate the catastrophic-failure literature.
>
> **Guideline.** Default to **per-channel symmetric MSE-calibrated uniform** for weights, **per-tensor (or per-token) percentile-99.9 + KL-calibrated** for activations, and **vector quantization only when you need sub-2-bit**. Override these defaults only with measured loss.

---

## Why this chapter exists

The previous chapter showed you what bit-pattern each format puts on disk. This chapter shows you the three orthogonal decisions that determine whether the bit-pattern is any good for your tensor. You can pick the right format and still get garbage if you fit it with the wrong calibration objective — a per-tensor MSE-min/max INT8 calibration on an LLM residual stream destroys 6+ bits of effective precision before any quantization arithmetic even runs.

Three things you should walk away with:

1. The Lloyd-Max iteration and its vector generalization (LBG / `k`-means / product quantization) — the underpinning of every non-uniform code from NF4 to AQLM.
2. The MSE / KL / cosine calibration tree — which objective matches which downstream op, and why TensorRT defaults to KL for activations into Softmax.
3. The error-propagation calculus that determines per-block bit allocation (BRECQ, HAWQ) and explains the structural failure mode of long-context LLM quantization.

---

## 1. Lloyd-Max revisited — and the LBG generalization

Recall from ch-01 §4 the two necessary conditions for an MSE-optimal `N`-level scalar quantizer:

```
1.  Nearest-neighbour:  b_k = (y_k + y_{k+1}) / 2
2.  Centroid:           y_k = E[X | X ∈ (b_{k-1}, b_k]]
```

The iteration that alternates them is the **scalar Lloyd iteration**. The vector generalization to `ℝ^d` — Linde-Buzo-Gray (LBG, 1980; [[vector-quantization]]) — replaces "decision boundary" with "Voronoi cell" and "conditional mean" with "centroid in `ℝ^d`":

```
S_k = { x ∈ ℝ^d : ||x − c_k|| ≤ ||x − c_j||  for all j }      # Voronoi cell
c_k = (1 / |S_k ∩ X|)  Σ_{x_t ∈ S_k}  x_t                      # cell centroid
```

This is exactly **`k`-means**. The LBG splitting initialization (grow `1 → 2 → 4 → … → N` by perturbing each centroid by `±ε` and re-Lloyd-ing) is the trick that avoids the bad local minima of random `k`-means init.

### Why VQ matters: closing the space-filling gap

Recall the 1.53 dB scalar penalty (ch-01 §3): any 1-D scalar code lies at least 1.53 dB above Shannon's `R(D)` for the Gaussian. Vector quantization in dimension `d` shrinks this gap. The dimension-`d` "cell shape constant" `G_d` evolves as

```
G_1 ≈ 2.72  (1.53 dB)     # scalar Lloyd-Max
G_2 ≈ 1.16  (0.65 dB)
G_4 ≈ 1.07  (0.30 dB)
G_∞ = 1     (0 dB; Shannon bound)
```

This is the structural reason additive vector quantization (AQLM, ch-14) achieves sub-2-bit at quality where INT/FP4 collapse — going from `d = 1` to `d = 8` recovers ~1.3 dB, which at 2-bit is the difference between coherent text and word salad.

> **Practical pitfall.** Brute-force VQ has encoding cost `O(N · d)` per vector. At sub-2-bit on a 70B model (`N = 2^{12}`, `d = 8`, `~10¹⁰` weights) this is infeasible. Tree-structured VQ, residual VQ, and **product quantization** (next section) trade some distortion for log-or-linear cost.

---

## 2. Product quantization — making VQ scale

From [[product-quantization]] (Jégou, Douze, Schmid 2011), PQ decomposes a `D`-dim vector into `M` disjoint sub-vectors of length `D/M`, each quantized independently with a `K`-entry codebook learned via `k`-means:

```
x = [u_1 ; u_2 ; … ; u_M],      u_m ∈ ℝ^{D/M}
code_m(u_m) = argmin_k ||u_m − c_{m,k}||²
PQ(x) = [c_{1,code_1(u_1)} ; … ; c_{M,code_M(u_M)}]
```

The **effective codebook size is `K^M`** — astronomically large — while storage is only `M · K · (D/M) = K · D` floats, and per-vector code is `M · log₂ K` bits. With `M = 16, K = 256`, a vector becomes 16 bytes regardless of `D`.

### Asymmetric Distance Computation (ADC)

For query `q = [q_1; …; q_M]`, precompute `M` lookup tables of size `K`:

```
T_m[k] = ||q_m − c_{m,k}||²        for m = 1…M, k = 1…K
```

Then for any database vector with PQ codes `(k_1, …, k_M)`:

```
||q − PQ(x)||²  ≈  Σ_m T_m[k_m]      # M lookups + M−1 adds
```

This is the ANN trick that made FAISS billion-scale. The same mechanism transferred directly to LLM quantization: **AQLM (ch-14) is essentially PQ + Hessian-aware codebook learning + sequential GPTQ-style update**.

### Optimized PQ → rotation preview

A learned rotation `R` applied *before* splitting (`x → R x`) balances variance across sub-vectors and substantially improves distortion — the **direct ancestor of rotation-based LLM quant** ([[quarot]], [[spinquant]], ch-13/14). The lineage: OPQ (2014) → QuIP (2023) → QuaRot (2024) → SpinQuant (2024).

---

## 3. Companding — the original "log-scale" non-uniform code

From [[companding-mu-law]], a companding quantizer applies a non-linear compressor `F` to the input, uniformly quantizes the compressed signal, then applies `F⁻¹` to decode. The recipe is

```
y = F(x) → uniform B-bit quantize → ŷ → x̂ = F⁻¹(ŷ)
```

Bell Labs' µ-law (Smith 1957; standard `µ = 255`) is the canonical example:

```math
F_\mu(x) \,=\, \text{sign}(x) \cdot \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}
```

The decoder `F_µ⁻¹(y) = sign(y) · (1/µ) · ((1 + µ)^{|y|} − 1)`. The effective quantization step in the original domain is

```
Δ_eff(x)  ≈  Δ / F'(x)  ∝  (1/µ + |x|)        for µ-law
```

i.e. step *grows linearly with `|x|`*, so the **relative error `|e/x|` is approximately constant** across the dynamic range — instead of constant *absolute* error as in uniform quantization. For 8-bit telephony, µ-law achieves SNR ≈ 38 dB over a 30 dB dynamic range, vs uniform 8-bit which gives 38 dB only at full scale and loses 6 dB per amplitude halving.

### Floating-point as companding

A floating-point number `x = (−1)^s · 1.m · 2^e` is exactly a piecewise-uniform quantizer on the log axis: within each exponent bin `[2^e, 2^{e+1})`, the `2^M` mantissa values are uniformly spaced; across bins the step doubles. **FP = piecewise-linear approximation to logarithmic companding.** FP4 / FP8 are low-resolution companders.

NF4 (ch-02 §6) is something stronger — the *Lloyd-Max-optimal compander tuned to the Gaussian source*, not a fixed log curve. The Gish-Pierce condition `F'(x) ∝ p(x)^{1/3}` (ch-01 §3) prescribes the optimum; NF4 tabulates it for `N(0,1)`; µ-law approximates it for log-distributed audio.

---

## 4. Calibration objectives — MSE vs KL vs cosine

From [[mse-vs-kl-calibration]], **no single calibration loss is universally best**. The three canonical options match three different downstream-propagation assumptions:

### MSE (the Bennett-justified default)

```
S* = argmin_S  Σ_n (x_n − S · round(clamp(x_n / S, Qmin, Qmax)))²
```

Under the high-rate uniform-noise model, quantization error has variance `Δ²/12 = S²/12`. Minimizing MSE is the maximum-likelihood scale assuming Gaussian post-noise output. **Use for weight calibration** (per-channel for conv / linear).

### KL divergence (the TensorRT activation default)

Bin the FP tensor into `M` bins; bin the quantized tensor into the corresponding levels; minimize

```
KL(P_fp ‖ P_quant)  =  Σ_i  P_fp(i) · log [ P_fp(i) / P_quant(i) ]
```

over candidate clip ranges. Sweep procedure: for each candidate clip `T_k` (typically log-spaced over `[128, 8192]` bins), quantize using `S = T_k / Q_max` and compute KL. Pick the `T_k*` minimizing KL.

**Use when the tensor feeds a Softmax** (attention scores, classifier logits) — KL is the divergence that matters when the next layer turns this into a probability distribution.

### Cosine similarity

```
cos(x, Q(x))  =  ⟨x, Q(x)⟩ / (‖x‖ · ‖Q(x)‖)
```

Maximized when `Q` preserves direction. **Use when only the angle matters** (e.g. attention scores where only the ranking survives Softmax). *Pathological* for tensors whose absolute magnitude propagates (residual stream) — cosine is happy with a globally rescaled output, but the residual sum requires absolute scale.

### Decision tree

| Tensor type | Recommended objective |
|-------------|----------------------|
| Weights (conv / linear) | **per-channel MSE** |
| Activation feeding Softmax | **KL** |
| Residual-stream activations with heavy outliers | **percentile-clipping + MSE** |
| Attention scores where only ranking matters | cosine |
| Post-Swish / post-GeLU heavy-tailed | per-token + percentile |

> **Practical pitfall.** Never use raw min/max alone for heavy-tailed activations. It is the degenerate `T = 100%` case of percentile clipping and routinely sacrifices 5+ bits of effective precision.

---

## 5. Percentile clipping — the universal outlier preprocessor

From [[percentile-clipping]], a single outlier can dilate the per-tensor scale enough to quantize the entire bulk to noise. Suppose 99% of activations lie in `[0, 10]` and 1% in `[10, 1000]`:

- **Min/max scale**: `S = 1000 / 255 = 3.92` → bulk `[0, 10]` maps to integers `[0, 2.5]` → only ~3 quantization levels for 99 % of the data. **Effective resolution: 1.6 bits.**
- **99% percentile clip**: clip at 10, `S = 10 / 255 = 0.039` → bulk uses full `[0, 255]`. **Effective resolution: 8 bits.** Tail saturates to 255.

The 1% saturation loss is far smaller than the 6.4-bit resolution gain on the bulk.

### Standard recipe

```
T = quantile_99.9(|x|)
x' = clip(x, −T, +T)
Then  S = T / Qmax,  Z = 0  (symmetric)  or  S = (max(x') − min(x')) / (Qmax − Qmin)  (asymmetric)
```

Sweep `p ∈ {99%, 99.5%, 99.9%, 99.99%, 100%}` and pick the one minimizing downstream KL or task accuracy. NVIDIA's TensorRT does this automatically; the 99.9% default is the empirical sweet spot for most CNN and transformer activations.

### When percentile clipping isn't enough

- **LLM residual stream at 6.7B+** (`outliers > 1000×`): even 99.99 % clip wastes most bits. → Structural fix: migrate the outlier load to weights ([[smoothquant]], ch-09), or split into a separate FP16 path ([[llm-int8]], ch-07).
- **Multimodal distribution** (post-Swish / GELU): percentile on `|x|` collapses the modes. Use per-side percentile (separate `+/−` clips; LSQ+ in ch-04).

---

## 6. Error propagation across a transformer block

From [[quantization-error-propagation]], the per-layer quantization noise variance under Bennett is

```
σ_ℓ²  =  Δ_ℓ² / 12  ·  ‖x_ℓ‖²
```

For a linear layer `y = W·x` with `x_quant = x + e_x, W_quant = W + e_W`, the dominant noise terms (ignoring second-order `e_W · e_x`):

```
Var(y)  ≈  ‖x‖² · σ_W²  +  ‖W‖_F² · σ_x²
```

### Residual stream — additive accumulator

A transformer block:

```
x_{ℓ+1}  =  x_ℓ + Attn(LN(x_ℓ)) + FFN(LN(x_ℓ + Attn(...)))
```

Noise injected at block `ℓ` accumulates additively into `x_{ℓ+L}` for all `L > 0`. After `D` blocks:

```
Var(residual noise)  ≈  Σ_{ℓ=1}^D  σ_{ℓ_block}²
```

Independent of layer depth — pure additive sum. **This is why mid-stack blocks dominate quantization sensitivity**: their noise persists through every subsequent block.

### LayerNorm contraction — and the outlier asymmetry

```
LN(x)  =  γ · (x − µ) / √(Var(x) + ε) + β
```

Noise added to `x` is rescaled by `1/√Var(x)`. If `x` has outlier channels with large variance (the LLM.int8 regime), the *bulk* noise is suppressed but the *outlier-channel* noise is amplified (because their pre-LN magnitudes dominate `Var`). This creates the per-channel noise asymmetry that motivates per-channel weight scaling ([[awq]], [[smoothquant]], ch-09).

### Attention compounding

`Softmax(QKᵀ / √d)` amplifies noise quadratically in the softmax temperature regime: a small noise `δ` on `QKᵀ` becomes `~e^δ` in attention probabilities. **This is why attention-input scale matters disproportionately in PTQ** — and why KL calibration of the attention input is non-negotiable.

### Mixed-precision implication

Layers with high noise amplification factor `Π_{k>ℓ} ‖W_k‖_F²` should get more bits. This is the HAWQ Hessian-trace criterion (ch-05); the Hessian encodes loss-sensitivity, which is the gradient of noise amplification.

Empirical fingerprint:

- Embedding and output layers: low amplification → 4-bit safe.
- **Mid-stack residual MLP: high amplification → ≥ 6-bit conservative.**
- Final-block attention: amplifies into the head → per-channel + percentile required.

---

## 7. The decision sheet

```
Shape:
  Scalar uniform (INT)       →  weight default; activation w/ percentile
  Scalar non-uniform (NF/FP) →  weight prior is Gaussian-ish, low-bit
  Vector (LBG / PQ / AQ)     →  sub-2-bit, accept encoding cost
  Companding (FP / mu-law)   →  log-distributed sources

Calibration objective:
  MSE        →  weight (per-channel)
  KL         →  activation feeding Softmax / classifier
  Cosine     →  attention scores where only ranking matters
  Min/max    →  never alone; degenerate case of T=100% clip

Clip preprocessor:
  Percentile (99.9% default) →  always for activations
  Per-side percentile         →  multimodal (post-GeLU)
  Learned clip (PACT / LSQ+) →  QAT setting, ch-04

Granularity (per Shannon reverse water-filling):
  per-tensor          →  weight base case
  per-channel         →  weight default; +1% at no cost
  per-group-128       →  sub-INT8 weight default
  per-token           →  activation under per-token scale (LLM.int8 / SmoothQuant)
  per-block (32 / 16) →  MX / NVFP4 sub-8-bit
```

---

## Connections and what's next

- **[[lloyd-max-quantizer]] / ch-01 §4** — scalar Lloyd-Max, parent of LBG and NF4.
- **[[vector-quantization]] / ch-14** — AQLM, VPTQ, GPTVQ are direct LBG / PQ descendants for sub-2-bit LLM weights.
- **[[product-quantization]] / ch-13–14** — the OPQ-style pre-rotation is the seed of QuIP / QuaRot / SpinQuant.
- **[[companding-mu-law]] / ch-02 §6** — FP formats are discrete companders; NF4 is the Lloyd-Max compander for Gaussians.
- **[[mse-vs-kl-calibration]] / ch-05** — calibration objectives are the search space PTQ surveys (Krishnamoorthi 2018, BRECQ, OmniQuant) optimize over.
- **[[percentile-clipping]] / ch-07** — extreme outliers at 6.7B+ break percentile clipping; structural fix via SmoothQuant / LLM.int8.
- **[[quantization-error-propagation]] / ch-05** — feeds directly into BRECQ block-wise reconstruction and HAWQ mixed-precision bit allocation.

## Further reading

- [[lloyd-max-quantizer]] — Lloyd 1957/1982; Max 1960.
- [[vector-quantization]] — Linde, Buzo, Gray 1980 (LBG, the `k`-means foundation).
- [[product-quantization]] — Jégou, Douze, Schmid 2011 (PQ → FAISS).
- [[companding-mu-law]] — Bernard Smith 1957; ITU-T G.711.
- [[mse-vs-kl-calibration]] — NVIDIA TensorRT 2017; Krishnamoorthi 2018.
- [[percentile-clipping]] — Migacz 2017 GTC; Wu 2020 integer-quant survey.
- [[quantization-error-propagation]] — Wu 2020; Park 2023 transformer-error consolidation.
