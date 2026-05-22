---
chapter: ch-01
course: model-quantization
phase: read
excerpt_of: "Deep Learning with Limited Numerical Precision (Gupta, Agrawal, Gopalakrishnan, Narayanan 2015)"
source_url: https://arxiv.org/abs/1502.02551
created_at: "2026-05-21"
raw_data_source: [[raw-data/stochastic-rounding]]
---

# Excerpt: Stochastic rounding — unbiased rounding that survives accumulation

**Authors:** Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, Pritish Narayanan (IBM Research).
**Year:** 2015.
**Venue:** ICML 2015.
**URL / arXiv ID:** see source_url / 1502.02551.

---

## The one-box rule

For real value `x` between adjacent representable values `⌊x⌋` and `⌈x⌉` separated by step `Δ`:

```math
\boxed{\;\text{SR}(x) =
\begin{cases}
\lceil x \rceil & \text{w.p. } p = (x - \lfloor x \rfloor) / \Delta \\
\lfloor x \rfloor & \text{w.p. } 1 - p
\end{cases}\;}
```

---

## Unbiasedness — one line

```math
\mathbb{E}[\text{SR}(x)] = p \cdot \lceil x \rceil + (1-p) \cdot \lfloor x \rfloor
        = \tfrac{(x - \lfloor x \rfloor)}{\Delta} (\lfloor x \rfloor + \Delta) + \tfrac{(\lceil x \rceil - x)}{\Delta} \lfloor x \rfloor
        = x \quad\checkmark
```

Variance: `Var[SR(x)] = p(1 − p) · Δ² ≤ Δ²/4`, maximum at `p = 1/2`. Compare RNE: bias up to `Δ/2` under adversarial accumulation, variance zero.

---

## Why RNE fails low-precision *training*

Consider the weight-update step `w ← w + Δw` with `|Δw| ≪ Δ_w` (ULP of `w` in the low-precision representation):

- **RNE:** `RNE(w + Δw) = w` whenever `|Δw| < Δ_w / 2`. For 99 %+ of typical SGD/Adam updates this holds → **no update is ever applied; learning stalls.**
- **SR:** `SR(w + Δw) = w + Δ_w` with probability `|Δw|/Δ_w`, else `w`. Over many minibatches, `E[w]` tracks the true update; learning proceeds.

The original 1502.02551 paper proves this empirically: 16-bit fixed-point training with RNE catastrophically diverges on MNIST/CIFAR; the same training with SR matches FP32.

---

## Bias-variance tradeoff

| Mode | Bias | Variance | Use case |
|------|-----:|---------:|----------|
| RNE  | `O(Δ)` per step | 0 | One-shot PTQ, deterministic forward |
| SR   | 0 | `O(Δ²)` per step | Accumulated low-precision arithmetic (training, reductions) |

For tensor reduction across `N` terms, RNE worst-case error scales as `O(Δ · N)` while SR error scales as `O(Δ · √N)`. The square-root factor is the entire reason FP8 / MXFP4 / NVFP4 training works.

---

## Modern usage map

- **FP8 master-weight update** ([[fp8-e5m2]], [[deepseek-v3-fp8]]): SR on cast from FP32 master → FP8/BF16 storage.
- **MXFP4 / NVFP4 pretraining** ([[mx-formats]], [[nvfp4-training]]): SR on activations during the *gradient cast*. The single load-bearing trick that makes sub-8-bit pretraining stable on Blackwell.
- **QK accumulator** in attention at low precision.

Forward activations are usually RNE — determinism within a forward pass matters for reproducibility, and per-forward noise hurts more than it helps when the same value is consumed once. The asymmetry "SR on gradient cast, RNE on forward" recurs across every low-precision training paper.

---

## Hardware cost

One uniform random draw `r ∈ [0, Δ)` per round. NVIDIA Hopper / Blackwell, Intel Gaudi, AMD MI300 provide SR as a single-instruction rounding mode. Older accelerators require a software PRNG, which can dominate the kernel — verify hardware support before specifying SR in a training recipe.

---

## Connections

- [[excerpts/uniform-quantization-noise]] — Bennett `Δ²/12` (RNE) vs SR `Δ²/4`; the 3× variance is the price of unbiasedness.
- [[fp8-e5m2]] / [[fp8-e4m3]] — typical low-precision targets where SR matters in training.
- [[mxfp-training]] / [[nvfp4-training]] — modern training recipes built on SR.
- [[ch-01]] — parent synthesis.
