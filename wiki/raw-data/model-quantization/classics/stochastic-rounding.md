<!-- scope: unbiased random rounding that preserves expectation; foundational for low-precision training
     deps: [[round-to-nearest-even]]
     see-also: [[uniform-quantization-noise]], [[fp8-e5m2]], [[mxfp-training]]
-->

# Stochastic Rounding for Low-Precision Training (Gupta et al., IBM 2015)
- **Core Insight:** Rounding x to the nearest representable value with probability proportional to nearness ("stochastic rounding") produces an unbiased estimator E[SR(x)] = x — preserving the expectation of accumulated quantities (gradients, weight updates) across many low-precision operations, where deterministic round-to-nearest would systematically lose small updates that don't reach the LSB.
- **Guideline:** Use stochastic rounding for the accumulator in low-precision training (FP16, BF16, FP8, MXFP) — especially for weight updates `W ← W − η·g` where η·g is often smaller than one ULP of W; SR keeps the *expected* update faithful, RNE silently zeroes it.
- **Authors:** Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, Pritish Narayanan (IBM Research)
- **Year:** 2015
- **URL:** https://arxiv.org/abs/1502.02551 ("Deep Learning with Limited Numerical Precision", ICML 2015)
- **Relevant topics:** stochastic rounding, low-precision training, unbiased quantizer, fixed-point training, FP8/MX training

## Abstract
Gupta et al. demonstrate that training deep networks in 16-bit fixed-point arithmetic *with stochastic rounding* matches FP32 accuracy on MNIST and CIFAR, while the same low-precision training with round-to-nearest-even (RNE) catastrophically diverges or stagnates. The mechanism: weight updates Δw = −η·g typically have magnitude well below one quantization step; RNE rounds these to zero (no update ever happens), but SR rounds to ⌈w⌉ or ⌊w⌋ probabilistically, so updates accumulate correctly *in expectation*. The paper makes stochastic rounding the standard recipe for sub-FP32 training and is cited by every subsequent FP8 / MX-format training study.

## Key Contributions
- Empirical demonstration that 16-bit fixed-point + SR matches 32-bit float on standard CV benchmarks.
- Establishes the "small-update vanishing" failure mode of RNE in low-precision training.
- Proves SR is unbiased: E[SR(x)] = x exactly.
- Provides a hardware-cheap implementation (single uniform random draw per round).
- Foundation cited by every FP8 / MXFP / NVFP4 training paper since 2022.

## Key Figures/Tables to Study
- **Training curves: RNE vs SR at 16-bit fixed-point** — RNE flatlines or diverges, SR tracks FP32. The single most-reproduced figure in the low-precision-training literature.
- **Update-magnitude histogram** showing that the bulk of Δw values are < 1 ULP of w → motivates SR conceptually.

## Technical Details

### Stochastic rounding definition
For real value x with adjacent representable values ⌊x⌋ and ⌈x⌉ (one step Δ apart, ⌊x⌋ ≤ x ≤ ⌈x⌉):
```
SR(x) = ⌈x⌉   with probability   p = (x − ⌊x⌋) / Δ
SR(x) = ⌊x⌋   with probability   1 − p = (⌈x⌉ − x) / Δ
```

### Unbiasedness
```
E[SR(x)] = p · ⌈x⌉ + (1−p) · ⌊x⌋
        = ((x − ⌊x⌋)/Δ) · (⌊x⌋ + Δ) + ((⌈x⌉ − x)/Δ) · ⌊x⌋
        = x                                                    ✓
```
Variance:
```
Var[SR(x)] = p(1−p) · Δ²
          ≤ Δ²/4
```
(maximum at p = 1/2). Compare deterministic RNE: variance 0 but bias up to Δ/2 in adversarial accumulation.

### Why RNE fails at low precision in training
Consider the weight-update step `w ← w + Δw` with |Δw| ≪ Δ_w (ULP of w):
- RNE: `RNE(w + Δw) = w` whenever |Δw| < Δ_w/2 (more than 99% of small SGD updates) → **no update is ever applied**; learning stalls.
- SR: `SR(w + Δw)` = w + Δ_w with prob |Δw|/Δ_w, else w. Over many minibatches, E[w] tracks the true update; learning proceeds.

### Hardware cost
One uniform random number r ∈ [0, Δ) per rounding operation. Modern accelerators (NVIDIA Hopper/Blackwell, Intel Gaudi, AMD MI300) provide SR as a single-instruction rounding mode.

### Where to apply SR in mixed-precision training
- **Weight master update** (the main use case): keep weights in FP32 master copy with SR cast to BF16/FP8 for forward — or, more aggressively, keep weights in FP8/MXFP and SR the update directly.
- **Gradient cast** before all-reduce (NVFP4 / MXFP4 training): SR preserves expectation across the reduction.
- **Activation cast** in forward: usually RNE (you want determinism within the forward; the noise hurts more than it helps for activations whose statistics matter immediately).

### Bias-variance tradeoff
- RNE: bias O(Δ), variance 0. Good for one-shot quantization (PTQ).
- SR: bias 0, variance O(Δ²). Good for *accumulated* low-precision arithmetic (training, long-running sums).
- For tensor reductions across N terms, SR error scales as O(Δ·√N), RNE error as O(Δ·N) in worst case.

### Modern relevance
- **FP8 training** ([[fp8-lm]], [[deepseek-v3-fp8]]): SR on weight update is standard.
- **MXFP4 / NVFP4 pretraining** ([[microscaling-formats]], [[nvfp4-training]]): SR on activations *during gradient cast* is the key trick making sub-8-bit training stable.
- **Cross-layer accumulation** in attention: SR on the QK^T accumulator at low precision.

## Connections
- [[round-to-nearest-even]] — the deterministic alternative; biased in adversarial accumulation but cheaper and deterministic.
- [[uniform-quantization-noise]] — SR variance Δ²/4 vs Bennett's Δ²/12; SR pays 3× variance for unbiasedness.
- [[fp8-e5m2]] / [[fp8-e4m3]] — typical low-precision targets where SR matters.
- [[mxfp-training]] / [[nvfp4-training]] — modern training recipes built on SR.
- [[fp8-lm]] / [[deepseek-v3-fp8]] — production FP8 training relies on SR for the master-weight update.
