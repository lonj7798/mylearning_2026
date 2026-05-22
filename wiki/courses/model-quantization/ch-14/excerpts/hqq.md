---
chapter: ch-14
course: model-quantization
phase: read
excerpt_of: "Half-Quadratic Quantization (HQQ)"
source_url: https://mobiusml.github.io/hqq_blog/
created_at: "2026-05-21"
---

# Excerpt: HQQ — half-quadratic splitting, data-free, fast

**Authors:** Hicham Badri, Appu Shaji (Mobius Labs)
**Year:** 2024
**URL:** https://mobiusml.github.io/hqq_blog/ ; https://github.com/mobiusml/hqq
**Raw-data source:** [[raw-data/hqq]]

---

## What HQQ solves

GPTQ and AWQ need calibration data (~128 sequences). AQLM needs ~24 h of calibration. For rapid iteration, edge deployment, fine-tune-then-quantize pipelines, and any case where you don't have or trust calibration data, **HQQ is the "iterate quickly" tool**: data-free, no STE, no gradient descent, no calibration corpus. Just closed-form alternating updates.

**Wall clock**: <10 minutes for LLaMA-2-70B at 4-bit on a single A100.

---

## The quantization rule (asymmetric, per-group)

Per-group (group_size typically 64):

```
W_q = round((W / s) + z),    Ŵ = s · (W_q - z)
```

- s ∈ ℝ: per-group scale.
- z ∈ ℝ: per-group zero-point.
- W_q ∈ {0, ..., 2^b - 1}: integer codes.

The unknowns are `(s, z, W_q)`. The challenge: solve for them jointly without calibration data.

---

## The lp objective (robust regression)

```math
\min_{s, z, W_q} \|W - s(W_q - z)\|_p^p
```

with `p ∈ (0, 1)` (typically 0.5 or 0.7).

**Why p < 1?** It down-weights outlier residuals. Equivalent to assuming a heavy-tailed (generalised Gaussian) prior on quantization error, which matches the real LLM weight distribution (heavy tails). L2 (p=2) would weight outliers ~50× too much; L1 (p=1) is closer; p<1 explicitly handles them as "expected".

```
L2:  large residuals weighted r²       (over-emphasises tail)
L1:  large residuals weighted |r|      (handles tail)
Lp:  large residuals weighted |r|^p    (down-weights tail when p<1)
```

---

## Half-quadratic splitting

Introduce auxiliary variable `W_e = W - s(W_q - z)`:

```math
\min_{W_e, s, z, W_q} \|W_e\|_p^p + \frac{\beta}{2} \|W - s(W_q - z) - W_e\|^2
```

β is a penalty parameter increased across iterations (continuation strategy). As `β → ∞`, the constraint `W_e = W - s(W_q - z)` is enforced exactly and the auxiliary problem matches the original.

**Alternating updates:**

```
1. W_e step (fix z, s, W_q):   closed-form proximal operator of lp.
   W_e ← prox_{||·||_p^p / β}(W - s(W_q - z))
   For p=0.5: half-shrinkage operator (closed form).
   For p=2/3: cubic-root closed form.

2. z step (fix W_e, s):
   z ← median_group(W_q - (W - W_e) / s)
   Median is the L1 proximal operator — the natural counterpart to the robust-loss prior.

3. (optional) s step: closed-form LS update for s.

4. Refresh W_q = round((W - W_e) / s + z); increase β; repeat.
```

**Convergence**: 4–8 iterations per group. Each step is elementwise / per-group, fully vectorizable. Per-layer cost: milliseconds per million weights.

---

## Why data-free

The objective minimises **raw weight reconstruction**; no input distribution X is needed. Equivalent to assuming `H = I` — gives up the Hessian weighting that GPTQ exploits, but the lp robustness mostly compensates by handling outliers explicitly.

In practice: HQQ at 4-bit ≈ GPTQ at 4-bit on standard benchmarks, **without calibration data**. At 2-bit HQQ trails AQLM by ~3 ppl (Hessian information matters more at extreme low-bit) but beats RTN by ~8 ppl.

---

## The numbers

LLaMA-2-7B WikiText-2:

| Method | Bits | ppl | Calib data? | Wall clock |
|--------|------|-----|-------------|------------|
| FP16 | 16 | 5.47 | — | — |
| RTN | 4 | 5.73 | no | seconds |
| GPTQ | 4 | 5.66 | yes (128 seqs) | minutes |
| AWQ | 4 | 5.62 | yes (128 seqs) | minutes |
| **HQQ** | **4** | **5.65** | **no** | **<1 min** |
| RTN | 3 | 7.20 | no | seconds |
| GPTQ | 3 | 6.18 | yes | minutes |
| **HQQ** | **3** | **6.30** | **no** | **<1 min** |
| RTN | 2 | NaN | no | — |
| GPTQ | 2 | ≥ 12 | yes | minutes |
| **HQQ** | **2** | **8.91** | **no** | **<1 min** |

HQQ-4 matches GPTQ-4 at zero calibration cost. HQQ-3 is competitive. HQQ-2 is much better than RTN but not as good as AQLM.

---

## Bit-widths supported

8, 4, 3, 2, 1 bits. 1-bit requires accepting larger ppl hit because no fine-tune.

---

## Hyperparameters

| Knob | Default |
|------|---------|
| Group size | 64 |
| p (lp norm) | 0.7 (safe), 0.5 (more outlier-robust) |
| β init | 1.0 |
| β increase | ×10 per outer iter |
| Outer iters | 4–8 |
| Quantize axis | per output channel (axis=0) or per input channel (axis=1) |

---

## Pitfalls

- **`p` matters.** p=0.5 over-emphasises outlier robustness for well-behaved layers (LayerNorm, embeddings); p=0.7 is the safer default. For aggressive outlier handling on FFN-up/down try p=0.5.
- **β continuation must be smooth.** Jumping β too fast (×100 per iter) overshoots and locks in suboptimal codes. Stick to ×10.
- **Group size 64 vs 128.** HQQ defaults to 64 because the per-group median is more responsive at smaller groups; 128 saves more memory but loses ~0.1 ppl.
- **Asymmetric is the default; symmetric option exists** but loses ~0.2 ppl on real LLM weights. Stick to asymmetric unless your kernel only supports symmetric.
- **HQQ + LoRA is the right combo** for resource-constrained fine-tuning. HQQ the base model (data-free, fast), train LoRA on top in BF16. This is what makes HQQ popular in PEFT pipelines.
- **No KV cache support.** HQQ is weight-only. For end-to-end deployment pair with KIVI/KVQuant (ch-15).

---

## Connections

- Classical ancestor: half-quadratic optimization (Geman & Reynolds 1992) for robust image restoration.
- [[excerpts/quarot]] / [[excerpts/spinquant]] — orthogonal direction; rotations vs robust loss as outlier handling.
- [[ch-08]] — GPTQ as the Hessian-aware baseline; HQQ trades Hessian info for zero calibration.
- [[ch-09]] — AWQ as the activation-aware baseline; HQQ trades activation info for speed.
- [[ch-12]] — QLoRA/NF4 ecosystem; HQQ + LoRA is a popular alternative to QLoRA for the base-quantize step.
