---
chapter: ch-12
course: model-quantization
phase: read
excerpt_of: "NF4 — Quantile-Based 4-bit Code for Gaussian Weights (QLoRA §3.1)"
source_url: https://arxiv.org/abs/2305.14314
created_at: "2026-05-21"
---

# Excerpt: NF4 — the 4-bit quantile code

**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
**Year:** 2023 (introduced in QLoRA §3.1)
**URLs:** https://arxiv.org/abs/2305.14314 ; bitsandbytes source `csrc/kernels.cu`
**Raw-data source:** [[raw-data/nf4]]

---

## What NF4 is

A 16-level 4-bit code where the **reconstruction values are placed at equally-spaced quantiles of the standard normal distribution N(0, 1)**, restricted to a symmetric range so |max| = 1.

The motivation: after per-block absmax normalization, LLM weight blocks are well-modelled as i.i.d. N(0, 1) samples. The MSE-optimal 4-bit code for this prior is approximately the Lloyd-Max quantizer ([[lloyd-max-quantizer]]) — NF4 is a near-optimal quantile-based approximation.

---

## The 16 NF4 values

From QLoRA Table 14 / bitsandbytes source:

```
[-1.0,
 -0.6961928,
 -0.5250730,
 -0.39491748,
 -0.28444138,
 -0.18477343,
 -0.09105003,
  0.0,
  0.07958029,
  0.16093020,
  0.24611230,
  0.33791524,
  0.44070983,
  0.56261432,
  0.72295684,
  1.0]
```

8 negative + zero + 7 positive = 16. **Asymmetric** (one extra negative) because the quantile placement isn't perfectly symmetric around zero in the chosen offset scheme.

---

## Construction

1. Compute the symmetric N(0, 1) quantile function `Q(p) = √2 · erf⁻¹(2p - 1)`.
2. Take 8 quantile points on the positive side (offset / shifted so the extreme = 1.0).
3. Mirror across zero for the negative side; include 0.0.
4. Normalize so |max| = 1 (matches per-block absmax).

---

## Per-block quantization

```
B = 64 weights per block
s = max(|w_i|)                                  (absmax scale, FP32 or FP16)
ŵ_i = w_i / s                                   (normalize to [-1, 1])
q_i = argmin_k |NF4[k] - ŵ_i|                   (encode to 4-bit index)
recover: w_recon = s · NF4[q_i]
```

Two NF4 indices pack into one byte.

---

## Why NF4 beats INT4

| | INT4 levels | NF4 levels |
|---|---|---|
| Spacing | uniform `0, ±1/7, ..., ±1` | dense near 0, sparse at tails |
| Match to weight density | wastes resolution at tails, too coarse near 0 | matches Gish-Pierce `p^(1/3)` optimum |
| LLM weight PPL | baseline | -0.3 to -0.5 PPL |

The Gish-Pierce result ([[information-theoretic-bounds]]): for a source with density `p(x)`, the MSE-optimal high-resolution non-uniform quantizer places levels at density `ρ(x) ∝ p(x)^(1/3)`. NF4's quantile spacing approximates this for the Gaussian source.

---

## Comparison with FP4 E2M1

| | NF4 | FP4 E2M1 |
|---|---|---|
| Levels | 16, quantile-spaced | 16, log-spaced (2 exp, 1 mantissa) |
| Best for | unimodal Gaussian weights | log-magnitude data |
| Hardware | software dequant only | Blackwell native |
| LLM weight PPL | best | second |

NF4 has **no native tensor-core support** — the non-uniform LUT prevents direct integer multiply. Modern kernels (Marlin, Machete, bitsandbytes' fused GEMV) dequantise NF4 to BF16 in shared memory, then run tensor-core matmul.

---

## Double quantization (the bit-budget recovery)

The per-block FP32 scales `{s_b}` are themselves quantized:
- Group every 256 block-scales.
- Quantize to FP8 with one FP32 outer scale per 256-block group.

```
total bits/weight = 4 + 8/64 + 32/(64·256) ≈ 4.127
```

Savings ~0.37 bits/weight vs naïve FP32-per-block scale.

---

## Failure cases

- **Heavy-tailed distributions** (some early transformer layers, RMSNorm gain parameters): NF4's symmetric Gaussian assumption breaks; INT4 with per-group scale may match or exceed.
- **Activations:** NF4 is **not** used for activations (post-GeLU/SiLU is heavy-tailed positive, not Gaussian).

---

## Connection to Lloyd-Max

NF4 is **not exactly** the Lloyd-Max 16-level quantizer for N(0, 1) — Dettmers chose quantile spacing for analytical simplicity. True Lloyd-Max levels differ slightly. Subsequent work (AF4 and academic followups) tabulates tighter codes that win another ~0.05 PPL.

The lesson: quantile-spaced is *near-optimal* but a tiny gap remains. NF4's adoption is about the engineering tradeoff (simple closed-form construction + competitive quality), not pure rate-distortion optimality.

---

## Connections

- [[lloyd-max-quantizer]] (ch-03) — theoretical ancestor; NF4 is a quantile-spaced approximation.
- [[information-theoretic-bounds]] (ch-01) — Gish-Pierce `p^(1/3)` optimum.
- [[int4]] — uniform 4-bit alternative; NF4 wins by ~0.5 PPL.
- [[qlora]] — the paper that introduced NF4.
- [[bitsandbytes-nf4]] — production implementation.
- [[companding-mu-law]] — companding theory NF4 instantiates for the Gaussian distribution.
