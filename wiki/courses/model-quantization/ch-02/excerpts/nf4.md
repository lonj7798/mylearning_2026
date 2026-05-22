---
chapter: ch-02
course: model-quantization
phase: read
excerpt_of: "NF4 (NormalFloat-4) — Quantile-Based 4-bit Code for Gaussian Weights (Dettmers et al. 2023, QLoRA §3.1)"
source_url: https://arxiv.org/abs/2305.14314
created_at: "2026-05-21"
raw_data_source: [[raw-data/nf4]]
---

# Excerpt: NF4 — quantile-spaced 4-bit code for the Gaussian weight prior

**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (UW).
**Year:** 2023.
**Venue:** NeurIPS 2023 (QLoRA paper, §3.1 NormalFloat).
**URLs:** see source_url; bitsandbytes source https://github.com/TimDettmers/bitsandbytes/blob/main/csrc/kernels.cu

---

## The 16 reconstruction values

NF4 is a 16-level non-uniform code with reconstruction values placed at equally-spaced quantiles of the standard normal CDF, normalized so `|max| = 1`:

```math
\text{NF4\_levels} = \big[
  -1.0,\ -0.6962,\ -0.5251,\ -0.3949,\ -0.2844,\ -0.1848,\ -0.0911,\ 0.0,\
  +0.0796,\ +0.1609,\ +0.2461,\ +0.3379,\ +0.4407,\ +0.5626,\ +0.7230,\ +1.0
\big]
```

---

## The one-box quantization rule

For a block of `G = 64` raw weights `w_1, …, w_G`:

```math
\begin{aligned}
s &= \max_i |w_i|                            && \text{(per-block absmax scale, FP32 / FP16)} \\
\hat{w}_i &= w_i / s                          && \text{(normalize to } [-1, 1]\text{)} \\
q_i &= \arg\min_k |\text{NF4\_levels}[k] - \hat{w}_i|  && \text{(encode to 4-bit index)} \\
w_{\text{recon},i} &= s \cdot \text{NF4\_levels}[q_i]
\end{aligned}
```

---

## Construction recipe (where the values come from)

1. Compute the symmetric `N(0,1)` quantile function `Q(p) = √2 · erf⁻¹(2p − 1)`.
2. Take 8 quantile points in `(0, 1]` on the positive side (offset so the extreme = 1.0).
3. Mirror across zero; include `0.0`.
4. Normalize so `|max| = 1` (so per-block absmax scale maps to `±1`).

This is essentially a Lloyd-Max approximation for `N(0,1)` at `N = 16`; the difference vs the true Lloyd output is `< 0.05 dB`. Dettmers chose the quantile form because it can be derived once from `erf⁻¹`.

---

## Double quantization

Per-block FP32 scales are themselves quantized: 256 scales (one per block) are grouped, an outer FP32 scale is stored per 256 inner scales, and the 256 inner scales are quantized to 8-bit. **Saves ~0.37 bits/weight at minimal quality cost.**

```math
\text{effective bits} = 4 + \frac{8}{64} + \frac{32}{64 \cdot 256} \approx 4.127 \text{ bits/weight}
```

---

## Why NF4 beats INT4 on LLM weights

- **INT4 spacing**: uniform in `[−1, +1]` → wastes resolution at tails (low Gaussian mass) and is too coarse near zero (high Gaussian mass).
- **NF4 spacing**: dense near 0, sparse at the tails — matches the **Gish-Pierce `p^{1/3}` optimal density** ([[excerpts/information-theoretic-bounds]]) up to the symmetric-normalization constraint.
- **Empirical gain**: ~0.3–0.5 PPL on Llama-class models at 4-bit; larger gain at lower bit-widths.

---

## Comparison with FP4 E2M1

| | NF4 | FP4 E2M1 |
|---|---|---|
| Levels | 16, quantile-spaced (Gaussian) | 16, log-spaced |
| Best for | unimodal Gaussian-ish weights | log-magnitude data |
| Hardware | software dequant only | Blackwell native |
| LLM weight PPL | best | second |

NF4 has no native tensor-core support — the non-uniform LUT prevents direct integer multiply. Modern Marlin / Machete kernels handle NF4 by `4-bit index → 16-entry LUT lookup → BF16 → tensor core`.

---

## Failure cases

- Heavy-tailed weight distributions (some early transformer layers, RMSNorm gain parameters): NF4's symmetric Gaussian assumption breaks; INT4 with per-group scale may match or exceed.
- **Activations**: NF4 is *not* used for activations (activations are not Gaussian; post-GeLU/SiLU is heavy-tailed positive).

---

## Connections

- [[excerpts/lloyd-max-quantizer]] — NF4 ≈ Lloyd-Max for `N(0,1)` at 16 levels.
- [[excerpts/information-theoretic-bounds]] — Gish-Pierce `p^{1/3}` is the theoretical optimum NF4 approximates.
- [[int4]] — uniform 4-bit alternative; NF4 wins by ~0.5 PPL on LLM weights.
- [[qlora]] — paper that introduced NF4 (ch-12).
- [[companding-mu-law]] — NF4 is the Lloyd-Max-optimal compander for unit-Gaussian inputs.
- [[ch-02]] — parent synthesis.
