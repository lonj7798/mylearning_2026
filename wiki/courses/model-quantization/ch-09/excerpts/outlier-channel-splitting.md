---
chapter: ch-09
course: model-quantization
phase: read
excerpt_of: "Improving Neural Network Quantization without Retraining using Outlier Channel Splitting (OCS)"
source_url: https://arxiv.org/abs/1901.09504
created_at: "2026-05-21"
---

# Excerpt: OCS — duplicate-and-halve as discrete equivalent transformation

**Authors:** Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Christopher De Sa, Zhiru Zhang
**Year:** 2019 (ICML)
**URL:** https://arxiv.org/abs/1901.09504
**Raw-data source:** [[raw-data/outlier-channel-splitting]]

---

## The splitting identity

For linear `y = W·x = Σ_c W_{·,c} · x_c`, pick input channel `c`. Replace the column `W_{·,c}` with two copies `W_{·,c}/2` (and duplicate input `x_c` → `x_c, x_c`):

```
y' = ... + (W_{·,c}/2) · x_c + (W_{·,c}/2) · x_c + ... = ... + W_{·,c} · x_c + ...
```

The output is **exactly** the original `y`. But the per-column maximum is now halved:

```
|W'_{·,c}|_max = |W_{·,c}|_max / 2
```

Each split reduces the quantizer's required clip range by 2× on that channel.

---

## Why OCS matters for the SmoothQuant lineage

OCS is the **discrete** version of what SmoothQuant generalises to a continuous per-channel scale. The shared insight: outliers don't have to be clipped (lossy) or assigned more bits (kernel-breaking) — you can *redistribute* them via an algebraically invisible reparameterisation.

| Method | Year | Transformation | Scope |
|---|---|---|---|
| OCS | 2019 | discrete split (`×1/2` and duplicate) | CNN, weight-only |
| Outlier Suppression / Gamma Migration | 2022 | `γ` migration into next Linear | BERT, activations |
| SmoothQuant | 2022 | continuous per-channel `diag(s)` | LLM, W8A8 |
| AWQ | 2023 | per-channel `s = mean|X|^α`, weight-only | LLM, W4A16 |
| OmniQuant | 2023 | learnable per-channel `(s, b)` | LLM, W4A4 |
| AffineQuant / FlatQuant | 2024 | learnable full / Kronecker affine | LLM, W4A4 |
| QuaRot / SpinQuant | 2024 | orthogonal rotation | LLM, W4A4 |

The line of descent: discrete redistribution → continuous diagonal → learned diagonal → learned affine → orthogonal rotation. Each step expands the search space of equivalent transformations.

---

## Greedy selection

For each input channel c, compute outlier score:

```
s_c = |W_{·,c}|_max / quantile_99(|W|)
```

Sort channels by `s_c`. Split the top-K channels iteratively until all `s_c ≤ threshold`. Typical: ~5–10% input dimension growth for clean 6-bit PTQ on ImageNet ResNet.

---

## Data-free property

Splitting decision uses only weight statistics — no calibration data. Activation-aware variants exist (split based on `max|X_c|`) but the base algorithm is purely weight-driven.

This is also a limit: OCS handles per-channel weight outliers, but **does not address shared activation outliers** that appear across many channels — the LLM regime where SmoothQuant and AWQ are needed.

---

## Empirical scope

- ImageNet ResNet-50, VGG: 0.5–2% accuracy gain over percentile-clipping baseline at 6-bit PTQ.
- 4-bit regime: needs combination with QAT or AdaRound.
- Negligible inference overhead given GEMM amortisation.

---

## Connections

- [[smoothquant]] — LLM-era continuous generalisation; same redistribution idea.
- [[oscar]] — BERT-era LayerNorm-γ migration (orthogonal but related).
- [[percentile-clipping]] — alternative outlier handling (lossy).
- [[awq]] — activation-aware per-channel scaling.
- [[llm-int8]] — outlier-FP16 alternative for the extreme LLM regime.
