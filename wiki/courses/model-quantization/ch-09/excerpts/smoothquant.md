---
chapter: ch-09
course: model-quantization
phase: read
excerpt_of: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
source_url: https://arxiv.org/abs/2211.10438
created_at: "2026-05-21"
---

# Excerpt: SmoothQuant — closed-form W8A8 equivalent transformation

**Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
**Year:** 2022 (ICML 2023)
**URL:** https://arxiv.org/abs/2211.10438
**Raw-data source:** [[raw-data/smoothquant]]

---

## The identity

For input `X ∈ R^{T × C_in}`, weight `W ∈ R^{C_in × C_out}`, and any positive diagonal `s ∈ R^{C_in}`:

```math
Y = X \cdot W = (X \cdot \mathrm{diag}(s)^{-1}) \cdot (\mathrm{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}
```

The matmul output is unchanged. What changes is the per-channel statistics: column `j` of X̂ shrinks by `s_j`; row `j` of Ŵ grows by `s_j`. If activation channel `j` has an outlier (large `max|X_{·,j}|`) and its paired weight row is small (`max|W_{j,·}|` small), choosing `s_j > 1` rebalances both.

---

## Closed-form choice of `s`

```math
s_j = \frac{\max(|X_{\cdot, j}|)^{\alpha}}{\max(|W_{j, \cdot}|)^{1-\alpha}}
```

Migration strength `α ∈ [0, 1]`:

| α | Effect | Where the difficulty lives |
|---|---|---|
| 0 | `s = 1/max|W|` | All in activations (trivial; no smoothing) |
| 0.5 (default) | Geometric mean of activation and weight maxes | Balanced — `max|X̂| = max|Ŵ|` |
| 1.0 | `s = max|X|` | All in weights |

At `α = 0.5`:

```math
\max|\hat{X}_{\cdot, j}| = \max|\hat{W}_{j, \cdot}| = \sqrt{\max|X_{\cdot, j}| \cdot \max|W_{j, \cdot}|}
```

The symmetric condition that makes both sides INT8-friendly.

---

## Per-model α defaults

| Model family | Recommended α |
|---|---|
| OPT, BLOOM | 0.5 |
| LLaMA | 0.85 |
| GLM | 0.75 |
| Falcon | 0.6 |

LLaMA has stronger activation outliers than OPT, so more of the difficulty must migrate into weights (higher α). Larger α → weights bear more dynamic range → use per-channel weight scales (not per-tensor) on the weight side.

---

## Architectural fusion (zero runtime overhead)

For the canonical `LayerNorm(γ, β) → Linear(W) → ...` pattern:

```
γ_j ← γ_j / s_j      (absorb diag(s)⁻¹ into LN affine)
β_j ← β_j / s_j
W_j ← s_j · W_j      (absorb diag(s) into Linear weight)
```

After fusion, `s` exists only in the offline calibration script — the deployed graph is identical to FP16 in op structure, just with INT8 GEMM dispatch and rebalanced parameters.

**Where it doesn't fuse:** when the previous op is not `LN`/`RMSNorm` (e.g. activation comes from a Softmax or post-GeLU output), the `diag(s)⁻¹` becomes an explicit elementwise multiply with non-trivial cost.

---

## Calibration recipe

- 512 sequences × 512 tokens from Pile / C4.
- One forward pass to collect per-channel `max|X_j|` across the corpus.
- Closed-form `s_j` per channel; fold into LN + W.
- No optimization, no gradients, no per-layer search.

---

## Final inference quant

After smoothing:
- **Activations:** per-token absmax INT8 (one scale per token row).
- **Weights:** per-channel absmax INT8 (one scale per output channel).
- **Symmetric** on both sides.
- Dispatched to INT8 tensor cores (CUTLASS / cuBLAS-INT8 / FasterTransformer).

---

## Empirical results (OPT-175B, WikiText-2 PPL)

| Method | PPL | Δ vs FP16 | Speedup |
|---|---|---|---|
| FP16 | 8.34 | — | 1.0× |
| LLM.int8() | 8.40 | +0.06 | <1× (FP16 outlier path) |
| ZeroQuant | 8.61 | +0.27 | ~1× |
| **SmoothQuant** | **8.35** | **+0.01** | **1.51×** |

First method to make W8A8 actually faster than FP16 at 175B scale; first viable single-node serving of OPT-175B and BLOOM-176B.

---

## Connections

- [[outlier-channel-splitting]] — discrete pre-LLM ancestor (duplicate channel, halve weight).
- [[oscar]] — BERT-era ancestor; SmoothQuant generalises Gamma Migration.
- [[awq]] — weight-only specialisation (W4A16 instead of W8A8).
- [[omniquant]] — learnable extension; replaces closed-form `s` with gradient-trained `(s, b)` plus learned weight clipping.
- [[quarot]], [[spinquant]] — rotation-based descendants that go further (unitary instead of diagonal).
- Framework: integrated into TensorRT-LLM and vLLM.
