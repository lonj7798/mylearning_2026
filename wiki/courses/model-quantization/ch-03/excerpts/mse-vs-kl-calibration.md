---
chapter: ch-03
course: model-quantization
phase: read
excerpt_of: "Calibration Objectives for PTQ — MSE vs KL vs Cosine (TensorRT 2017; Krishnamoorthi 2018; Wu 2020)"
source_url: https://developer.download.nvidia.com/compute/machine-learning/tensorrt/docs/8.0.0/whitepaper.pdf
created_at: "2026-05-21"
raw_data_source: [[raw-data/mse-vs-kl-calibration]]
---

# Excerpt: MSE vs KL vs cosine — which calibration loss matches which downstream

**Sources:** TensorRT calibration whitepaper (NVIDIA), Krishnamoorthi 2018 (Google), Wu 2020 (integer quant survey).
**Year:** 2017–2021 (consolidated).
**URLs:** TensorRT whitepaper — see source_url; Wu 2020 https://arxiv.org/abs/2004.09602; Krishnamoorthi 2018 https://arxiv.org/abs/1806.08342

---

## The three canonical objectives

### MSE calibration

```math
S^* \,=\, \arg\min_S \; \sum_n \big( x_n - S \cdot \text{round}(\text{clamp}(x_n / S, Q_{\min}, Q_{\max})) \big)^2
```

Under the high-rate uniform-noise model (Bennett 1948), quantization error has zero mean and variance `Δ²/12 = S²/12`. Minimizing MSE is the maximum-likelihood scale assuming Gaussian post-noise output.

### KL calibration (TensorRT default for activations)

Bin the FP tensor into `M` bins; bin the quantized tensor into the corresponding levels; minimize

```math
\text{KL}(P_{\text{fp}} \,\|\, P_{\text{quant}}) \,=\, \sum_i P_{\text{fp}}(i) \log \frac{P_{\text{fp}}(i)}{P_{\text{quant}}(i)}
```

over candidate clip ranges. Search procedure:

- Build a fine histogram of `|x|` with `B` bins (TensorRT: `B = 2048`).
- For each candidate clip threshold `T_k` (log-spaced over `[128, 8192]` bins):
  - Quantize using `S = T_k / Q_max`.
  - Compute KL between FP histogram (clipped at `T_k`) and rebinned quantized histogram.
- Pick `T_k*` minimizing KL.

### Cosine similarity

```math
\cos(x, Q(x)) \,=\, \frac{\langle x, Q(x) \rangle}{\|x\| \cdot \|Q(x)\|}
```

Maximized when `Q` preserves direction. Useful for attention scores where only the ranking matters. **Pathological for tensors whose absolute magnitude propagates** (residual stream).

---

## Percentile clipping — universal preprocessor

Before any of the above, clip outliers:

```
x' = clip(x, −T, +T)   with  T = quantile_99.9(|x|)
```

This prevents a handful of large activations (LLM attention residuals) from blowing up the scale and quantizing the bulk to noise. Universal first step in modern LLM PTQ — see [[excerpts/percentile-clipping]].

---

## Decision tree

| Tensor / consumer | Objective |
|-------------------|-----------|
| Weight quantization (conv / linear) | **MSE (per-channel)** |
| Activation calibration for Softmax-input logits | **KL** |
| Activation calibration for residual-stream tensors with heavy outliers | **percentile-clipping + MSE** |
| Attention scores where only ranking matters | cosine |
| Highly non-uniform distributions (post-Swish, attention) | per-token + percentile |

---

## Implementation cost

- **MSE:** closed-form per-tensor (one sweep).
- **KL:** requires histogram + sweep over candidate clip thresholds (~100 quantize ops).
- **Cosine:** one inner product per candidate.

All are cheap relative to QAT alternatives ([[lsq]], [[pact]]) — they run on a CPU in seconds for a 70B model with 128 calibration sequences.

---

## Connections

- [[excerpts/uniform-quantization-noise]] — Bennett model that justifies MSE.
- [[excerpts/percentile-clipping]] — universal outlier preprocessor.
- [[adaround]] — moves beyond per-tensor calibration to per-weight rounding learning (ch-04).
- [[llm-int8]] — LLM-era outlier-handling motivation for percentile + mixed-precision (ch-07).
- [[gptq]] — uses Hessian-weighted MSE (`X Xᵀ`-weighted), a principled extension of MSE (ch-08).
- [[ch-03]] — parent synthesis.
