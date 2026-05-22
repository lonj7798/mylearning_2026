<!-- scope: Survey of calibration objectives — when MSE / KL / cosine matter
     deps: quantization-mapping, uniform-quantization-noise
     see-also: percentile-clipping, adaround, gptq, llm-int8
-->

# Calibration Objectives for PTQ: MSE vs KL vs Cosine
- **Core Insight:** No single calibration loss is universally best — MSE matches the high-rate uniform-noise model, KL minimises the Kullback-Leibler divergence between fp and quantized output distributions and dominates for Softmax-tailed activations, cosine ignores magnitude and works best for attention scores where direction matters; the right choice depends on what downstream the quantized tensor feeds into.
- **Guideline:** Use MSE for weight calibration (matches the orthogonal-noise assumption); use KL for the input to Softmax (preserves the probability distribution); use cosine when only the angle matters (e.g. score similarity); never use min-max alone for heavy-tailed activations — clip to a percentile first.
- **Authors:** practitioner/survey consolidation (TensorRT calibration whitepaper, Krishnamoorthi 2018, Wu 2020)
- **Year:** 2018–2021 (consolidated)
- **URL:** https://developer.download.nvidia.com/compute/machine-learning/tensorrt/docs/8.0.0/whitepaper.pdf (TensorRT calibration whitepaper); https://arxiv.org/abs/2004.09602 (Wu integer quant survey)
- **Relevant topics:** PTQ calibration, MSE, KL divergence, cosine similarity, percentile clipping

## Abstract
PTQ accuracy is dominated by the choice of calibration objective — the loss minimised when fitting the per-tensor scale S (and zero-point Z). The three canonical choices are MSE between fp and quantized tensor, KL divergence between fp and quantized output distributions (TensorRT's default), and cosine similarity between fp and quantized activations. Each is principled under a different assumption about how quantization noise propagates. This page consolidates the comparative analysis: MSE is optimal under the uniform-noise / orthogonal-error assumption (Bennett 1948); KL is optimal when the tensor is a logit/score that gets softmaxed; cosine is preferred when only directional information survives the next layer.

## Key Contributions
- Maps each calibration objective to the noise / propagation assumption it minimises.
- Documents the per-objective failure modes: MSE wastes resolution on outliers; KL needs binning; cosine ignores magnitude calibration that downstream depends on.
- Provides the standard percentile-clipping recipe (clip to 99.9% then MSE-fit) as the universal default.
- Frames why Softmax-input calibration uses KL while weight calibration uses MSE.

## Key Figures/Tables to Study
- **TensorRT whitepaper Figure 4** — KL-divergence calibration sweep across clip ranges.
- **Krishnamoorthi 2018 Table 7** — MSE vs KL vs percentile across CNN models.

## Technical Details

### MSE calibration
Choose S, Z to minimise per-tensor `‖x − Q(x)‖²` over calibration data:
`S* = arg min_S Σ_n (x_n − S · round(clamp(x_n/S, Qmin, Qmax)))²`
Under the high-rate uniform-noise model (Bennett 1948), quantization error has zero mean and variance Δ²/12 = S²/12 — minimising MSE is the maximum-likelihood scale assuming Gaussian post-noise output.

### KL calibration (TensorRT default)
Bin the fp tensor into M bins; bin the quantized tensor into the corresponding levels; minimise
`KL(P_fp || P_quant) = Σ_i P_fp(i) log [P_fp(i) / P_quant(i)]`
over candidate clip ranges. Search procedure:
- Build a fine histogram of |x| with B bins.
- For each candidate clip threshold T_k (typically log-spaced over [128, 8192] bins):
  - Quantize using S = T_k / Q_max.
  - Compute KL between FP histogram (clipped at T_k) and rebinned quantized histogram.
- Pick T_k* minimising KL.

KL is preferred when the tensor feeds a Softmax or other distribution-shape-sensitive op.

### Cosine similarity
`cos(x, Q(x)) = ⟨x, Q(x)⟩ / (‖x‖ · ‖Q(x)‖)`
Maximised when Q preserves direction. Useful for attention scores where only the ranking matters. Pathological for tensors whose absolute magnitude propagates (residual stream).

### Percentile clipping
Before any of the above, clip outliers:
`x' = clip(x, −T, T)  with T = quantile_99.9(|x|)`
This prevents a handful of large activations (LLM attention residuals) from blowing up the scale and quantizing the bulk to noise. Universal first step in modern LLM PTQ — see [[llm-int8]] for the outlier-channel fix.

### Decision tree
- **Weight quantization**: MSE (per-channel for conv/linear).
- **Activation calibration for Softmax-input logits**: KL.
- **Activation calibration for residual-stream tensors with heavy outliers**: percentile-clipping + MSE.
- **Attention scores where only ranking matters**: cosine.
- **Highly non-uniform distributions (post-Swish, attention)**: per-token + percentile.

### Implementation cost
- MSE: closed-form per-tensor (one sweep).
- KL: requires histogram + sweep over candidate clip thresholds (~100 quantize ops).
- Cosine: one inner product per candidate.

All cheap relative to the QAT alternatives ([[lsq]], [[pact]]).

## Connections
- [[quantization-mapping]] — taxonomy of PTQ choices; calibration objective is one cell.
- [[uniform-quantization-noise]] — Bennett model that justifies MSE.
- [[percentile-clipping]] — the universal outlier-handling preprocessor.
- [[adaround]] — moves beyond per-tensor calibration to per-weight rounding learning.
- [[llm-int8]] — LLM-era outlier-handling motivation for percentile + mixed-precision.
- [[gptq]] — uses Hessian-weighted MSE (X Xᵀ-weighted) — a more principled extension of MSE.
