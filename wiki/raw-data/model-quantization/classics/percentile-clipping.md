<!-- scope: Percentile clipping — outlier-aware calibration before scale fitting
     deps: mse-vs-kl-calibration, quantization-mapping
     see-also: llm-int8, smoothquant, awq, oscar
-->

# Percentile Clipping for Activation Quantization
- **Core Insight:** A single outlier activation can dilate the per-tensor scale enough to quantize the entire bulk of the distribution to noise; clipping at the 99.9% percentile before fitting S sacrifices a handful of saturated values but recovers 3–5 bits of effective resolution for the other ~10⁶ entries.
- **Guideline:** Always sweep clip thresholds T ∈ {99%, 99.9%, 99.99%, max}; pick the T minimising downstream MSE or KL; for LLM residual streams expect 99.5–99.9% to be optimal; never use raw min/max — it is a degenerate special case (T = 100%).
- **Authors:** practitioner consolidation (Wu 2020, Migacz/NVIDIA TensorRT 2017, Krishnamoorthi 2018)
- **Year:** 2017–2020 (consolidated)
- **URL:** https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf (Migacz GTC); https://arxiv.org/abs/2004.09602 (Wu survey)
- **Relevant topics:** outlier handling, calibration clip, percentile, robust scale fitting

## Abstract
Naive PTQ uses min/max of a calibration tensor to set the affine scale. This is fatal for transformer activations: a few outlier channels (typically 0.1–1% of entries) carry 10–100× the magnitude of the bulk. A min/max-fit scale wastes most of the quantization grid on the outliers, leaving the bulk underresolved. Percentile clipping fixes this by clipping at a high percentile (99.9% typical) before fitting S, then saturating the few clipped entries. This page documents the recipe used by TensorRT, by Krishnamoorthi's TF Lite whitepaper, and as the implicit prelude to every later LLM outlier-handling method (LLM.int8, SmoothQuant, AWQ).

## Key Contributions
- Establishes percentile clipping as the universal PTQ activation-calibration preprocessor.
- Documents the empirical sweet spot at 99.9% for most CNN and transformer activations.
- Explains the bit-resolution arithmetic: 100× outlier → 6.6 bits wasted on tail.
- Motivates the structural outlier-fix line that culminates in LLM.int8 / SmoothQuant.

## Key Figures/Tables to Study
- **Migacz GTC slides Figure** — histogram with outlier tail; KL sweep across percentile clips.
- **Wu 2020 Table** — accuracy vs clip percentile across CNN and transformer activations.

## Technical Details

### The arithmetic of outlier dilation
Suppose 99% of activations lie in [0, 10] and 1% in [10, 1000].
- Min/max scale: S = 1000/255 = 3.92 → bulk [0,10] mapped to integers [0, 2.5] → only ~3 quantization levels for 99% of the data. Effective resolution: 1.6 bits.
- 99% percentile clip: clip at 10, S = 10/255 = 0.039 → bulk uses full [0, 255] range. Effective resolution: 8 bits. Tail saturates to 255.

The 1% saturation loss is far smaller than the 6.4-bit resolution gain on the bulk.

### Percentile estimation
Two implementations:
1. **Exact**: sort |x| over the calibration set, take the ⌈p·N⌉-th element.
2. **Histogram-approximate**: build a B-bin histogram of |x| over calibration, take the inverse CDF (TensorRT, B=2048).

For N > ~10⁴ entries, the histogram is sufficient and much cheaper.

### Threshold sweep with KL
The exact percentile is task-dependent; sweep candidate p ∈ {99%, 99.5%, 99.9%, 99.99%, 99.999%, 100%} and pick the one minimising downstream KL or task accuracy. NVIDIA's TensorRT does this automatically.

### Per-channel vs per-tensor
Outliers in transformer activations are concentrated in a small number of channels (the "emergent outlier channels" of [[llm-int8]]). Per-channel percentile calibration is cleaner — each channel's clip threshold is fit independently. But per-channel activation quant is GEMM-unfriendly; the standard compromise is per-tensor calibration with aggressive percentile + outlier offloading.

### When percentile clipping fails
- Outliers span 1000× (LLM residual stream at 6.7B+) — even 99.99% clip wastes most bits. Solution: structural — migrate the outlier load to weights ([[smoothquant]]) or split it into a separate fp16 path ([[llm-int8]]).
- Distribution is multimodal (post-Swish/GELU) — percentile on |x| collapses the modes. Use per-side percentile (separate +/− clips, see [[lsq-plus]]).

### Recommended default
- 99.9% percentile (with EMA across calibration batches).
- Per-tensor for activations < 6.7B model scale.
- Per-channel + migration to weights ([[smoothquant]]-style) above that scale.

## Connections
- [[mse-vs-kl-calibration]] — percentile clipping is the preprocessor; MSE/KL is the loss.
- [[quantization-mapping]] — sits inside the calibration cell of the taxonomy.
- [[llm-int8]] — extreme case: even percentile clipping is insufficient at 6.7B+; mixed-precision outlier path needed.
- [[smoothquant]] — structural alternative: migrate the outlier magnitude into the weight matrix via per-channel rescale, then percentile-clip cleanly.
- [[awq]] — same intuition: weight scale absorbs activation difficulty so percentile clipping suffices.
- [[oscar]] — early transformer-era variant: outlier suppression with equalization, direct pre-SmoothQuant ancestor.
