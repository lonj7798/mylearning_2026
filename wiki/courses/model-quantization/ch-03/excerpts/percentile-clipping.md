---
chapter: ch-03
course: model-quantization
phase: read
excerpt_of: "Percentile Clipping for Activation Quantization (Migacz 2017 GTC; Wu 2020)"
source_url: https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
created_at: "2026-05-21"
raw_data_source: [[raw-data/percentile-clipping]]
---

# Excerpt: Percentile clipping — the universal outlier preprocessor

**Sources:** Szymon Migacz (NVIDIA TensorRT, GTC 2017); Krishnamoorthi 2018 TF Lite whitepaper; Wu 2020 integer-quant survey.
**Year:** 2017–2020 (consolidated).
**URLs:** Migacz GTC — see source_url; Wu 2020 https://arxiv.org/abs/2004.09602

---

## The arithmetic of outlier dilation (the one-box example)

Suppose 99% of activations lie in `[0, 10]` and 1% in `[10, 1000]`.

- **Min/max scale**: `S = 1000 / 255 = 3.92` → bulk `[0, 10]` mapped to integers `[0, 2.5]` → only ~3 quantization levels for 99% of the data. **Effective resolution: 1.6 bits.**
- **99% percentile clip**: clip at 10, `S = 10 / 255 = 0.039` → bulk uses full `[0, 255]` range. **Effective resolution: 8 bits.** Tail saturates to 255.

The 1% saturation loss is far smaller than the 6.4-bit resolution gain on the bulk.

---

## Standard recipe

```
T = quantile_p(|x|)              # p ∈ {99%, 99.5%, 99.9%, 99.99%}
x' = clip(x, −T, +T)
S = T / Qmax                     # symmetric
# or
S = (max(x') − min(x')) / (Qmax − Qmin)
z = round(−min(x') / S)          # asymmetric
```

---

## Percentile estimation

Two implementations:

1. **Exact**: sort `|x|` over the calibration set, take the `⌈p · N⌉`-th element.
2. **Histogram-approximate**: build a `B`-bin histogram of `|x|` over calibration, take the inverse CDF (TensorRT: `B = 2048`).

For `N > ~10^4` entries, the histogram is sufficient and much cheaper.

---

## Threshold sweep with KL

The exact percentile is task-dependent. Sweep candidate `p ∈ {99%, 99.5%, 99.9%, 99.99%, 99.999%, 100%}` and pick the one minimizing downstream KL or task accuracy. **NVIDIA's TensorRT does this automatically; the 99.9% default is the empirical sweet spot for most CNN and transformer activations.**

---

## Per-channel vs per-tensor

Outliers in transformer activations are concentrated in a small number of channels (the "emergent outlier channels" of [[llm-int8]]). Per-channel percentile calibration is cleaner — each channel's clip threshold is fit independently. But per-channel activation quant is GEMM-unfriendly; the standard compromise is per-tensor calibration with aggressive percentile + outlier offloading.

---

## When percentile clipping fails

- **Outliers span 1000×** (LLM residual stream at 6.7B+): even 99.99% clip wastes most bits. Solution: structural — migrate the outlier load to weights ([[smoothquant]]) or split into a separate FP16 path ([[llm-int8]]).
- **Distribution is multimodal** (post-Swish / GELU): percentile on `|x|` collapses the modes. Use per-side percentile (separate `+/−` clips, see [[lsq-plus]] in ch-04).

---

## Recommended default

- 99.9% percentile (with EMA across calibration batches).
- Per-tensor for activations `< 6.7B` model scale.
- Per-channel + migration to weights ([[smoothquant]]-style) above that scale.

---

## Connections

- [[excerpts/mse-vs-kl-calibration]] — percentile clipping is the preprocessor; MSE / KL is the loss.
- [[llm-int8]] — extreme case at 6.7B+: even percentile clipping is insufficient; mixed-precision outlier path needed (ch-07).
- [[smoothquant]] — structural alternative: migrate outlier magnitude into the weight matrix via per-channel rescale, then percentile-clip cleanly (ch-09).
- [[awq]] — same intuition: weight scale absorbs activation difficulty so percentile clipping suffices (ch-09).
- [[ch-03]] — parent synthesis.
