---
chapter: ch-09
course: model-quantization
phase: read
excerpt_of: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
source_url: https://arxiv.org/abs/2306.00978
created_at: "2026-05-21"
---

# Excerpt: AWQ — activation-aware weight-only W4A16

**Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
**Year:** 2023 (MLSys 2024 **Best Paper Award**)
**URL:** https://arxiv.org/abs/2306.00978
**Raw-data source:** [[raw-data/awq]]

---

## The salient-channel observation (the empirical kernel)

For weight `W ∈ R^{C_out × C_in}` and activation `X`, the per-output RTN-quantization error is

```math
\mathrm{err}_o = \sum_j \Delta W_{o, j} \cdot X_j
```

**Empirically, the error is dominated by the j's with largest `mean(|X_j|)`.** AWQ's Figure 2:

- Keep the top-1% of channels (by activation magnitude) in FP16: ~near-FP16 PPL.
- Keep a random 1% of channels in FP16: substantial PPL drop.

Activation magnitude is the right saliency signal — not weight magnitude, not random.

---

## Per-channel scaling (the same SmoothQuant identity, weight-only)

```math
Y = W \cdot X = (W \cdot \mathrm{diag}(s)^{-1}) \cdot (\mathrm{diag}(s) \cdot X)
```

- `diag(s) · X` is **not** quantized (we're in A16; activations stay FP16).
- `W · diag(s)⁻¹` is quantized to INT4. Dividing salient weight columns by `s_j > 1` reduces their dynamic range so per-group RTN preserves them.
- `diag(s)` is absorbed into the preceding LayerNorm or upstream Linear → no runtime cost.

---

## Per-channel scale from activation magnitude

```math
s_j = \big(\mathrm{mean}(|X_j|)\big)^{\alpha}, \qquad \alpha \in [0, 1]
```

Note the differences from SmoothQuant:
- Uses **mean** (not max) of activation magnitude.
- No weight term — single scalar `α` per layer instead of per-channel closed form.

---

## Grid search for α (no backprop)

```
for α in linspace(0, 1, 20):
    s = mean(|X_j|) ** α
    W_scaled = W * s
    X_scaled = X / s
    W_q = quantize_group(W_scaled, group_size=128, bits=4)
    loss[α] = ||(W · X) − (dequant(W_q) · X_scaled)||²
α* = argmin loss
```

- 20 forward passes per layer.
- The loss-vs-α curve is a single concave bowl → reliable.
- No gradients, no Hessian, no calibration overfit.

---

## Hyperparameters (the W4A16 default)

| Knob | Value |
|---|---|
| Bits | 4 (also 3) |
| Group size | 128 (8 groups per 1024-dim row) |
| α grid | 20 points in [0, 1] |
| Calibration | 128 sequences × 512 tokens (Pile / C4) |
| Activations | FP16 (weight-only) |
| Effective bits/weight | 4 + 16/128 ≈ 4.125 |
| Inference kernel | TinyChat / AutoAWQ / Marlin |

---

## Empirical results (LLaMA-2, WikiText-2 PPL)

| Method | Bits | 7B | 13B | 70B |
|---|---|---|---|---|
| FP16 | 16 | 5.47 | 4.88 | 3.32 |
| RTN | 4 | 6.66 | 5.51 | 3.67 |
| GPTQ (g=128) | 4 | 5.69 | 4.98 | 3.42 |
| **AWQ (g=128)** | 4 | **5.60** | **4.97** | **3.41** |

AWQ matches GPTQ at W4 while generalising better to instruction-tuned and vision-language variants (Table 8) — because grid-searching one `α` per layer is far less calibration-dependent than GPTQ's Hessian fit to the calibration covariance.

---

## Why AWQ beats GPTQ on OOD data

GPTQ minimises `||W·X − Ŵ·X||²` against a specific calibration X — the rounding pattern is **overfit to that calibration's covariance**. Move to instruction-tuned, multilingual, or multimodal data and GPTQ regresses 1–3% perplexity.

AWQ uses calibration only to estimate `mean(|X_j|)` and pick **one scalar α per layer**. The per-channel scale is data-cheap and shifts gracefully across domains. Practical consequence: AWQ became the W4A16 default for general-purpose deployment (HuggingFace, vLLM, TensorRT-LLM all consume AWQ checkpoints).

---

## Connections

- [[smoothquant]] — same equivalent-transformation identity at W8A8 (closed-form, not grid).
- [[gptq]] — Hessian-based weight-only rival; same accuracy at W4, worse OOD.
- [[omniquant]] — learnable extension; replaces grid-search α with gradient-trained `(s, b)`.
- [[squeezellm]], [[spqr]], [[owq]] — sibling sub-4-bit weight-only methods that protect outliers via sparse FP16 sidecar instead of scaling.
- [[autoawq]] — production implementation.
- [[marlin-kernel]] — the W4A16 GEMM kernel.
