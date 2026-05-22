---
chapter: ch-08
course: model-quantization
phase: read
excerpt_of: "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers (Yao et al. 2022)"
source_url: https://arxiv.org/abs/2206.01861
arxiv: 2206.01861
created_at: "2026-05-21"
---

# Excerpt: ZeroQuant — the granularity-focused W8A8 PTQ

**Authors:** Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He
**Year:** 2022 (NeurIPS 2022)
**Raw-data source:** [[raw-data/papers/zeroquant]]

---

## The three-piece recipe

ZeroQuant argues the right granularity + a cheap data-free distillation is enough for W8A8 LLM PTQ — no OBS, no AdaRound, no block reconstruction.

### (1) Group-wise symmetric weight quant

For weight `W ∈ ℝ^{d_out × d_in}`, partition each row into groups of G consecutive input dims:

```math
s \;=\; \max(|W_g|) / (2^{b-1} - 1)
```

```math
\hat{W}_g \;=\; \text{round}(W_g / s) \cdot s
```

- b = 8 (typical), G = 128–256.
- Per-row (G = d_in) is the default for W8; smaller G needed for W4.
- Same grouping as GPTQ.

### (2) Per-token (dynamic) activation quant — the load-bearing formula

For activation `X ∈ ℝ^{B × S × d}`, one scale per token at runtime:

```math
c_{b, s} \;=\; \max_d |X_{b, s, d}| / 127
```

```math
\hat{X}_{b, s, d} \;=\; \text{round}(X_{b, s, d} / c_{b, s})
```

**No calibration set needed**; scale recomputed every forward pass. Dequantization folds into the next op via `c_{b,s} · s_w`.

Why this works at LLM scale: per-token scale lets one outlier token shrink only its own row's scale, leaving other rows with full dynamic range. Conceptually similar to LLM.int8's per-token but without the FP16 outlier path — group-wise weight quant absorbs enough of the gap.

### (3) Layer-wise Knowledge Distillation (LKD)

```text
for each transformer block f_i in order:
    1. Run FP teacher + partially-quantized student up to input of block i.
    2. Quantize block f_i's weights → student block f̂_i.
    3. Optimize quant params to minimize:
       L_i = ||f_i(h) − f̂_i(h)||²  on calibration samples.
    4. Only block i in full precision in memory → fits 20B model on single GPU.
```

Memory-cheap: only one block at full precision at a time. Works even without original training data (use any short calibration text — C4, WikiText). LKD is decisive at W4: pure RTN W4 loses 5–10 ppl; LKD recovers within 1 ppl of FP.

The MSE objective at block grain is literally [[brecq]]'s objective without the Fisher weighting.

---

## Recipe

| Knob | Value |
|---|---|
| Weight bits | 8 (FC + attn) or W4-FC / W8-attn mixed |
| Weight group | 128–256, symmetric |
| Activation bits | 8 |
| Activation scale | per-token dynamic |
| LKD samples | 128–1024 |
| LKD optimizer | Adam, lr 5e-7, ≤ 1 epoch |

---

## Empirical effect

| Setting | GPT-NeoX-20B PPL | Speedup vs FP16 |
|---|---|---|
| FP16 baseline | 9.40 | 1.0× |
| **W8A8 ZeroQuant** | **9.45 (Δ +0.05)** | **4.16×** |
| W4-FC/W8-attn + LKD | 9.62 (Δ +0.22) | ~5× |

W8A8 essentially lossless; W4A8 within 0.3 ppl. 4-5× inference speedup on DeepSpeed-Inference's INT8 GEMM kernels.

---

## How ZeroQuant differs from GPTQ

| Axis | GPTQ | ZeroQuant |
|---|---|---|
| Weight quant | OBS column-by-column with Hessian correction | Group-wise RTN (round-to-nearest) |
| Activation quant | none (weight-only) | per-token dynamic INT8 |
| Calibration | 128 sequences × 2048 tokens, used for Hessian | 128–1024 samples, used for LKD |
| Recovery if loss too high | (none built-in) | LKD per block |
| Output bit-width | W3 / W4 | W4A8 / W8A8 |
| Deployment kernel | [[marlin-kernel]] | DeepSpeed-Inference INT8 GEMM |

**They're complementary, not competing.** GPTQ is the right choice when memory dominates (W4A16 deployment, FP16 activations). ZeroQuant is the right choice when both compute and memory dominate (W8A8 with INT8 tensor cores).

---

## What ZeroQuant established for LLM PTQ

1. **Group-wise weight + per-token activation is the right granularity.** Adopted by every later W8A8 method.
2. **LKD as cheap recovery.** Idea inherited by [[smoothquant]]'s migration strategy, [[awq]]'s search loop.
3. **Per-token activation scale doesn't need calibration.** Dynamic at runtime. Eliminates the calibration-distribution gap.
4. **DeepSpeed-Inference integration.** First production LLM-PTQ deployment stack.

---

## Common pitfalls

- **Per-tensor activation scale.** Doesn't survive past ~7B; use per-token (the whole point of ZeroQuant).
- **Skipping LKD for W4.** Pure RTN W4 loses 5–10 ppl. LKD recovery is mandatory at W4.
- **LKD overfitting.** With LR > 1e-6, LKD can drift quant params away from the per-layer optimum; stay at 5e-7.
- **Forgetting to fold the activation scale.** `c_{b,s}` must be propagated to the next op as a multiply — otherwise the activation magnitudes are wrong.

---

## Connections

- [[excerpts/gptq]] — the weight-only sibling.
- [[excerpts/zeroquant-fp]] — the FP8/FP4 extension.
- [[excerpts/zeroquant-v2]] — LoRC + sensitivity study extension.
- [[ch-08]] — parent synthesis.
- [[ch-09]] — [[smoothquant]] generalises per-token activation handling via outlier migration.
