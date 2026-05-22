---
chapter: ch-16
course: model-quantization
phase: read
excerpt_of: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)"
source_url: https://arxiv.org/abs/2402.17764
created_at: "2026-05-21"
---

# Excerpt: BitNet b1.58 — ternary weights via per-tensor absmean rounding

**Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft)
**Year:** 2024
**URL:** https://arxiv.org/abs/2402.17764
**Raw-data source:** [[raw-data/bitnet-b158]]

---

## The thesis

Add a single extra state — **zero** — to BitNet's `{−1, +1}` weights, giving ternary `{−1, 0, +1}`. Per-weight cost: log₂(3) ≈ 1.58 bits. Result: **matches FP16 perplexity from scratch starting at the 3B-parameter scale**.

The zero state lets the model express "this connection is off" without consuming a sign bit, dramatically improving expressivity at no inference cost (zero-multiplies are free).

---

## The ternary weight rule (absmean rounding)

For each layer's latent FP weight `W ∈ ℝ^{d_out × d_in}`:

```math
\begin{aligned}
\gamma &= \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \sum_{i,j} |W_{i,j}|        \quad \text{(absmean, scalar per-tensor)} \\
\tilde{W} &= \mathrm{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, +1\right) \\
\text{where } \mathrm{RoundClip}(x, a, b) &= \max(a, \min(b, \mathrm{round}(x)))
\end{aligned}
```

The result `W̃_{i,j} ∈ {−1, 0, +1}`. The per-tensor scale γ is the **absmean** — provably the minimum-MSE 1-level magnitude under absolute-error, and a tight choice for the ternary code as well.

```python
def weight_quant_b158(W, eps=1e-5):
    scale = W.abs().mean().clamp(min=eps)
    W_q = (W / scale).round().clamp(-1, 1)
    return W_q * scale
```

Forward: `y = γ · (W̃ · x̃)` where `x̃` is the INT8-quantized activation.

Backward: STE through RoundClip — gradient passes through as identity.

---

## Why log₂(3) = 1.58

A ternary cell has 3 states. Information-theoretic minimum storage = log₂(3) ≈ 1.585 bits/weight.

**Practical packing**: pack 5 ternary weights into 8 bits because `3^5 = 243 ≤ 256`. Density: **1.6 bits/weight**.

```
Base-3 encoding:  weight_byte = w_0 + 3·w_1 + 9·w_2 + 27·w_3 + 81·w_4
                  each w_i ∈ {0, 1, 2}  (mapped from {-1, 0, +1})
                  8 bits / 5 weights = 1.6 bits/weight
                  243 codes used, 13 unused (~5% waste)
```

This is the "1.58-bit LLM" framing in headline form.

---

## Why the zero matters

Pure binary `{−1, +1}` cannot express "this connection is off". Empirically real LLM weights cluster heavily near zero (approximately Gaussian prior). Binarising zero to ±1 forces every weight to contribute, which costs both accuracy and effective capacity.

Ternary `{−1, 0, +1}` lets the model express "off" at the cost of 0.58 bits/weight. Empirically this is the difference between b1.58 reaching FP16-parity at 3B and pure BitNet trailing by a few ppl indefinitely.

The ablation: at fixed 1.6 bits storage, ternary beats binary-with-extra-precision by 0.3–0.5 ppl. Zero is doing real work.

---

## Architecture and training

Same Llama-style backbone (SwiGLU, RoPE, RMSNorm). **BitLinear** replaces every Linear. SubLN unchanged from [[bitnet]]. Trained from scratch with the same corpora and tokenizer as the FP16 baseline.

---

## Scaling behavior — the headline result

| Params | b1.58 ppl | FP16 ppl | Gap |
|--------|-----------|----------|-----|
| 700M | ~ +1.5 | baseline | b1.58 worse |
| 1.3B | ~ +1.0 | baseline | b1.58 worse |
| **3B** | **same** | baseline | **parity** |
| 13B | slightly better | baseline | b1.58 ≥ FP16 |
| 70B (extrapolated in follow-up) | better | baseline | b1.58 > FP16 |

The "crossover at 3B" claim is the central empirical result. At ≥ 3B, the discrete weight space acts as **regularization** — the latent FP weights have plenty of redundancy to encode within the ternary codebook, and the discrete restriction prevents overfitting to noise.

---

## Inference

- Weight stored as ternary, 5-per-byte base-3 packing.
- Activation INT8 per-token.
- Matmul: integer multiply-accumulate with output rescale by `γ · γ_x / 127`.
- **Zero-skipping**: any 0-weight contributes nothing → skip in custom kernels, gaining ~30% sparsity speedup empirically.

```
For w ∈ {-1, 0, +1}:
    w = +1:   y += x
    w = -1:   y -= x
    w = 0:    skip      ← ~30% of weights typically
```

No multiplier needed at all. This is what bitnet.cpp exploits — see [[bitnet-models]].

---

## Inference cost reductions (vs FP16)

- **Memory**: ~10× reduction (1.6 bits vs 16 bits, accounting for SubLN parameters).
- **Latency**: ~4× faster on commodity hardware (HBM bandwidth × 10× weight compression, partially offset by Int8 activation overhead).
- **Energy**: 55–82% reduction (bitnet.cpp measurements on x86 + ARM).

---

## Hyperparameters (recipe)

| Knob | Value |
|------|-------|
| Weight states | {−1, 0, +1} |
| Per-tensor scale | absmean γ |
| Effective bits/weight | log₂(3) ≈ 1.58 (1.6 with 5-per-byte packing) |
| Activation bits | 8 (per-token absmax) |
| Optimizer | AdamW |
| LR | 5× FP16 baseline |
| Backward | STE through RoundClip |
| Pretrain data | same as FP16 baseline |
| Crossover scale | ~3B parameters |

---

## Pitfalls

- **Below 3B you're not at parity.** Don't promise FP16-equivalent quality on 700M / 1.3B BitNet b1.58; you get a 1-1.5 ppl gap.
- **Per-tensor scale γ means the whole layer shares one β.** No per-group scales here (unlike GPTQ). This keeps inference math simple but limits expressivity slightly; group-scale variants exist in follow-ups.
- **`eps = 1e-5` matters.** For layers where `W.abs().mean()` is close to zero (rare but possible in untrained heads), clamping the scale prevents division blow-up.
- **The 5-per-byte packing wastes 13 codes.** Negligible (5%) but ensure your decoder maps valid bytes only — don't accidentally read unused codes as valid ternary.
- **Zero-skipping requires kernel support.** Stock INT8 GEMM kernels don't skip zeros; the speedup requires bitnet.cpp or equivalent.
- **`scale = W.abs().mean()` is computed every forward.** This is fine for training but for inference, freeze γ after the final training step.
- **AdamW with β₂ = 0.95** — the LLM standard; β₂ = 0.999 oversmooths and BitNet training gets sluggish.

---

## Connections

- [[excerpts/bitnet]] — direct predecessor; same recipe with binary {−1, +1} only.
- [[excerpts/onebit]] — alternative ~1-bit path via SVID decomposition, fine-tunable from FP base.
- [[excerpts/bitnet-models]] — production releases (`microsoft/BitNet-b1.58-2B-4T` + bitnet.cpp).
- [[era-of-1bit-llms]] — survey-style consolidation of the 1.58-bit scaling-law thesis.
- [[ch-04]] — [[lsq]] / [[adaround]] / [[bnn]] as the QAT lineage.
- [[ch-14]] — sub-2-bit PTQ ceiling at ~2 bits ([[aqlm]] / [[quip-sharp]]); BitNet b1.58 is the only path below that bar.
