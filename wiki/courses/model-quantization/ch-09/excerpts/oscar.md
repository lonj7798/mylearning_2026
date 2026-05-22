---
chapter: ch-09
course: model-quantization
phase: read
excerpt_of: "Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models"
source_url: https://arxiv.org/abs/2209.13325
created_at: "2026-05-21"
---

# Excerpt: Outlier Suppression — LayerNorm γ migration as ancestor of SmoothQuant

**Authors:** Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, Xianglong Liu
**Year:** 2022 (NeurIPS)
**URL:** https://arxiv.org/abs/2209.13325
**Raw-data source:** [[raw-data/oscar]]

---

## The structural source of activation outliers

In a transformer, the typical pattern is:

```
y_i = γ_i · (x_i − μ)/σ + β_i      (LayerNorm)
z = W · y                          (next Linear)
```

LayerNorm's per-channel `γ` is the structural amplifier. If `γ_i` is 10× the median (which Wei et al. show is common in BERT/BART), then output channel `i` after LN has 10× larger magnitude than its neighbours. Downstream activations inherit this scaling. **Activation outliers are not random — they are LayerNorm γ outliers propagated forward.**

---

## Gamma Migration (the closed-form transformation)

Set `γ ← 1` and absorb `γ` into the next Linear:

```math
W' = W \cdot \mathrm{diag}(\gamma), \qquad \beta' = \beta \cdot \gamma, \qquad \gamma' = 1
```

The forward is algebraically identical:

```math
W' \cdot \mathrm{LN}(x; 1, \beta') = W \cdot \mathrm{diag}(\gamma) \cdot y_{\text{normalized}} = W \cdot \mathrm{LN}(x; \gamma, \beta)
```

But now `y' = LN(x; γ=1, β')` has **uniform per-channel magnitude** — the structural outlier source is gone.

**Constraint:** γ must feed *directly* into a Linear. Doesn't work across a non-linearity (GeLU before next Linear blocks it).

---

## Token-Wise Clipping

After γ migration, residual activation outliers concentrate at specific *token positions* (typically `[SEP]`, `[CLS]` in BERT). Per-tensor clip wastes scale on these positions.

- For each token position `t`, compute per-token max `m_t = max_c |x_{t, c}|`.
- Identify the top-K outlier tokens by `m_t`.
- Use a separate (higher) clip threshold for outlier tokens; standard percentile for the rest.

Per-token scales `S_t` are negligible memory (one scalar per token) and don't break GEMM.

---

## Empirical effect (BERT-Base GLUE, 8-bit PTQ)

| Method | Score | Δ vs FP |
|---|---|---|
| FP baseline | 84.6 | — |
| 8-bit PTQ (percentile) | 82.3 | −2.3 |
| + Gamma Migration | 83.7 | −0.9 |
| + Gamma Migration + Token Clipping | **84.2** | **−0.4** |

8-bit BERT PTQ within 0.5 GLUE without QAT.

---

## Why this is SmoothQuant's direct ancestor

| Aspect | Outlier Suppression (2022) | SmoothQuant (2022, weeks later) |
|---|---|---|
| What migrates | exactly `γ` (LN affine) | arbitrary per-channel `s` |
| Why | LN is the structural outlier source | activation/weight max rebalancing |
| Scope | BERT / BART (small models) | OPT / BLOOM / LLaMA up to 175B |
| Free parameter | none (γ fixed by LN) | migration strength α |
| Result | 8-bit BERT | 8-bit OPT-175B |

SmoothQuant generalises: instead of migrating exactly `γ`, migrate an arbitrary per-channel scale chosen to minimise the *combined* X+W quant difficulty. Same fusion mechanic; richer parameterisation.

---

## Limits identified by the paper

- 4-bit regime: γ migration alone insufficient; needs QAT.
- LN-followed-by-non-linearity: γ migration blocked.
- Larger LLMs (>10B): didn't test; SmoothQuant proved the idea scales.

---

## Connections

- [[smoothquant]] — LLM-era generalisation with tunable α.
- [[outlier-channel-splitting]] — orthogonal discrete approach.
- [[llm-int8]] — concurrent FP16-outlier-column path for extreme LLMs.
- [[awq]] — weight-only equivalent transformation; uses the same fusion trick on the previous LN.
- [[percentile-clipping]] — token-wise clipping is the per-position generalisation.
