---
chapter: ch-06
course: model-quantization
phase: read
excerpt_of: "Q8BERT: Quantized 8Bit BERT (Zafrir et al. 2019)"
source_url: https://arxiv.org/abs/1910.06188
arxiv: 1910.06188
created_at: "2026-05-21"
---

# Excerpt: Q8BERT — the first INT8 transformer (and its load-bearing gap)

**Authors:** Ofir Zafrir, Guy Boudoukh, Peter Izsak, Moshe Wasserblat
**Year:** 2019
**Raw-data source:** [[raw-data/classics/q8bert]]

---

## What Q8BERT actually does

Applies Jacob 2018's simulated-quantization QAT recipe to BERT-Base. Inserts fake-quant ops on every GEMM input, fine-tunes for one extra epoch on the downstream task, and ships an INT8 model that runs ~4× smaller and 2-4× faster on AVX-512.

```python
# Q8BERT linear-layer wrapping
class QuantLinear(nn.Module):
    def forward(self, x):
        x_fq = fake_quant_per_tensor(x, self.act_scale, self.act_zero)
        w_fq = fake_quant_per_channel(self.weight, self.weight_scale, 0)
        return F.linear(x_fq, w_fq, self.bias)
```

`fake_quant` rounds to the int8 grid in fp space; backward is STE (identity within clip range).

---

## What Q8BERT does NOT quantize (the load-bearing list)

| Module | Quantized? | Why kept fp |
|---|---|---|
| Q / K / V projections | INT8 | standard GEMM |
| Attention output projection | INT8 | standard GEMM |
| Both FFN linears | INT8 | standard GEMM |
| Classifier head | INT8 | standard GEMM |
| **Softmax** | **fp32** | needs `exp` |
| **GELU** | **fp32** | needs `erf` |
| **LayerNorm** | **fp32** | needs `1/√(Var+ε)` |
| Residual add | fp32 | mismatched scales between branches |
| Embedding lookup | fp32 | lookup, not GEMM |

This is **INT8-GEMM-with-fp32-non-linearities**, not integer-only inference. To get true integer-only, I-BERT replaces Softmax/GELU/LayerNorm with integer approximations.

---

## Quantizer specifics

**Activations** — per-tensor asymmetric:

```math
S \;=\; (\max - \min) / (Q_{\max} - Q_{\min}), \qquad Z \;=\; \text{round}(Q_{\min} - \min/S)
```

**Weights** — per-row (per-output-channel) symmetric:

```math
S_c \;=\; \max|W_c| / 127, \qquad Z_c \;=\; 0
```

Per-channel weight scale is **mandatory** — Q8BERT shows per-tensor weight scale costs 1.5–2 GLUE points on MNLI. This is the first hard evidence transformers need per-channel weight quant.

---

## Calibration

- Activation `(S, Z)`: EMA over the first ~5 batches, momentum 0.99. Single-batch calibration is "unlucky" — Q8BERT documents this as a stability fix.
- Weight `(S, Z)`: recomputed every step during QAT (weights move during training).

---

## QAT recipe

| Knob | Value |
|---|---|
| Init from | fp BERT fine-tuned on the task |
| Extra epochs | 1 |
| LR | 2e-5 (same as SFT, no scaling) |
| Loss | original task loss, no auxiliary quant loss |
| Calibration | EMA momentum 0.99 over first 5 batches |

---

## Empirical effect

| Setting | GLUE-avg | Model size |
|---|---|---|
| BERT-Base FP32 | 82.5 | 415 MB |
| BERT-Base Q8BERT | 82.3 (Δ = −0.2) | 105 MB (4×) |
| BERT-Large FP32 | 85.5 | 1.24 GB |
| BERT-Large Q8BERT | 85.3 (Δ = −0.2) | 311 MB (4×) |

CPU latency: 2–4× speedup on Intel Cascade Lake.

---

## What Q8BERT taught the field (the durable lessons)

1. **Per-channel weight scales are mandatory for transformers.** Per-tensor loses 1.5–2 GLUE points.
2. **Calibrate activation scales with EMA over multiple batches.** Single-batch calibration is fragile.
3. **Non-linearities can be left fp without much speed cost.** GEMM is the dominant cost.
4. **INT8 transformer = ~0.2 GLUE drop with 1 epoch QAT.** Sets the bar for later PTQ methods to beat (and they do, via vector-wise scales).
5. **LayerNorm params (γ, β) stay fp.** Too few parameters to compress, too sensitive.

---

## What Q8BERT did not solve

- **Non-linearities in integer**: closed by [[i-bert]].
- **Sub-8-bit BERT**: requires mixed-precision allocation — closed by [[q-bert]].
- **PTQ without retraining**: requires Hessian-based methods — closed by [[obc]] and later [[gptq]] at LLM scale.
- **Outliers at 6.7B+**: doesn't appear in BERT-Base; closed by [[llm-int8]] at LLM scale.

---

## Common pitfalls

- **Forgetting to fold LayerNorm's γ into the per-tensor activation scale.** The γ shift breaks the fake-quant grid alignment if not absorbed.
- **Quantizing the embedding lookup table.** Embeddings have a wide dynamic range (rare-token rows dominate); quantizing them tanks accuracy on long-tail tasks.
- **Per-tensor activation scale on the attention output.** Has per-head outliers; needs group-wise scale (per-head) as Q-BERT later established.

---

## Connections

- [[excerpts/integer-only-inference]] — Jacob 2018 QAT recipe Q8BERT directly applies.
- [[excerpts/i-bert]] — direct successor; closes the non-linearity gap.
- [[ch-06]] — parent synthesis.
- [[ch-07]] — [[llm-int8]] is the LLM-era successor that handles outliers without QAT.
