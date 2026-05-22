---
chapter: ch-06
course: model-quantization
phase: read
excerpt_of: "I-BERT: Integer-only BERT Quantization (Kim et al. 2021)"
source_url: https://arxiv.org/abs/2101.01321
arxiv: 2101.01321
created_at: "2026-05-21"
---

# Excerpt: I-BERT — the three integer approximations

**Authors:** Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer
**Year:** 2021 (ICML)
**Raw-data source:** [[raw-data/classics/i-bert]]

---

## Why this paper exists

Jacob 2018 ([[integer-only-inference]]) solved the GEMM half of integer-only inference. Q8BERT ([[q8bert]]) applied that to BERT but **kept Softmax, GELU, and LayerNorm in fp32** — meaning the "integer-only" claim was false for transformers. I-BERT closes the gap with three polynomial / shift approximations.

---

## i-GELU — polynomial fit of erf (the load-bearing formula)

GELU: `GELU(x) = (x/2)(1 + erf(x/√2))`. I-BERT approximates `erf` with a single squared polynomial:

```math
L(x) \;=\; \text{sign}(x) \cdot \Bigl[\, a \cdot \bigl(\text{clip}(|x|,\, 0,\, -b) + b\bigr)^2 + 1 \,\Bigr]
```

with `a = −0.2888`, `b = −1.769`. In integer arithmetic against quantized input `q_x` (real `x ≈ S_x · q_x`):

```math
\text{i-erf}(q_x) \;=\; \text{sign}(q_x) \cdot \Bigl[\, a \cdot \bigl(\text{clip}(|q_x|,\, 0,\, -b/S_x) + b/S_x\bigr)^2 \cdot S_x^2 + 1 \,\Bigr]
```

```math
\text{i-GELU}(q_x) \;=\; (q_x / 2) \cdot \bigl(1 + \text{i-erf}(q_x)\bigr)
```

One squared-polynomial + a handful of shifts. **Max approximation error vs true GELU: < 1e-3** over the active range — well below the int8 grid resolution, so the polynomial is numerically free.

---

## i-Softmax — the `2^x` shift trick

Standard softmax: `s_i = exp(x_i − max_x) / Σ_j exp(x_j − max_x)`. The trick:

```math
\exp(x) \;=\; 2^{x \cdot \log_2 e} \;=\; 2^{x \cdot 1.4427}
```

For integer `x' = round(x · 1.4427)`, decompose:

```text
x' = q · 1  +  r,    integer q, fractional r ∈ [−1, 0)
2^{x'} = 2^q · 2^r
2^q     = bit shift
2^r    ≈ 0.3585·r² + 0.353·r + 0.344        # tiny polynomial, |r| ≤ 1
```

Normalisation: compute `1/Σ` via an integer reciprocal table or one Newton-Raphson step. The exponential is now a shift + 3-term polynomial. **No fp `exp` in the kernel.**

---

## i-LayerNorm — Newton-Raphson integer sqrt

LayerNorm needs `1/√(Var + ε)`. Variance is an int32 sum-of-squares (straightforward). The hard part is `sqrt(v)` in integer:

```text
x_{n+1} = (x_n + ⌊v / x_n⌋) >> 1
```

Newton-Raphson converges in `⌈log₂ bit_width⌉ ≈ 5` steps for int32 — five integer divides + shifts. Final scale absorbs `γ`, shift absorbs `β`; rescale output to int8.

The "Newton sqrt" trick is older than computing (Hero of Alexandria, 1st century AD); in integer form it's mature CS textbook material. I-BERT's contribution is realising you can stitch it into a transformer's hot path without an fp32 fallback.

---

## The pipeline contract

Every module — matmul, GELU, Softmax, LayerNorm, residual-add, embedding lookup — takes `(q ∈ int8, S ∈ fp metadata, Z ∈ int8)` and emits the same triple. Scales fold into the next layer's `M = S_in · S_op / S_out` at compile time (Jacob 2018 requantization).

**Runtime: zero fp ops.**

---

## Empirical effect

| Metric | BERT-Base FP | I-BERT |
|---|---|---|
| GLUE-avg | 83.2 | 83.0 (Δ = −0.2) |
| GLUE-avg, BERT-Large | 85.5 | 85.4 (Δ = −0.1) |
| Latency on Intel Cascade Lake AVX-512 (Base) | 1.0× | **2.4×** |
| Latency (Large) | 1.0× | **4.0×** |

The non-linearity polynomials cost ~5% of inference time; the GEMM savings dwarf them.

---

## Why this doesn't transfer to GPU LLMs (mostly)

GPUs have *fast fp16/bf16 paths*. The non-linearities cost essentially nothing relative to the matmul. The integer-only constraint exists only when you cannot afford a floating-point unit — mobile CPU, NPU, FPGA, edge MCU.

For **GPU LLM serving**, the dominant cost is matmul memory bandwidth, not non-linearity compute. The right tradeoff is INT4 / INT8 weights with FP16 activations and FP16 non-linearities ([[gptq]], [[awq]]) — keep the non-linearities in fp16 where they're fast, only quantize the bandwidth-bound weights.

For **edge LLMs** (mobile chips, TFLite, llama.cpp on phones), I-BERT-style integer approximations are coming back into fashion — see GGML's INT8 transformer kernels and Apple's Core ML quantization pipeline.

---

## Common pitfalls

- **Polynomial fit fails outside training range.** i-erf is accurate for `|x| ≤ 5`; values outside (which do appear at low precision) need clip handling.
- **Softmax `2^r` polynomial coefficients depend on `r`'s range.** I-BERT's coefficients assume `r ∈ [−1, 0)`; using `[0, 1)` flips signs and silently corrupts attention.
- **Newton sqrt seed point matters.** Bad initial `x_0` (e.g. 1 for large `v`) costs extra iterations and may overflow the int32 division. Use a magic-constant seed like `2^(31 − leading_zeros(v))/2`.

---

## Connections

- [[excerpts/integer-only-inference]] — Jacob 2018 supplied the matmul + requantize pipeline I-BERT plugs into.
- [[ch-06]] — parent synthesis.
- [[ch-07]] — [[llm-int8]] takes the opposite path for LLMs: keep fp16 non-linearities, INT8 GEMM, FP16 outlier path.
- [[ch-09]] — [[smoothquant]] keeps fp16 non-linearities but solves the outlier problem at the GEMM-input level.
