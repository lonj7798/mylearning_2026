<!-- scope: I-BERT — integer-only BERT inference with INT-GELU / INT-Softmax / INT-LayerNorm
     deps: integer-only-inference, q8bert
     see-also: quantization-mapping, smoothquant
-->

# I-BERT: Integer-only BERT Quantization
- **Core Insight:** Transformer inference can be made fully integer if and only if the non-linearities (GELU, Softmax, LayerNorm) are replaced by integer polynomial / shift-based approximations; the matmuls were already int8 by Jacob 2018, but the activation functions were the bottleneck blocking pure-integer transformers.
- **Guideline:** Use i-GELU (second-order polynomial approximation of erf), i-Softmax (replace exp with `2^x` via shift, normalise by integer reciprocal), i-LayerNorm (Newton-Raphson integer square root); ship all three as INT8 modules so the entire transformer is integer end-to-end.
- **Authors:** Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer
- **Year:** 2021 (ICML)
- **URL:** https://arxiv.org/abs/2101.01321
- **Relevant topics:** integer-only transformer, INT-GELU, INT-Softmax, INT-LayerNorm, BERT INT8

## Abstract
I-BERT is the first end-to-end integer-only inference pipeline for transformers. Building on Jacob 2018's integer matmul + requantize pipeline, it replaces the three non-linearities that previously forced fp32 fallback — GELU, Softmax, LayerNorm — with integer-only approximations that achieve negligible accuracy loss. On GLUE benchmarks, I-BERT matches FP BERT-Base / BERT-Large to within 0.4 average points while running 2.4–4.0× faster on Intel CPUs without any fp32 ops in the hot path.

## Key Contributions
- **i-GELU**: second-order polynomial approximation of erf usable in integer arithmetic.
- **i-Softmax**: replace exp with shift-based `2^x` and integer reciprocal normalisation.
- **i-LayerNorm**: integer square-root via Newton-Raphson iteration.
- End-to-end integer BERT pipeline: every op in {matmul, GELU, Softmax, LayerNorm, residual, embedding lookup} integer.
- GLUE accuracy preserved; 4× wall-clock speedup on AVX-512 CPUs.

## Key Figures/Tables to Study
- **Figure 3** — i-GELU polynomial fit overlaid on true GELU; sub-1e-3 error.
- **Algorithm 2 / 3** — integer Softmax and LayerNorm pseudocode.
- **Table 4** — GLUE scores: I-BERT vs FP BERT vs Q8BERT.

## Technical Details

### i-GELU (the polynomial trick)
GELU(x) = (x/2)·(1 + erf(x/√2)). Approximate erf via a 2nd-order polynomial L(x):
`L(x) = sign(x) · [a · (clip(|x|, 0, −b) + b)² + 1],  a = −0.2888,  b = −1.769`
Then `i-erf(q_x) = sign(q_x) · [a · (clip(|q_x|, 0, −b/S_x) + b/S_x)² · S_x² + 1]`
and `i-GELU(q_x) = (q_x / 2) · (1 + i-erf(q_x))`
all in int operations with one squared-poly and a few shifts.

### i-Softmax (shift + reciprocal)
Standard softmax: `s_i = exp(x_i − max_x) / Σⱼ exp(x_j − max_x)`.
Use `exp(x) = 2^{x · log₂ e} = 2^{(x · 1.4427)}`. For integer x' = round(x · 1.4427):
- Split `x' = q·log₂(e) + r` so `2^{x'} = 2^q · 2^r` with r ∈ [−1, 0].
- Approximate `2^r ≈ 0.3585·r² + 0.353·r + 0.344` (small polynomial).
- The 2^q is a shift.
Normalisation: compute `1/Σ` via integer reciprocal table or one Newton step.

### i-LayerNorm (integer sqrt via Newton-Raphson)
LayerNorm needs `1/√(Var + ε)`. Compute Var as int32 sum-of-squares. Then int sqrt by iterating
`x_{n+1} = (x_n + ⌊v/x_n⌋) >> 1`
converging in ⌈log₂ bit_width⌉ steps. Final scale absorbs γ, shift absorbs β; rescale to int8.

### Integer pipeline contract
Every module takes (q ∈ int8, S ∈ fp metadata, Z ∈ int8) and emits the same triple. Scales are folded into the next layer's M = S_in · S_op / S_out (Jacob-2018 requantization) at compile time. Runtime: zero fp ops.

### Empirical effect
- GLUE avg: BERT-Base FP 83.2 → I-BERT 83.0 (Δ = −0.2).
- Latency: 2.4× (Base) / 4.0× (Large) on Intel Cascade Lake AVX-512.

## Connections
- [[integer-only-inference]] — Jacob 2018 supplied the matmul + requant pipeline I-BERT plugs into.
- [[quantization-mapping]] — affine (S, Z) scheme; per-tensor activations, per-channel weights.
- [[q8bert]] — int8 BERT via QAT but kept Softmax / LayerNorm in fp; I-BERT closes that gap.
- [[smoothquant]] — LLM-era heir for the activation-outlier problem I-BERT side-steps via QAT.
- [[bitnet]] — pushes the integer-only philosophy to the extreme of 1-bit weights.
