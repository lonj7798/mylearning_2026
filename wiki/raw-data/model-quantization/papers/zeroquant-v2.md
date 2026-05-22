<!-- scope: ZeroQuant-V2 — comprehensive PTQ study + Low-Rank Compensation
     deps: [[zeroquant]], [[gptq]]
     see-also: [[zeroquant-fp]], [[loftq]], [[spqr]]
-->

# ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation
- **Core Insight:** When INT4 weight or INT8 activation PTQ leaves a residual quality gap, you can recover most of it by storing the per-layer quantization error as a low-rank correction `U V^⊤` — adding only a few extra bits per weight on average.
- **Guideline:** After running RTN/GPTQ at W4A8, compute the per-layer error `E = W − Ŵ` and replace it with the top-r SVD `E ≈ U Σ V^⊤` (r = 8–32); apply at inference as `(Ŵ + U V^⊤) x`. Useful when GPTQ alone isn't enough but full fine-tuning is too expensive.
- **Authors:** Zhewei Yao, Xiaoxia Wu, Cheng Li, Stephen Youn, Yuxiong He
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2303.08302
- **Relevant topics:** low-rank compensation, comprehensive PTQ study, INT4 weight + INT8 activation, sensitivity scaling

## Abstract
ZeroQuant-V2 is two papers in one. First, a systematic evaluation of RTN, GPTQ, and ZeroQuant across model families (BLOOM, OPT, GPT-NeoX, GLM, GPT-J) from 125M to 176B, breaking down sensitivity by (weight-only vs activation-only vs both) × (bit-width) × (model size). Key empirical findings: (a) activation quantization is more sensitive than weight quantization, (b) smaller models are sometimes *more* robust to activation quant than larger ones, and (c) no existing PTQ holds quality at W4A4 or W4A8 universally. Second, the paper proposes **Low-Rank Compensation (LoRC)**: store the residual quantization error as a low-rank product `U V^⊤` to recover quality with minimal extra memory.

## Key Contributions
- Comprehensive sensitivity matrix across model families × sizes × bit-widths — the reference taxonomy used by later papers.
- Identifies that activation quantization, not weight quantization, is the binding constraint for low-bit LLM PTQ.
- Introduces **LoRC**: a low-rank residual added to the quantized weight that bridges the W4A8 quality gap with ~1–2 bits of extra effective storage.
- Demonstrates LoRC on top of both RTN and GPTQ — orthogonal to the base PTQ algorithm.

## Key Figures/Tables to Study
- **Figure 2:** sensitivity heatmap (model family × bit-width × W-only vs A-only) — useful when picking a PTQ recipe for a new model.
- **Table 5/6:** LoRC ablation across rank r = {4, 8, 16, 32} — diminishing returns after r=16 for 7B-class.

## Technical Details

### Low-Rank Compensation (LoRC)
Given quantized weight `Ŵ` (from RTN or GPTQ) and full-precision `W`, the residual is `E = W − Ŵ`. LoRC stores
```
E ≈ U V^⊤,   U ∈ R^{d_out × r},  V ∈ R^{d_in × r}
```
via truncated SVD `E = U_1 Σ_1 V_1^⊤` keeping the top-r singular vectors, with U = U_1 sqrt(Σ_1), V = V_1 sqrt(Σ_1).

At inference:
```
y = (Ŵ + U V^⊤) x = Ŵ x + U (V^⊤ x)
```
- The right-multiply `V^⊤ x` is rank-r and cheap (FP16).
- Memory cost: `r · (d_out + d_in) · 16` bits — for r=8 on a 4096×4096 layer this is ~0.06 bits/weight extra.

### Two scaling constraints
ZeroQuant-V2 proposes two scale-alignment tricks (for the W4A8 setting where weight and activation scales must align for INT GEMM):
1. Constraint A: tie weight scale per output channel to a power of 2 → cheap re-scale into the INT8 accumulator.
2. Constraint B: enforce that the activation scale and weight scale product is representable in the accumulator without overflow.

Both lose <0.05 ppl vs unconstrained W4A8.

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Base PTQ | RTN or GPTQ |
| Weight bits | 4 (FC) |
| Activation bits | 8 |
| LoRC rank r | 8 (large models), 16 (small) |
| SVD basis | computed once per layer on calibration data |

## Connections
- Predecessor: [[zeroquant]] (group-wise + per-token + LKD).
- FP-format successor: [[zeroquant-fp]] (FP8/FP4 instead of INT).
- Same low-rank residual idea applied to LoRA initialisation: [[loftq]].
- Orthogonal outlier-preserving cousin: [[spqr]].
