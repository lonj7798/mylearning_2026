<!-- scope: ZeroQuant-FP — first systematic FP8/FP4 LLM PTQ study; W4A8 with FP4 weights + FP8 activations
     deps: [[zeroquant]], [[zeroquant-v2]], [[fp8-formats-paper]]
     see-also: [[fp8-e4m3]], [[fp4-e2m1]], [[llm-fp4]], [[microscaling-formats]]
-->

# ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats
- **Core Insight:** For LLM PTQ at low bit-widths, floating-point formats (FP8 for activations, FP4 for weights) consistently beat integer formats of the same width because the wider dynamic range of FP absorbs outliers that INT clips.
- **Guideline:** If targeting H100 (FP8-native) or Blackwell (FP4-native) hardware, use FP8-E4M3 for activations and FP4-E2M1 for weights with per-tensor scale alignment; expect FP8-A to beat INT8-A by a clear margin on ≥1B-parameter models and FP4-W to roughly tie INT4-W with simpler kernels.
- **Authors:** Xiaoxia Wu, Zhewei Yao, Yuxiong He
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.09782
- **Relevant topics:** FP8 LLM PTQ, FP4 weights, H100, format comparison, LoRC

## Abstract
ZeroQuant-FP extends the ZeroQuant line to floating-point quantization formats motivated by NVIDIA H100 FP8 hardware. Across LLaMA / OPT / BLOOM at 1–66B, the authors show that (i) **FP8 activations outperform INT8 activations**, with the gap widening above 1B parameters, and (ii) **FP4 weights match or beat INT4 weights** while being more hardware-friendly on FP-native silicon. They combine FP4 weights with FP8 activations into a W4A8-FP recipe, add LoRC ([[zeroquant-v2]]) for residual quality, and propose two scaling constraints that align weight/activation scales so the GEMM stays in the FP accumulator path. The recipe matches W4A16 INT baselines while exploiting Hopper FP8 throughput.

## Key Contributions
- Side-by-side benchmark of INT vs FP at 4, 8 bits across model scales — the cleanest empirical answer to "FP8 or INT8?" of 2023.
- Establishes **FP8-A > INT8-A** is robust above ~1B params.
- Demonstrates **FP4 weight quant** as a viable alternative to GPTQ-INT4 for FP-native hardware (no integer GEMM needed).
- Combines FP4 weights + FP8 activations into a unified W4A8-FP pipeline, the predecessor of NVFP4 deployments.
- Two scale-alignment constraints that preserve INT-GEMM-style hardware fusion in the FP path.

## Key Figures/Tables to Study
- **Table 1/2:** FP8 vs INT8 activation perplexity on LLaMA/OPT — FP8 wins consistently above 1B.
- **Table 5:** FP4 vs INT4 weight ablation — within noise at 7B, FP4 wins at 30B+.
- **Figure 3:** LoRC ablation on W4A8-FP — adds ~0.3 ppl recovery at r=8.

## Technical Details

### FP formats used
- **E4M3** (4 exp + 3 mantissa, bias = 7): used for weights and activations at FP8. Range ≈ [-448, 448]; one NaN bit-pattern; no infinities. See [[fp8-e4m3]].
- **E5M2** (5 exp + 2 mantissa, bias = 15): used for gradients in FP8 training (not relevant here — ZeroQuant-FP is inference-only).
- **E2M1** (2 exp + 1 mantissa, FP4): used for weights at FP4. Represents the 16 values {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. See [[fp4-e2m1]].

### Quantization rule
For each weight tensor `W`:
```
s = max(|W|) / max_repr(format)
Ŵ = nearest_fp(W / s) · s
```
where `nearest_fp(·)` is round-to-nearest in the FP format's representable set.

For activations: same rule per-token (dynamic) at FP8, with `max_repr(E4M3) = 448`.

### Scaling constraints (the W4A8-FP twist)
For W4-E2M1 × A8-E4M3, the per-tensor scale product `s_w · s_a` must align with what the accumulator can absorb without overflow. ZeroQuant-FP enforces:
1. `s_w` constrained to a power-of-two so the rescale is a bit-shift.
2. `s_a · s_w` bounded so the partial sum stays in FP16/FP32 accumulator.

### LoRC integration
After FP quantization, compute residual `E = W − Ŵ_FP` and add a rank-r SVD correction (as in [[zeroquant-v2]]). r=8 typical; recovers ~0.2–0.4 ppl on 7B models.

### Hyperparameters
| Knob | Value |
|------|-------|
| Weight format | FP4-E2M1 |
| Activation format | FP8-E4M3 |
| Weight scale | per-tensor (power-of-2 constrained) |
| Activation scale | per-token, dynamic |
| LoRC rank | 8 |
| Hardware target | H100 (FP8 tensor cores) |

## Connections
- Predecessors: [[zeroquant]], [[zeroquant-v2]].
- Format references: [[fp8-e4m3]], [[fp8-e5m2]], [[fp4-e2m1]], [[fp8-formats-paper]].
- Companion FP4 LLM PTQ: [[llm-fp4]].
- Production format successor: [[mxfp4]] / [[nvfp4]] in [[microscaling-formats]].
- FP8 training (orthogonal axis): [[fp8-lm]], [[deepseek-v3-fp8]].
