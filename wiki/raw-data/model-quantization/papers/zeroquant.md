<!-- scope: ZeroQuant — INT8 PTQ with group-wise weight + token-wise activation + LKD distillation
     deps: [[int8]], [[straight-through-estimator]]
     see-also: [[zeroquant-v2]], [[zeroquant-fp]], [[smoothquant]], [[gptq]]
-->

# ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
- **Core Insight:** A practical W8A8 LLM PTQ doesn't need a fancy algorithm — what it needs is finer granularity (group-wise on weights, per-token on activations) plus a cheap layer-by-layer self-distillation pass to repair the dropped accuracy, all in minutes without training data.
- **Guideline:** As an INT8 baseline, use group-wise (G=128) symmetric weight quant + per-token dynamic activation quant; if W4 needed, add layer-wise knowledge distillation (LKD) from the FP teacher one transformer block at a time.
- **Authors:** Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He
- **Year:** 2022 (NeurIPS 2022)
- **URL:** https://arxiv.org/abs/2206.01861
- **Relevant topics:** PTQ granularity, token-wise activation quant, layer-wise KD, DeepSpeed integration

## Abstract
ZeroQuant is a hardware-friendly INT8 PTQ pipeline for BERT-scale and early GPT-scale transformers (up to GPT-NeoX-20B). It combines three pieces: (i) group-wise quantization for weights — one scale per group of consecutive output channels; (ii) per-token (dynamic) quantization for activations — one scale per token rather than per tensor; and (iii) Layer-by-Layer Knowledge Distillation (LKD), a data-free repair step that distills one transformer block at a time from the FP teacher using the previous block's quantized output as input. The result is up to 5.19× BERT-base and 4.16× GPT-J inference speedup against FP16, and W4A8 mixed-precision GPT-NeoX-20B with ~3× memory reduction.

## Key Contributions
- Establishes the standard PTQ recipe for LLM linears: **group-wise weight + per-token activation**, both symmetric.
- **LKD**: a memory-cheap distillation that only requires one transformer block resident at a time — works even without the original training data (use any short calibration text).
- Optimized INT8 GEMM kernels integrated into DeepSpeed-Inference; fuses dequant into subsequent ops to eliminate intermediate FP32 traffic.
- Demonstrates 4× memory reduction for GPT-NeoX-20B at W4-FC / W8-attn / A8 mixed config.

## Key Figures/Tables to Study
- **Figure 1:** the per-token-vs-per-tensor activation scale visualization — why per-tensor fails when token magnitudes vary 10×.
- **Algorithm in §3.3:** LKD pseudocode — one block at a time, MSE on the block output.
- **Table 4:** GPT-J/GPT-NeoX-20B W8A8 + W4A8 results — W8A8 essentially lossless.

## Technical Details

### Weight quantization (group-wise symmetric)
For weight `W ∈ R^{d_out × d_in}`, partition each row into groups of G consecutive input dims. For each group:
```
s = max(|W_g|) / (2^{b-1} − 1)
Ŵ_g = round(W_g / s) · s
```
- b = 8 (typical), G = 128–256.
- Per-row (G = d_in) is the default for W8; smaller G needed for W4.

### Activation quantization (token-wise dynamic)
For activation `X ∈ R^{B×S×d}`, compute one scale per token (per (b, s) index across the d channels) at runtime:
```
c_{b,s} = max_d |X_{b,s,d}| / 127
X̂_{b,s,d} = round(X_{b,s,d} / c_{b,s})
```
No calibration set is needed; the scale is recomputed every forward pass. Dequantization is folded into the next op via output scale `c_{b,s} · s_w`.

### Layer-wise Knowledge Distillation (LKD)
For each transformer block `f_i` in order:
1. Run the FP teacher and the partially-quantized student up to the input of block `i` on the same data.
2. Quantize block `f_i`'s weights; create student block `f̂_i`.
3. Optimize the quantization parameters (or fine-tune `f̂_i`) to minimize
   ```
   L_i = || f_i(h) − f̂_i(h) ||²
   ```
   on a few hundred calibration samples.
4. Only block `i`'s parameters need to be in memory at full precision → fits in a single GPU even for 20B-scale.

LKD is decisive at W4: pure RTN W4 loses 5–10 ppl, LKD recovers within 1 ppl of FP.

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Weight bits | 8 (FC + attn) or W4-FC / W8-attn mixed |
| Weight group | 128–256, symmetric |
| Activation bits | 8 |
| Activation scale | per-token dynamic |
| LKD samples | 128–1024 |
| LKD optimizer | Adam, lr 5e-7, ≤ 1 epoch |

## Connections
- Companion: [[zeroquant-v2]] (LoRC for low-rank compensation), [[zeroquant-fp]] (FP8/FP4 variant).
- Activation-difficulty-migration alternative: [[smoothquant]].
- Hessian-based weight-only alternative: [[gptq]].
- DeepSpeed inference integration: [[tensorrt-llm-quant]] equivalent on Microsoft stack.
- Baseline for KV-cache quant studies in 2023: [[flexgen]], [[kvquant-2023]].
