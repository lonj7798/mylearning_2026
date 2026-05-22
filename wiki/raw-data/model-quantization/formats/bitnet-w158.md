<!-- scope: BitNet ternary {-1, 0, 1} weight format; 1.58 bits/weight = log2(3)
     deps:
     see-also: [[bitnet]], [[bitnet-b158]], [[bitnet-a48]], [[era-of-1bit-llms]]
-->

# BitNet W1.58 (Ternary {-1, 0, +1} Weight Format)
- **Core Insight:** Restricting LLM weights to three values {-1, 0, +1} — encoding each weight in log₂(3) ≈ 1.58 bits — converts every matmul into pure adds and subtracts (no multiplies), and the BitNet b1.58 paper empirically demonstrates that LLMs trained from scratch at this constraint match FP16 perplexity at ≥ 3B parameters.
- **Guideline:** Use ternary weights *only with end-to-end pretraining from scratch (QAT)* — naïve post-training quantization to ternary catastrophically fails; you must train with the ternary constraint and the BitLinear straight-through estimator from step 1.
- **Authors:** Shuming Ma et al. (Microsoft Research, BitNet b1.58); originally introduced in BitNet (Wang et al. 2023, binary); extended to ternary by Ma 2024
- **Year:** 2024 (b1.58 paper); 2023 (original BitNet binary)
- **URL:** https://arxiv.org/abs/2402.17764 (BitNet b1.58, "The Era of 1-bit LLMs"); https://arxiv.org/abs/2310.11453 (BitNet binary)
- **Relevant topics:** ternary weights, BitNet, 1-bit LLM, multiplication-free inference, native QAT

## Abstract
BitNet b1.58 constrains every weight in a Transformer's linear layers to the set {-1, 0, +1}, encoding each in 1.58 = log₂(3) bits. The forward pass is implemented by a custom **BitLinear** layer that replaces matrix multiplication with integer add/subtract over INT8 activations; gradients flow through the quantization step via a straight-through estimator with absmean weight scaling. Trained from scratch at 3B parameters on 100B tokens, the b1.58 model matches FP16 perplexity and downstream task accuracy of standard Llama-class models of the same size — while reducing weight memory ~10× and replacing multiplications with adds in the hot path. This is the strongest empirical evidence for sub-2-bit LLM inference at scale.

## Key Contributions
- Ternary {-1, 0, +1} weight constraint; 1.58 bits/weight = log₂(3).
- Multiplication-free matmul: every operation is ±activation accumulated into an integer register.
- BitLinear layer + absmean scaling + STE gradient for end-to-end pretraining.
- Demonstrates parity with FP16 baseline at 3B+ scale on Llama-like architectures.
- Lower energy / area / latency vs FP8 by ~10× at iso-quality.

## Key Figures/Tables to Study
- **BitNet b1.58 vs Llama-FP16 perplexity scaling plot** — the canonical figure showing parity from 3B onward.
- **Add-only inference pseudocode** — single-paragraph illustration that multiplies disappear.

## Technical Details

### Weight representation
Each weight W_ij ∈ {-1, 0, +1}. Bit budget per weight:
```
H({−1, 0, +1}) ≤ log₂(3) ≈ 1.58 bits
```
Storage layout: pack 5 ternary values into 8 bits (3^5 = 243 ≤ 256) or use simpler 2-bit per weight with one unused code (2 bits/weight, slightly wasteful but byte-aligned).

### BitLinear forward pass
For input X ∈ ℝ^{B × N} and weight W ∈ {-1, 0, +1}^{M × N}:
```
1. Quantize X to INT8 per-token: X_q = round(X / s_x) where s_x = max(|X|) / 127
2. Compute Y_int = X_q · Wᵀ                         (integer adds/subtracts only)
3. Dequant: Y = s_x · Y_int / scale_w                (multiply by accumulated FP scale)
```
The **inner matmul is multiplication-free** because W ∈ {-1, 0, +1}:
```
(X_q · Wᵀ)_im = Σ_n X_q[i,n] · W[m,n]
              = Σ_{n: W[m,n] = +1} X_q[i,n]   −   Σ_{n: W[m,n] = −1} X_q[i,n]
```

### Weight quantization (during training)
Absmean scaling:
```
scale_w = mean(|W_fp|)        (per-tensor scalar)
W_scaled = W_fp / scale_w
W_int = clip(round(W_scaled), −1, +1)
```

### Straight-through estimator (STE)
Forward uses quantized W; backward treats quantization as identity:
```
∂L/∂W_fp = ∂L/∂W_int        (gradient flows through unchanged)
```
This is the standard [[straight-through-estimator]] device.

### Activation quantization
Activations are 8-bit symmetric per-token:
```
s_x = max(|X|) / 127       (per-token across hidden dim)
X_q = round(X / s_x)        (INT8)
```

### Why exactly 1.58 bits (not 2)?
Information content of 3 symbols = log₂(3) = 1.585 bits. Optimal entropy coding of long ternary sequences asymptotically uses log₂(3) bits/symbol; the 2-bit per-weight on-disk layout wastes 0.42 bits/weight that could be recovered by group-packing (5 ternary in 1 byte = 1.6 bits/weight, near-optimal).

### Comparison to binary BitNet
| | BitNet binary (2023) | BitNet b1.58 (2024) |
|---|---|---|
| Weight set | {-1, +1} | {-1, 0, +1} |
| Bits/weight | 1 | 1.58 |
| Sparsity | none | natural (W=0 → no add) |
| Quality vs FP16 | lags at small scale | parity at ≥3B |

The extra "0" code in ternary lets the model express sparsity, which empirically matters more than the saved bit.

### Hardware implications
- **Mainstream GPU (CUDA, ROCm)**: no native ternary matmul; emulate via INT8 (pack ternary, do INT8 multiply with W ∈ {-1, 0, +1}). Effective speedup vs FP16 mostly from memory bandwidth (10× weight compression).
- **Specialized accelerators**: Microsoft, Intel, Cerebras have demonstrated dedicated ternary-multiply-free units; ~10× energy efficiency vs FP8.
- **CPU**: pure adds map well to vector adds on AVX-512/SVE; high throughput at low power.

### Limitations
- **Pretraining from scratch required**: cannot convert an existing FP16 model to b1.58 via PTQ without massive quality loss.
- **3B+ parameter floor**: b1.58 matches FP16 only from 3B onward; smaller models still lag.
- **Activations at INT8**: KV cache and intermediate activations still use 8-bit storage; not "true 1-bit" end-to-end.
- **No autograd-native support** in PyTorch — BitLinear is a custom kernel.

### Extension: BitNet a4.8
[[bitnet-a48]] (2024) adds 4-bit activations on top of 1.58-bit weights, pushing toward fully-multiplication-free transformers.

### Packing layout
Most efficient: 5 ternary values per byte (3^5 = 243 codes used out of 256). Decoder uses a 243-entry LUT or arithmetic decode. Some implementations use 2 bits/weight (1 wasted code) for simplicity at the cost of 0.4 bits/weight overhead.

## Connections
- [[bitnet]] — original binary {-1, +1} BitNet (2023).
- [[bitnet-b158]] — the 2024 ternary paper introducing this format.
- [[bitnet-a48]] — 4-bit activations on top of 1.58-bit weights.
- [[era-of-1bit-llms]] — survey-style consolidation of BitNet results.
- [[straight-through-estimator]] — gradient device used in BitLinear.
- [[bnn]] / [[xnor-net]] — pre-LLM binary weight precursors.
