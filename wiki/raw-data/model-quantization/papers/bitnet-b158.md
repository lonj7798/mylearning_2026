<!-- scope: BitNet b1.58 — ternary {-1, 0, 1} weights via per-tensor absmean rounding; the "1.58-bit" LLM
     deps: [[bitnet]], [[straight-through-estimator]]
     see-also: [[bitnet-a48]], [[era-of-1bit-llms]], [[onebit]], [[bitnet-w158]]
-->

# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)
- **Core Insight:** Adding a single extra state — zero — to BitNet's {−1, +1} weights, giving ternary {−1, 0, +1}, raises the per-weight cost from 1 bit to log₂(3) ≈ 1.58 bits but matches FP16 perplexity from scratch starting at the 3B-parameter scale; the zero state lets the model express "this connection is off" without consuming a sign bit, dramatically improving expressivity at no inference cost.
- **Guideline:** When pretraining at ≥3B parameters and willing to pay the b1.58 training-recipe cost (5–10% higher pretrain perplexity at intermediate scales), use ternary BitLinear with per-tensor absmean quantization `W̃ = RoundClip(W / (γ+ε), −1, +1)` and the rest of the BitNet recipe; the resulting model matches FP16 quality with ~10× inference memory savings and integer-only matmul.
- **Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.17764
- **Relevant topics:** ternary weights, 1.58-bit LLM, absmean quantization, FP16-equivalent quality from scratch, integer-only matmul

## Abstract
BitNet b1.58 extends BitNet ([[bitnet]]) by quantizing weights to **ternary {−1, 0, +1}** instead of binary {−1, +1}. The third state, zero, costs only log₂(3) ≈ 1.58 bits per weight (when packed efficiently) but recovers the missing expressivity of pure binary nets. Trained from scratch on standard pretraining corpora using a per-tensor absmean rounding rule, b1.58 reaches **FP16-equivalent perplexity starting at 3B parameters**; at 700M and 1.3B it lags by 1–3 ppl, at 3B+ it ties, and at 13B+ it slightly *exceeds* FP16 because the discrete weight space acts as regularization. With integer-only matmul and zero-skipping, inference memory drops ~10× and latency ~4× vs FP16 on commodity hardware.

## Key Contributions
- **Ternary weights** via per-tensor absmean rounding — a one-line change from BitNet.
- Demonstrates **FP16-equivalent quality from-scratch** at 3B+ parameters, the breakthrough that triggered the "1-bit LLM era" framing.
- New scaling-law recipe for ternary pretraining.
- Practical packing: 5 ternary weights fit into 8 bits (3⁵ = 243 < 256), giving the 1.6 bits/weight memory floor.
- Released model checkpoints (BitNet b1.58 700M, 1.3B, 3B; later 2B/4B in follow-up).

## Key Figures/Tables to Study
- **Figure 1:** the b1.58 vs FP16 perplexity-vs-parameter-scale curves — they cross at 3B.
- **Table 1:** zero-shot accuracy on standard suite at 700M, 1.3B, 3B — parity at 3B.
- **Table 3:** inference memory and latency vs FP16 — 10× and ~4× respectively.

## Technical Details

### Ternary weight quantization (absmean rule)
For each layer's latent FP weight `W ∈ R^{d_out × d_in}`:
```
γ = (1 / (d_out · d_in)) · Σ_{i,j} |W_{i,j}|    # absmean (scalar, per-tensor)
W̃ = RoundClip( W / (γ + ε), −1, +1 )
       where RoundClip(x, a, b) = max(a, min(b, round(x)))
```
The result `W̃_{i,j} ∈ {−1, 0, +1}`. The per-tensor scale `γ` is the absmean — provably the minimum-MSE 1-level magnitude under absolute-error and a tight choice for the ternary code as well.

Forward: `y = γ · (W̃ · x̃)` where `x̃` is the INT8-quantized activation (same per-token absmax as in [[bitnet]]).

### Why log₂(3) = 1.58
A ternary cell has 3 states. Information-theoretic minimum storage = log₂(3) ≈ 1.58 bits. Practical packing: pack 5 ternary weights into 8 bits (3⁵ = 243 codes ≤ 256) → 1.6 bits/weight. This is the "1.58-bit LLM" framing.

### Backward (STE)
```
∂L/∂W ≈ ∂L/∂W̃        (straight-through through RoundClip)
```
Latent FP weight is updated by the optimizer; ternary is recomputed each forward.

### Architecture and training recipe
- Same Llama-style backbone as BitNet (SwiGLU, RoPE, RMSNorm).
- BitLinear replaces every Linear.
- SubLN unchanged from BitNet.
- Trained from scratch with the same corpora and tokenizer as the FP16 baseline used in scaling-law comparisons.

### Scaling behavior (the headline result)
| Params | b1.58 ppl | FP16 ppl | Gap |
|--------|-----------|----------|-----|
| 700M | ~ +1.5 | baseline | b1.58 worse |
| 1.3B | ~ +1.0 | baseline | b1.58 worse |
| **3B** | **same** | baseline | parity |
| 13B | slightly better | baseline | b1.58 ≥ FP16 |
| 70B (extrapolated in follow-up) | better | baseline | b1.58 > FP16 |

(Numbers approximate; exact values in Table 1 of the paper.)

### Inference
- Weight stored as ternary, 5-per-byte packing.
- Activation INT8 per-token.
- Matmul: integer multiply-accumulate with output rescale by `γ · γ_x / 127`.
- **Zero-skipping**: any 0-weight contributes nothing → can be skipped in custom kernels, gaining ~30% sparsity speedup empirically.
- Custom kernels described in follow-up bitnet.cpp.

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Weight states | {−1, 0, +1} |
| Per-tensor scale | absmean γ |
| Effective bits/weight | log₂(3) ≈ 1.58 (1.6 with 5-per-byte packing) |
| Activation bits | 8 (per-token absmax) |
| Optimizer | AdamW |
| Backward | STE through RoundClip |
| Pretrain data | same as FP16 baseline |
| Crossover scale | ~3B parameters |

## Connections
- Direct predecessor: [[bitnet]] (binary {−1, +1} only).
- Activation-precision extension: [[bitnet-a48]] (4-bit activations on top of b1.58 weights).
- Survey companion: [[era-of-1bit-llms]].
- 1-bit weight via SVD alternative: [[onebit]].
- Format spec: [[bitnet-w158]].
- 2B/4B official models: [[bitnet-b158-2b]].
- Scaling-law follow-up: [[bitnet-scaling-laws]].
- Lab summary: [[microsoft-bitnet]].
