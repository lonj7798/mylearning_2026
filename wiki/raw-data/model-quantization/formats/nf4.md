<!-- scope: Dettmers NormalFloat-4 — 4-bit non-uniform code with quantile-based reconstruction levels for Gaussian weights
     deps: [[lloyd-max-quantizer]], [[information-theoretic-bounds]], [[int4]]
     see-also: [[qlora]], [[bitsandbytes-nf4]], [[af4]]
-->

# NF4 (NormalFloat-4) — Quantile-Based 4-bit Code for Gaussian Weights
- **Core Insight:** Pre-trained LLM weights are approximately zero-mean Gaussian after per-block normalization; the *information-theoretically optimal* 4-bit code for this distribution puts 16 reconstruction levels at the 16-quantiles of the standard normal CDF rather than uniformly — yielding ~0.5 PPL improvement over INT4 at the same bit-width with no extra hardware cost.
- **Guideline:** Use NF4 (not INT4) when quantizing LLM weights *and* you know they are unimodal-ish post-normalization; it's the default in QLoRA / bitsandbytes 4-bit and gives free quality over RTN INT4.
- **Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (UW)
- **Year:** 2023 (introduced in the QLoRA paper)
- **URL:** https://arxiv.org/abs/2305.14314 (QLoRA paper, §3.1 NormalFloat); bitsandbytes source: https://github.com/TimDettmers/bitsandbytes/blob/main/csrc/kernels.cu
- **Relevant topics:** non-uniform quantization, quantile code, Gaussian prior, QLoRA, 4-bit

## Abstract
NF4 (NormalFloat 4-bit) is a 16-level quantization code where the 16 reconstruction values are placed at equally-spaced quantiles of the standard normal distribution N(0,1), restricted to the symmetric range. The motivation: after per-block absmax normalization, LLM weight blocks are well-modelled as i.i.d. N(0,1) samples; the MSE-optimal 4-bit code for this prior is essentially the Lloyd-Max quantizer for N(0,1). Introducing NF4 as the storage code for QLoRA's base-model weights, Dettmers et al. showed it consistently outperforms RTN INT4 and FP4 E2M1 by ~0.3–0.5 perplexity on Llama-class models. NF4 is the *default* 4-bit quantizer in bitsandbytes and the foundation of every QLoRA-fine-tuned LoRA adapter in HuggingFace.

## Key Contributions
- 4-bit non-uniform code with quantile-based reconstruction values.
- Demonstrates ~0.3–0.5 PPL improvement over INT4 / FP4 at the same 4-bit budget.
- Per-block (block-size 64) double-quantized scales: scale of scales also quantized, saving ~0.3 bits/weight.
- Adopted as default 4-bit format in bitsandbytes; deployed by every QLoRA / PEFT user.

## Key Figures/Tables to Study
- **NF4 vs FP4 vs INT4 reconstruction-value plot**: shows NF4's denser packing near zero (where Gaussian mass is) and sparser packing at the tails.
- **QLoRA Table 4**: NF4 perplexity vs FP4 / INT4 at the same bit budget on Llama / OPT — NF4 wins consistently.

## Technical Details

### Reconstruction values
NF4 has 16 values, symmetric around 0 (technically 15 non-zero + zero), restricted to a normalized range so |max| = 1. The canonical values (from QLoRA paper Table 14, bitsandbytes source):

```
[-1.0,
 -0.6961928009986877,
 -0.5250730514526367,
 -0.39491748809814453,
 -0.28444138169288635,
 -0.18477343022823334,
 -0.09105003625154495,
  0.0,
  0.07958029955625534,
  0.16093020141124725,
  0.24611230194568634,
  0.33791524171829224,
  0.44070982933044434,
  0.5626170039176941,
  0.7229568362236023,
  1.0]
```

These are *empirically* derived by taking 16 equal-quantile splits of the symmetric standard normal CDF (with adjustment so the extreme values are ±1 exactly to match per-block absmax normalization).

### Construction recipe
1. Compute the symmetric N(0,1) quantile function Q(p) = √2 · erf⁻¹(2p − 1).
2. Take 8 quantile points in (0, 1] on the positive side (offset / shifted so the extreme = 1.0).
3. Mirror across zero for the negative side; include 0.0.
4. Normalize so |max| = 1 (so per-block absmax scale maps to ±1).

### Per-block scaling
Given a block of G = 64 raw weights w_1, …, w_G:
```
s = max(|w_i|)                         (absmax scale, FP32 or FP16)
ŵ_i = w_i / s                          (normalized to [−1, +1])
q_i = argmin_k |NF4_levels[k] − ŵ_i|    (encode to 4-bit index)
```
Reconstruction: `w_recon = s · NF4_levels[q_i]`.

### Double quantization
The per-block FP32 scales themselves are quantized: 256 scales (one per block) are grouped, an "outer" FP32 scale is stored per 256 inner scales, and the 256 inner scales are quantized to 8-bit. This saves ~0.37 bits/weight overhead at minimal quality cost.

### Bit budget (with double quant, block-64)
```
4 (weight) + 8/64 (inner scale FP8) + 32 / (64·256) (outer scale FP32)
= 4 + 0.125 + 0.002
≈ 4.127 effective bits/weight
```

### Why NF4 beats INT4
- INT4 reconstruction values: uniform spacing in [−1, +1] → 0, ±1/7, ±2/7, …, ±1. Wastes resolution at tails (low Gaussian mass) and too coarse near zero (high Gaussian mass).
- NF4 spacing: dense near 0 (where p(w) is largest), sparse at the tails — matches the Gish-Pierce p^{1/3} optimal density up to the *symmetric* normalization constraint.
- Empirical gain: ~0.3–0.5 PPL on Llama-class models at 4-bit; larger gain at lower bit-widths.

### Connection to Lloyd-Max
NF4 is *not* exactly the Lloyd-Max 16-level quantizer for N(0,1) — Dettmers used quantile spacing (which is Voronoi-suboptimal) for analytical simplicity. The true Lloyd-Max levels differ slightly; subsequent work ([[af4]] and academic followups) tabulates better-tuned codes.

### Comparison with FP4 E2M1
| | NF4 | FP4 E2M1 |
|---|---|---|
| Levels | 16, quantile-spaced | 16, log-spaced |
| Best for | unimodal Gaussian-ish weights | log-magnitude data |
| Hardware | software dequant only | Blackwell native |
| Storage | 4 bits + per-block scale | 4 bits + per-block scale |
| LLM weight PPL | best | second |

### Hardware support
- **Pure software** (bitsandbytes CUDA kernels, HuggingFace 4-bit): dequant to BF16 → BF16 tensor-core matmul.
- **No native NF4 tensor cores** — the non-uniform LUT prevents direct integer multiply.
- Modern Marlin / Machete kernels handle NF4 the same way: 16-entry LUT lookup → BF16 → tensor core.

### Failure cases
- Heavy-tailed weight distributions (some early transformer layers, RMSNorm gain parameters): NF4's symmetric Gaussian assumption breaks; INT4 with per-group scale may match or exceed.
- Activations: NF4 is *not* used for activations (activations are not Gaussian, post-GeLU/SiLU is heavy-tailed positive).

## Connections
- [[lloyd-max-quantizer]] — NF4 is a quantile-spaced approximation of the Lloyd-Max code for N(0,1).
- [[information-theoretic-bounds]] — Gish-Pierce density p^{1/3} is the theoretical optimum NF4 approximates.
- [[int4]] — uniform 4-bit alternative; NF4 wins by ~0.5 PPL on LLM weights.
- [[af4]] — abstract / asymmetric float 4 variants that further refine NF4.
- [[qlora]] — the paper that introduced NF4.
- [[bitsandbytes-nf4]] — production implementation.
- [[companding-mu-law]] — companding theory NF4 instantiates for the Gaussian distribution.
