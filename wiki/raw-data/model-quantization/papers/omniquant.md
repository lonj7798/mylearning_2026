<!-- scope: OmniQuant — block-wise learned weight clipping + learned equivalent transformation for PTQ
     deps: [[smoothquant]], [[awq]], [[brecq]]
     see-also: [[affinequant]], [[quarot]], [[spinquant]]
-->

# OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models
- **Core Insight:** SmoothQuant's diagonal `diag(s)` and AWQ's grid-searched per-channel α are both special cases of a richer learnable equivalent transformation; if you train (s, b) for the diagonal + bias and the per-channel clipping thresholds (γ, β) jointly with a block-wise MSE objective, you can match QAT-quality W4A4 PTQ in 1–16 hours on a single A100.
- **Guideline:** When 0.5 ppl matters and a few GPU-hours are available, use OmniQuant with `LWC` (learnable per-channel clip) + `LET` (learnable diag-scale + shift), train one transformer block at a time with MSE against the FP teacher block, ~20 epochs, AdamW lr 5e-3 on the quant parameters only.
- **Authors:** Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo
- **Year:** 2023 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2308.13137
- **Relevant topics:** learnable equivalent transformation, learned clipping, block-wise PTQ, W4A4 LLM, gradient-based PTQ

## Abstract
OmniQuant unifies the activation-aware family ([[smoothquant]], [[awq]]) under a learnable framework with two new modules: (1) Learnable Weight Clipping (LWC) that learns per-channel upper/lower clipping bounds via a sigmoid parameterisation so the quantization grid covers the most useful dynamic range; (2) Learnable Equivalent Transformation (LET) that generalises SmoothQuant's `diag(s)` to `diag(s)·X + b` with both scale and shift learned. Trained block-wise with MSE against the FP teacher block (1 transformer block resident at a time), OmniQuant supports W4A4 / W6A6 / W4A16 / W3A16 / W2A16 on LLaMA-2 7–70B in 1–16 hours on a single A100-40G with 128 calibration samples. It is the first PTQ that holds W4A4 LLM accuracy.

## Key Contributions
- **LWC** — sigmoid-parameterised per-channel clipping bounds learned by gradient.
- **LET** — learnable per-channel diagonal scale + shift `(s, b)` generalising SmoothQuant and AWQ.
- Block-wise training: one transformer block at a time, MSE loss, only quant parameters trainable → fits on a single A100 even at 70B.
- First PTQ to achieve viable **W4A4** on LLaMA-class models.
- Maintains PTQ wall-clock (1–16 hr) instead of full QAT (days).

## Key Figures/Tables to Study
- **Figure 2:** the LWC/LET module diagram — how the learnable parameters slot into a transformer block.
- **Table 2/3:** W4A4 / W6A6 / W4A16 results vs SmoothQuant + GPTQ + AWQ — OmniQuant is the only one that survives W4A4 on LLaMA.

## Technical Details

### Learnable Weight Clipping (LWC)
Per output channel j, learnable upper/lower bounds `γ_j, β_j ∈ R`:
```
W_clip = clip(W, σ(β) · min(W), σ(γ) · max(W))
Ŵ = round((W_clip − z) / s) · s + z
```
- σ(·) is sigmoid so bounds stay in (0,1) of the per-channel range.
- s, z derived from `(W_clip)`'s max/min.
- Only γ, β are trainable (2 params per channel).

### Learnable Equivalent Transformation (LET)
Per input channel j, learnable scale `s_j > 0` and shift `b_j ∈ R`:
```
X̂ = (X − b) ⊘ s
Ŵ = diag(s) · W
b̂ = (added back as bias term: + W · diag(s) · (b ⊘ s))
```
- Mathematically equivalent: `(X − b)/s · diag(s)·W = X·W − b·W`.
- `s` is parameterised via softplus to stay positive.
- LET is applied only on the channels most prone to outliers (typically the input of qkv and FFN-up projections).

### Block-wise training objective
For each transformer block `f_i` in order:
```
L_i = || f_i^{FP}(h_i) − f_i^{quant}(h_i; γ, β, s, b) ||²
```
- `h_i` = output of the previous (already-quantized) block on calibration data.
- Only (γ, β, s, b) are trainable; original weights and activations FP during forward, quantization simulated via straight-through round.
- ~20 epochs per block, AdamW lr 5e-3 on quant params, batch size 1.

### Memory footprint
At any time only block `i`'s FP teacher + quantized student + ~2 calibration batches resident → ≤40 GB even for 70B. This is what lets OmniQuant run on a single A100.

### Hyperparameters
| Knob | Value |
|------|-------|
| Supported configs | W4A4, W6A6, W4A16, W3A16, W2A16 |
| Calibration samples | 128 |
| Epochs per block | 20 |
| Optimizer | AdamW, lr 5e-3 (quant params only) |
| LET placement | qkv-in, FFN-up-in |
| LWC placement | every weight matrix |
| Wall-clock LLaMA-2 70B | ~16 hr on A100-40G |

## Connections
- Generalises: [[smoothquant]] (diagonal scale only, no learning) and [[awq]] (one scalar α grid-searched).
- Same block-wise reconstruction lineage: [[brecq]] (pre-LLM), [[bitdistiller]].
- Affine-transformation extension (full affine not just diagonal): [[affinequant]].
- Rotation-based descendants that supersede diagonal LET: [[quarot]], [[spinquant]], [[duquant]], [[flatquant]].
- Sub-2-bit + fine-tuning successor: [[pv-tuning]].
