---
chapter: ch-10
course: model-quantization
phase: read
excerpt_of: "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models"
source_url: https://arxiv.org/abs/2308.13137
created_at: "2026-05-21"
---

# Excerpt: OmniQuant — LWC + LET via block-wise gradient PTQ

**Authors:** Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo
**Year:** 2023 (ICLR 2024)
**URL:** https://arxiv.org/abs/2308.13137
**Raw-data source:** [[raw-data/omniquant]]

---

## Learnable Weight Clipping (LWC)

Per output channel `j`, two learnable bounds `γ_j, β_j ∈ R`:

```math
W_{\text{clip}} = \mathrm{clip}\Big(W,\ \sigma(\beta) \cdot \min(W),\ \sigma(\gamma) \cdot \max(W)\Big)
```

```math
\hat{W} = \mathrm{round}\Big(\frac{W_{\text{clip}} - z}{s}\Big) \cdot s + z
```

- `σ(·)` = sigmoid → bounds stay in (0, 1) of per-channel range (can contract, never expand).
- `s, z` derived from `W_clip`'s post-clip max/min.
- Only `γ, β` trainable: **2 params per output channel** per weight matrix.
- Backward: STE through `round`, normal grad through `clip`.

Intuition: outlier weights that don't carry output mass get clipped away; the INT-k grid spacing tightens on the rest.

---

## Learnable Equivalent Transformation (LET)

Per input channel `j`, learnable scale `s_j > 0` (softplus) and shift `b_j ∈ R`:

```math
\hat{X} = (X - b) \oslash s, \qquad \hat{W} = \mathrm{diag}(s) \cdot W
```

Identity:

```math
\hat{X} \cdot \hat{W} = (X - b)/s \cdot \mathrm{diag}(s) \cdot W = X \cdot W - b \cdot W
```

The `-b·W` term is absorbed as an additive bias.

**Placement.** LET is applied only where it matters most:
- input of **qkv** projections
- input of **FFN-up**

Other inputs use identity (`s = 1, b = 0`).

**Why this beats AWQ / SmoothQuant:** AWQ has one scalar α per layer (`s = mean|X|^α`); SmoothQuant has closed-form `s = max|X|^α / max|W|^(1−α)`. Both have `b = 0` and force `s` to a parametric form. LET frees both.

---

## Block-wise training objective

For each transformer block `f_i` in order:

```math
\mathcal{L}_i = \big\lVert f_i^{FP}(h_i) - f_i^{\text{quant}}(h_i;\ \gamma, \beta, s, b) \big\rVert^2
```

- `h_i` = output of the previous (already-quantised) block on calibration data.
- Only `(γ, β, s, b)` trainable; original weights stay FP during forward; quantization is *simulated* (`dequant(quant(·))`).
- STE for the round backward.
- 20 epochs per block, AdamW lr=5e-3, batch=1.

---

## Memory profile (the engineering trick)

At any moment, only block `i`'s FP teacher + quantised student + ~2 calibration batches are resident → **≤ 40 GB even for LLaMA-2-70B**. This is what lets OmniQuant fit on a single A100-40G.

Contrast with full QAT, which needs the entire model + activations + optimizer states in memory at once.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Supported configs | W4A4, W6A6, W4A16, W3A16, W2A16 |
| Calibration | 128 sequences |
| Epochs per block | 20 |
| Optimizer | AdamW, lr 5e-3 (quant params only) |
| LWC placement | every weight matrix |
| LET placement | qkv-in, FFN-up-in |
| Wall-clock LLaMA-2-7B (W4A4) | ~1 hr on A100-40G |
| Wall-clock LLaMA-2-70B (W4A4) | ~16 hr on A100-40G |

---

## Why this is the first viable W4A4 PTQ on LLaMA

Pre-OmniQuant:
- AWQ: weight-only (A16). One scalar α can't compensate for A4 activation noise.
- SmoothQuant: A8 only. The closed-form `s` cannot absorb the additional difficulty A4 imposes.
- LLM-QAT: works at A4 but needs full QAT (days, 100k self-generated sequences).

OmniQuant lives between: **PTQ wall-clock with QAT-quality accuracy** at W4A4. The block-wise decomposition trades the joint optimum for tractability; the lost optimality is small because each transformer block's output is a strong proxy for downstream loss.

---

## Connections

- Generalises: [[smoothquant]] (diagonal scale, closed-form) and [[awq]] (one scalar α grid-searched).
- Block-wise reconstruction lineage: [[brecq]] (pre-LLM CNN), [[bitdistiller]].
- Full-affine extension: [[affinequant]] (gradual mask schedule).
- Kronecker affine + flatness target: [[flatquant]].
- Rotation-based descendants: [[quarot]], [[spinquant]], [[duquant]].
- Sub-2-bit fine-tuning successor: [[pv-tuning]].
