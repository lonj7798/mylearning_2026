---
chapter: ch-10
course: model-quantization
phase: read
excerpt_of: "RPTQ: Reorder-based Post-training Quantization for Large Language Models"
source_url: https://arxiv.org/abs/2304.01089
created_at: "2026-05-21"
---

# Excerpt: RPTQ — channel reordering as orthogonal complement to LET

**Authors:** Zhihang Yuan, Lin Niu, Jiawei Liu, Wenyu Liu, Xinggang Wang, Yuzhang Shang, Guangyu Sun, Qiang Wu, Jiaxiang Wu, Bingzhe Wu
**Year:** 2023
**URL:** https://arxiv.org/abs/2304.01089
**Raw-data source:** [[raw-data/rptq]]

---

## The reframing

The activation-quant problem is not "isolated outliers" but **inter-channel range variance** — different hidden dimensions have systematically different magnitudes. Per-tensor quant burns its dynamic range on the highest-range channels; per-token quant doesn't address channel-axis variance.

Fix: **cluster channels by per-channel max, assign one scale per cluster**.

---

## Channel clustering

For input activation `X ∈ R^{T × C_in}`, compute per-channel statistic over calibration:

```math
r_j = \max_t |X_{t, j}|
```

Run k-means on `{r_j}` with K = 32–64 clusters. Partition `{1, ..., C_in} = ⊔_k S_k`. Channels in the same cluster have similar magnitude → per-cluster scale fits all of them tightly.

---

## Permutation π

Define `π` so each cluster occupies a contiguous index range: `S_1 → [0, |S_1|)`, `S_2 → [|S_1|, |S_1|+|S_2|)`, etc.

---

## Per-cluster quantization

```math
s_k = \frac{\max_{j \in S_k} r_j}{2^{b-1} - 1}, \qquad \hat{X}_{\cdot, S_k} = \mathrm{round}(X_{\cdot, S_k} / s_k) \cdot s_k
```

One scale per cluster (K = 32–64 scales total) — far fewer than per-channel, but range-matched within cluster.

---

## Folding π into the graph (zero runtime overhead)

For the canonical `LayerNorm(γ, β) → Linear(W) → ... → Linear(V)` pattern:

- Reorder LayerNorm affine: `γ ← γ[π]`, `β ← β[π]`.
- Reorder Linear input dim: `W ← W[π, :]`.
- For the output side (producer Linear → consumer Linear → activation quant), apply π on the producing Linear's output channels: `W ← W[:, π_out]`.

The permutation is **a compile-time index remap**, not a runtime gather/scatter.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Clusters K | 32–64 |
| Activation bits | 3 (target), 4 / 8 (easier) |
| Weight bits | 4 (GPTQ companion) |
| Calibration | 512 sequences × 512 tokens |
| Cluster algorithm | k-means on `r_j = max_t|X_{t,j}|` |
| Symmetric / asymmetric | symmetric |

---

## Why RPTQ is the *complement* to OmniQuant's LET

| Aspect | OmniQuant LET | RPTQ |
|---|---|---|
| What changes | activation **values** via `(X - b)/s` | activation **grouping** for scale assignment |
| Mechanism | per-channel learned scale + shift | k-means clustering + permutation |
| Trainable? | yes (gradient through STE) | no (one-pass calibration) |
| Runtime overhead | zero (folded into LN, weights) | zero (permutation folded into ops) |
| Composes with the other? | **yes** — first cluster, then learn per-cluster `(s, b)` |

In 2024 this composition becomes Atom and QServe — W4A4-KV4 production stacks that reorder + scale + quantize in one pipeline (see [[ch-14]]).

---

## Empirical reach

First viable **A3** (3-bit activation) PTQ on **OPT-175B**:

| Config | OPT-175B PPL |
|---|---|
| FP16 | 8.34 |
| W4A8 (RPTQ) | 8.84 |
| W4A4 (RPTQ) | 11.05 |
| **W4A3** (RPTQ, K=64) | within ~2 PPL of W4A4 |

---

## Connections

- The activation-outlier-as-channel-variance reframing: extends [[llm-int8]], [[smoothquant]].
- Per-cluster scale lineage: [[atom]] (W4A4 + KV4 with sub-channel reorder).
- Activation-aware weight quant cousin: [[awq]].
- Rotation-based successors that obviate clustering by Gaussianising activations: [[quarot]], [[spinquant]], [[duquant]].
- Learnable equivalent transformation: [[omniquant]] LET (composes with RPTQ).
