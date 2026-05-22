---
chapter: ch-05
course: model-quantization
phase: read
excerpt_of: "Data-Free Quantization Through Weight Equalization and Bias Correction (Nagel et al. 2019)"
source_url: https://arxiv.org/abs/1906.04721
arxiv: 1906.04721
created_at: "2026-05-21"
---

# Excerpt: DFQ — the equivalent-transformation seed

**Authors:** Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling
**Year:** 2019 (ICCV)
**Raw-data source:** [[raw-data/classics/data-free-quantization]]

---

## The positive-scaling invariance (the load-bearing identity)

For consecutive layers `W_i → ReLU → W_{i+1}` and any positive per-output-channel diagonal `S ∈ ℝ₊^{C}`:

```math
W_{i+1}\, \text{ReLU}(W_i\, x + b_i) + b_{i+1}
\;=\;
\bigl(W_{i+1}\, S^{-1}\bigr)\, \text{ReLU}\bigl(S\, W_i\, x + S\, b_i\bigr) + b_{i+1}
```

ReLU is positive-homogeneous: `ReLU(α x) = α ReLU(x)` for `α > 0`. So you can absorb a per-channel rescale `S` into the boundary between two layers **without changing the output**. This is a free knob — and the seed of every "equivalent transformation" PTQ paper since.

---

## Cross-Layer Equalization (CLE)

Choose `S` to equalise the per-channel range across the two layers:

```math
S^c \;=\; \sqrt{\frac{r_i^c}{r_{i+1}^c}}
```

where `r_ℓ^c` is the range of channel `c` in layer `ℓ`. After applying `S`, both layers have geometric-mean ranges — minimising the loss when per-tensor scales are then applied. Iterate over all adjacent `(L_i, L_{i+1})` pairs to convergence (~5 sweeps).

**The intuition:** per-tensor quant fails when one channel's range dominates; CLE flattens the range histogram across two layers using only the free invariance.

---

## Bias Correction (BC)

Quantization introduces a *systematic* mean error per output channel. Let `E[x_prev]` be the previous layer's BatchNorm running mean. Then:

```math
\varepsilon_c \;=\; \bigl(W_c - Q(W_c)\bigr) \cdot \mathbb{E}[x_{\text{prev}}]
```

```math
b_c \;\leftarrow\; b_c - \varepsilon_c
```

Absorbs the mean quantization error into the bias. Closed-form, no data, uses BN's free training-time statistics.

---

## Why this matters for LLMs (the SmoothQuant connection)

DFQ's CLE requires (a) positive-homogeneous activation (ReLU/ReLU6), and (b) BatchNorm for `E[x]`. LLMs satisfy **neither** — they have GELU/SiLU and LayerNorm.

**But the equivalent-transformation idea is exactly what [[smoothquant]] (ch-09) does.** Instead of equalising weight ranges across two CNN layers, SmoothQuant migrates activation outliers into weights via a per-channel scale, absorbed into the previous LayerNorm's weight via:

```math
\hat{X} \cdot \hat{W} \;=\; \bigl(X \cdot \text{diag}(s)^{-1}\bigr) \cdot \bigl(\text{diag}(s) \cdot W\bigr)
```

where `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)`. Same invariance argument, different host architecture, different choice of `s`.

**AWQ** ([[awq]], ch-09) is another variant: per-channel scale `s_j` driven by activation magnitude, grid-searched over `α`, absorbed into adjacent ops. Same DFQ pattern.

---

## Empirical effect (MobileNetV2 — the CNN worst case)

| Method | ImageNet top-1 |
|---|---|
| FP32 | 71.7 |
| Naive per-tensor INT8 | 0.1 (collapsed) |
| **DFQ (CLE + BC)** | **71.2** |
| QAT (lower bound) | 71.4 |

MobileNetV2 with depthwise convs has notoriously large per-channel weight range variance — naive per-tensor INT8 collapses to random. CLE + BC recovers nearly all of it with zero data.

---

## Limits

- **CLE requires positive-homogeneous activation.** ReLU/ReLU6 ✓; Swish/h-swish ✗; GELU/SiLU ✗.
- **BC requires BatchNorm.** LayerNorm-only networks (transformers) cannot use the BN running mean shortcut. Workaround: replace `E[x]` with a calibration-set mean — costs you the "data-free" claim.
- **CLE is iterative.** ~5 sweeps to convergence; not closed-form.

---

## Common pitfalls

- **Applying CLE to non-ReLU activations.** The invariance breaks — the transformed model will not be identical in the FP forward pass. Some papers apply CLE to Swish networks and report mild accuracy *gain* on FP, evidence the implementation is wrong.
- **Using a stale BN mean.** Some PTQ pipelines re-batch-norm the model first; BC needs the BN mean from the *original* FP model, not a re-folded one.

---

## Connections

- [[excerpts/obc]] — orthogonal: closed-form Hessian-based weight editing instead of equivalent transformation.
- [[ch-05]] — parent synthesis; DFQ is the no-calibration corner of the playbook.
- [[ch-09]] — SmoothQuant and AWQ are the LLM-era heirs of the equivalent-transformation idea.
- [[ch-13]] — QuIP's rotation idea generalises "absorb a transformation into adjacent weights" to arbitrary orthogonal matrices.
