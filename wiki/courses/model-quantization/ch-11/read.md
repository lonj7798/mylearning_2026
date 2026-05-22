<!-- chapter: ch-11
     phase: 2023-refinements
     title: SqueezeLLM + SpQR — Sensitivity-Aware + Sparse-and-Dense
     sources: [[squeezellm]], [[spqr]], [[owq]]
-->

# Chapter 11 — SqueezeLLM + SpQR: Sensitivity-Aware + Sparse-and-Dense

> **Core insight.** Outliers in LLM weights are not noise to clip or rebalance away — they are **signal that disproportionately drives the loss**. The right move at ≤3-bit is to (a) weight the quant-grid design by per-weight sensitivity (Fisher diagonal `(∂L/∂w)²`) so the codebook concentrates where it matters, and (b) extract the top ~0.5–1% most sensitive/outlier weights into a tiny sparse FP16 sidecar, then quantize the rest aggressively. This is dense-and-sparse decomposition: structured precision allocation, not uniform aggression.
>
> **Guideline.** For ≤3-bit weight-only deployment use SqueezeLLM (Fisher-weighted k-means, 8 centroids for 3-bit, 0.4–0.5% sparse outliers in CSR) or SpQR (bilevel groups G_in=16/G_out=128, 1% outlier sparse, ~3.44 effective bits). For a structured-outlier (whole-column) variant that doubles as a PEFT method, use OWQ + Weak Column Tuning.

---

## Why this chapter exists

[[ch-09]] and [[ch-10]] handle outliers by **equivalent transformation**: rescale or rotate them away so they fit a uniform grid. That works down to W4A4 with learned affine transforms ([[omniquant]], [[flatquant]]), but at extreme weight-only compression (≤3-bit) the strategy hits a wall:

1. The quantization grid only has 8 levels (3-bit) or 4 levels (2-bit). Even a perfectly-flattened distribution wastes resolution because the levels are uniformly spaced.
2. A small fraction of weights carries an outsized portion of the loss derivative. Treating them like average weights — even after smoothing — is the wrong allocation of precision.

The 2023 refinements that crack this regime take a different stance: **outliers are signal, not noise**. Preserve them precisely; aggress on the rest.

Three methods, three flavours of the same idea:

- **SqueezeLLM** ([[squeezellm]], ICML 2024) — Fisher-information-weighted **non-uniform** k-means quantization. The codebook is data-driven; the levels are *not* uniformly spaced. Plus a 0.4–0.5% sparse FP16 outlier matrix.
- **SpQR** ([[spqr]], 2023) — **Bilevel uniform** quantization (G_in=16 inner groups + G_out=128 outer groups + per-weight outlier extraction at ~1%). The first near-lossless 4-bit.
- **OWQ** ([[owq]], AAAI 2024) — Structured **whole-column** outliers (1–5% of input dimension stays FP16); doubles as a parameter-efficient fine-tune via Weak Column Tuning.

Throughout, the loss-attribution problem is solved with two different sensitivity estimators: **Fisher diagonal** `(∂L/∂w)²` (SqueezeLLM, OWQ) and **GPTQ Hessian** `(W - Q(W))² / [H⁻¹]_{jj}` (SpQR). Both proxy "how much does the loss change if I quantize this weight wrong."

---

## 1. The setup: single-batch LLM inference is memory-bound

SqueezeLLM opens with a measurement that justifies the entire chapter: at batch=1 / batch=2 LLM decode, **memory bandwidth — not compute — is the bottleneck**. A 7B model in FP16 is 14 GB of weight reads per token. The HBM bandwidth ceiling (A100 ~2 TB/s, A6000 ~700 GB/s) bounds tokens/sec well below what the SM compute can deliver.

Consequence: cutting weight bits is the highest-leverage knob. Going from 16-bit to 4-bit ≈ 4× memory bandwidth headroom ≈ 4× tokens/sec (modulo dequant cost). Going to 3-bit ≈ 5.3×. The dequant cost itself is hidden behind the load if the kernel is fused properly.

This is *why* sub-4-bit weight-only is worth chasing even when activations stay FP16. SqueezeLLM reports a **2.3× decode speedup over FP16 on an A6000** at 3-bit.

---

## 2. SqueezeLLM — Fisher-weighted non-uniform k-means

The conventional approach: pick a uniform grid (INT-k absmax) and round each weight to the nearest level. The implicit objective is to minimise unweighted reconstruction MSE.

SqueezeLLM's reframe: weights don't contribute equally to the loss. A first-order expansion gives

```math
\Delta L \approx \mathbb{E}\Big[\sum_i \frac{\partial L}{\partial w_i} \cdot \Delta w_i\Big] = \sum_i \mathbb{E}\Big[\frac{\partial L}{\partial w_i}\Big] \cdot \Delta w_i
```

and second-order (with diagonal Hessian approximation `F = diag(∂L/∂w)²`):

```math
\Delta L \approx \frac{1}{2} \sum_i F_i \cdot (\Delta w_i)^2
```

So minimising the *Fisher-weighted* squared error is the loss-aware objective.

### Fisher-weighted k-means

For each row (or per-group), find codebook `C = {c_1, ..., c_K}` (K = 2^b for b-bit) by **weighted Lloyd iteration**:

```math
\min_{C,\ \text{assign}} \sum_i F_i \cdot (w_i - c_{\text{assign}(i)})^2
```

Weighted Lloyd update:

```math
c_k = \frac{\sum_{i :\ \text{assign}(i) = k} F_i\, w_i}{\sum_{i :\ \text{assign}(i) = k} F_i}
```

- Assignment = nearest neighbour to `c_k`.
- ~30 iterations to convergence; minutes per layer.
- `F_i` estimated by squared gradient over a calibration corpus (next-token loss on Pile / C4).

The codebook is **non-uniform**: levels concentrate near weights that carry loss mass. This is the LLM-era analogue of Lloyd-Max for non-uniform sources ([[lloyd-max-quantizer]], ch-03).

### Per-row codebook + LUT decode

Each row stores:
- A K-entry FP16 codebook (K = 8 or 16).
- A k-bit index per weight (b ∈ {3, 4}).

Effective bits per weight: `b + 16 · K / G` where G is the group size. For row-wise (G = d_in) and K=16, the codebook overhead is negligible.

Decode at inference: per-weight k-bit index → row-LUT lookup → FP16 value → standard FP16 GEMV.

> **Pitfall.** Per-row codebook storage means an extra 16 · K bytes per row. At small models this is non-trivial. SqueezeLLM uses per-row at 3-bit but groups rows at 4-bit to amortise.

---

## 3. SqueezeLLM — dense-and-sparse decomposition

The Fisher-weighted k-means handles the bulk of the weight distribution. The long tail — the top ~0.4–0.5% of weights by Fisher · weight² — gets pulled out into a separate **FP16 sparse matrix** stored in CSR.

```
W = Q(W_dense) + S_sparse
y = (Q-decoded GEMV)(W_dense, x) + (CSR-SpMV)(S_sparse, x)
```

- Sparse matrix is ~0.5% nnz → CSR overhead trivial.
- FP16 outliers fully preserved — zero quant error on the most sensitive weights.
- Fused into one CUDA kernel: dequant + GEMV + sparse-add in a single pass.

### The 0.5% number

Why 0.5% specifically? Below 0.5%, residual outliers degrade PPL. Above 0.5%, the sparse CSR overhead starts dominating without quality gain — the long tail is genuinely thin. SqueezeLLM's Figure 4 ablation: 0.45% is the sweet spot for LLaMA-7B at 3-bit.

### Hyperparameters

| Knob | Value |
|---|---|
| Bits b | 3 or 4 |
| Codebook size K | 8 (3-bit), 16 (4-bit) |
| Sparse outlier % | 0.4–0.5% |
| Sensitivity | diag-Fisher = mean squared grad over calibration |
| Calibration | 128 sequences C4 |
| k-means iterations | ~30 |
| Effective bits | ~3.0 + 0.5% · 16 ≈ 3.08 (3-bit + sparse) |

### Empirical (LLaMA-7B WikiText-2 PPL)

| Method | Bits | PPL |
|---|---|---|
| FP16 | 16 | 5.68 |
| GPTQ | 3 | 8.06 |
| **SqueezeLLM** | 3 | **7.08** |
| GPTQ | 4 | 5.85 |
| **SqueezeLLM** | 4 | **5.79** |

At 3-bit SqueezeLLM beats GPTQ by ~1.0 PPL. At 4-bit the gap is smaller — both are near-FP16 — but SqueezeLLM still wins.

---

## 4. SpQR — bilevel uniform + per-weight sparse

[[spqr]] (Dettmers et al. 2023) takes a different angle on the same problem: **keep the grid uniform** (cheap dequant, no LUT) but use **tiny inner groups** so the per-group scale fits a small range tightly. Then quantize the inner scales themselves to recoup the storage overhead.

### Bilevel group-wise quantization

For weight `W ∈ R^{d_out × d_in}`, partition each row first into outer groups of size `G_out = 128`, then each outer group into inner groups of size `G_in = 16`.

For each inner group (16 weights):

```math
s_{\text{inner}} = \frac{\max(|W_g|)}{2^{b-1} - 1}, \qquad W_q = \mathrm{round}\Big(\frac{W_g}{s_{\text{inner}}}\Big), \quad b = 3
```

The inner scales `{s_inner}` (8 per outer group of 128) are themselves quantized to 3 bits within the outer group:

```math
s_{\text{outer}} = \frac{\max(s_{\text{inner-group}})}{2^3 - 1}, \qquad \hat{s}_{\text{inner}} = \mathrm{round}\Big(\frac{s_{\text{inner}}}{s_{\text{outer}}}\Big)
```

Net effect: a 3-bit-of-3-bit nested scale structure.

**Bit budget arithmetic:**

```
bits/weight = b + 3/G_in + 16/G_out + 16/G_out
            = 3 + 3/16 + 32/128
            = 3 + 0.1875 + 0.25
            ≈ 3.44 effective bits
```

This gives near-4-bit quality at ~3.44 bits — closing the gap that plain group-wise GPTQ leaves.

### Per-weight outlier extraction (the "Sp" in SpQR)

Run a GPTQ-style per-weight sensitivity score:

```math
\mathrm{sens}_{i, j} = \frac{(W_{i, j} - Q(W_{i, j}))^2}{[H^{-1}]_{j, j}}
```

Pick the top ~1% globally per layer (or threshold by τ). These go into a sparse CSR matrix `S` at FP16. The dense quantization above is re-run with the outlier indices masked to zero.

```
y = (SpQR-decoded GEMV)(W_dense, x) + (CSR-SpMV)(S, x)
```

The sparse kernel is ~1% nnz → negligible compute.

### Hyperparameter table

| Knob | Value |
|---|---|
| Inner group G_in | 16 |
| Outer group G_out | 128 |
| Weight bits | 3 |
| Inner-scale bits | 3 |
| Outer-scale bits | 16 |
| Outlier fraction | ~1% |
| Outlier sensitivity | GPTQ-style |
| Average bits/weight | ~3.4–4.0 |
| Calibration | 128 × 2048 tokens |

### Empirical reach

SpQR's headline: **first near-lossless 4-bit on LLaMA-65B (≤0.1 PPL gap to FP16)**. On LLaMA-2-7B WikiText-2:

| Method | Bits | PPL | Δ |
|---|---|---|---|
| FP16 | 16 | 5.47 | — |
| GPTQ (g=128) | 4 | 5.69 | +0.22 |
| **SpQR** | ~4 | **5.49** | **+0.02** |

A 33B model in SpQR fits in a single 24 GB GPU with a 15% inference speedup. This is the recipe that lets LLaMA-33B run on a 4090 at near-FP quality.

---

## 5. SqueezeLLM vs SpQR — same idea, different parameterisation

| Aspect | SqueezeLLM | SpQR |
|---|---|---|
| Grid | non-uniform (Fisher-weighted k-means) | uniform (bilevel scales) |
| Codebook | per-row, K=8 or 16 | none (uniform spacing) |
| Sensitivity proxy | Fisher diagonal `(∂L/∂w)²` | GPTQ Hessian sensitivity |
| Outlier extraction | top 0.4–0.5% per-weight, FP16 sparse | top ~1% per-weight, FP16 sparse |
| Inference | LUT decode + sparse add (fused) | dequant (2-level scale) + sparse add (fused) |
| Best at | 3-bit (1.0 PPL gain over GPTQ) | 4-bit (closes gap to FP16) |
| Backend | custom kernel | Marlin / AutoGPTQ + sparse |

Both prove the same thesis from different directions: **outliers are signal; structure precision around sensitivity**.

The taxonomy of "what counts as an outlier":
- **SpQR / SqueezeLLM**: individual weights with high Hessian/Fisher score (per-weight sparse).
- **OWQ**: whole input columns aligned with activation-outlier channels (structured sparse).
- **LLM.int8()**: activation columns above magnitude threshold (per-token FP16 path).
- **AWQ / SmoothQuant**: per-channel scale to rebalance (no FP16 storage).

---

## 6. OWQ — structured whole-column outliers + PEFT bonus

[[owq]] (Lee et al. AAAI 2024) takes the "outlier weights stay FP16" idea but defines an outlier as a **whole input column** (per-channel structured) rather than individual weights.

### Weak column score

For weight `W ∈ R^{d_out × d_in}` and calibration activations `X ∈ R^{N × d_in}`, per input channel j:

```math
\text{score}_j = \max_t |X_{t, j}| \cdot \lVert W_{\cdot, j} \rVert_2
```

This combines:
- activation-outlier magnitude `max_t |X_{t,j}|` (which activation channel hits hard)
- weight contribution `||W_{·,j}||_2` (does the weight column actually use that signal)

Pick the top `k = ⌈p · d_in⌉` columns (p = 1–5%) as the **weak set**.

### Mixed-precision storage

- Weak columns (1–5% of d_in): stored as FP16, no quant.
- Dense columns (95–99%): GPTQ-quantized at b=3 bits, group_size=128, with weak columns masked from the Hessian update.

```math
\text{bits/weight} \approx 0.95 \cdot 3.125 + 0.05 \cdot 16 \approx 3.77
```

(Drops to ~3.30 at p=1%.)

### Why structured > per-weight

Per-weight sparse (SqueezeLLM, SpQR) needs CSR storage and a separate SpMV kernel. Whole-column sparse is **a simple concat**: dense INT-k columns + sparse FP16 columns. Far easier to kernelise. Trades a small accuracy gap for kernel simplicity.

### Weak Column Tuning (the PEFT bonus)

The same FP16 outlier columns can serve as the **only** trainable parameters for parameter-efficient fine-tuning:

- Freeze all dense (quantised) weights.
- Make only `W_weak_fp16` trainable (shape `d_out × k`, k ≈ 1–5% · d_in).
- AdamW on a task loss.

Trainable parameter count ~3–5% of full fine-tune — comparable to LoRA at r=8. **And it lives natively in the quantised format** — no merge step, no FP16 LoRA branch at inference.

This makes OWQ both a quant scheme and a PEFT scheme. The OWQ-fine-tuned model is just OWQ — same kernel, updated FP16 columns.

---

## 7. The lesson, stated plainly

```
Outliers are signal, not noise.
```

Pre-2023 quantization (and the earlier chapters in this course) treated outliers as a problem: clip them, smooth them, rebalance them, rotate them away. SqueezeLLM / SpQR / OWQ flip the framing: outliers are **the most information-dense weights in the matrix**. Preserve them in FP16 (CSR or whole-column); spend the bit budget aggressively on the rest.

This decomposition is now the standard recipe for sub-4-bit weight-only LLM quant. The 2024 successors ([[aqlm]], [[hqq]], [[quip-sharp]] — all in [[ch-14]]) continue the same dense-and-sparse philosophy, sometimes with different sparse formats (lattice codebooks, additive quantization) but always with the same precision-allocation discipline.

### Hyperparameter cheat-sheet — pick your tool

| Goal | Tool | Effective bits | LLaMA-7B WikiText-2 PPL |
|---|---|---|---|
| Near-lossless 4-bit | SpQR | ~3.4–4.0 | 5.49 |
| Aggressive 3-bit | SqueezeLLM | ~3.08 | 7.08 |
| Quant + PEFT in one | OWQ + WCT | ~3.77 | 6.51 (3.01-bit avg) |
| Pure dense uniform 4-bit | AWQ / GPTQ | 4.125 | 5.60 / 5.69 |

---

## Connections and what's next

- **[[squeezellm]]** — full extract; Fisher-weighted k-means + dense-and-sparse.
- **[[spqr]]** — full extract; bilevel uniform + sparse outlier.
- **[[owq]]** — full extract; structured whole-column outliers + Weak Column Tuning.
- **[[gptq]]** — the Hessian sensitivity score reused for outlier picking in SpQR.
- **[[hawq]]** — Hessian-aware mixed-precision predecessor.
- **[[llm-int8]]** — the activation-side analogue: outlier *activation columns* in FP16.
- **[[awq]]** — the equivalent-transformation alternative for the dense path; composes with sparse outlier extraction.
- **[[ch-14]] / [[aqlm]]** — sub-2-bit additive quantization successor; same codebook + sparse philosophy at higher compression.
- **[[ch-14]] / [[quip-sharp]]** — E8 lattice 2-bit codebook; structured-codebook alternative to SqueezeLLM's per-row k-means.
- **[[ch-19]] / kernels** — production sparse-add fused GEMV kernels.

## Further reading

- The SqueezeLLM Figure 3 (weight magnitude vs Fisher diagonal scatter) is the strongest argument for **Fisher-weighting over magnitude-weighting** — they really don't agree.
- SpQR's bilevel-scale construction is worth a whiteboard pass; the 3.44-bit arithmetic is non-obvious until you write it out.
- OWQ's Weak Column Tuning is the simplest PEFT recipe in the literature — read the algorithm box in §4 of the paper.
