---
chapter: ch-11
course: model-quantization
phase: read
excerpt_of: "SqueezeLLM: Dense-and-Sparse Quantization"
source_url: https://arxiv.org/abs/2306.07629
created_at: "2026-05-21"
---

# Excerpt: SqueezeLLM — Fisher-weighted k-means + dense-and-sparse

**Authors:** Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer
**Year:** 2023 (ICML 2024)
**URL:** https://arxiv.org/abs/2306.07629
**Raw-data source:** [[raw-data/squeezellm]]

---

## The memory-bandwidth premise

Single-batch LLM decode is **memory-bandwidth-bound**, not compute-bound. Weight bits ↔ memory bandwidth ↔ tokens/sec is roughly linear. Aggressive weight-only quantization is therefore the highest-leverage knob for decode latency.

Measured: **2.3× decode speedup over FP16 at 3-bit on an A6000.**

---

## Sensitivity-weighted (Fisher) non-uniform quantization

Loss perturbation under weight quantization, second-order with diagonal Hessian:

```math
\Delta L \approx \frac{1}{2} \sum_i F_i \cdot (\Delta w_i)^2, \qquad F_i \approx \mathbb{E}\Big[\Big(\frac{\partial L}{\partial w_i}\Big)^2\Big]
```

Per row (or per-group), find codebook `C = {c_1, ..., c_K}` (K = 2^b) by **weighted Lloyd iteration**:

```math
\min_{C,\ \text{assign}} \sum_i F_i \cdot (w_i - c_{\text{assign}(i)})^2
```

Weighted Lloyd update:

```math
c_k = \frac{\sum_{i :\ \text{assign}(i) = k} F_i\, w_i}{\sum_{i :\ \text{assign}(i) = k} F_i}
```

- Assignment = nearest neighbour to `c_k` (plain L2).
- ~30 iterations to convergence; minutes per layer.
- F_i = squared gradient over a small calibration corpus (next-token loss on C4 / Pile).

The codebook is **non-uniform** — levels concentrate where Fisher-mass is.

---

## Dense-and-sparse decomposition

Identify weights with the largest `F_i · w_i²` product (top 0.4–0.5%); pull them into a sparse FP16 matrix `S` (CSR). The remaining "dense" weights are quantized as above.

```
W = Q(W_dense) + S_sparse
y = (Q-decoded GEMV)(W_dense, x) + (CSR-SpMV)(S, x)
```

- Sparse matrix ~0.5% nnz → CSR overhead trivial.
- FP16 outliers fully preserved.

---

## LUT inference kernel

At decode time:
- Per-weight k-bit index decodes through a 16-or-8-entry LUT (the row's codebook) → FP16.
- Standard FP16 GEMV proceeds.
- Sparse outlier add is **fused into the same kernel**.

---

## Hyperparameters

| Knob | Value |
|---|---|
| Bits b | 3 or 4 |
| Codebook size K | 8 (3-bit), 16 (4-bit) |
| Sparse outlier % | 0.4–0.5% |
| Sensitivity | diag-Fisher = mean squared grad |
| Calibration | 128 sequences C4 |
| k-means iterations | ~30 |

---

## Empirical (LLaMA-7B WikiText-2 PPL)

| Method | Bits | PPL |
|---|---|---|
| FP16 | 16 | 5.68 |
| GPTQ | 3 | 8.06 |
| **SqueezeLLM** | 3 | **7.08** |
| GPTQ | 4 | 5.85 |
| **SqueezeLLM** | 4 | **5.79** |

At 3-bit: **+1.0 PPL over GPTQ**. At 4-bit the gap narrows (both near FP16), but SqueezeLLM still wins.

---

## The key visualisation (Figure 3)

Scatter plot: weight magnitude vs Fisher diagonal per weight. **The two don't agree.** This is the empirical argument for Fisher-weighted (vs magnitude-weighted) codebook design. The most-quant-sensitive weights are not the largest weights; they are the ones the loss cares about.

---

## Connections

- Sensitivity-aware-bit-allocation ancestor: [[hawq]] (Hessian-aware mixed precision).
- Outlier-preserving cousins: [[spqr]] (similar idea, different threshold / grid), [[owq]] (whole-column outliers).
- Uniform-grid + Hessian rival: [[gptq]].
- Activation-aware uniform-grid rival: [[awq]].
- LUT-GEMM kernel lineage: [[nuqmm]], [[gguf-k-quants]] (q3_K, q4_K, q5_K families).
- Sub-2-bit codebook successor: [[aqlm]] (in [[ch-14]]).
