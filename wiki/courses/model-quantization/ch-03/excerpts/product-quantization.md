---
chapter: ch-03
course: model-quantization
phase: read
excerpt_of: "Product Quantization for Nearest Neighbor Search (Jégou, Douze, Schmid 2011)"
source_url: https://hal.inria.fr/inria-00514462/document
created_at: "2026-05-21"
raw_data_source: [[raw-data/product-quantization]]
---

# Excerpt: PQ — factorized codebooks that make VQ scale

**Authors:** Hervé Jégou, Matthijs Douze, Cordelia Schmid (INRIA).
**Year:** 2011.
**Venue:** IEEE TPAMI 33(1).
**URL:** see source_url.

---

## The one-box decomposition

Split a `D`-dim vector `x` into `M` sub-vectors of length `D' = D / M`:

```math
x = [u_1 ; u_2 ; \ldots ; u_M], \quad u_m \in \mathbb{R}^{D/M}
```

Quantize each `u_m` with its own sub-codebook `C_m = {c_{m,1}, \ldots, c_{m,K}}` of size `K` (typically `K = 256` so each code fits in one byte):

```math
\text{code}_m(u_m) = \arg\min_k \|u_m - c_{m,k}\|^2
\qquad
\text{PQ}(x) = [c_{1, \text{code}_1(u_1)} ; \ldots ; c_{M, \text{code}_M(u_M)}]
```

---

## Storage and rate

- **Codebook storage:** `M · K · (D/M) = K · D` floats.
- **Per-vector code:** `M · log₂ K` bits (e.g. `M = 16, K = 256` ⇒ **16 bytes per vector regardless of `D`**).
- **Effective codebook size:** `K^M` (e.g. `256^16 = 2^128`).

---

## Asymmetric Distance Computation (ADC)

For query `q = [q_1; …; q_M]`, precompute `M` tables of size `K`:

```math
T_m[k] = \|q_m - c_{m,k}\|^2  \qquad \text{for } m = 1\ldots M, \; k = 1\ldots K
```

Then for any database vector with PQ codes `(k_1, …, k_M)`:

```math
\|q - \text{PQ}(x)\|^2 \,\approx\, \sum_m T_m[k_m]
```

`M` table lookups + `M − 1` adds per database comparison. **This is the trick that made FAISS billion-scale ANN possible.**

---

## Sub-codebook training

Run `k`-means independently on the empirical distribution of each sub-vector `u_m` across the training set:

```
for m = 1..M:
    C_m ← KMeans(K, {u_m^(t) : t = 1..T})
```

---

## Optimized PQ (OPQ)

A **learned rotation** `R` applied before splitting (`x → R · x`) balances variance across sub-vectors → significantly better distortion at the same bit budget. **Direct precursor to rotation-based LLM quant** ([[quarot]], [[spinquant]] in ch-13/14).

---

## Additive / Residual quantization

- **Residual VQ (RVQ):** quantize `x`, then quantize the residual `x − c_1`, then residual of residual, …
- **Additive quantization:** `PQ(x) = Σ_m c_m` where each `c_m` comes from its own full-dim codebook (not sub-vector). More expressive than PQ, harder encoding (combinatorial).
- **AQLM = additive quantization for LLM weights** ([[aqlm]], ch-14).

---

## Bit budget for LLM weights

For per-row weight quantization at sub-2-bit: typical setting `D = 512–4096, M = 64–256, K = 256`, giving ~1.5–2 bits/weight at ~3–5× compression vs INT4.

---

## Connections

- [[excerpts/vector-quantization]] — PQ is a tractable factorization of LBG codebook.
- [[aqlm]] — additive VQ for LLM weights (PQ → AQ generalization), ch-14.
- [[vptq]] / [[gptvq]] — direct PQ-style LLM weight compression with GPTQ-style Hessian update.
- [[quip-sharp]] — E8 lattice + Hadamard rotation; rotation idea inherited from OPQ.
- [[quarot]] / [[spinquant]] — Hadamard / learned rotations on LLM weights/activations; lineage from OPQ.
- [[ch-03]] — parent synthesis.
