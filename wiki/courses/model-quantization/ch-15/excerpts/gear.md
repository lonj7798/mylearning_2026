---
chapter: ch-15
course: model-quantization
phase: read
excerpt_of: "GEAR: An Efficient KV Cache Compression Recipe"
source_url: https://arxiv.org/abs/2403.05527
created_at: "2026-05-21"
---

# Excerpt: GEAR — quant + low-rank residual + sparse outliers

**Authors:** Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao (Georgia Tech)
**Year:** 2024
**URL:** https://arxiv.org/abs/2403.05527
**Raw-data source:** [[raw-data/gear]]

---

## The thesis

Single-technique KV quantization (uniform low-bit, or pure outlier isolation, or pure low-rank) leaves accuracy on the table. The residual after aggressive low-bit quant has **two distinct components**:

1. **A low-rank systematic pattern** (the same error appears across many tokens for the same channels) — captures most of the signal that quant lost.
2. **Sparse outlier entries** (a few values that even the low-rank capture misses) — handled by a small COO buffer.

The three together — quant Q, low-rank L, sparse S — give near-lossless 4-bit KV with the interplay being essential. Removing any one component materially degrades accuracy.

---

## The decomposition

For a per-head KV matrix `M ∈ ℝ^{T × d}` (T tokens, d head_dim):

```math
M \approx Q + L + S
```

| Component | Definition | Bit budget |
|-----------|------------|------------|
| **Q** | dequant(quant_b(M)), uniform b-bit | b bits / element |
| **L** | A · B^⊤, rank r SVD truncation of residual | 2(T+d)·16 / (T·d) ≈ 0.3 / element |
| **S** | top-K residual entries as sparse FP16 | 1% × 32 ≈ 0.3 / element |

Total at b=4: ~4.6 bits/element. At b=2: ~2.6 bits/element.

---

## Computing L (truncated SVD)

After choosing Q, compute residual `R = M − Q`. Truncated SVD:

```math
R \approx U_r \Sigma_r V_r^\top
```

Set:

```math
A = U_r \sqrt{\Sigma_r}, \quad B = V_r \sqrt{\Sigma_r}
```

```python
def gear_init_L(M, Q, rank=2):
    R = M - Q
    U, s, Vt = torch.linalg.svd(R, full_matrices=False)
    U_r = U[:, :rank]
    s_r = s[:rank]
    V_r = Vt[:rank, :].T
    A = U_r * s_r.sqrt()       # (T, r)
    B = V_r * s_r.sqrt()       # (d, r)
    return A, B
```

For r = 2–4 the low-rank capture is most of the residual's Frobenius norm — empirically ~80% of the systematic error.

---

## Streaming SVD

For decode-time inference, M grows by one token per step. GEAR maintains A, B incrementally:

```python
def gear_stream_update(A, B, M_new_chunk, Q_new_chunk):
    r = M_new_chunk - Q_new_chunk        # new residual chunk
    # project onto current B
    coeffs = r @ B                       # (chunk_size, rank)
    # orthogonalise against current B (Gram-Schmidt or QR)
    residual_after_proj = r - coeffs @ B.T
    # append to A
    A_new = torch.cat([A, coeffs])
    # B unchanged unless drift threshold exceeded
    return A_new, B
```

**Periodic full refresh** (every ~256 tokens) prevents drift. The refresh is amortised over the interval.

---

## Sparse S

After Q + L, scan `M − Q − L`; pick top-K elements by absolute value (`K = ⌈0.01 · T · d⌉`); store as `(token_idx, channel_idx, FP16 value)` tuples.

These are the entries that the rank-r approximation cannot capture — typically extreme outliers that don't fit any low-rank structure.

---

## Attention math

```
attn_logits = Q · (Q_K + L_K + S_K)^⊤ / √d
            = Q · Q_K^⊤      (INT4 GEMM — bulk of compute)
            + Q · L_K^⊤      = (Q · A_K)·B_K^⊤ (two small FP16 matmuls)
            + Q · S_K^⊤      (sparse SpMV)
```

Three independent matmuls + sum. The Q-path dominates compute; the L and S paths add ~5% overhead.

---

## Why all three are non-redundant

GEAR ablations (LLaMA-7B Wikitext-2 at 4-bit):

| Components | ppl | Δppl |
|-----------|-----|------|
| FP16 | 5.47 | — |
| Q only | 6.95 | +1.48 |
| Q + L | 6.05 | +0.58 |
| Q + S | 6.27 | +0.80 |
| **Q + L + S** | **5.52** | **+0.05** |

The residual after Q has both low-rank structure (captured by L) and sparse outliers (captured by S). Removing either component loses 0.3–1.0 ppl.

---

## Bit-budget details

For b=4, r=2, K=1% T·d, T=2048, d=128:

```
Q:  4 bits/element                                      (HBM dense)
L:  2 · (T+d) · 16 bits / (T·d) = 32(T+d)/(Td)         (HBM dense small)
   ≈ 32/2048 + 32/128 ≈ 0.265 bits/element
S:  1% · (24 bits index + 8 bits FP16 value)            (HBM sparse)
   ≈ 0.32 bits/element
Total ≈ 4.59 bits/element
```

vs FP16 (16 bits/element): **3.49× compression**.

---

## Throughput numbers

LLaMA-7B at 32K context, batch 8:
- **2.38× decoding throughput** vs FP16 KV.
- **2.29× peak memory reduction**.

Comparable to KIVI INT4 on memory; slightly better on accuracy because of the L + S compensation.

---

## When to pick GEAR over KIVI/KVQuant

| | Strength | Weakness |
|--|----------|----------|
| **KIVI** | Tuning-free, INT2 viable | No residual compensation; PPL hit at INT2 |
| **KVQuant** | Sub-2-bit avg; outlier-aware | Calibration + non-uniform codebook complexity |
| **GEAR** | Near-lossless at 4-bit; flexible Q+L+S | Streaming SVD complexity; higher avg bits |

GEAR is the choice when you need **near-lossless 4-bit KV** — for example, agent / tool-use workloads where small accuracy regressions matter a lot. KIVI and KVQuant are the right pick at sub-4-bit where lossiness is the point.

---

## Pitfalls

- **Streaming SVD drifts** without periodic refresh. Refresh every ~256 tokens to bound the orthogonality error.
- **Rank r choice**: r=2 is the sweet spot; r=4 overfits to calibration; r=1 misses too much systematic error.
- **Sparse S's 1% threshold** is fixed per-paper; for distribution-shifted inputs, periodic re-thresholding may be needed.
- **L_K and L_V are separate** — don't reuse SVD bases across K and V. Their residuals have different structure.

---

## Connections

- [[excerpts/kivi]] — the canonical axis-asymmetry recipe; GEAR adds Q+L+S on top of KIVI's quantizer.
- [[excerpts/kvquant]] — alternative sub-2-bit path via non-uniform + dense-and-sparse.
- [[ch-12]] — [[lq-lora]] uses the same low-rank-residual idea for weights.
- [[ch-19]] — production KV-cache kernels.
