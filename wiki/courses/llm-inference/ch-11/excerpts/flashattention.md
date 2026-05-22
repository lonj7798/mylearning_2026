---
chapter: ch-11
course: llm-inference
phase: read
excerpt_of: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
source_url: https://arxiv.org/abs/2205.14135
created_at: "2026-05-21"
---

# Excerpt: FlashAttention — IO-aware exact attention

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
**Year:** 2022
**Venue:** NeurIPS 2022
**URL:** https://arxiv.org/abs/2205.14135
**Raw-data source:** [[raw-data/flashattention]]

---

## The reframing (the load-bearing insight)

Conventional wisdom: attention's cost is O(L²·d) FLOPs.
FlashAttention's reframing: attention's cost is dominated by HBM traffic, not FLOPs. The L×L scores matrix gets written to HBM and read back, dwarfing the L·d-sized inputs.

A100 numbers (paper Table 1):

| Method | HBM traffic (GB) for L=4k, d=128 | Wall time (ms) |
|---|---|---|
| Standard attention (PyTorch) | 16 GB | 17 ms |
| Memory-efficient (xFormers) | 3 GB | 12 ms |
| **FlashAttention** | **1 GB** | **7 ms** |

FA1 does ~9× less HBM traffic and ~2× less wall time. The compute side is *unchanged* (same exact softmax); the only change is that the L×L matrix never lives in HBM.

---

## The online softmax recurrence (Algorithm 1)

For each Q-block `Q_i` (rows), iterate over K/V-blocks `K_j, V_j` (cols). Maintain per-row running max `m` and normalizer `ℓ`:

```math
\begin{aligned}
S_{ij} &= Q_i K_j^\top / \sqrt{d_k} \\
\tilde{m}_{ij} &= \text{rowmax}(S_{ij}) \\
m_i^{\text{new}} &= \max(m_i, \tilde{m}_{ij}) \\
P_{ij} &= \exp(S_{ij} - m_i^{\text{new}}) \\
\ell_i^{\text{new}} &= e^{m_i - m_i^{\text{new}}} \ell_i + \text{rowsum}(P_{ij}) \\
O_i^{\text{new}} &= \mathrm{diag}\bigl(\ell_i^{\text{new}}\bigr)^{-1}\bigl(\mathrm{diag}(\ell_i) e^{m_i - m_i^{\text{new}}} O_i + P_{ij} V_j\bigr)
\end{aligned}
```

The key step: whenever a new block's max `tilde{m}_ij` exceeds the running `m_i`, **rescale all prior output by `exp(m_i − m_i^new)`**. This is the same rescaling identity standard log-sum-exp uses, applied incrementally.

After all blocks are processed, `O_i` equals the standard softmax output — *exactly*.

---

## IO complexity proof (Theorem 2)

For SRAM size `M` (per SM), block size `B = Θ(M/d)`:

```math
\text{HBM accesses} \;=\; \Theta\Bigl(L \cdot d \;+\; L^2 d^2 / M\Bigr)
```

vs standard attention's `Θ(L·d + L²)`. For typical `M ≈ 100 KB`, `d = 128`:

- standard: ~L² · 2 bytes accesses
- FlashAttention: ~L² · 128² / 100000 · 2 ≈ L² · 0.3 bytes — **~7× fewer**.

The improvement grows with d and shrinks with M; modern d=128 GPUs see the full ~7–10× benefit.

---

## Lower-bound theorem (Theorem 3)

The paper also proves that no algorithm can do attention with substantially fewer HBM accesses than FlashAttention for a range of SRAM sizes. FlashAttention is **asymptotically optimal** in IO complexity.

---

## Empirical results

| Workload | Speedup vs standard | Memory reduction |
|---|---|---|
| GPT-2 small training, L=1024 | 3.0× | linear in L |
| BERT-large MLM, L=512 | 1.15× e2e | 10× attention memory |
| Long Range Arena, L=16k | enables (OOM otherwise) | quadratic → linear |
| GPT-2 generation, L=2k | 2× | enables larger batch |

---

## Connections

- [[ch-11]] — parent chapter; FA1 is the foundation of the lineage.
- [[excerpts/flashattention-2]] — better work partitioning and lower non-matmul FLOPs.
- [[excerpts/flashattention-3]] — Hopper async + FP8 follow-on.
- [[excerpts/flashdecoding]] — decode-specific split-K reformulation.
- [[attention-complexity]] (ch-02) — the O(L²·d) compute reality FA1 reshapes.
