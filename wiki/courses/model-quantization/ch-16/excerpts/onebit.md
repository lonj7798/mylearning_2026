---
chapter: ch-16
course: model-quantization
phase: read
excerpt_of: "OneBit: Towards Extremely Low-bit Large Language Models"
source_url: https://arxiv.org/abs/2402.11295
created_at: "2026-05-21"
---

# Excerpt: OneBit — SVID decomposition for true 1-bit weights

**Authors:** Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, Wanxiang Che
**Year:** 2024
**Venue:** NeurIPS 2024
**URL:** https://arxiv.org/abs/2402.11295
**Raw-data source:** [[raw-data/onebit]]

---

## What OneBit is

A different route to ~1-bit LLM weights than BitNet's ternary recipe. Instead of `{−1, 0, +1}` with per-tensor scale, factor each weight matrix as:

```math
W \approx \mathrm{diag}(a) \cdot S \cdot \mathrm{diag}(b)
```

- `S ∈ {−1, +1}^{m \times n}`: binary sign matrix, 1 bit/element.
- `a ∈ ℝ^m`: per-row scale (FP16).
- `b ∈ ℝ^n`: per-column scale (FP16).

This is **Sign-Value-Independent Decomposition (SVID)** — disentangles the sign (1-bit) from the magnitude (two FP vectors), preserving the rank-1 dominant magnitude structure that single-scale binarisation loses.

Effective bits/weight = `1 + 16/n + 16/m ≈ 1` for large m, n.

---

## OneBit vs BitNet b1.58 — pick which

| | OneBit | BitNet b1.58 |
|--|--------|--------------|
| States | {−1, +1} (binary) | {−1, 0, +1} (ternary) |
| Bits/weight | ~1 | log₂(3) ≈ 1.58 |
| Scale | per-row + per-column (rank-1) | per-tensor (scalar) |
| Training | QAT fine-tune of FP base | scratch |
| Quality | 81% of FP at 1-bit | 100% at 3B+ |
| Calibration | yes (fine-tune) | no (full pretrain) |

**OneBit's strength**: reachable by *fine-tuning an existing FP LLM* — no need to retrain from scratch. The cost: trails FP by ~20%.

**BitNet b1.58's strength**: full parity at 3B+. The cost: requires scratch training.

---

## Decomposition initialisation

The initial assignment of S, a, b matters enormously — random S leads to massive initial loss that fine-tuning cannot recover. The OneBit init:

```python
def svid_init(W):
    # W: (m, n)
    m, n = W.shape
    # 1) per-column mean
    b = W.abs().mean(dim=0)        # (n,)
    # 2) per-row mean of |W| / b_j
    a = (W.abs() / b.unsqueeze(0)).mean(dim=1)   # (m,)
    # 3) sign matrix
    S = torch.sign(W).to(torch.int8)
    # 4) alternating SVD-style refinement
    for _ in range(N_refine):
        # re-derive a given S, b
        a = ((W * S) @ torch.diag(b ** -1)).sum(dim=1) / (S.shape[1])
        # re-derive b given S, a
        b = (torch.diag(a ** -1) @ (W * S)).sum(dim=0) / (S.shape[0])
    return S, a, b
```

The refinement minimises `||W − diag(a) S diag(b)||_F` via closed-form alternating updates. Initial loss after init is within 2× the final FP loss; without init it's > 100×.

---

## Quantization-aware fine-tuning

After SVID init, fine-tune the model with:
- **S** updated by STE through `sign()`.
- **a, b** updated by standard SGD (FP16 parameters).
- Loss: standard causal-LM cross-entropy on a small corpus, or distillation from the FP teacher.

A few thousand steps on a small corpus suffice — much less than full pretraining.

---

## Inference math

```
y = W x ≈ diag(a) · (S · (diag(b) · x))
```

1. `x' = diag(b) · x` — n FP multiplies.
2. `y' = S · x'` — each `y'_i = Σ_j S_{ij} · x'_j`. With S ∈ {−1, +1}, each MAC is a sign-flipped add. Implementable as XOR + popcount on bit-packed S.
3. `y = diag(a) · y'` — m FP multiplies.

Total: O(mn) sign-MACs + O(m+n) FP MACs (amortised negligible).

The bit-packed S occupies `mn / 8` bytes — for a 4096 × 4096 linear, 2 MB instead of 16 MB (8× compression). Plus `(m + n) · 2 = 16 KB` for a, b.

---

## Why SVID beats single-scale binarisation

A single scalar scale `s` with `W ≈ s · sign(W)` discards all per-row / per-column magnitude variation. SVID recovers **two rank-1 vectors** of magnitude info per matrix — captures the dominant low-rank component of W's magnitude pattern.

The math: if you SVD `|W| ≈ σ_1 u_1 v_1^⊤ + ...`, the rank-1 truncation `σ_1 u_1 v_1^⊤` is what `diag(a) diag(b)` (with `a = σ_1 u_1`, `b = v_1`) approximates. So SVID captures the dominant singular component of the weight magnitude. Single-scale binarisation captures only the spectral norm.

---

## The numbers

LLaMA-7B 1-bit average accuracy (across ARC-c, Hellaswag, PIQA, Winogrande):

| Method | Bits | Avg acc | % of FP |
|--------|------|---------|---------|
| FP16 baseline | 16 | 65.4 | 100% |
| GPTQ-1bit | 1 | NaN | collapse |
| PB-LLM | 1 | 49.5 | 75% |
| BiLLM | 1 | 51.2 | 78% |
| **OneBit** | **1** | **53.4** | **81%** |
| BitNet b1.58 (scratch) | 1.58 | 65.4 | 100% |

OneBit at 1 bit reaches 81% of FP; BitNet b1.58 at 1.58 bits reaches 100% but requires from-scratch training. For practitioners who can't retrain, OneBit is the best ~1-bit option.

---

## Pitfalls

- **SVID init is critical.** Random S → unrecoverable training. Use the absmean-based init from the paper.
- **Fine-tune must update S via STE.** Forgetting STE means S stays at its init forever; quality plateaus at the init quality.
- **Per-row + per-column scales must be applied in the right order.** `diag(a) · S · diag(b)` is associative but the runtime ordering `(diag(a) · (S · (diag(b) · x)))` is the cheap one.
- **Don't share a, b across layers.** Each layer's magnitude structure is unique; sharing breaks the recovery.
- **The 81% number is the headline.** Don't read it as "OneBit is bad" — it's the best 1-bit-from-FP result. For 100% you need BitNet b1.58 from scratch.

---

## Connections

- [[excerpts/bitnet-b158]] — ternary alternative, requires from-scratch training but reaches FP parity.
- [[excerpts/bitnet]] — original sign-quant BitNet; OneBit's S matrix is structurally similar but with per-row/col rather than per-tensor scale.
- [[ch-04]] — [[bnn]] / [[xnor-net]] / [[dorefa-net]] as the binary-network ancestors.
- [[ch-14]] — [[aqlm]] / [[quip-sharp]] as sub-2-bit PTQ alternatives (no fine-tune; better accuracy but more bits).
- Sub-1-bit alternatives: BiLLM (binary residual), PB-LLM.
