---
chapter: ch-14
course: model-quantization
phase: read
excerpt_of: "Extreme Compression of Large Language Models via Additive Quantization (AQLM)"
source_url: https://arxiv.org/abs/2401.06118
created_at: "2026-05-21"
---

# Excerpt: AQLM — additive multi-codebook quantization

**Authors:** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh
**Year:** 2024
**Venue:** ICML 2024
**URL:** https://arxiv.org/abs/2401.06118
**Raw-data source:** [[raw-data/aqlm]]

---

## Position in the design space

At ≥ 3 bits scalar quantization (GPTQ, AWQ, RTN) is close to optimal. At ≤ 2 bits the Gish-Pierce rate-distortion bound says scalar is provably suboptimal — vector quantization wins by ~0.3 bits at the same MSE. AQLM takes the **additive** (multi-codebook) route, generalizing classical Product Quantization.

The 2024 sub-2-bit Pareto frontier:
- 2 bits: AQLM ≈ QuIP# (lattice).
- 2.5 bits: AQLM is clearly best.
- 1.5 bits: only AQLM survives among PTQ methods.

---

## The additive quantization rule

Split each linear's weight into groups of d (typically d = 8 or 16). Each group `w ∈ ℝ^d` is approximated by:

```math
\hat{w} = \sum_{m=1}^{M} C_m[i_m]
```

- `C_m ∈ ℝ^{2^B \times d}` is the m-th codebook (each row is a length-d codeword).
- `i_m ∈ \{0, ..., 2^B - 1\}` is the m-th selected index.
- M codebooks, M indices per group.

**Bits per group** = M·B (indices) + amortised codebook overhead.
**Bits per weight** = (M·B) / d + ε.

| M | B | d | bits/weight | codebook size | amortised ε |
|---|---|---|-------------|---------------|-------------|
| 2 | 8 | 8 | 2.0 | 8 KB / linear | ~0.05 |
| 1 | 12 | 8 | 1.5 | 32 KB / linear | ~0.05 |
| 4 | 8 | 16 | 2.0 | 16 KB / linear | ~0.05 |
| 1 | 16 | 16 | 1.0 | 1 MB / linear | ~0.5 |

---

## Why additive beats product quantization

Product quantization (Jégou 2011) splits each d-dim group into M **disjoint** sub-vectors and quantizes each independently. Codewords are constrained to live in a Cartesian product of M lower-dim subspaces.

Additive quantization allows each codeword to span the **full** d dimensions, with the sum recovering the group:

```
PQ:   ŵ = [C_1[i_1] | C_2[i_2] | ... | C_M[i_M]]    # concat
AQ:   ŵ = C_1[i_1] + C_2[i_2] + ... + C_M[i_M]      # sum
```

The expressive power: PQ can represent `2^{M·B}` codewords but they're constrained to a Cartesian-product structure. AQ can also represent `2^{M·B}` codewords but with arbitrary cross-coordinate correlations — strictly more expressive at the same bit budget.

---

## Encoding (the combinatorial subproblem)

Given codebooks, picking `(i_1, ..., i_M)` to minimise `||w - Σ C_m[i_m]||²` is a discrete-OPT problem. AQLM uses **beam search + iterative residual encoding**:

```python
def aqlm_encode(w, codebooks, beam_width=1):
    # codebooks: list of M tensors of shape (2^B, d)
    # initial: nearest single-codeword in each codebook to current residual
    r = w.clone()
    indices = []
    for C_m in codebooks:
        dists = ((C_m - r) ** 2).sum(dim=1)
        i_m = dists.argmin().item()
        indices.append(i_m)
        r = r - C_m[i_m]

    # local refinement: swap one i_m at a time, keep if loss drops
    for _ in range(N_refine):
        for m in range(len(codebooks)):
            current = indices[m]
            best_loss = compute_loss(w, codebooks, indices)
            for cand in range(2 ** B):
                indices[m] = cand
                loss = compute_loss(w, codebooks, indices)
                if loss < best_loss:
                    best_loss = loss
                    current = cand
            indices[m] = current
    return indices
```

For beam width > 1, keep the top-k partial assignments after each m and prune; small improvement at sub-2-bit, negligible at 2-bit.

---

## Hessian-aware codebook learning

Codebooks are *not* learned by raw weight MSE. They're learned by **output reconstruction MSE**, which is a weighted-MSE over weights with weight `diag(X^⊤ X) / n` (the per-column input variance — same as GPTQ's diagonal Hessian):

```math
\min_{\{C_m\}, \{i_m^{(g)}\}} \sum_g (\hat{w}_g - w_g)^\top H_g (\hat{w}_g - w_g)
```

where `H_g` is the sub-block of the Hessian corresponding to group g.

For full-matrix dependency (off-diagonal H), AQLM applies the GPTQ sequential column update: process groups in some order; after fixing each group's indices, propagate the residual to later groups via the Cholesky factor.

---

## Block-level joint optimization (the +0.3 ppl recovery)

After per-linear AQLM, freeze indices and fine-tune all codebooks **across all linears in a single transformer block jointly** by minimising block-output MSE on a small calibration set. This captures cross-layer dependencies that per-linear calibration misses.

Recovers ~0.3 ppl on LLaMA-2-70B at 2-bit. Computationally cheap (only codebook params, no index updates).

---

## The numbers

LLaMA-2-70B WikiText-2 perplexity (lower = better):

| Method | Bits | ppl | Δppl |
|--------|------|-----|------|
| FP16 | 16 | 3.32 | — |
| GPTQ | 3 | 3.61 | +0.29 |
| QuIP# | 2 | 3.81 | +0.49 |
| SqueezeLLM | 2 | 4.51 | +1.19 |
| **AQLM** | **2** | **3.83** | **+0.51** |
| AQLM | 2.5 | 3.50 | +0.18 |
| AQLM | 1.5 | 5.13 | +1.81 |

AQLM at 1.5 bits is the only Pareto point in that regime. At 2 bits it ties QuIP#.

---

## GPU kernel

Decode = M cache-resident table lookups + horizontal sum, then a standard FP16 matmul against the dequantized W.

```
For each output row:
  ŵ = 0_d
  for m in 1..M:
      ŵ += C_m[i_m]              # 8-element LUT read, fma into ŵ
  y_partial = ŵ · x_block        # 8-way dot product
  accumulate into y
```

The codebook is L1-resident (small, per-layer). Bottleneck shifts from HBM-bandwidth (weight reads) to ALU (decode + dot). Ships fused CUDA kernels reaching ≥ FP16 throughput on A100/H100 at 2 bits.

---

## Hyperparameters

| Knob | Value |
|------|-------|
| d (group size) | 8 (typical), also 16 |
| M (#codebooks) | 1 (low-bit) to 4 (high-bit) |
| B (bits per index) | 8 (256 codewords / codebook) |
| Calibration | 1024 sequences × 4096 tokens |
| Beam width | 1 (2-bit) to 4 (sub-2-bit) |
| Block-level FT | yes, ~100 iter Adam |
| Wall clock | ~24 h LLaMA-2-70B on 1 × A100 |

---

## Pitfalls

- **Calibration time is the limiting factor.** AQLM takes ~24h on LLaMA-2-70B vs ~2h for GPTQ. [[vptq]] cuts this 10× by going channel-independent; consider VPTQ if calibration is bottleneck.
- **Beam search is finicky.** Width 1 + refinement is fine for 2-bit; for 1.5-bit, width 4 helps measurably.
- **Codebook overhead grows with K and d.** For K=65536 (B=16), the codebook is ~1 MB per linear → must be amortised over a large weight matrix; otherwise the effective bits/weight exceeds the nominal.
- **STE fine-tuning on AQLM is suboptimal.** Use [[pv-tuning]] for proper alternating P/V optimisation; the gains are real (~0.3 ppl).
- **Doesn't compose with rotation methods.** AQLM and QuaRot are orthogonal directions in the design space; you don't usually stack them (the rotation flatness already gives the Gaussian assumption AQLM needs, but redundancy costs calibration time without quality gain).

---

## Connections

- Classical ancestor: Product Quantization (Jégou et al. 2011, "Product Quantization for Nearest Neighbor Search").
- [[excerpts/quarot]] — orthogonal direction; rotations for end-to-end, AQLM for sub-2-bit.
- [[excerpts/quip-sharp]] (ch-13) — lattice alternative; AQLM and QuIP# tie at 2-bit but use very different machinery.
- [[ch-14]] — VPTQ / GPTVQ as siblings; PV-Tuning as the proper fine-tune.
- [[ch-08]] — GPTQ Hessian as the calibration backbone.
