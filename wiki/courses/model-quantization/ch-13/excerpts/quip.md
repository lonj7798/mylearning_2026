---
chapter: ch-13
course: model-quantization
phase: read
excerpt_of: "QuIP: 2-Bit Quantization of Large Language Models With Guarantees"
source_url: https://arxiv.org/abs/2307.13304
created_at: "2026-05-21"
---

# Excerpt: QuIP — incoherence processing + LDLQ

**Authors:** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa (Cornell)
**Year:** 2023
**Venue:** NeurIPS 2023
**URL:** https://arxiv.org/abs/2307.13304
**Raw-data source:** [[raw-data/quip]]

---

## The 2-bit wall and what causes it

At 2 bits there are only 4 representation levels per channel. Naively, RTN fits its quantization grid to the max-magnitude entry, so `Δ ≈ max/2`. If `max = 50 · avg`, the bulk of weights gets ~one bin of resolution and the round error eats the signal. GPTQ helps but the residual still scales with the column max — so GPTQ at 2-bit on LLaMA-65B collapses to ≥ 30 ppl vs FP16's 3.53.

**QuIP's diagnosis** (Section 2): the failure is a property of the **distribution**, not the algorithm. Concretely, the GPTQ error bound has the form

```math
\|W X - \hat{W} X\|^2 \le \mathrm{const} \cdot d^2 \cdot \max_{ij} |W_{ij}|^2 \cdot \Delta^2
```

The `max|W_{ij}|²` factor is exactly what kills 2-bit. If you make `max|W_{ij}|` only `O(log d)` larger than the average, the bound becomes noise-limited.

---

## µ-incoherence (Definition 2)

A pair `(W, H)` is **µ-incoherent** if:

```math
\begin{aligned}
|W_{i,j}| &\le \mu \cdot \frac{\|W\|_F}{\sqrt{d_{\text{out}} \cdot d_{\text{in}}}} \quad \forall i, j \\
|\langle v_k, e_j \rangle|^2 &\le \frac{\mu}{d} \quad \forall \text{ eigvec } v_k \text{ of } H,\ \forall j
\end{aligned}
```

The first clause: max entry of W is at most µ times the RMS. The second: eigenvectors of H are not axis-aligned.

Real LLM weights routinely have `max / RMS ≈ 50–100` → µ ≈ 50–100. After random rotation: `µ = O(\sqrt{\log d}) ≈ 3–4` for d=8192.

---

## Theorem 1 (informal): incoherence kills the outlier dependence

> If `(W, H)` is µ-incoherent, then LDLQ achieves `||W X - Ŵ X||² ≤ C · μ² · d² · Δ²` for a universal constant C.

The dependence on `max|W_{ij}|` is replaced by µ. For real LLMs this is a 100×–1000× reduction in the rounding-error bound. Empirically this *exactly* tracks the transition from GPTQ-2bit collapse to QuIP-2bit functional.

---

## Incoherence processing — the rotation

Draw uniformly random orthogonal `U ∈ O(d_out)`, `V ∈ O(d_in)`. Replace:

```
W ← U^⊤ W V
H ← V^⊤ H V
```

With high probability `(U^⊤ W V, V^⊤ H V)` is `μ = O(\log d)`-incoherent. Both clauses become satisfied by the concentration-of-measure fact that a uniformly random unit vector on `S^{d-1}` has max coord `O(\sqrt{\log d / d})` w.h.p.

To make rotations cheap at inference, QuIP factors `U = U_1 U_2` where `U_1` is a random permutation and `U_2` is a random Hadamard-like structured orthogonal. Successor QuIP# replaces this with a single randomized Hadamard transform — same incoherence guarantee, `O(d \log d)` instead of `O(d^2)`.

---

## LDLQ adaptive rounding

Compute `H = L D L^⊤` (lower-triangular L with unit diagonal, diagonal D). Process columns in order:

```
for j = 1 .. d_in:
    δ_j ← quant(w_j) − w_j           # rounding residual
    for k = j+1 .. d_in:
        w_k ← w_k − L[k, j] · δ_j    # propagate to remaining columns
```

GPTQ is the special case where the propagation uses the Cholesky factor of `H^{-1}`. LDLQ's lower-triangular factor of H lets the optimality proof go through cleanly under incoherence.

**Why LDLQ is "optimal" under incoherence:** the off-diagonal `L_{k,j}` is the analytical solution to the per-column least-squares compensation problem assuming the bound on rounded entries is uniform. Under incoherence, that uniformity assumption holds; under non-incoherence, GPTQ and LDLQ both undershoot the compensation on outlier columns.

---

## Inference math

For the layer `y = W x`:

```
y_rot = (U^⊤ W V) (V^⊤ x)        # quantized GEMV in the rotated frame
y     = U y_rot                  # un-rotate
```

The right-rotation `V^⊤ x` fuses into the previous layer's output projection. The left-rotation `U y_rot` fuses into the next layer's input. With Hadamard-structured `U, V`, each fused op is `O(d \log d)`.

For embeddings: `E ← E V` so the input to the first transformer block is already rotated. For the LM head: `W_{lm} ← W_{lm} U^⊤` so the final unrotation is absorbed.

---

## The numbers — first viable 2-bit PTQ

LLaMA-65B WikiText-2 perplexity (lower = better):

| Bits | RTN | GPTQ | QuIP |
|------|-----|------|------|
| FP16 | 3.53 | 3.53 | 3.53 |
| 4 | 3.92 | 3.62 | 3.55 |
| 3 | NaN | 4.21 | 3.69 |
| 2 | NaN | ≥ 30 | **4.50** |

The 2-bit row is the headline: QuIP is the only method that doesn't collapse. Pre-QuIP, 2-bit PTQ was considered impossible at 65B; post-QuIP it loses ~1 ppl.

---

## Hyperparameters

| Knob | Value |
|------|-------|
| Bits | 2 (also 3, 4) |
| Rotations U, V | random orthogonal, Hadamard-structured |
| Adaptive rounding | LDLQ |
| Calibration | 128 sequences × 2048 tokens, C4 |
| Group size | per-row (or G=128 + scale) |
| Wall clock | ~30 min per LLaMA-65B layer on 1 × A100 |

---

## Pitfalls

- **Random seeds are part of the model.** The U, V sign vectors define the quantized weights; serialize them with the checkpoint.
- **`H` is built on rotated inputs.** Apply V to the calibration activations first, then compute `H = X_rot^⊤ X_rot`, then LDL. Reversing the order silently corrupts the Hessian.
- **µ-incoherence is *probabilistic*.** With low probability the random rotation does not flatten enough; QuIP runs a quick post-check on the rotated max and re-draws if needed. The check costs nothing.
- **Combining with `act_order` (GPTQ trick) is redundant.** LDLQ already processes columns in the LDL order, which is determined by H. `act_order` would override this and hurt the optimality.

---

## Connections

- [[excerpts/quip-sharp]] — the lattice + RHT + fine-tune upgrade.
- [[excerpts/quik]] — the W4A4 outlier-sidecar contemporary at the same lab.
- [[ch-08]] — GPTQ as the LDLQ ancestor (and the special case without rotation).
- [[ch-14]] — QuaRot / SpinQuant inherit the rotation idea and extend it to activations + KV cache.
