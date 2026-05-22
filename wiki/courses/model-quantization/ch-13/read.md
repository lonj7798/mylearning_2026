<!-- chapter: ch-13
     track: 2023-refinements
     title: QuIP + Incoherence Processing (Rotation Preview)
     sources: [[quip]], [[quip-sharp]], [[quik]]
     figures: (none — rotation visuals deferred to ch-14)
-->

# Chapter 13 — QuIP + Incoherence Processing (Rotation Preview)

> **Core insight.** Adaptive rounding (GPTQ / AdaRound) is provably near-optimal *exactly when* the weight matrix and its calibration Hessian are **incoherent** — no single entry of W and no single eigendirection of H dominates. Real LLM weights and Hessians violate this badly. You can fix it by sandwiching `W` between random orthogonal matrices `U^⊤ W V`, running Hessian-aware rounding in the rotated frame, then absorbing the rotations at inference. This single trick — the **incoherence-processing rotation** — is the seed that explodes into [[quarot]] / [[spinquant]] / [[duquant]] in ch-14.
>
> **Guideline.** For 2-bit weight-only LLM PTQ, do this in order: (1) draw a randomized Hadamard transform `R = (1/√d) diag(S) H_d`, (2) apply `W ← R^⊤ W R'` and `H ← R'^⊤ H R'`, (3) run LDLQ adaptive rounding in the rotated frame, (4) optionally swap scalar rounding for an E₈ lattice codebook (QuIP#) and add ~1 epoch of codebook fine-tuning. At inference, the Hadamards are O(d log d) and fuse into adjacent ops.

---

## Why this chapter exists

By ch-08 you have [[gptq]] — Hessian-aware sequential rounding that holds 4-bit zero-shot accuracy. By ch-09 you have [[smoothquant]] / [[awq]] for activation-aware migration. None of these crack 2 bits cleanly. RTN at 2-bit gives garbage; GPTQ at 2-bit gives a 5–30 ppl hit on LLaMA-65B. The bottleneck is structural, not algorithmic: the weight and Hessian distributions have a *small number of dominant entries / directions* that consume the entire INT2 dynamic range, leaving the bulk underquantized.

QuIP (Chee et al., NeurIPS 2023) is the first paper to (a) state precisely what condition on `(W, H)` makes adaptive rounding optimal, (b) prove a tight error bound under it, and (c) give a cheap preprocessing step that makes the condition hold. The preprocessing step is a random orthogonal rotation. Once you absorb that idea, everything in ch-14 follows: QuaRot extends it to activations + KV cache, SpinQuant *learns* the rotation, DuQuant adds a permutation, FlatQuant relaxes orthogonal to affine. Without QuIP you cannot follow the 2024 rotation literature.

Three takeaways:

1. The µ-incoherence condition and why it bounds quantization error to `O(Δ²)` instead of `O(M² Δ²)` where `M` is the largest outlier.
2. The **LDLQ** algorithm: GPTQ generalised to use the LDL factor of H directly, which lets the optimality proof go through.
3. Why a **randomized Hadamard transform** is the practical realisation — O(d log d), structure-preserving, fits in adjacent ops — and why the E₈ lattice is the optimal codebook *after* you've Gaussianised the distribution.

All of these come from [[quip]] and [[quip-sharp]] in the raw-data library. [[quik]] adds the W4A4 outlier-sidecar instance for context.

---

## 1. The 2-bit wall and what causes it

Recall the Bennett uniform-noise bound from ch-01: for round-to-nearest at bin width `Δ`, the squared error on a uniformly-distributed input is `Δ²/12`. This is the floor adaptive rounding can hope to approach.

Now look at a real LLaMA weight column. The maximum-magnitude entry can be 50–100× the average magnitude. If you fit `Δ` to the max (so nothing clips), `Δ ≈ max/2^(b-1)`. At b=4, Δ ≈ max/7. At b=2, Δ ≈ max/1 — *the bulk of weights has only one bin* and all the resolution is being burnt on the long tail. RTN error explodes from `Δ²/12 ≈ (avg)²/12` (good) to `Δ²/12 ≈ (50·avg)²/12` (catastrophic).

GPTQ helps by spreading quant error from each column into later columns via the inverse Hessian, but the residual after compensation still scales with the column max. At 2 bits the residual eats the whole signal budget.

You cannot fix this by being cleverer about rounding alone. You have to **change the distribution before you round.**

---

## 2. The µ-incoherence condition

From [[quip]] (Definition 2): a pair `(W, H)` is **µ-incoherent** if for all `i, j` and all eigenvectors `v_k` of H:

```
|W_{i,j}|       ≤  μ · ||W||_F / √(d_out · d_in)
|⟨v_k, e_j⟩|²  ≤  μ / d
```

Two clauses. The first bounds the largest entry of W by a constant multiple of the average. The second bounds the inner product of any Hessian eigenvector with any axis-aligned standard basis vector by `√(μ/d)` — i.e. no eigenvector of H is concentrated on a few coordinates.

Heavy-tailed real LLM weights violate clause 1 (the outlier entry blows up the bound). Hessians with axis-aligned eigenstructure violate clause 2.

**Theorem 1 of QuIP** (informal). For µ-incoherent `(W, H)`, adaptive rounding methods (GPTQ / LDLQ family) achieve a quantization error scaling as `O(d² · μ² · Δ²)` — *independent of the largest entry of W*. Without incoherence, the bound is `O(d² · M² · Δ²)` with `M = max|W_{ij}|`.

So the 2-bit wall has a precise diagnosis: real LLM weights have `M / avg ≈ 50–100`, blowing up the rounding bound by 2500–10000×. If you can drive `M / avg → O(log d)`, you recover the noise-limited regime.

---

## 3. Incoherence processing — the rotation step

The fix is one of the cleanest tricks in the literature. Draw **uniformly random orthogonal matrices** `U ∈ O(d_out)`, `V ∈ O(d_in)`. Replace:

```
W ← U^⊤ W V
H ← V^⊤ H V
```

(H transforms with V on both sides because H = X^⊤ X and X carries the V rotation through the input projection.)

**Standard random-matrix-theory fact:** with `U, V` uniformly sampled from the orthogonal groups, `U^⊤ W V` is µ-incoherent with `μ = O(log d)` with high probability. The maximum entry of the rotated matrix is bounded by `O(√(log d / d)) · ||W||_F` — a `√d / √log d` reduction from the worst-case unrotated bound. For `d = 8192`, that's ~30× reduction in the largest entry — exactly the cliff the 2-bit wall sits on.

The intuition is sphere-packing. A random rotation spreads the energy of any concentrated vector uniformly over the d-sphere; the maximum coordinate of a unit vector picked uniformly on `S^{d-1}` is `O(√(log d / d))` w.h.p. (a classical concentration result). QuIP exports this concentration onto the rows / columns of W.

**Inference.** The layer matmul `y = W x` becomes:

```
y_rot = (U^⊤ W V) · (V^⊤ x)        # quantized GEMV in the rotated frame
y     = U · y_rot                  # un-rotate
```

The right-rotation `V^⊤ x` fuses into the previous layer's output projection. The left-rotation `U y_rot` fuses into the next layer's input. As long as `U, V` are structured (Hadamard), each fused op is `O(d log d)` — small overhead.

---

## 4. LDLQ — adaptive rounding under incoherence

GPTQ processes weight columns in some order (default: input order, or `act_order` = descending `diag(H)`) and propagates the quant error of each column into the remaining ones via the inverse-Hessian Cholesky factor. QuIP introduces **LDLQ**, which is GPTQ generalised to use the **LDL decomposition** of H directly.

Compute `H = L D L^⊤` (lower-triangular L with unit diagonal, diagonal D ≻ 0). Process columns `j = 1, ..., d_in` in the order implied by L. For each column j of the (rotated) weight W:

```
δ_j = quant(w_j) − w_j                          # quantization error of column j
w_{k > j} ← w_{k > j} − L_{k,j} · δ_j           # propagate to later columns
```

The off-diagonal entry `L_{k,j}` tells you precisely how much of column j's quant error should be subtracted from column k to leave the output-MSE objective minimised conditional on column j being fixed at its quantized value.

**Why LDL not Cholesky.** GPTQ's classical formulation uses the Cholesky factor of `H^{-1}` (upper triangular). LDLQ's lower-triangular LDL factor of H itself simplifies the proof: under µ-incoherence the off-diagonal entries of L are uniformly bounded, and the propagated error stays in the noise-limited regime. With Cholesky-of-H⁻¹ the bookkeeping is less clean.

**Equivalence to GPTQ when no rotation.** With `U = V = I` (no rotation), LDLQ and GPTQ produce essentially the same quantized weights — they're two views of the same adaptive-rounding scheme. LDLQ is the version whose error bound goes through cleanly *combined with* the incoherence preprocessing. GPTQ is a special case.

---

## 5. Putting it together — the QuIP algorithm

```python
def quip_quantize_layer(W, H, bits=2):
    # 1) draw random orthogonal U, V (Hadamard-structured in practice)
    U = sample_random_orthogonal(W.shape[0])
    V = sample_random_orthogonal(W.shape[1])

    # 2) incoherence processing
    W_rot = U.T @ W @ V
    H_rot = V.T @ H @ V

    # 3) LDLQ adaptive rounding in the rotated frame
    L, D = ldl_decompose(H_rot)
    W_q = torch.zeros_like(W_rot)
    for j in range(W_rot.shape[1]):
        # round column j to nearest grid point on the b-bit scalar grid
        W_q[:, j] = scalar_quantize(W_rot[:, j], bits)
        # propagate residual to remaining columns
        delta = W_q[:, j] - W_rot[:, j]
        for k in range(j + 1, W_rot.shape[1]):
            W_rot[:, k] -= L[k, j] * delta

    # 4) store W_q + the U, V seeds; inference will absorb them
    return W_q, U, V
```

At inference time the per-layer call is:

```python
def quip_linear_forward(x, W_q, U, V):
    x_rot   = V.T @ x          # fast Hadamard transform if V is RHT
    y_rot   = W_q @ x_rot      # quantized GEMV
    y       = U @ y_rot        # fast Hadamard transform if U is RHT
    return y
```

The two rotations are O(d log d) each via FWHT and fuse into adjacent LayerNorm / residual operations in a fused-kernel implementation.

---

## 6. QuIP# — randomized Hadamard + E₈ lattice + fine-tune

[[quip-sharp]] (Tseng et al., ICML 2024) upgrades QuIP along three axes. This is the version you'd actually deploy.

### 6.1 Randomized Hadamard Transform replaces uniform random orthogonal

Let `S ∈ {±1}^d` be a uniform random sign vector and `H_d` the Sylvester Hadamard (defined recursively, `d` a power of 2). Define

```
R = (1/√d) · diag(S) · H_d
```

R is orthogonal, achieves the **same** µ = O(log d) incoherence guarantee as a uniform random orthogonal, but the matmul `R x` costs `O(d log d)` instead of `O(d²)` via the Fast Walsh–Hadamard Transform. This is the difference between a feasible inference path and a research curiosity.

```python
def fast_walsh_hadamard(x):
    """In-place FWHT, O(d log d). d must be a power of 2."""
    h = 1
    while h < len(x):
        for i in range(0, len(x), h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j], x[j + h] = a + b, a - b
        h *= 2
    return x  # caller divides by √d outside

def randomized_hadamard(x, S):
    # x ← diag(S) x, then FWHT, then divide by √d
    return fast_walsh_hadamard(S * x) / math.sqrt(len(x))
```

### 6.2 E₈ lattice codebook replaces scalar rounding

After incoherence processing, the entries of `R^⊤ W R'` are approximately i.i.d. Gaussian (concentration of measure). For an isotropic Gaussian source, scalar round-to-nearest is provably suboptimal — vector quantization on the right lattice beats it by `~0.3` bits at the same MSE.

The **E₈ lattice** in ℝ⁸ is the densest sphere packing in 8 dimensions (kissing number 240; Viazovska proved its optimality in 2017, Fields Medal). For an isotropic Gaussian, lattice quantization with E₈ reaches within `~0.1` bit of the Shannon rate-distortion lower bound.

QuIP# groups every 8 rotated weights into a vector `v ∈ ℝ⁸`, then snaps `v` to the nearest E₈ lattice point. The codebook in practice is **E₈P**, a half-integer-shifted subset of E₈ containing 256 codewords (so each 8-vector costs 8 bits = 1 bit/weight nominal; combined with a sign bit and a per-block scale, the effective rate is 2 bits/weight).

| Bits | Codebook | Bits/weight |
|------|----------|-------------|
| Scalar INT2 (RTN) | 4 levels | 2.0 |
| E₈P (256 entries) | 8-dim lattice | 2.0 (8 bits / 8 weights) |
| AQLM (M=2, B=8) | additive 2× 256 | 2.0 |

The lattice version dominates the rate-distortion frontier whenever the source is Gaussian — which is exactly what incoherence processing guarantees.

### 6.3 Fine-tuning the codebook assignments

After PTQ, freeze the layer structure and run ~1 epoch of cross-entropy distillation from the FP teacher on a small calibration corpus (256 sequences from C4). Update:

- The per-layer scale factors (continuous).
- Occasionally, the codeword index for the worst-error 8-vectors (discrete).
- The Hadamard seeds (the sign vector S) for small per-layer perturbations.

Recovers ~0.2–0.5 ppl on LLaMA-2 70B at 2-bit. This pattern — "PTQ, then briefly fine-tune the indices" — is exactly what [[pv-tuning]] (ch-14) systematises.

---

## 7. Numbers from the paper

**QuIP (Chee 2023), LLaMA-65B at 2-bit weight-only:**

| Method | Bits | WikiText-2 ppl | Δppl vs FP16 |
|--------|------|----------------|--------------|
| FP16 | 16 | 3.53 | — |
| GPTQ | 2 | ≥ 30 | +25+ (collapse) |
| QuIP | 2 | 4.5 | +1.0 |
| QuIP | 3 | 3.69 | +0.16 |

QuIP is the **first** PTQ method that doesn't collapse at 2-bit on 65B.

**QuIP# (Tseng 2024), LLaMA-2-70B at 2-bit:**

| Method | Bits | WikiText-2 ppl |
|--------|------|----------------|
| FP16 | 16 | 3.32 |
| QuIP | 2 | 4.40 |
| AQLM | 2 | 3.83 |
| QuIP# | 2 | 3.81 |
| QuIP# (3-bit) | 3 | 3.41 |

QuIP# at 2-bit ≈ QuIP at 3-bit; the lattice + fine-tune recovers a whole bit. Beats GPTQ-4bit on memory and matches GPTQ-3bit on accuracy.

---

## 8. The bridge to ch-14 — rotations escape QuIP

QuIP applies rotations to **weights only**. At inference the rotations are absorbed into the matmul and the activations never see them explicitly. This is enough for weight-only PTQ but it leaves the activation outlier problem untouched.

The 2024 rotation explosion (ch-14) makes one move: **insert the rotations into the residual stream itself**, so activations are rotated in-place. Once you commit to this, you can quantize activations and KV cache too, because the activation distribution after rotation is also Gaussian-ish. The chain runs:

- [[quarot]] — random Hadamard rotation of the residual stream; enables full W4A4KV4.
- [[spinquant]] — *learn* the rotation via Cayley parametrization on the Stiefel manifold; up to 13 points better than random Hadamard.
- [[duquant]] — block-diagonal rotation + zigzag channel permutation for the "massive outlier" regime that single rotations don't flatten enough.
- [[flatquant]] — relax orthogonal to affine; trade spectral preservation for more aggressive flattening.

All of these inherit the µ-incoherence framing of QuIP. The math is the same; the deployment surface expands from weights-only to end-to-end.

The other branch — sub-2-bit via vector/additive codebooks — comes from the QuIP# lattice insight: [[aqlm]] (additive multi-codebook), [[vptq]] / [[gptvq]] (Hessian-aware VQ), plus [[pv-tuning]] (proper discrete fine-tuning to replace STE).

---

## 9. QUIK — the W4A4 outlier-sidecar instance (context)

[[quik]] (Ashkboos 2023, same IST-Austria lab as GPTQ) is a different attack on the same problem: instead of rotating the distribution flat, keep the 99% bulk at INT4 and route the top 1% outlier rows/columns through a parallel INT8 GEMM, fused in a single kernel. Achieves real W4A4 end-to-end speedup (3.4× over FP16 on A100, LLaMA-2-7B).

QUIK is the "outlier sidecar" school. QuaRot (ch-14) is the "rotate the outliers away" school. The two compete head-to-head in 2024; QuaRot wins on accuracy (no special path needed) and QUIK persists as a deployment option when you want to avoid the online Hadamard cost. The same lab eventually publishes both — they are not philosophical enemies, they are different points on the design space.

---

## 10. Pitfalls

- **`d` must be a power of 2 for FWHT.** If your hidden dim isn't (e.g. d=6144 = 6×1024), block the Hadamard: apply RHT to chunks of 2048 or 1024 separately. You lose some incoherence but keep the speed.
- **Random seed `S` must be persisted.** The sign vector S is part of the model now. Lose it and you can't run the quantized weights at all. Treat it like a checkpoint hyperparameter.
- **Hadamard does not commute with RMSNorm scaling factors.** Folding the rotation into adjacent linears works cleanly; folding through RMSNorm requires absorbing the gain into the rotation. [[quarot]] writes this out carefully.
- **The H matrix in LDLQ is the per-layer Hessian `X^⊤ X` over calibration activations**, *after* the input rotation V is applied. Order of operations matters: rotate, then compute H, then LDL.
- **2-bit + no fine-tune still loses ~0.5–1 ppl.** If you absolutely need PTQ (no calibration backprop), QuIP gets you 2-bit functional but expect a measurable quality gap. The QuIP# fine-tune closes most of it.

---

## Connections and what's next

- **[[gptq]] / ch-08** — LDLQ is GPTQ's adaptive-rounding step viewed through the LDL factor; QuIP is the *preprocessing* that makes the GPTQ bound actually tight.
- **[[adaround]] / ch-04** — the spiritual parent of adaptive PTQ; AdaRound learns per-weight rounding direction via a rectified sigmoid. GPTQ closed-forms it; QuIP makes the closed form provably near-optimal.
- **[[quarot]] / ch-14** — same Hadamard trick, but rotations live in the residual stream so activations + KV cache can also be quantized.
- **[[spinquant]] / ch-14** — learn the rotation instead of using a random Hadamard. The optimality of random Hadamard is an *average-case* result; SpinQuant exploits the structured-outlier nature of real LLM activations.
- **[[aqlm]] / [[vptq]] / [[gptvq]] / ch-14** — vector/additive quantization successors of QuIP#'s lattice idea, at sub-2-bit.
- **[[pv-tuning]] / ch-14** — replaces QuIP#'s STE-style fine-tune with a proper alternating P-step / V-step on (continuous codebook, discrete indices).

## Further reading

- [[quip]] — Chee et al. 2023, the original paper with the µ-incoherence theorem.
- [[quip-sharp]] — Tseng et al. 2024, the lattice + RHT + fine-tune upgrade.
- [[quip-sharp-2024]] — community follow-up notes and kernel benchmarks.
- [[quik]] — Ashkboos et al. 2023, the outlier-sidecar W4A4 contemporary.
- Sphere-packing background: Conway & Sloane, *Sphere Packings, Lattices and Groups* (Springer 1999).
- Concentration of measure (the "max coord of random unit vector ≈ √(log d / d)" result): Vershynin, *High-Dimensional Probability*, Ch. 5.

## Excerpts

- [excerpts/quip.md](excerpts/quip.md) — µ-incoherence definition, the LDLQ algorithm, the 2-bit LLaMA-65B numbers.
- [excerpts/quip-sharp.md](excerpts/quip-sharp.md) — randomized Hadamard transform, E₈P codebook, the fine-tune pass.
- [excerpts/quik.md](excerpts/quik.md) — W4A4 mixed-precision GEMM with the INT8 outlier sidecar.
