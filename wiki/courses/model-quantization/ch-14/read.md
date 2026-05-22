<!-- chapter: ch-14
     track: 2024-maturation
     title: 2024 W4A4 + Sub-2-bit — Rotations + AQLM + HQQ
     sources: [[quarot]], [[spinquant]], [[duquant]], [[rotation-and-quantization]], [[aqlm]], [[pv-tuning]], [[vptq]], [[gptvq]], [[hqq]], [[atom]], [[qserve]]
     figures: (none — interactive rotation viz deferred; sufficient diagrams in source papers)
-->

# Chapter 14 — 2024 W4A4 + Sub-2-bit: Rotations + AQLM + HQQ

> **Core insight.** 2024 splits the low-bit LLM PTQ frontier into two parallel breakthroughs. (1) **Rotations escape weight-only.** QuIP's incoherence trick (ch-13) moves into the residual stream itself, so activations and the KV cache become quantizable. Random Hadamard ([[quarot]]) → learned orthogonal on the Stiefel manifold ([[spinquant]]) → block-rotation + zigzag permutation ([[duquant]]) → affine non-orthogonal ([[flatquant]]) all instantiate the same algebra. End state: lossless **W4A4KV4** at 70B. (2) **Sub-2-bit via vector codebooks.** [[aqlm]] generalises product quantization to additive multi-codebook; [[vptq]] / [[gptvq]] make the Hessian update vector-aware; [[pv-tuning]] replaces STE fine-tuning with a proper alternating P-step / V-step on (continuous codebook, discrete indices). [[hqq]] becomes the "iterate fast" data-free tool. Production stacks ([[atom]] = W4A4KV4, [[qserve]] = W4A8KV4) ship.
>
> **Guideline.** For 2024-style production W4A4: insert a QuaRot Hadamard rotation, run GPTQ on the rotated weights, dynamic per-token INT4 on activations, INT4 KV cache. If you need more accuracy, upgrade the rotation to SpinQuant (learn it). If you need *less* than 2 bits/weight, use AQLM (additive codebook) + PV-Tuning. If you need fast data-free, use HQQ. If you need a serving stack, pick **QServe (W4A8KV4)** on Hopper, **Atom (W4A4KV4)** on Ampere.

---

## Why this chapter exists

This is the densest chapter in the course because 2024 was the year low-bit LLM quantization went from research curiosity to production reality. The W4A4 papers stop collapsing to NaN. The sub-2-bit papers reach the Pareto frontier where 2-bit AQLM matches 3-bit GPTQ. Serving stacks ship measurable end-to-end speedups.

You need to come out of this chapter with five working models:

1. The **QuaRot computational-invariance rotation pattern** — where the rotations live, why folding into weights is free, why the Hadamard kills outliers.
2. **SpinQuant's Cayley parametrization** of the Stiefel manifold — how you optimise on the orthogonal group with vanilla SGD.
3. The **AQLM additive quantization rule** `w ≈ Σ_m C_m[i_m]` — why it beats scalar at low bits, why PQ is the wrong baseline.
4. **PV-Tuning's P-step / V-step** alternation — why STE on discrete indices is biased and what the principled replacement looks like.
5. **HQQ's half-quadratic splitting** — closed-form alternating updates, data-free, the right tool when you can't afford calibration.

Plus the production framings: Atom for W4A4KV4 deployment, QServe for W4A8KV4 on Hopper.

Lean on the chapter; readers should spend 2–3× a normal chapter's time here.

---

## 1. QuaRot — rotations move into the residual stream

[[quarot]] (Ashkboos et al. 2024) is the first paper to make rotations operate **end-to-end**, not just on weights. The key construction is **computational invariance**: insert an orthogonal `Q` into the residual stream and into all the projections, in a pattern such that the FP outputs are unchanged.

### 1.1 The invariance trick

For a residual block `y = x + f(x)` and any orthogonal Q (`Q^⊤ Q = I`):

```
y = Q x + Q · f(Q^⊤ · Q x) = Q (x + f(x))
```

The residual stream is rotated by Q **everywhere** after insertion. The LM head absorbs `Q^⊤` at the end. Logits are bit-identical in FP.

### 1.2 Where Q lives in the weights (folded offline)

```
Embedding:    E    ← E Q^⊤
QKV:          W_q, W_k, W_v   ← W_{q,k,v} · Q^⊤     # input is Q x
Out:          W_o  ← Q · W_o                          # output re-enters residual
FFN-up/gate:  W_up, W_gate    ← W_{up,gate} · Q^⊤
FFN-down:     W_down          ← Q · W_down
LM head:      W_lm  ← W_lm · Q
```

Compute the offline-folded weights once; the rotation matrix Q never appears at inference. Zero runtime cost for the residual-stream rotation.

### 1.3 The four rotation slots

| Slot | Where | Online cost | Purpose |
|------|-------|-------------|---------|
| R1 | residual stream | offline-folded, free | per-token outliers across all blocks |
| R2 | between V and W_o | online FWHT | per-head V outliers |
| R3 | between SwiGLU output and W_down | online FWHT | gated-activation spikes |
| R4 | on K after RoPE | online FWHT (fused with attn) | K-cache outliers |

R1 is free. R2/R3/R4 cost `O(d \log d)` per token each, fused into the surrounding kernels.

### 1.4 Why a Hadamard kills outliers

For a vector x with one outlier of magnitude M among d coordinates: `(H_d x)_i ≈ M / √d` for all i (spread uniformly). The post-rotation max coordinate is reduced by ~√d.

| d | √d outlier reduction |
|---|--------------------|
| 4096 (LLaMA-7B hidden) | 64× |
| 8192 (LLaMA-70B hidden) | 90× |
| 14336 (LLaMA-70B FFN) | 120× |

A 100× outlier becomes ~1.1× the bulk after rotation in 70B hidden state. Activations are now in the regime where INT4 RTN works.

### 1.5 Numbers

LLaMA-2-70B WikiText-2, W4A4KV4 PTQ:

| Method | ppl | Δppl vs FP16 |
|--------|-----|--------------|
| FP16 | 3.32 | — |
| SmoothQuant W4A4 | NaN | collapse |
| OmniQuant W4A4 | 6.11 | +2.79 |
| **QuaRot W4A4KV4** | **3.79** | **+0.47** |

First lossless-grade W4A4KV4 at 70B. 99% of zero-shot accuracy retained.

---

## 2. SpinQuant — learn the rotation on the Stiefel manifold

[[spinquant]] (Liu et al. 2024) makes one observation: among the infinite family of computationally-invariant rotations, different choices differ by **up to 13 points** in downstream accuracy. Random Hadamard is not optimal. The right move is to *learn* the rotation.

### 2.1 The optimization problem

Each rotation Q must satisfy `Q^⊤ Q = I`. The set of such matrices is the **Stiefel manifold** `St(d, d) = {R ∈ ℝ^{d×d} : R^⊤ R = I}`. Vanilla SGD on R does not preserve orthogonality; you'd have to project after each step.

### 2.2 Cayley parametrization

Parametrize R via a skew-symmetric matrix `A = -A^⊤`:

```math
R = (I - A)(I + A)^{-1}
```

This is the **Cayley map**: a diffeomorphism from the space of skew-symmetric matrices onto a dense subset of SO(d). Any unconstrained gradient step on A produces a valid orthogonal R — no projection needed, no QR retraction.

Equivalent view: optimise R on St(d, d) via Riemannian SGD; the Cayley map is the closed-form retraction.

```python
def cayley(A):
    # A is skew-symmetric (A = -A.T)
    I = torch.eye(A.shape[0])
    return (I - A) @ torch.linalg.solve(I + A, I)

# learnable rotation
A = torch.nn.Parameter(0.01 * torch.randn(d, d))
A.data = (A.data - A.data.T) / 2   # project to skew at init

def get_rotation():
    A_skew = (A - A.T) / 2          # re-skew after each step
    return cayley(A_skew)
```

The `(A − A^⊤)/2` projection is the only enforcement step needed — and it's cheap. (`A.grad` may push A off the skew-symmetric subspace; this re-symmetrisation keeps it on.)

### 2.3 Loss

Per-block reconstruction:

```math
\mathcal{L}(R) = \sum_{\text{blocks}} \| f_{\text{FP}}(x) - f_{\text{quant}}(x; R) \|^2
```

Optimised over a few hundred SGD steps on a small calibration corpus (WikiText-2 segments). After convergence, freeze R, quantize weights with GPTQ in the R-rotated frame, run activations with dynamic per-token INT4.

### 2.4 Why learning beats random Hadamard

Random Hadamard uniformly spreads any *single* outlier coordinate over all d dimensions — optimal under a uniform prior on outlier location. But real LLM activations have **structured outliers**: specific channels, specific heads, with persistent locations. Learned R can match this structure, concentrating its rotation mass where the outliers actually live.

Empirically: SpinQuant improves over QuaRot by up to **45% on LLaMA-3-8B** at W4A4KV4, closing the gap to FP16 to 2.9 points on LLaMA-2-7B.

### 2.5 What's learned where

- **R1** (residual): dense d×d learnable rotation, full Cayley parametrization.
- **R2** (V→O): dense d×d learnable.
- **R3, R4** (FFN-down, K-cache): block-diagonal Hadamards for inference speed; only the per-block sign vectors are learnable.

R1 and R2 dominate the accuracy gain; R3/R4 are kept structured to preserve the FWHT speed.

---

## 3. DuQuant — block rotation + zigzag permutation for "massive" outliers

[[duquant]] (Lin et al. NeurIPS 2024 Oral) observes that real LLM activations have **two outlier regimes**: "normal" outliers (broad, ~10× bulk, SmoothQuant-style) and "massive" outliers (a few coordinates, >100× bulk, layer-specific). A single Hadamard handles normal outliers but doesn't fully flatten massive ones because the **block** containing the outlier still has elevated variance vs other blocks.

### 3.1 Step 1 — outlier-prior block rotation

Identify outlier channel indices `{i_1, ..., i_k}` from calibration. Build a block-diagonal rotation `R = blkdiag(R_1, ..., R_B)` where each block `R_b` (size 128×128 typical) includes the outlier channels falling into block b. Each `R_b` is a Greedy-Householder rotation aligning the dominant outlier direction with the block mean, redistributing the spike across all 128 in-block channels.

### 3.2 Step 2 — zigzag channel permutation

After rotation, blocks still have unequal variance. Sort blocks by σ², then interleave high and low:

```
σ²(b_1) ≥ σ²(b_2) ≥ ... ≥ σ²(b_B)
→ permute to b_1, b_B, b_2, b_{B-1}, b_3, b_{B-2}, ...
```

Adjacent blocks have complementary variance → per-group quant scales sit near a global average rather than swinging.

### 3.3 Step 3 — uniform W4A4 PTQ

Standard per-channel weight INT4 + per-token activation INT4 RTN. No outlier path needed — both rotation and permutation are fused into surrounding weights.

### 3.4 Numbers

LLaMA-3-8B WikiText-2 W4A4:

| Method | ppl |
|--------|-----|
| FP16 | 6.13 |
| SmoothQuant W4A4 | NaN |
| QuaRot | 8.16 |
| SpinQuant | 7.36 |
| **DuQuant** | **7.18** |

DuQuant wins on LLaMA-3 specifically because LLaMA-3 has more pronounced massive outliers than LLaMA-2 — block rotation matches the structure.

---

## 4. The rotation design space (consolidated)

[[rotation-and-quantization]] gives the unified view. Every rotation-based method inserts orthogonal Q in computationally-invariant positions; they differ in three axes:

| Method | Q type | Calibration | Online cost |
|--------|--------|-------------|-------------|
| [[quip]] (ch-13) | random orth + Hadamard | none for R | O(d log d) for residual rotation |
| [[quarot]] | random Hadamard | none | online FWHT for R2/R3/R4 |
| [[spinquant]] | learned dense Stiefel | ~500 SGD steps | same as QuaRot |
| [[duquant]] | block-diag rotation + permutation | outlier-index calibration | small block matmul |
| [[flatquant]] | learned affine (non-orthogonal) | block-MSE optimization | small Kronecker matmul |
| [[quip-sharp]] | RHT + lattice codebook | E₈ codebook tabulated | lattice-decode kernel |

**Orthogonality preserves the L² spectrum of W** → no condition-number blowup. Non-orthogonal A (FlatQuant) flattens more aggressively but introduces a numerical-stability cost.

---

## 5. AQLM — additive quantization for sub-2-bit weights

[[aqlm]] (Egiazarian et al. ICML 2024) takes the **vector-quantization route** to sub-2-bit. The motivation: at 2 bits scalar quantization is provably suboptimal (Gish-Pierce bound is loose). Vector or **additive** quantization with multiple small learned codebooks reaches the Pareto frontier by exploiting joint structure across groups of weights.

### 5.1 The additive quantization rule

Split each linear's weight into groups of d (typically d = 8 or 16). Each group `w ∈ ℝ^d` is approximated by:

```math
\hat{w} = \sum_{m=1}^{M} C_m[i_m]
```

- `C_m ∈ ℝ^{2^B × d}` is the m-th codebook (each row is a length-d codeword).
- `i_m ∈ {0, ..., 2^B − 1}` is the m-th selected index.
- M codebooks, M indices per group.

**Bits per group** = M·B (indices) + amortised codebook overhead.
**Bits per weight** = (M·B)/d + ε.

Example: M=2, B=8, d=8 → 16 bits per 8 weights = **2.0 bits/weight**. Codebook size: `2 · 256 · 8 · FP16 = 8 KB per linear`, amortised over millions of weights → ε ≈ 0.05.

### 5.2 Why additive beats product quantization

Product quantization (PQ, Jégou 2011) splits the d-dim group into M **disjoint** sub-vectors and quantizes each independently. Codewords are constrained to the disjoint subspace.

AQ allows each codeword to span the **full** d dimensions, with the sum recovering the group. Captures cross-coordinate structure that PQ rules out by construction. At the same bit budget, AQ has strictly more expressive power.

### 5.3 Encoding (the discrete-OPT problem)

Picking the best `(i_1, ..., i_M)` given codebooks is combinatorial. AQLM uses beam search + iterative residual encoding:

```python
def aqlm_encode(w, codebooks):
    # w ∈ ℝ^d, codebooks: list of M tensors of shape (2^B, d)
    r = w.clone()
    indices = []
    for C_m in codebooks:
        # nearest codeword to current residual
        i_m = ((C_m - r) ** 2).sum(dim=1).argmin()
        indices.append(i_m.item())
        r = r - C_m[i_m]
    # local refinement: swap one i_m at a time
    indices = local_search(w, codebooks, indices, swap_radius=8)
    return indices
```

### 5.4 Hessian-aware codebook learning

Codebooks are not learned by raw weight MSE; they're learned by **output reconstruction MSE**, which is a Hessian-weighted MSE over weights with weight `diag(X^⊤ X) / n` (the per-column input variance — same as GPTQ's diagonal Hessian). For full-matrix dependency, AQLM applies the GPTQ sequential column update.

### 5.5 Block-level joint optimization

After per-linear AQLM, freeze indices and fine-tune all codebooks in a transformer block jointly to minimize block-output MSE on a small calibration set. Recovers ~0.3 ppl.

### 5.6 Numbers

LLaMA-2-70B WikiText-2:

| Method | Bits | ppl |
|--------|------|-----|
| FP16 | 16 | 3.32 |
| GPTQ | 3 | 3.61 |
| QuIP# | 2 | 3.81 |
| **AQLM** | **2** | **3.83** |
| AQLM | 2.5 | 3.50 |

AQLM at 2 bits is Pareto-optimal in the sub-3-bit regime. The Pareto frontier from this chapter:
- 4 bits → GPTQ / AWQ.
- 3 bits → AQLM or QuIP#.
- 2.5 bits → AQLM.
- 2 bits → AQLM ≈ QuIP#.
- 1.58 bits → BitNet b1.58 (ch-16, scratch training only).

---

## 6. PV-Tuning — proper fine-tuning for compressed codes

[[pv-tuning]] (Malinovskii et al. 2024) addresses a subtle but important problem: STE fine-tuning of AQLM / QuIP# checkpoints is **biased**. STE backprops through the discrete quantizer with `dq/dx ≈ 1` — but the true derivative is 0 a.e. and δ on bin boundaries. In the extreme low-bit regime the bin boundaries dominate and STE's bias drives optimization off-direction.

### 6.1 The setup

Compressed weights:
- AQLM form: `ŵ_g = Σ_m C_m[i_m^{(g)}]`
- QuIP# form: `ŵ_g = LatticeDecode(i^{(g)})`

Variables:
- Codebooks `{C_m}`: continuous, in `ℝ^{K×d}`.
- Indices `{i_m^{(g)}}`: discrete, in `[0..K)`.

Objective (calibration cross-entropy):
```math
\mathcal{L}(C, i) = \mathbb{E}_{x,y \sim \mathcal{D}_{\text{cal}}}[-\log p_\theta(y|x; C, i)]
```

### 6.2 The P/V alternation

```
1. P-step (parameter): freeze indices i; SGD on codebooks C.
                       Fully differentiable, standard backprop.

2. V-step (value):     freeze codebooks C; for each group g, find better indices:
                       i_m^{(g)} ← argmin_{j ∈ N(i_m^{(g)})} L_g(C, i with i_m^{(g)} ← j)
                       N = k-nearest-neighbour set in codebook space (k=8 typical).

3. Alternate until convergence.
```

The V-step makes **discrete** jumps explicitly, with the **true** loss difference as the criterion — no gradient estimation needed for the discrete part.

### 6.3 Why this beats STE

STE assumes a continuous gradient flows through the quantizer. The true derivative of `argmin` over a discrete set is not continuous; STE's identity approximation is *biased* and the bias does not vanish at the optimum. PV's V-step uses the actual loss change from each candidate swap — unbiased, and the bias-free guarantee gives a `O(1/√T)` convergence rate that STE doesn't have.

### 6.4 Cost

P-step: standard quantized-forward / dequantize-backward, same cost as STE.
V-step: per-group loss evaluation across k=8 candidates → adds ~30% wall clock.
Total: a few hundred steps on a small calibration set (hundreds of MB).

### 6.5 Numbers

LLaMA-2-7B WikiText-2 at 2-bit AQLM:

| Fine-tune | ppl |
|-----------|-----|
| none | 6.93 |
| STE | 6.32 |
| **PV-Tuning** | **5.99** |
| FP16 baseline | 5.47 |

PV-Tuning recovers ~50% of the STE → FP16 gap. The same ~0.3 ppl gain extends to QuIP# 2-bit.

---

## 7. VPTQ / GPTVQ — Hessian-aware vector PTQ

Two sibling papers, same lineage as AQLM, different design choices.

### 7.1 VPTQ (Liu et al. EMNLP 2024)

Channel-Independent Second-Order Optimization (CISO): per output channel,

```
1. Build per-channel diagonal Hessian H = diag(X^⊤ X) restricted to that channel.
2. Init indices by nearest-codeword in H-weighted norm.
3. Sequentially update groups g = 1..G with GPTQ-style Cholesky residual propagation.
4. Re-fit codebook C via weighted k-means on (w_g, H_g).
5. Iterate 3-4 a few times.
```

Each output channel is solved independently → trivially parallel across channels → 10–19% of AQLM's calibration time. Plus residual + outlier sub-codebooks for the top 1% high-error groups.

### 7.2 GPTVQ (van Baalen et al. Qualcomm 2024)

Interleaves vector quantization of weight columns with GPTQ-style Hessian-residual updates. Generalises GPTQ from scalar (d=1) to vector (d=2, 4, 8) and shows the Pareto frontier **monotonically improves with d** at fixed bits/weight — the "blessing of dimensionality" thesis.

```python
# GPTVQ column-by-column update (sketch)
for j in range(0, d_in, d):  # block of d input dims at a time
    for each output channel:
        w_g = W[c, j:j+d]
        # snap to nearest codeword in H-weighted norm
        c_best = argmin_{c in C} (c - w_g)^T H_g (c - w_g)
        W[c, j:j+d] = c_best
        delta = c_best - w_g
        # propagate residual to remaining input dims via H Cholesky
        propagate_via_cholesky(delta, j, d_in)
```

Codebooks are SVD-compressed + INT-quantized for storage — keeps the overhead small. Calibration cost: 3–11 h on a single H100 for LLaMA-2-70B; cheaper than AQLM at similar quality.

### 7.3 When to use which

- **AQLM**: best raw accuracy at sub-2-bit; longest calibration.
- **VPTQ**: ~10× faster calibration; competitive accuracy at 2.0–2.5 bits.
- **GPTVQ**: native GPTQ-friendly recipe; sweet spot if you already have GPTQ infra.

All three are within ~0.1 ppl of each other on LLaMA-2-70B 2-bit. The choice is mostly about engineering integration.

---

## 8. HQQ — half-quadratic splitting, data-free, fast

[[hqq]] (Badri & Shaji, Mobius Labs 2024) is the **"iterate quickly" tool**. No calibration data, no STE, no gradient descent — just closed-form alternating updates. Quantizes LLaMA-2-70B at 4-bit in <10 minutes on a single A100.

### 8.1 Quantization rule (asymmetric)

Per-group (e.g. group_size=64):

```
W_q = round((W / s) + z),   Ŵ = s · (W_q − z)
```

s ∈ ℝ scale, z ∈ ℝ zero-point, W_q ∈ {0, ..., 2^b − 1}.

### 8.2 The lp objective (robust regression)

```math
\min_{s, z, W_q} \|W - s(W_q - z)\|_p^p
```

with `p ∈ (0, 1)` (typically 0.5 or 0.7). `p < 1` down-weights outlier residuals — equivalent to assuming a heavy-tailed (generalised Gaussian) prior on quantization error, which matches real LLM weight distributions.

### 8.3 Half-quadratic splitting

Introduce auxiliary `W_e = W − s(W_q − z)`:

```math
\min \|W_e\|_p^p + \frac{\beta}{2} \|W - s(W_q - z) - W_e\|^2
```

β increased across iterations (continuation). Alternating updates:

```
1. W_e step (fix z, s, W_q):
   W_e ← prox_{||·||_p^p / β}(W − s(W_q − z))
   # closed-form half-shrinkage for p=0.5, cubic-root for p=2/3.

2. z step (fix W_e, s):
   z ← median_group(W_q − (W − W_e) / s)
   # median is the L1 proximal — matches the robust-loss prior.

3. (optional) s step: closed-form LS update for s.

4. Refresh W_q = round((W − W_e)/s + z); increase β; repeat.
```

**Convergence**: 4–8 iterations per group.

### 8.4 Why data-free

The objective minimises raw weight reconstruction; no input distribution X needed. Equivalent to assuming `H = I` — gives up Hessian weighting but the lp robustness compensates for outlier-heavy weights without calibration.

### 8.5 Numbers

LLaMA-2-7B WikiText-2:

| Method | Bits | ppl | Calib data | Wall clock |
|--------|------|-----|------------|------------|
| FP16 | 16 | 5.47 | — | — |
| RTN | 4 | 5.73 | no | seconds |
| GPTQ | 4 | 5.66 | 128 seqs | minutes |
| AWQ | 4 | 5.62 | 128 seqs | minutes |
| **HQQ** | **4** | **5.65** | **no** | **<1 min** |
| HQQ | 2 | 8.91 | no | <1 min |

HQQ at 4-bit ≈ GPTQ at 4-bit, **without calibration**. At 2-bit it trails AQLM but beats RTN by miles. HQQ is the right tool when calibration is expensive or unavailable — fine-tune frameworks (PEFT + LoRA), rapid iteration, edge-side.

---

## 9. Atom — production W4A4KV4 (Ampere)

[[atom]] (Zhao et al. MLSys 2024) is the W4A4KV4 serving stack for A100/A6000-era GPUs. End-to-end **7.7× FP16 / 2.5× INT8 throughput** at same latency.

### 9.1 Sub-channel reorder

For each weight matrix W:

1. From calibration, identify top-K input channels by activation max (K ≈ 128, ~3% of channels).
2. Permute these K channels to the front: `W = [W_outlier | W_normal]`, `x = [x_outlier; x_normal]`.
3. Quantize W_outlier and x_outlier to INT8; W_normal and x_normal to INT4.
4. `y = W_outlier · x_outlier (INT8) + W_normal · x_normal (INT4)`.

The INT8 path takes K/C_in ≈ 3% of FLOPs → negligible runtime overhead, recovers the accuracy.

This is QUIK's idea (ch-13) extended to also cover the KV cache.

### 9.2 Per-token dynamic activation quantization

```
scale_t = max_i |x_{t,i}| / 7
x̂_{t,i} = round(x_{t,i} / scale_t)
```

No static calibration scales — robust to prompt distribution shift.

### 9.3 KV cache INT4

- K: per-head, per-token, group size 128 along channel dim.
- V: per-head, per-token, group size 128 along channel dim.

Stored packed INT4, dequantized on-the-fly inside the attention kernel.

### 9.4 Single fused CUDA kernel

INT4 weight tile + INT4 activation tile → dequant in registers → tensor-core matmul → accumulate; outlier path runs in a small parallel SM block. CUTLASS mixed-precision tiles.

---

## 10. QServe — production W4A8KV4 (Hopper)

[[qserve]] (Lin et al. MIT HAN Lab 2024) is the Hopper-optimised cousin. The bet: **W4A8KV4** beats W4A4 on Hopper because the A8 activation path keeps tensor-core utilization high while the W4 weight path still delivers HBM-bandwidth savings.

### 10.1 Progressive group quantization (the kernel core)

The problem with naive W4A8: dequantizing INT4 → FP16 in registers adds 2–3 instructions per element and pushes register pressure. QServe keeps everything integer:

- **Stage 1** (per-channel INT8): quantize each output channel of W to INT8 with a single per-channel FP16 scale. Store as INT8 `W_s`.
- **Stage 2** (per-group INT4): within each group of g=128 weights along input axis, quantize the INT8 values to INT4 with a per-group **INT8** scale (note: scale is itself integer). Store as INT4 `W_g`.
- **At inference**: dequantize INT4 → INT8 entirely in registers (INT8 × INT8 → INT16, cheap on tensor cores); feed INT8 operand into INT8 tensor-core GEMM with A8 activation. No FP dequant in the critical path.

### 10.2 SmoothAttention

KV4 is sensitive because INT4 K introduces noise that softmax amplifies. QServe applies a SmoothQuant-style per-head scaling:

```
Q' = Q · s,   K' = K / s
```

QK^⊤ unchanged but K' has reduced dynamic range → KV4 quantization gentler. s calibrated to minimise softmax KL.

### 10.3 Throughput numbers

| Model | Hardware | QServe vs TRT-LLM W8A8 |
|-------|----------|------------------------|
| Llama-3-8B | H100 | 1.2× |
| Llama-3-8B | L40S | 2.4× |
| Qwen-1.5-72B | A100 | 3.5× vs Atom W4A4 |

Atom is hurt by A4-induced softmax instability on the larger model; QServe's W4A8 sidesteps this.

### 10.4 Atom vs QServe — when to pick which

| | Atom (W4A4KV4) | QServe (W4A8KV4) |
|--|----------------|------------------|
| Target HW | Ampere (A100, A6000) | Hopper (H100, L40S) |
| Activation | INT4 dynamic | INT8 dynamic |
| KV cache | INT4 | INT4 |
| Accuracy | slightly lower (A4 cost) | higher (A8 preserves) |
| Throughput | higher (less compute) | balanced |
| Use case | memory-bound | balanced compute + memory |

---

## 11. The 2024 production decision tree

```
Need PTQ?                                           → yes (training-free)
├── Need <2 bits?                                   → AQLM + PV-Tuning
├── Need fast / no calibration?                    → HQQ
├── Need W4 weight-only at best quality?           → AWQ (ch-09) or GPTQ + group
├── Need W4A4 inference speedup?
│   ├── Ampere?                                     → Atom (W4A4KV4)
│   └── Hopper?                                     → QServe (W4A8KV4)
└── Need lossless W4A4KV4?                          → QuaRot (free) or SpinQuant (learn)

Need <1.58 bits?                                    → can't PTQ. Train from scratch with
                                                       BitNet b1.58 (ch-16).
```

---

## 12. Pitfalls

- **Don't mix rotations and outlier-sidecars naively.** QuaRot makes the outlier path unnecessary; running QUIK-style outlier preservation on top of QuaRot's already-flat distribution wastes channels at INT8.
- **SpinQuant convergence depends on initialization.** Initialise A from a random Hadamard's matrix log to start near the QuaRot solution; pure random init can land in bad local minima.
- **AQLM's beam search has tuning sensitivity.** Beam width 1 is fine for 2 bits; below 2 bits use width 4–8 and pay the calibration cost.
- **PV-Tuning's V-step neighbourhood `N`** should be the k-NN in codebook space, not in index space. Indices are arbitrary labels; codebook distance is the actual loss-relevant signal.
- **HQQ's `p`** is critical. p=0.5 over-emphasises outlier robustness for well-behaved layers (LayerNorm); p=0.7 is the safer default. Tune per layer in production.
- **Atom's outlier set is fixed at calibration.** Distribution shift at long contexts or out-of-domain prompts can degrade accuracy; revalidate.
- **QServe SmoothAttention scales are per-head learnable but per-layer applied.** Don't share scales across heads or layers.
- **Hadamard requires `d_in` power of 2.** For GLM-style models with odd dims, block the rotation; you lose some incoherence.

---

## Connections and what's next

- **[[quip]] / ch-13** — incoherence processing as the algorithmic ancestor; QuaRot generalises rotation from weight-only to end-to-end.
- **[[gptq]] / ch-08** — the weight quantizer all rotation methods pair with for the W4 step.
- **[[smoothquant]] / [[awq]] / ch-09** — activation-aware migration; SmoothAttention is the QServe extension.
- **[[kivi]] / [[kvquant]] / ch-15** — KV-cache quantization that complements the W4A4 stack and dominates long-context inference.
- **[[bitnet-b158]] / ch-16** — sub-2-bit territory that PTQ cannot reach; requires training from scratch.
- **[[turboquant]] / ch-18** — data-oblivious successor where the rotation reused here is reused for KV cache without any calibration.
- **[[marlin-kernel]] / ch-19** — the production W4A16 GEMM kernel that GPTQ/AWQ ship on; Atom and QServe extend the same dequant pattern to W4A4 / W4A8.

## Further reading

- [[quarot]] — Ashkboos et al. 2024, the end-to-end rotation paper.
- [[spinquant]] — Liu et al. 2024 (ICLR 2025), learned rotations on the Stiefel manifold.
- [[duquant]] — Lin et al. NeurIPS 2024 Oral, block rotation + zigzag.
- [[rotation-and-quantization]] — the unified-view synthesis.
- [[aqlm]] — Egiazarian et al. ICML 2024, additive multi-codebook sub-2-bit.
- [[pv-tuning]] — Malinovskii et al. 2024, alternating P/V for discrete codes.
- [[vptq]] — Liu et al. EMNLP 2024, channel-independent Hessian VQ.
- [[gptvq]] — van Baalen et al. Qualcomm 2024, blessing of dimensionality.
- [[hqq]] — Badri & Shaji 2024, half-quadratic data-free PTQ.
- [[atom]] — Zhao et al. MLSys 2024, W4A4KV4 serving.
- [[qserve]] — Lin et al. MIT HAN Lab 2024, W4A8KV4 on Hopper.

## Excerpts

- [excerpts/quarot.md](excerpts/quarot.md) — computational invariance, the four rotation slots, fold rules, Hadamard outlier reduction.
- [excerpts/spinquant.md](excerpts/spinquant.md) — Stiefel manifold, Cayley parametrization, per-slot learnability.
- [excerpts/aqlm.md](excerpts/aqlm.md) — additive quantization rule, beam-search encoding, Hessian-aware codebook.
- [excerpts/hqq.md](excerpts/hqq.md) — lp robust regression, half-quadratic splitting, alternating updates.
- [excerpts/qserve.md](excerpts/qserve.md) — progressive group quant, register-level INT4 → INT8 dequant, SmoothAttention.
