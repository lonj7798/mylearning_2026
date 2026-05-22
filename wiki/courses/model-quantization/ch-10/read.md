<!-- chapter: ch-10
     phase: 2023-refinements
     title: OmniQuant + Learnable Equivalent Transformations
     sources: [[omniquant]], [[rptq]], [[affinequant]], [[flatquant]]
-->

# Chapter 10 — OmniQuant: Learnable Equivalent Transformations

> **Core insight.** SmoothQuant's closed-form `s` and AWQ's grid-searched scalar `α` are both single points in a much larger space of equivalent transformations. Make `s` (and the per-channel clipping bounds for W) **learnable** via a block-wise reconstruction objective, optimise them with gradient descent through STE, and you get QAT-quality W4A4 in 1–16 hours on a single A100 — without ever backpropagating through the whole model.
>
> **Guideline.** When ≤0.5 PPL matters and a few GPU-hours are available, use OmniQuant with Learnable Weight Clipping (LWC) on every weight matrix + Learnable Equivalent Transformation (LET) on the input of qkv and FFN-up projections. Train block-by-block, one transformer block resident at a time, 20 epochs per block, AdamW lr=5e-3 on the quant parameters only. For the affine-generalised variant use AffineQuant or FlatQuant.

---

## Why this chapter exists

[[ch-09]] established the activation-aware equivalent transformation as the central trick for sub-W8A8 LLM PTQ. SmoothQuant gave it a closed form (`s_j = max|X|^α / max|W|^(1−α)`). AWQ specialised it to weight-only with a 20-point grid search over a single per-layer α.

Both are point estimates inside a much richer parameter space. The closed-form picks `α` heuristically; the grid search picks the *single* α that minimises layer output MSE. Neither lets the per-channel `s` deviate from the parametric form. And neither addresses **weight-clipping** — the choice of where to set the INT-k absmax bounds, which is the other axis besides scaling.

OmniQuant ([[omniquant]], ICLR 2024) makes both axes learnable:

- **LWC (Learnable Weight Clipping):** per-channel learned upper/lower clipping bounds via a sigmoid parameterisation.
- **LET (Learnable Equivalent Transformation):** per-channel learnable scale + shift `(s, b)` — strict generalisation of SmoothQuant.

Trained block-wise with MSE against the FP teacher block — one transformer block at a time, only quant parameters trainable. This is the **first PTQ method to hold W4A4 accuracy on LLaMA**, with wall-clock comparable to PTQ (1–16 hours) rather than QAT (days).

The chapter then covers three sibling ideas: [[rptq]] (channel reordering — complementary to learnable scaling), [[affinequant]] (lift `s` from diagonal to full invertible affine), and [[flatquant]] (Kronecker-structured affine targeting flatness directly).

---

## 1. The block-wise reconstruction objective

The key engineering insight that lets OmniQuant fit on a single A100 even at 70B: don't backprop through the whole model. Instead, train **one transformer block at a time**.

For block `i` with input `h_i` (collected from the previous already-quantised block) and FP teacher block `f_i^{FP}`:

```math
\mathcal{L}_i = \big\lVert f_i^{FP}(h_i) - f_i^{\text{quant}}(h_i;\ \gamma, \beta, s, b) \big\rVert^2
```

- Only quant parameters `(γ, β, s, b)` are trainable.
- Original weights and activations stay FP during forward; quantization is *simulated* via `dequant(quant(·))`.
- Backward uses straight-through estimator ([[straight-through-estimator]]) for the round operation.
- 20 epochs per block, AdamW lr=5e-3, batch=1.

**Memory profile.** At any moment only one block's FP teacher + quantized student + ~2 calibration batches are resident → ≤40 GB even for LLaMA-2-70B. This is what makes the method run on a single A100-40G.

Contrast with full QAT, which needs the entire model + activations + optimizer states for all parameters in memory at once. OmniQuant is **PTQ wall-clock with QAT quality** because the block-wise decomposition trades the joint optimum for tractability — and the lost optimality is small because each transformer block's output is a strong proxy for downstream loss.

---

## 2. Learnable Weight Clipping (LWC)

The fixed-clip problem: standard symmetric INT-k uses `s = max(|W|) / (2^{b-1} − 1)`. This wastes resolution on weight outliers that contribute little to layer output. AdaRound and related methods learn the round direction; LWC instead learns where to **clip** before rounding.

Per output channel `j`, two learnable scalars `γ_j, β_j ∈ R`:

```math
W_{\text{clip}} = \mathrm{clip}\Big(W,\ \sigma(\beta) \cdot \min(W),\ \sigma(\gamma) \cdot \max(W)\Big)
```

```math
\hat{W} = \mathrm{round}\Big(\frac{W_{\text{clip}} - z}{s}\Big) \cdot s + z
```

- `σ(·)` is sigmoid → bounds stay in `(0, 1)` of the per-channel range.
- `s, z` are derived from `W_clip`'s per-channel max/min after clipping.
- Trainable: only `γ, β` — 2 parameters per output channel.
- Backward: STE through `round`, normal grad through `clip`.

The intuition: if the largest weight in a channel is a quant-disrupting outlier that doesn't carry much output mass, learning `σ(γ) < 1` clips it away, tightens the INT-k grid spacing on the rest, and reduces overall MSE.

> **Pitfall.** LWC's outer-bound parameterisation `σ(β) · min(W)` means you can never expand beyond the original range — only contract. This is intentional: it prevents the optimization from over-expanding into never-seen-weight territory. The price is that any genuine outlier you should *keep* is reachable only via LET (next section), not LWC.

---

## 3. Learnable Equivalent Transformation (LET)

LET generalises SmoothQuant's `diag(s)` and AWQ's `mean(|X|)^α` to:

- per-channel learnable scale `s_j > 0`
- per-channel learnable shift `b_j ∈ R`

Per input channel j:

```math
\hat{X} = (X - b) \oslash s, \qquad \hat{W} = \mathrm{diag}(s) \cdot W
```

The bias term `b` is absorbed back as `+ W · diag(s) · (b ⊘ s)`, mathematically:

```math
(X - b)/s \cdot \mathrm{diag}(s) \cdot W = X \cdot W - b \cdot W
```

- `s` parameterised via softplus to stay positive.
- LET is applied only where it matters most: **input of qkv projections** and **input of FFN-up**. Other inputs use identity (no LET).

This is strictly richer than SmoothQuant (which has `b = 0` and `s` closed-form) and AWQ (which has `b = 0` and `s` from one scalar α). The added freedom of `b` lets LET re-centre the activation distribution before quantization, which matters for asymmetric distributions like post-GeLU/SiLU outputs.

---

## 4. The full OmniQuant recipe

Combining LWC + LET, the per-block training loop:

```python
# Per transformer block i:
freeze(model.params)                          # original weights stay FP
train_params = [γ, β]_LWC + [s, b]_LET        # only quant params
optimizer = AdamW(train_params, lr=5e-3)

for epoch in range(20):
    for batch in calibration_set:             # 128 sequences
        h_i = collect_input_from_prev_block(batch)
        target = f_FP(h_i)                    # teacher (FP, no grad)
        pred = f_quant(h_i; γ, β, s, b)       # student (simulated quant)
        loss = mse(pred, target)
        loss.backward()                       # STE through round
        optimizer.step()

# Freeze γ, β, s, b after block i is done; move to block i+1
```

Total parameter count trained per block: roughly `2 · d_out` (LWC) + `2 · d_in` (LET at two positions) ≈ tens of thousands. Trivial compared to the model weights, which is why the optimization is fast and well-conditioned.

### Hyperparameter table

| Knob | Value |
|---|---|
| Supported configs | W4A4 / W6A6 / W4A16 / W3A16 / W2A16 |
| Calibration samples | 128 |
| Epochs per block | 20 |
| Optimizer | AdamW, lr 5e-3 (quant params only) |
| LWC placement | every weight matrix |
| LET placement | input of qkv and FFN-up |
| Wall-clock LLaMA-2-7B (W4A4) | ~1 hr on A100-40G |
| Wall-clock LLaMA-2-70B (W4A4) | ~16 hr on A100-40G |

---

## 5. AWQ contrast: grid search vs gradient training

| Aspect | AWQ | OmniQuant |
|---|---|---|
| Per-channel scale | `s = mean|X|^α`, one α per layer | per-channel `s, b` trained by gradient |
| Weight clipping | fixed `max|W|` | per-channel learned `γ, β` via LWC |
| Calibration cost | 20 forward passes per layer | ~20 epochs × 128 batches per block |
| Wall-clock LLaMA-2-7B (W4A16) | minutes | ~1 hr |
| Activation precision floor | A16 (weight-only) | A4 viable |
| W4A4 LLaMA PPL (paper) | doesn't survive | works |

AWQ is faster and simpler for W4A16. OmniQuant is the right tool when you need W4A4 (4-bit activations) or W2A16 (sub-4-bit weights) — regimes where one-scalar-α-per-layer no longer captures enough of the equivalent-transformation space.

The line of descent:
```
SmoothQuant → AWQ → OmniQuant
(closed-form)  (1 α/layer grid)  (per-channel s, b trained)
```

---

## 6. RPTQ — the orthogonal complement

[[rptq]] (Yuan et al. 2023) attacks the same activation-quant problem from a structurally different angle: **reorder channels into clusters of similar magnitude**, then assign one INT-k scale per cluster.

For input activation X, compute per-channel max `r_j = max_t |X_{t, j}|` over calibration. Run k-means on `{r_j}` with K=32–64 clusters. Permute the channels in-place so each cluster's channels are contiguous, and apply per-cluster INT-k absmax scaling.

```math
s_k = \frac{\max_{j \in S_k} r_j}{2^{b-1} - 1}, \qquad \hat{X}_{\cdot, S_k} = \mathrm{round}(X_{\cdot, S_k} / s_k) \cdot s_k
```

The permutation `π` is folded into the preceding LayerNorm (`γ[π], β[π]`) and the next Linear (`W[π, :]`) → zero runtime overhead. First viable A3 (3-bit activation) PTQ on OPT-175B.

**Why this is complementary to OmniQuant.** RPTQ doesn't change activation values — it only changes the *grouping* of channels for quant scale assignment. LET changes the values via `(X - b)/s`. The two can stack: reorder channels into similar-range clusters, then apply LET within each cluster. In 2024 this idea becomes Atom and QServe (W4A4-KV4 production stacks, see [[ch-14]]).

> **Pitfall.** RPTQ's k-means runs once on calibration; if the deployment distribution shifts so the cluster assignment is wrong, the per-cluster scales mis-fit. Validate with a held-out calibration before shipping.

---

## 7. AffineQuant — full invertible affine

[[affinequant]] (Ma et al. ICLR 2024) lifts the OmniQuant LET from diagonal `diag(s)` to **general invertible affine** `A ∈ R^{d×d}`:

```math
y = (W A^{-1})(A x) = W' x'
```

W' is folded offline; x' is computed online (or A is folded into the previous Linear when possible).

The challenge with arbitrary A: invertibility is not automatic. The fix is a **gradual mask schedule** `M_t`:

```math
A_t = I + M_t \odot (A - I)
```

- `t = 0`: `M_0 = I` → only diagonal entries trainable; `A_0 = diag` (SmoothQuant-equivalent start).
- `t = T/2`: M banded around diagonal.
- `t = T`: M = all-ones → full A trainable.

This schedule keeps A close to diagonal early, when the optimization is most prone to singularity. The **Levy-Desplanques theorem** (strict diagonal dominance ⇒ invertibility) guarantees A stays invertible as the off-diagonal entries grow slowly.

**Results.** LLaMA-2-7B W4A4 C4 PPL: 15.76 (AffineQuant) vs 18.02 (OmniQuant baseline) — a meaningful gain. SOTA at W4A4 across LLaMA-30B/65B in early 2024.

---

## 8. FlatQuant — Kronecker affine for flatness

[[flatquant]] (Sun et al. ICML 2025) makes two further moves:

1. **Drop orthogonality.** QuaRot/SpinQuant (see [[ch-14]]) use orthogonal rotations that preserve L²-norm. Orthogonal preserves heavy-tailedness; affine can flatten. **Flatness — not just outlier elimination — is the right target for uniform-interval quant** (Bennett 1/12 noise model is tight only for uniform sources).
2. **Kronecker decomposition for cheap storage.** For `d = d_1 · d_2` (e.g. `4096 = 64 · 64`), let `A = A_1 ⊗ A_2` with small `A_1 ∈ R^{d_1 × d_1}`, `A_2 ∈ R^{d_2 × d_2}`. Storage `d_1² + d_2² ≪ d²`. Online cost: `Ax = vec(A_2 · X · A_1^⊤)` with X = reshape(x, d_2 × d_1) — two small matmuls per token, fused with INT4 GEMM in one kernel.

Reported overhead: <5% prefill, <10% decode vs raw INT4. LLaMA-3-70B W4A4 within 1% of FP16, beating SpinQuant by 7.5%.

---

## 9. The trajectory from W8A8 to W4A4

| Method | Year | Scope | Transformation | Wall-clock 7B |
|---|---|---|---|---|
| LLM.int8() | 2022 | W8A8 (FP16 outlier) | none (mixed-precision) | minutes |
| SmoothQuant | 2022 | W8A8 | closed-form per-channel `s` | minutes |
| AWQ | 2023 | W4A16 | grid-searched `α` (1/layer) | ~10 min |
| GPTQ | 2022 | W4A16 | Hessian-based weight update | ~1 hr |
| OmniQuant | 2023 | W4A4 / W2A16 | learnable per-channel `(s, b, γ, β)` | ~1 hr |
| AffineQuant | 2024 | W4A4 | learnable full affine A | ~few hr |
| FlatQuant | 2024 | W4A4 | Kronecker affine, flatness target | ~few hr |
| QuaRot / SpinQuant | 2024 | W4A4 + KV4 | orthogonal rotation | ~1 hr |

The search space of equivalent transformations expands monotonically: scalar → per-channel diag → per-channel diag+shift → full affine → Kronecker affine → orthogonal rotation. Each expansion buys ~1-2 PPL at W4A4 on LLaMA.

---

## Connections and what's next

- **[[omniquant]]** — full extract; LWC + LET + block-wise training.
- **[[awq]]** — direct predecessor; OmniQuant replaces grid-search α with trained `(s, b)`.
- **[[smoothquant]]** — even earlier predecessor; OmniQuant generalises the closed-form `s`.
- **[[rptq]]** — orthogonal trick (channel reordering); composes with OmniQuant.
- **[[affinequant]]** — lifts diagonal to invertible affine via gradual mask schedule.
- **[[flatquant]]** — Kronecker affine targeting flatness for W4A4.
- **[[brecq]]** — pre-LLM block-wise reconstruction ancestor (CNN-era).
- **[[ch-11]] / [[squeezellm]], [[spqr]]** — orthogonal axis: instead of learning the transformation, **identify and preserve outliers in FP16** via dense-and-sparse decomposition.
- **[[ch-14]] / [[quarot]], [[spinquant]], [[duquant]]** — 2024 rotation-based descendants. SpinQuant explicitly trains the rotation on the Stiefel manifold; DuQuant adds channel permutation.
- **[[pv-tuning]]** — sub-2-bit fine-tuning successor.

## Further reading

- OmniQuant's Figure 2 (LWC/LET module diagram) and Table 2 (W4A4 LLaMA results) are the visual + numerical anchors.
- The **Levy-Desplanques theorem** (1881) — strict diagonal dominance implies invertibility — is the AffineQuant guarantee. Worth a 5-minute detour to internalise.
- FlatQuant's Figure 2 (activation histograms: original vs rotation vs affine) is the single best argument for *flatness* as the right optimization target.
