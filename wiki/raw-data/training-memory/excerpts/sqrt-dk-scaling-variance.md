# Why 1/√d_k — The Variance Derivation and Softmax Saturation
<!-- slug: sqrt-dk-scaling-variance · type: paper · source: https://arxiv.org/abs/1706.03762 (§3.2.1 + footnote 4) -->

**Core Insight.** The `1/√d_k` factor is not cosmetic normalisation — it is a **necessary condition** for dot-product attention to work at all. A dot product of two `d_k`-dimensional vectors with i.i.d. zero-mean unit-variance components has variance exactly `d_k`, so raw scores grow as `√d_k`. Feed those into a softmax and it saturates toward one-hot; the softmax Jacobian `p_i(1−p_i)` collapses toward zero and gradients stop flowing to every key but the winner. Dividing by `√d_k` restores unit score variance *at any head dimension*, keeping softmax in its sensitive region. Vaswani et al. measured this: unscaled dot-product attention loses to Bahdanau's additive attention at large `d_k`; **scaled** dot-product matches it while being far faster because it is a plain GEMM.

**Guideline.** Any time you introduce a dot-product similarity into a network (attention, retrieval, contrastive loss, router logits), compute the variance of the resulting logits. If it grows with dimension, divide by the standard deviation — otherwise the softmax saturates and gradients vanish. Implementation rule: **fold the scale into `Q` before the GEMM** (`Q ← Q/√d_k`), never as an out-of-place divide on the `N×N` score tensor.

## Technical Details

- **The paper's own derivation (footnote 4, verbatim):** *"To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Σ_{i=1}^{d_k} q_i k_i, has mean 0 and variance d_k."*
- **Full derivation.** For `q, k ∈ ℝ^(d_k)` with `q_i, k_i ~ i.i.d. 𝒩(0,1)`:
  `E[q·k] = Σ_i E[q_i]E[k_i] = 0`
  `Var(q·k) = Σ_i Var(q_i k_i) = Σ_i E[q_i²]E[k_i²] = Σ_i 1·1 = d_k`
  `sd(q·k) = √d_k`
  `Var(q·k / √d_k) = d_k / d_k = 1`  ← unit variance at every `d_k`
- **Concrete σ values:** `d_k = 64` → `√d_k = 8`; `d_k = 128` → `√d_k = 11.3137`; `d_k = 512` → `√d_k = 22.6274`.
- **Worked numeric example — the gradient actually dies.** Take `d_k = 64` (`√d_k = 8`), two keys with raw dot products `s₁ = 24.0`, `s₂ = 16.0` (both inside ±3σ = ±24, so entirely ordinary draws):
  - *Unscaled:* softmax over `{24, 16}` → `p₁ = 1/(1+e^{−8}) = 0.99966`, `p₂ = 0.00034`.
    Softmax Jacobian diagonal `∂p₁/∂s₁ = p₁(1−p₁) = 3.3524×10⁻⁴`.
  - *Scaled by 8:* softmax over `{3.0, 2.0}` → `p₁ = 1/(1+e^{−1}) = 0.731059`, `p₂ = 0.268941`.
    `p₁(1−p₁) = 0.196612`.
  - **Ratio = 0.196612 / 3.3524×10⁻⁴ ≈ 586×** more gradient into the score with the scaling. That factor is the whole argument.
- **Why saturation is fatal, mechanically.** Softmax Jacobian: `∂p_i/∂s_j = p_i(δ_ij − p_j)`. As `p_i → 1`, every entry → 0. The gradient reaching `W_Q` and `W_K` passes through this Jacobian, so a saturated head stops learning *where* to look — it is frozen into whatever routing its random initialisation happened to produce.
- **Scale of the problem at realistic `N`.** The max of `N` i.i.d. `𝒩(0,1)` scores is `≈ √(2 ln N)`: `N=1024 → 3.72`, `N=32768 → 4.56`. Unscaled at `d_k = 128` those become `11.31 × 3.72 = 42.1` and `11.31 × 4.56 = 51.6` — softmax logit gaps of tens of nats, i.e. exactly one-hot.
- **Empirical evidence (§3.2.1, verbatim):** *"While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k. We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."*
- **Table 3 row (B) — head dimension matters independently** (verified from the paper): shrinking `d_k` at fixed compute hurts. `d_k = 16` → dev PPL **5.16**, dev BLEU **25.1**, 58M params; `d_k = 32` → PPL **5.01**, BLEU **25.4**, 60M params; base `d_k = 64` → PPL **4.92**, BLEU **25.8**, 65M params. Paper's reading: *"determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial."*
- **The assumption's fine print.** The variance argument assumes unit-variance, independent components. Real `Q`/`K` come from `X W_Q` after LayerNorm/RMSNorm, so the assumption holds approximately at init (with standard `1/√d_model` weight init) and degrades during training — which is why some models add QK-norm (RMSNorm on `Q` and `K`) to re-enforce it. `1/√d_k` is the *init-time* correct constant, kept fixed thereafter.
- **Training-memory angle:** The scale itself is free in FLOPs, but *where you apply it* costs gigabytes. A naive implementation writes `S = Q @ K.transpose(-1,-2)` then `S_scaled = S / math.sqrt(d_k)` out-of-place — two live `(B, a, N, N)` tensors simultaneously. At `B=1, a=32, N=32768`, bf16, that second tensor is another `32 × 32768² × 2 B = 68.7 GB` per layer, doubling the attention peak. Correct practice is to pre-scale the query (`Q *= d_k**-0.5`, cost `B·N·d_model` elements) so the `N×N` tensor is produced already-scaled; FlashAttention folds the constant into the `Q` tile at load time inside SRAM, so no scaled score tensor ever reaches HBM. Secondarily: saturated (unscaled) logits force an fp32 softmax to avoid bf16 range loss, adding a further `B·a·N²·4` bytes of fp32 intermediate — with correct scaling the softmax input stays within a few units of zero and bf16/fp16 accumulation with the running-max trick (see [[online-softmax]]) is numerically safe.

## Citation
Ashish Vaswani et al. "Attention Is All You Need." NeurIPS 2017, §3.2.1 and footnote 4, Table 3 rows (A)/(B). https://arxiv.org/abs/1706.03762
