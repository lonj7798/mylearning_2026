# Pre-LN vs Post-LN: Why Normalization Moved Inside the Residual Branch

<!-- slug: pre-ln-vs-post-ln · type: doc · source: wiki:llm-arch:wiki/courses/llm-arch/ch-03/excerpts/layer-norm-placement.md -->

**Core Insight.** Post-LN (`x ← LN(x + Sub(x))`, the 2017 Transformer) puts a normalization layer *on* the residual highway, so the backward pass must traverse `L` LayerNorm Jacobians in series; gradient magnitude then scales as `O(1/√ℓ)` and the first-vs-last-layer imbalance reaches `O(L)`, which is why the original paper needed 4,000 warmup steps. Pre-LN (`x ← x + Sub(LN(x))`, GPT-2 onward) moves LN *inside* the residual branch, leaving a clean identity `I` on the highway; Xiong et al. proved gradients are then `O(1)` across depth and warmup becomes optional. **The two placements cost identical activation bytes** — the win is entirely trainability, not memory.

**Guideline.** Use Pre-RMSNorm. Never choose Post-LN for a new model at depth ≥ 24: it buys at most a marginal quality ceiling and costs you warmup, gradient clipping, and a fragile LR schedule. And when a recipe demands an unusual stabilization trick, first ask whether a *structural* fix (moving a norm) removes the need for the trick — structural fixes add robustness, tricks add fragility.

## Technical Details

- **The two update rules, written as the residual stream sees them:**

```
Post-LN (Transformer 2017, GPT-1):     Pre-LN (GPT-2, GPT-3, LLaMA, everything since):
  h'  = LayerNorm(x  + Attn(x))          h'  = x  + Attn(LayerNorm(x))
  h'' = LayerNorm(h' + FFN(h'))          h'' = h' + FFN(LayerNorm(h'))
```

- **The gradient argument (Xiong et al. 2020).** Pre-LN gives `∂x_{ℓ+1}/∂x_ℓ = I + ∂Sub(LN(x_ℓ))/∂x_ℓ`. The `I` is a straight wire from the loss to layer 1 — an *additive* path that cannot shrink. Post-LN has no such term: every `∂x_m/∂x_{m-1}` factor in the product `∏_{m=ℓ+1}^{L} ∂x_m/∂x_{m-1}` is a LayerNorm Jacobian applied to a sum. Result: `E‖∂L/∂θ_ℓ‖ ∝ O(1/√ℓ)` for Post-LN vs `O(1)` for Pre-LN.
- **Empirical comparison table** (verbatim from the llm-arch source):

| Property | Post-LN | Pre-LN |
|---|---|---|
| Warmup required | Yes — 4,000 steps in the original paper | No |
| Gradient magnitude across layers | Imbalanced, `O(1/√ℓ)` | Balanced, `O(1)` |
| Training stability | Fragile without careful hyperparameters | Robust |
| Final model quality | Slightly higher ceiling (with tuning) | Slightly lower ceiling |
| Adoption (2026) | Rarely used alone | Universal (with RMSNorm) |

- **LayerNorm mechanics.** `LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β`, with `μ, σ²` reduced over the `d_model` axis **per position, per example** (`ε ≈ 1e-5`). Per-position statistics is why LN works at batch size 1, at variable sequence length, and identically at train and inference time — no running statistics, unlike batch norm.
- **RMSNorm** (Zhang & Sennrich 2019, used by LLaMA/Gemma/Mistral/Qwen): `RMSNorm(x) = γ ⊙ x / √( (1/d)Σx_i² + ε )`. Drops mean-centering and the `β` bias. ~10% faster on GPU (fewer memory reads); empirically equivalent quality. The mean removes the DC component, which carries little task-relevant signal in deep transformers.
- **Memory accounting — the exact claim.** Both placements save the same tensors: LayerNorm's VJP needs its **input** (`2sbh` bytes) plus the reduction statistics. In Korthikanti's coefficient that is the `4 sbh` term = two norms × `2 sbh` (see [[transformer-block-tensor-ledger]]). At the reference config `B=1, T=4096, h=4096, bf16`, that is **67.11 MB per block**, or 2 × 33.55 MB — identical whether the norm sits before or after the add.
- **RMSNorm's one real byte saving.** LayerNorm saves `(mean, rstd)` = `[B,T] × 2 fp32` = **32,768 B per norm** at the reference config; RMSNorm saves only `rstd` = `[B,T] × 1 fp32` = **16,384 B**. Two norms per block × 80 blocks: 5.24 MB → 2.62 MB. Real, exact, and utterly negligible next to the 1.07 GB attention matrix — which is the point worth teaching: *normalization placement is a gradient-flow decision, not a memory decision.*
- **What Pre-LN actually costs in memory, indirectly.** Because nothing normalizes the highway, the residual stream's variance grows roughly linearly with depth. GPT-2's fix was to initialize residual-branch output weights at scale `1/√N` (`N` = number of residual layers), keeping each block's contribution `O(1/√N)`. The practical memory consequence appears only in low-precision training: a growing-magnitude stream is the tensor most likely to be kept in fp32 or to need a per-tensor scale in FP8 ([[fp8-training]], [[ch-02]]) — a precision decision forced by a placement decision.
- **Post-LN's checkpoint boundary is different but the same size.** With gradient checkpointing, the stored boundary tensor is the block's output: for Pre-LN that is the residual-2 sum, for Post-LN it is the final `LN(...)` output. Both are `[B, T, h]` = **33.55 MB**. The `2·s·b·h·L` full-recompute floor from [[ch-03]] is placement-independent.
- **Training-memory angle:** Pre-LN changes *zero* bytes in the activation ledger but changes *everything* about whether the ledger is worth computing — a Post-LN 80-layer model that diverges at step 300 has an activation budget you never get to spend. The second-order memory effects are real though: (a) Pre-LN's unnormalized, depth-growing residual stream is what makes the fp32 master-weight / per-tensor-scale machinery of [[mixed-precision-training]] and [[fp8-training]] non-optional; (b) Post-LN's need for warmup interacts with the "OOM on step 2" failure mode in [[ch-01]] §1.6, since Adam's fp32 moments materialize only after the first `.step()` — a run that must warm up is a run that must survive that allocation.

## Citation
Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu. "On Layer Normalization in the Transformer Architecture." ICML 2020, arXiv:2002.04745. Biao Zhang, Rico Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019, arXiv:1910.07467. Vaswani et al., "Attention Is All You Need," NeurIPS 2017, arXiv:1706.03762. Sourced from `wiki:llm-arch:wiki/courses/llm-arch/ch-03/excerpts/layer-norm-placement.md` and `.../ch-04/excerpts/gpt-architecture-diff.md`.
