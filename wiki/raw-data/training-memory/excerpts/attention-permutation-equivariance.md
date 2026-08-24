# Self-Attention Is a Set Function: The Permutation-Equivariance Proof
<!-- slug: attention-permutation-equivariance · type: paper · source: wiki:llm-arch:wiki/courses/llm-arch/ch-06/read.md -->

**Core Insight.** Bare scaled-dot-product self-attention is exactly permutation-equivariant — `Attn(PX) = P·Attn(X)` for every permutation matrix `P` — so a transformer with no positional signal and no causal mask cannot distinguish "고객이 보험을 해지했다" from "보험이 고객을 해지했다". Position is not a refinement of attention; it is a missing input that must be injected deliberately, and *where* you inject it is an architectural (and memory) decision, not a cosmetic one.

**Guideline.** Inject position at the place where it is consumed — inside the `QKᵀ` score, never into `V` and never into the residual stream. Of the three possible injection points (input-additive, Q/K-multiplicative, logit-additive), only the first two leave the attention kernel's memory contract untouched; logit-additive schemes put a `T×T`-shaped object into the exact tensor FlashAttention exists to never materialize.

## Technical Details
- **Setup.** `X ∈ ℝ^{T×d}` (rows = tokens), `Q = XW_Q`, `K = XW_K`, `V = XW_V`, all `W` shared across positions. Let `P ∈ {0,1}^{T×T}` be a permutation matrix with `(PX)_i = X_{π(i)}`, `PᵀP = I`.
- **Step 1 — projections commute with P.** Row-wise linear maps: `(PX)W_Q = P(XW_Q) = PQ`. Same for `K`, `V`.
- **Step 2 — scores conjugate.** `S' = (PQ)(PK)ᵀ/√d_k = P(QKᵀ)Pᵀ/√d_k = P S Pᵀ`, i.e. `S'_{ij} = S_{π(i)π(j)}`.
- **Step 3 — softmax is row-wise, so it passes through.** Row `i` of `PSPᵀ` is row `π(i)` of `S` with its *entries* permuted by `π`. The softmax denominator `Σ_j exp(·)` is a sum, hence permutation-**invariant**, so `softmax(PSPᵀ) = P·softmax(S)·Pᵀ = P A Pᵀ`.
- **Step 4 — the Pᵀ cancels against V.** `Out' = (P A Pᵀ)(P V) = P A (PᵀP) V = P A V = P·Out`. ∎
- **Extends to the whole block.** LayerNorm/RMSNorm and the FFN act independently and identically on each row, and the residual add is elementwise, so `Block(PX) = P·Block(X)`. Stacking preserves it. **The full pre-norm transformer stack is permutation-equivariant**, so the *multiset* of output representations is literally unchanged by shuffling the input — only relabelled.
- **Equivariant ≠ invariant.** The outputs move with the permutation (equivariance). What is invariant is the *set* of outputs and therefore any bag-of-tokens loss. This is why "attention outputs change when I shuffle" is not a counter-argument.
- **The causal-mask caveat (important, and absent from the llm-arch pages).** The proof requires the mask `M` to satisfy `P M Pᵀ = M`. A causal mask does **not**: it is lower-triangular, so only `π = id` preserves it. A decoder-only LM is therefore *not* permutation-equivariant even with zero positional encoding — token `i` can see exactly `i+1` predecessors, and that count is a usable absolute-position signal. Haviv et al. (2022) show NoPE decoder LMs are competitive with explicit-PE models and probe out an implicit absolute-position representation; Kazemnejad et al. (2023) find NoPE can even *beat* explicit PE on length generalization. So: the equivariance proof is exact for **bidirectional/encoder** attention, and is a *near*-degeneracy (weak, learned-not-given signal) for causal decoders. Explicit PE is still used everywhere because the mask-derived signal is weak, indirect and hard to sharpen.
- **The three injection points and what each touches:**

  | Where | Examples | Shape of the added object | Touches the `T×T` score matrix? |
  |---|---|---|---|
  | Input embedding (additive) | sinusoidal, learned absolute | `[L, d_model]` table, added once before layer 0 | No |
  | Q and K (multiplicative) | RoPE, iRoPE | `[L, d_head/2]` cos/sin cache, applied pre-kernel | No |
  | Attention logits (additive) | T5 RPE, ALiBi | `[H, T, T]` bias (or on-the-fly slope) | **Yes** |
- **Training-memory angle:** this proof is the reason position never has to cost activation bytes. Because the `Pᵀ` cancels against `V`, the fix only has to reach `Q` and `K` — the value path and hence the residual stream can stay completely position-free, so RoPE adds **zero** bytes to the per-layer saved activations. The failure mode is the third row of the table: any scheme that adds a bias to the logits needs a `B·H·T·T` object to exist, which is **4.000 GiB at B=1, H=32, T=8192 in bf16, per layer** (`1×32×8192×8192×2 B`) — 10% of an entire A100-40GB for one layer's positional bias. FlashAttention/SDPA only avoid that if the kernel has the bias baked in (FA2 has an `alibi_slopes` argument precisely for this); a custom relative-position bias with no kernel support silently falls back to the MATH path and re-materializes the full score matrix. RoPE's real memory virtue is not that its cos/sin table is small — it is that it is *kernel-transparent*.

## Citation
Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762. — Haviv, A., Ram, O., Press, O., Izsak, P., Levy, O. "Transformer Language Models without Positional Encodings Still Learn Positional Information." Findings of EMNLP 2022. arXiv:2203.16634. — Kazemnejad, A. et al. "The Impact of Positional Encoding on Length Generalization in Transformers." NeurIPS 2023. arXiv:2305.19466. — Proof and injection-point table restated from `wiki:llm-arch:wiki/courses/llm-arch/ch-06/read.md` §1.
