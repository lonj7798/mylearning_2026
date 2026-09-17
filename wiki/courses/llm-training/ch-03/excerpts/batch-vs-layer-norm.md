---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: batch-vs-layer-norm (composite) — Ba et al. 2016 (LayerNorm); Zhang & Sennrich 2019 (RMSNorm); Xiong et al. 2020 (Pre-LN vs Post-LN); OLMo et al. 2025 (OLMo 2 §2.1, §3.3); Qwen2.5 and Qwen3 reports (§2)
source_url: https://arxiv.org/abs/1607.06450 ; https://arxiv.org/abs/1910.07467 ; https://arxiv.org/abs/2002.04745 ; https://arxiv.org/abs/2501.00656 ; https://arxiv.org/abs/2412.15115 ; https://arxiv.org/abs/2505.09388
created_at: "2026-04-23"
revised_at: "2026-09-15"
note: "Rewritten for the 2026-09 revision. The library card classics/batch-vs-layer-norm.md was not yet re-verified on 2026-09-15 and contains errors: OLMo 2's reordered norm described as a norm after the residual add, QK-norm attributed to Qwen-2.5, RMSNorm's 7%-64% described as a norm-op speedup, and a post-norm divergence threshold ('depth ≥ 24 ... around step 1k') with no source."
---

# Excerpt: normalization formulas, placement, and reported evidence

## LayerNorm and RMSNorm

- LayerNorm over the d features of one token: y = γ ⊙ (x − μ)/√(σ² + ε) + β, with μ and σ² the mean and variance
  over the d features (Ba et al. 2016).
- RMSNorm (Zhang & Sennrich 2019, arXiv:1910.07467 §4, Eq. 4): ā_i = (a_i / RMS(a)) · g_i, RMS(a) = √((1/n) Σ a_i²).
  The authors "hypothesize that re-centering invariance in LayerNorm is dispensable" (Abstract). Abstract: RMSNorm
  "achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models."
  The figure is total model running time across the paper's tasks, not the time of the normalization op alone.
- OLMo 2 (arXiv:2501.00656 §3.3.1): OLMo-0424 used non-parametric LayerNorm; "Our ablations show no difference
  between the two, so we switch back to RMSNorm." No numbers printed.

## Post-LN versus Pre-LN (Xiong et al. 2020, arXiv:2002.04745v2)

- Post-LN: x_{l+1} = LN(x_l + F(x_l)). Pre-LN: x_{l+1} = x_l + F(LN(x_l)), plus "Final LayerNorm" on x_{L+1} (Table 1).
  §3.3: "in the Pre-LN Transformer, the scale of the input to the final layer normalization is linear in L, and thus
  the gradients of all parameters will be normalized by √L."
- Lemma 2 (§3.3): at initialization, Post-LN E‖x^{post,5}_{l,i}‖² = (3/2)d for all l; Pre-LN
  (1 + l/2)d ≤ E‖x^{pre}_{l,i}‖² ≤ (1 + 3l/2)d, so the Pre-LN hidden-state scale "grows linearly along with the depth".
- Lemma 3: ‖J_LN(x)‖₂ = O(√d / ‖x‖₂).
- Theorem 1 (§3.3): gradient of the last FFN layer is O(d√(ln d)) for Post-LN, "independent of L", and
  O(d√(ln d / L)) for Pre-LN. §3.4 Fig. 3: at initialization the last-FFN gradient norm "remains in the Post-LN
  Transformer (around 1.6) and decreases in the Pre-LN Transformer" as layers grow from 6-6 to 14-14.
- §3.2 Fig. 2 (IWSLT14 De-En, Post-LN): without warmup, Adam reaches BLEU 8.45; with warmup "around 34"; with
  T_warmup = 500, BLEU 31.16 (lr_max 5e-4) and 2.77 (lr_max 1e-3). The paper's Pre-LN experiments remove warmup and
  report comparable results with less training time (Abstract, §4).

## OLMo 2 reordered norm and QK-norm (arXiv:2501.00656)

- §2.1: "We normalize the outputs to the attention and feedforward (MLP) layers within each transformer block,
  instead of the inputs." §3.3.2 table: OLMo-0424 `h := x + Attention(LN(x))`, `h_out := h + MLP(LN(h))`; OLMo 2
  `h := x + RMSNorm(Attention(x))`, `h_out := h + RMSNorm(MLP(h))`. (§2.1 Eq. 2 prints `MLP(x)`; §3.3.2 prints
  `MLP(h)`.) The residual stream itself is not normalized after the add. "This strategy was first proposed by Liu
  et al. (2021)."
- QK-norm: "we normalize the key and query projections with RMSNorm before calculating attention. This avoids
  attention logits being too large, which can lead to training loss divergence" (§2.1).
- §3.3.2, Fig. 7: "In isolation, neither of these changes yield good results, but together they improve both the
  growth and the spikiness of the L2 norm of the gradient." Spike score of the gradients 0.108 → 0.069 "when applied
  together".
- Table 1 and `OLMo2-7B-stage1.yaml`: `norm_after: true`, `attention_layer_norm: true`, `layer_norm_type: rms`,
  `layer_norm_eps: 1e-6`.

## z-loss, AdamW ε, and embedding weight decay in OLMo 2

- §3.3.3: z-loss adds "10^-4 · log² Z" where Z is the softmax denominator. Table 1 lists Z-Loss Weight 10^-5 and the
  7B config sets `auxiliary_loss_multiplier: 1e-5` (conflict inside the source).
- §3.3.3 Fig. 8: Flash Attention's fused z-loss and a PyTorch implementation match in the forward pass but differ in
  the backward pass; the team re-trained from the divergence point with the original implementation.
- §3.4.1 Fig. 9: AdamW ε lowered from 10^-5 to 10^-8; "the gradient norm settles much more quickly and remains
  permanently lower."
- §3.4.2 Fig. 10: weight decay on embeddings shrinks embedding norm and raises gradient norm; spike scores 0.16 (with
  decay) vs 0.092 (without).

## QK-norm adoption statements in primary reports

- Qwen2.5 (arXiv:2412.15115 §2): GQA, SwiGLU, RoPE, "QKV bias", RMSNorm with pre-normalization. Card: [[qwen-2.5]].
- Qwen3 (arXiv:2505.09388 §2): "we remove QKV-bias used in Qwen2 ... and introduce QK-Norm (Dehghani et al., 2023)
  to the attention mechanism to ensure stable training for Qwen3."
