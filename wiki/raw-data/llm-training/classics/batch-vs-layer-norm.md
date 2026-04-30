<!-- scope: normalization layers — BatchNorm vs LayerNorm vs RMSNorm; pre-norm vs post-norm
     deps: [[weight-init]]
     see-also: [[mixed-precision]], [[gradient-clipping]]
-->

# Normalization: BatchNorm, LayerNorm, RMSNorm; Pre-Norm vs Post-Norm
- **Core Insight:** Normalizing intermediate activations decouples per-layer training dynamics from batch composition and from weight magnitude, dramatically stabilizing deep networks.
- **Guideline:** Use **RMSNorm + pre-norm** for any new Transformer in 2025 (Llama/Qwen/DeepSeek default); compute the normalization in fp32 even under bf16/fp8 training.
- **Authors:** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (LayerNorm 2016); Biao Zhang & Rico Sennrich (RMSNorm 2019); Ioffe & Szegedy (BatchNorm 2015)
- **Year:** 2015 / 2016 / 2019
- **URL:** https://arxiv.org/abs/1607.06450 ; https://arxiv.org/abs/1910.07467 ; https://arxiv.org/abs/1502.03167
- **Relevant topics:** training stability, normalization, residual networks, Transformer architecture

## Abstract (composite)
**BatchNorm (2015)**: normalizes each feature across the batch dimension; enabled the training of very deep CNNs (ResNet-152, Inception-v3); requires running statistics for inference; couples examples in a batch.
**LayerNorm (2016)**: normalizes across the feature dimension within each example; batch-independent, RNN-friendly; the standard normalizer for Transformers (Vaswani 2017).
**RMSNorm (2019)**: drops the mean-subtraction step of LayerNorm — normalizes only by root-mean-square magnitude. ~7–64% faster than LayerNorm with negligible quality difference; adopted by T5, Llama family, and almost all 2023+ frontier LLMs.
**Pre-norm vs post-norm** (Xiong 2020 "On Layer Normalization in the Transformer Architecture"): placing LN *inside* the residual branch (pre-norm) gives strictly better gradient flow than the original post-norm Vaswani placement, eliminating the need for warmup-only-or-divergence behavior of deep post-norm Transformers.

## Key Contributions
- **BatchNorm** introduced internal-covariate-shift framing (later debated); enabled higher learning rates and deeper CNNs.
- **LayerNorm** decoupled normalization from batch — essential for autoregressive RNNs/Transformers and for variable-batch RL rollouts.
- **RMSNorm** showed that the *re-centering* step of LayerNorm contributes nothing to representation quality; dropping it gains throughput.
- **Pre-norm placement** makes Transformers depth-stable: 100+ layer pre-norm Transformers train without exotic warmup, while post-norm Transformers diverge.
- Modern norm placement: pre-norm + a final norm before the LM head (the "ln_f" of GPT-2). Some 2024 architectures (Qwen-2.5, OLMo-2) experiment with adding an extra norm between attention and MLP outputs ("QK-norm", "double-norm") for further stability.

## Key Figures/Tables to Study
- **LayerNorm Figure 1**: training-curve comparison BN vs LN on RNNs.
- **RMSNorm Table 1**: speedups across model sizes — concrete win for LLM scale.
- **Xiong 2020 Figures 2–3**: gradient norm at initialization for pre- vs post-norm Transformers — pre-norm is `O(1)`, post-norm grows with depth, explaining its training fragility.
- **OLMo-2 architecture diagram**: shows the modern "QK-norm + reordered-norm" recipe used to stabilize 7B–70B training.

## Technical Details

**LayerNorm** (over feature dim `d`, per token):
```
mu = mean(x)
sigma2 = var(x)                                # both reduced over feature dim
x_hat = (x - mu) / sqrt(sigma2 + eps)
y = gamma * x_hat + beta                       # learnable scale + bias
```

**RMSNorm**:
```
rms = sqrt(mean(x^2) + eps)
y = gamma * x / rms                            # no mean subtract, no learnable bias
```
- Compute saving: 1 fewer reduction (no mean), 1 fewer subtract, 1 fewer parameter set (no `beta`).
- Memory saving: ~30% fewer FLOPs in the norm op (small, but norms are called twice per block in pre-norm).

**Pre-norm vs post-norm**:
```
# Post-norm (Vaswani 2017 original)
x = LN(x + Sublayer(x))

# Pre-norm (Llama, GPT-2, every modern LLM)
x = x + Sublayer(LN(x))
```
Pre-norm preserves the residual stream identity and keeps gradient norm bounded with depth. The price: pre-norm representations grow in magnitude with depth, requiring a final `LN` before the LM head. Post-norm produces sharper representations but requires careful warmup.

**Modern variants seen in 2024–2025 frontier models**:
- **QK-Norm**: `LayerNorm(Q)` and `LayerNorm(K)` before attention dot product. Used by ViT-22B, OLMo-2, Qwen-2.5. Eliminates attention-logit explosion in long-sequence training.
- **Sandwich-Norm**: norm both before *and* after each sub-layer. Used in some early 2024 architectures; adds compute, marginal gains.
- **Reordered-Norm** (OLMo-2): place the second norm *after* the residual addition for the MLP — empirically eliminates a class of mid-training loss spikes.
- **DeepNorm** (Wang 2022): a residual-scaling + init combination that stabilizes 1000-layer post-norm Transformers. Niche.

**Hyperparameters**:
- `eps`: `1e-5` (Llama, default) or `1e-6` (T5). Smaller `eps` is fine in fp32; in fp16/bf16, `1e-5` avoids NaN.
- `gamma` init: `1.0`. Some recipes init gamma slightly less (`0.1`) on the second norm in a sandwich layout.
- Learnable bias `beta`: present in LayerNorm, **absent** in RMSNorm. Modern code typically removes biases everywhere (Llama removes bias from linear layers too).

**Numerical-precision pitfall** (mixed precision): the mean-and-var reduction *must* happen in fp32. Computing `mean(x^2)` in bf16 over a 4096-dim vector accumulates errors that bias the normalization. Frameworks default to fp32 reductions inside the norm; **never** override this.

**Common pitfalls**:
- Using BN in a Transformer / RL setting → batch-coupling breaks variable-length sequences and rollout-batch reuse. Always LN/RMS in NLP.
- Post-norm at depth ≥ 24 without explicit warmup tuning → training diverges around step 1k.
- Forgetting the final `ln_f` after the last residual block in a pre-norm model → unbounded LM-head logits.
- Dropping LN's `gamma` (well-meaning "parameter saving") → loses the per-feature scaling that the model relies on.

## Connections
- **[[weight-init]]**: pre-norm models are far more init-tolerant than post-norm; LayerNorm/RMSNorm absorbs init scale errors.
- **[[mixed-precision]]**: norm reductions stay in fp32 even under bf16/fp8; this is the single most common precision bug.
- **[[gradient-clipping]]**: post-norm training without aggressive clipping is unworkable; pre-norm relaxes the requirement.
- **OLMo-2** ([[olmo-2]]): publicly documents the reordered-norm + QK-norm recipe and the loss spikes it eliminated.
- **Llama-3 / Qwen / DeepSeek**: all use RMSNorm + pre-norm; differ only in the QK-norm decision.
- **Karpathy** ([[karpathy-training-neural-net-recipe]]): "begin with the simplest model + LayerNorm + Adam — don't innovate on the norm."
