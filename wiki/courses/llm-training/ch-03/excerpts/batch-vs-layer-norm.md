---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: batch-vs-layer-norm (Ioffe & Szegedy 2015 + Ba 2016 + Zhang & Sennrich 2019 + Xiong 2020 + Henry 2020 / Dehghani 2023)
source_url: https://arxiv.org/abs/1607.06450 ; https://arxiv.org/abs/1910.07467 ; https://arxiv.org/abs/1502.03167 ; https://arxiv.org/abs/2002.04745
created_at: "2026-04-23"
---

# Excerpt: Normalization — BatchNorm, LayerNorm, RMSNorm; Pre-Norm vs Post-Norm; QK-Norm

**Sources (composite family):**
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML 2015 — arxiv 1502.03167
- Ba, Kiros, Hinton, "Layer Normalization," 2016 — arxiv 1607.06450
- Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019 — RMSNorm — arxiv 1910.07467
- Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020 — pre-norm vs post-norm — arxiv 2002.04745
- Henry et al., "Query-Key Normalization for Transformers," 2020 — arxiv 2010.04245
- Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters" (ViT-22B), 2023 — QK-norm at scale — arxiv 2302.05442
- OLMo-2 technical report, 2024 — reordered-norm + QK-norm recipe

---

## Why norms decide depth-scaling

From [[batch-vs-layer-norm]]:

> "Normalizing intermediate activations decouples per-layer training dynamics from batch composition and from weight magnitude, dramatically stabilizing deep networks."

That is the core insight, but it conceals three independent decisions the architect must make:

1. **Which normaliser** — BatchNorm, LayerNorm, RMSNorm, something custom?
2. **Where does it sit** relative to the residual connection — pre-norm or post-norm?
3. **What precision** does the reduction run in — fp32 always, or does it follow the ambient dtype?

This excerpt covers all three plus the 2024–2025 variants (QK-norm, reordered-norm, sandwich-norm) that every frontier team is experimenting with.

---

## 1. BatchNorm (Ioffe & Szegedy, 2015) — why it doesn't belong in Transformers

BatchNorm normalizes each **feature channel** across the **batch dimension**:

```math
\mu_c = \frac{1}{B \cdot T}\sum_{b,t} x_{b,t,c}, \quad \sigma^2_c = \frac{1}{B \cdot T}\sum_{b,t} (x_{b,t,c} - \mu_c)^2
```

Then `x_hat = (x - μ) / √(σ² + ε)`, followed by learned affine `y = γ · x_hat + β`.

The conceptual problem for language modelling: the batch dimension is a **variable-meaning axis**. During pretraining with sequence packing, "batch" conflates many documents. During inference, batch size is whatever arrived. During RL rollouts, batches are variable-length trajectories. You cannot maintain stable running statistics across this zoo.

From [[batch-vs-layer-norm]]:

> "BatchNorm: requires running statistics for inference; couples examples in a batch. Using BN in a Transformer / RL setting → batch-coupling breaks variable-length sequences and rollout-batch reuse. Always LN/RMS in NLP."

Internal-covariate-shift — Ioffe's original framing for *why* BN works — has been largely debunked (Santurkar et al. 2018 show BN's benefit is about loss-landscape smoothness, not covariate shift). The benefit was real; the explanation was wrong; and for Transformers the cost (batch coupling) outweighs the benefit.

What does this mean for LLM training in 2025? **Never use BatchNorm in a Transformer.** If you see BN in a language model, it's a bug or a legacy codebase. Vision Transformers use LayerNorm for the same reason — they want batch-independence.

---

## 2. LayerNorm (Ba, Kiros, Hinton, 2016)

LayerNorm normalizes across the **feature dimension** within each example, independent of batch:

```math
\mu = \frac{1}{d}\sum_{i=1}^d x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2
```

```math
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma_i \hat{x}_i + \beta_i
```

Key properties:
- Batch-independent — per-token, per-example reduction over the feature dim.
- Same compute at train and inference time (no running statistics).
- Scale-invariant: if you replace `x` with `α x` for any `α > 0`, `LN(α x) = LN(x)` (assuming `β = 0`).

The last property is load-bearing. It is why LayerNorm *absorbs* init-scale errors — scaling up all weights in a block does not change the block's output after LN. This is formalised in [[excerpts/weight-init]]: pre-LN Transformers are "init-tolerant" precisely because LN kills the multiplicative scale.

### The `eps` hyperparameter

`ε` prevents division by zero when `σ² = 0`. Convention:
- Llama, default: `ε = 1e-5`.
- T5: `ε = 1e-6`.

The [[batch-vs-layer-norm]] note:

> "Smaller `eps` is fine in fp32; in fp16/bf16, `1e-5` avoids NaN."

`1e-6` added to a bf16 squared-norm can be lost to rounding — bf16 has only 7 mantissa bits. If the norm then divides by `√0`, you get NaN. For bf16/fp8 training, prefer `1e-5`.

---

## 3. RMSNorm (Zhang & Sennrich, 2019) — drop the mean

The observation: LayerNorm does two things — **re-centering** (subtract mean) and **re-scaling** (divide by std). The paper asks: is re-centering necessary? Answer: empirically, no.

RMSNorm:

```math
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}, \qquad y_i = \gamma_i \cdot \frac{x_i}{\mathrm{RMS}(x)}
```

Notice: **no mean subtraction, no `β` bias**. Just a variance-only normalization with a learnable per-feature scale.

### What did we save?

From [[batch-vs-layer-norm]]:

> "Compute saving: 1 fewer reduction (no mean), 1 fewer subtract, 1 fewer parameter set (no `beta`). Memory saving: ~30% fewer FLOPs in the norm op (small, but norms are called twice per block in pre-norm)."

Concretely:
- LayerNorm: 2 reductions (mean, var), 1 subtract, 2 learnable parameters per feature (`γ`, `β`).
- RMSNorm: 1 reduction (mean of squares), 0 subtracts, 1 learnable parameter per feature (`γ`).

RMSNorm Table 1 (the paper) reports **7–64% speedup** on the norm op across model sizes. In a 32-layer Transformer with pre-norm, there are 64 norm calls per forward pass. The savings compound.

### Empirical quality — the surprising part

The original RMSNorm paper tested across machine translation, language modelling, and image classification and reported **no measurable quality difference** between LayerNorm and RMSNorm. Why? The re-centering step of LN moves the pre-activation's mean to zero, but in practice:
- The subsequent weight matrix's bias parameter can absorb any constant shift.
- The residual connection preserves the original mean through the stream.
- The softmax in attention is translation-invariant — mean-shifts in QK don't affect weights.

So the mean subtraction turns out to be decorative. The variance normalization is what actually stabilises training.

What does this mean for LLM training in 2025? **Use RMSNorm for any new Transformer.** Llama, Qwen, DeepSeek, OLMo, Mistral, Gemma — all RMSNorm. The `[[ch-03]]` source code uses RMSNorm with fp32 reduction as the default. From [[batch-vs-layer-norm]]:

> "Llama-3 / Qwen / DeepSeek: all use RMSNorm + pre-norm; differ only in the QK-norm decision."

---

## 4. Pre-norm vs Post-norm (Xiong et al., 2020)

```python
# Post-norm (Vaswani 2017 original)
x = LN(x + Sublayer(x))

# Pre-norm (every modern LLM)
x = x + Sublayer(LN(x))
```

Xiong et al.'s Figure 2–3 is the mic-drop. At initialization, measure gradient norm at each layer, post-norm vs pre-norm.

Post-norm:

```math
\|\nabla_{\theta_\ell}\| = \mathcal{O}(\ell) \;\text{at depth } \ell
```

Gradient norm grows linearly with depth. At layer 32, the gradient is 32× what it was at layer 1. AdamW's `v̂` adapts, but only over many steps — for the first few iterations the 32× ratio shows up as a loss spike.

Pre-norm:

```math
\|\nabla_{\theta_\ell}\| = \mathcal{O}(1) \;\text{at all depths}
```

Depth-invariant. The residual stream carries the gradient straight through (the `x +` path is an identity for gradients), so depth doesn't accumulate in the gradient norm.

### Why this kills post-norm at scale

The [[batch-vs-layer-norm]] source:

> "Post-norm placement at depth ≥ 24 without explicit warmup tuning → training diverges around step 1k."

The original Transformer (Vaswani 2017) was 6 layers encoder + 6 decoder. Post-norm worked — barely. GPT-2 went to 12 layers of decoder with post-norm and struggled with training instability. GPT-3 switched to pre-norm at 96 layers. Every frontier model since is pre-norm.

The trade-off: pre-norm's **residual-stream magnitude grows with depth** (see [[excerpts/weight-init]] §residual-scaling derivation). So you need a final `ln_f` before the LM head. Every modern model has one.

### DeepNorm — the post-norm holdout

Wang et al. 2022 proposed DeepNorm, a residual scaling + init combination that stabilises 1000-layer post-norm Transformers. Niche. Microsoft used it for some Swin variants. No frontier LLM uses DeepNorm — pre-norm is cleaner.

---

## 5. QK-Norm (Henry 2020, Dehghani 2023 ViT-22B)

Standard attention computes `softmax(QK^T / √d_k)`. The `QK^T` can explode in magnitude on long contexts — especially when Q or K entries drift during training. QK-norm inserts a LayerNorm on Q and K *before* the dot product:

```math
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{\mathrm{LN}(Q) \cdot \mathrm{LN}(K)^T}{\sqrt{d_k}}\right) V
```

### Why

Henry et al. (2020) observed that attention logits in deep/trained Transformers occasionally reach magnitudes of `±1000`. Softmax of such logits is effectively a hard-argmax — attention becomes one-hot, gradients through the softmax vanish, and the model stops learning. ViT-22B (Dehghani 2023) hit this at 22B-parameter vision scale.

QK-norm clamps the logit scale: after LN, `|Q_i|, |K_i| = O(1)`, so `|QK^T| = O(√d_k)`. Bounded, stable, no softmax saturation.

Adopted by:
- ViT-22B (Dehghani 2023) — original motivation.
- OLMo-2 (2024) — full recipe public.
- Qwen-2.5 (2024).
- Various 2024+ long-context experiments.

What does this mean for LLM training in 2025? If you're training beyond 32K context or scaling past 13B parameters, **add QK-norm**. The compute cost is trivial (two LN calls per attention block) and it eliminates a class of long-context training failures.

---

## 6. Reordered-norm (OLMo-2) and sandwich-norm

### Reordered-norm

OLMo-2's twist: place the *second* norm of the pre-norm block **after** the residual addition, not before the sublayer.

```python
# Standard pre-norm MLP block
x = x + MLP(LN(x))

# OLMo-2 reordered for MLP
x = LN(x + MLP(LN(x)))
```

The outer LN re-normalises the residual stream at every depth, fighting the residual-growth problem at a small compute cost. OLMo-2 reports this eliminates a class of mid-training loss spikes at 7B+ scale.

### Sandwich-norm

Norm both before *and* after each sub-layer:

```python
x = x + LN_post(Sublayer(LN_pre(x)))
```

Used by some 2024 experiments (Anthropic has mentioned it publicly; not confirmed for frontier production). Adds compute for marginal gains. Niche.

From [[batch-vs-layer-norm]]:

> "OLMo-2 architecture diagram: shows the modern 'QK-norm + reordered-norm' recipe used to stabilize 7B–70B training."

What does this mean for LLM training in 2025? Reordered-norm is cheap insurance against loss spikes. Worth the extra LN if you've seen spikes. Sandwich-norm is optional — skip unless you have a specific reason.

---

## 7. The precision rule — fp32 reductions are non-negotiable

The single most common precision bug in home-grown training code. From [[batch-vs-layer-norm]]: "the mean-and-var reduction *must* happen in fp32. Computing `mean(x^2)` in bf16 over a 4096-dim vector accumulates errors that bias the normalization."

In bf16 (7 mantissa bits, relative precision ~1%), summing 4096 squared-gaussians silently drops low-magnitude contributions once the running sum exceeds ~128. The result is a 5–15% underestimate of the true sum — the normalization over-scales by a matching factor, every step, every layer, silently.

The fix: cast to fp32 for the reduction, cast back to ambient dtype. See `[[ch-03]]` §5 for the reference implementation and [[excerpts/mixed-precision]] for the full numerical analysis including fp8.

---

## 8. Common pitfalls (quoting the source)

From [[batch-vs-layer-norm]]:

> - "Using BN in a Transformer / RL setting → batch-coupling breaks variable-length sequences and rollout-batch reuse."
> - "Post-norm at depth ≥ 24 without explicit warmup tuning → training diverges around step 1k."
> - "Forgetting the final `ln_f` after the last residual block in a pre-norm model → unbounded LM-head logits."
> - "Dropping LN's `gamma` (well-meaning 'parameter saving') → loses the per-feature scaling that the model relies on."

The final `ln_f` one is the easiest to forget. In a pre-norm stack, the residual stream exits the last block with magnitude `O(√(2L))`. Feeding that into an LM head gives logits of magnitude `O(√(2L) · √d)` — absurdly large. The `ln_f` brings it back to `O(1)` before the LM head matmul. GPT-2's `ln_f` is a half-dozen-line function that has been silently load-bearing for every LLM since.

---

## 9. The modern recipe (what to actually ship)

From [[batch-vs-layer-norm]] and [[ch-03]] synthesis:

1. **RMSNorm** with `ε = 1e-5`, `γ` init to 1.0, no bias.
2. **Pre-norm** placement: `x = x + Sublayer(RMSNorm(x))`.
3. **Final `ln_f`** before LM head.
4. **Reductions in fp32** even under bf16/fp8 training.
5. **Optional: QK-norm** for >13B or >32K context.
6. **Optional: reordered-norm** if you hit mid-training loss spikes.
7. **No BatchNorm. Anywhere. Ever.**

---

## Connections

- [[excerpts/weight-init]] — residual-scaling `1/√(2L)` complements pre-norm; without it pre-norm still has depth-growing magnitude.
- [[excerpts/mixed-precision]] — norm reductions must stay in fp32 under bf16/fp8; this is the single most common precision bug.
- [[excerpts/lr-schedules]] — pre-norm's `O(1)` gradient-norm property lets schedules stay aggressive; post-norm requires gentler warmup.
- [[excerpts/adam]] — norm layers have γ and β parameters typically excluded from weight decay in the optimizer's no-decay group.
- [[ch-03]] — synthesis with reference RMSNorm implementation.
- [[ch-02]] — mixed-precision details for the fp32-reduction rule.
- [[olmo-2]] — reference public implementation of QK-norm + reordered-norm.
