# Chapter 9: Normalization and Residual Connections

<!-- scope: normalization techniques (LayerNorm, RMSNorm, QK-norm), pre-norm vs post-norm placement, residual stream as information highway, initialization strategies
     deps: [[ch-03]]
     see-also: [[ch-24]], [[ch-08]]
-->

## Overview

Normalization and residual connections are the two mechanisms that make deep Transformers trainable. Without them, a 96-layer GPT model would be a mathematical curiosity — gradients would vanish or explode within the first few hundred training steps, and no amount of hyperparameter tuning would save you.

This chapter traces the evolution from Batch Normalization (which fails for Transformers) through LayerNorm, RMSNorm, and QK-norm, and explains why each simplification works. The answer is consistently surprising: the component you'd expect to matter (mean-centering, the bias term, full statistical normalization) turns out to be dispensable. What survives is the minimal operation that keeps activations on a stable scale. We'll also formalize the "residual stream" view of Transformers — the interpretive framework from mechanistic interpretability where each layer reads from and writes to a shared information highway — and show how norm placement relative to this stream determines whether your model trains stably or requires fragile warmup hacks.

---

## 1. Why Normalization Exists

Each layer's input distribution shifts as preceding layers update — **internal covariate shift** (Ioffe & Szegedy, 2015). Without normalization, training becomes unstable as depth increases: you need increasingly small learning rates, careful initialization, and warmup schedules to prevent divergence. Normalization standardizes activations to a known scale at each layer boundary, decoupling the layers. The question is: *what statistics do you normalize over?*

---

## 2. BatchNorm, LayerNorm, and Why the Normalization Axis Matters

### Batch Normalization (BatchNorm)

BatchNorm normalizes across the **batch dimension** — for each feature $d$, it computes $\mu_d$ and $\sigma_d^2$ across all $B$ examples in the mini-batch: $\text{BatchNorm}(x_{b,d}) = \gamma_d \cdot (x_{b,d} - \mu_d) / \sqrt{\sigma_d^2 + \epsilon} + \beta_d$.

**Why it fails for Transformers:** (1) Variable-length sequences corrupt batch statistics — padding tokens mix with real tokens at the same position. (2) Per-GPU micro-batch sizes can be small under model parallelism, making statistics noisy. (3) At inference, autoregressive generation has no "batch" — BatchNorm's running averages from training are a poor substitute.

### Layer Normalization (LayerNorm)

LayerNorm normalizes across the **feature dimension** for each individual example (Ba, Kiros & Hinton, 2016, [[layer-norm|paper]]):

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$ are computed across the hidden dimension $d$ for a single token's activation vector.

This solves all three of BatchNorm's problems: it's independent of batch size, handles variable-length sequences naturally, and computes identically at training and inference time. LayerNorm became the standard normalization for Transformers from the original "Attention Is All You Need" paper onward.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Normalization Axis Comparison</div>
<table style="width:100%; border-collapse:collapse; font-size:13px; color:#e0e0e0;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:10px 12px; color:#e94560;">Property</th>
<th style="text-align:center; padding:10px 12px; color:#e94560;">BatchNorm</th>
<th style="text-align:center; padding:10px 12px; color:#e94560;">LayerNorm</th>
<th style="text-align:center; padding:10px 12px; color:#e94560;">RMSNorm</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #2a2a4a;">
<td style="padding:10px 12px; font-weight:bold;">Normalization axis</td>
<td style="padding:10px 12px; text-align:center;">Batch (B)</td>
<td style="padding:10px 12px; text-align:center;">Features (d)</td>
<td style="padding:10px 12px; text-align:center;">Features (d)</td>
</tr>
<tr style="border-bottom:1px solid #2a2a4a;">
<td style="padding:10px 12px; font-weight:bold;">Mean subtraction</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">No</td>
</tr>
<tr style="border-bottom:1px solid #2a2a4a;">
<td style="padding:10px 12px; font-weight:bold;">Learnable bias (beta)</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">No</td>
</tr>
<tr style="border-bottom:1px solid #2a2a4a;">
<td style="padding:10px 12px; font-weight:bold;">Batch-size dependent</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">No</td>
<td style="padding:10px 12px; text-align:center;">No</td>
</tr>
<tr style="border-bottom:1px solid #2a2a4a;">
<td style="padding:10px 12px; font-weight:bold;">Train/test identical</td>
<td style="padding:10px 12px; text-align:center;">No</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
</tr>
<tr style="border-bottom:1px solid #2a2a4a;">
<td style="padding:10px 12px; font-weight:bold;">Re-centering invariance</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">No</td>
</tr>
<tr>
<td style="padding:10px 12px; font-weight:bold;">Re-scaling invariance</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
<td style="padding:10px 12px; text-align:center;">Yes</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">Re-scaling invariance (invariance to input magnitude scaling) is sufficient for training stability. Re-centering invariance (invariance to constant shifts) is dispensable.</div>
</div>

---

## 3. RMSNorm: The Essential Simplification

[Deep Dive: RMSNorm vs LayerNorm Implementation Comparison](excerpts/rmsnorm-vs-layernorm.md) — side-by-side mathematical definitions, pseudocode, fused CUDA kernel comparison, and performance benchmarks.

Zhang & Sennrich (2019) ([[rmsnorm|paper]]) asked a simple question: does the mean-centering in LayerNorm actually help? The answer is no.

**RMSNorm** drops the mean subtraction entirely and normalizes by the root mean square:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

No mean computation, no mean subtraction, no learnable bias $\beta$. Just divide by the RMS and apply a learned scale $\gamma$.

### Why Dropping the Mean Works

The key theoretical insight is the distinction between two invariance properties:

- **Re-centering invariance:** $f(x + c\mathbf{1}) = f(x)$ for any constant $c$. LayerNorm has this (mean subtraction absorbs constant shifts). RMSNorm does not.
- **Re-scaling invariance:** $f(\alpha x) = f(x) \cdot \text{sign}(\alpha)$ for any scalar $\alpha$. Both LayerNorm and RMSNorm have this.

Re-scaling invariance is what actually matters for training stability. It ensures that the magnitude of activations flowing through the network stays bounded regardless of how weights are initialized or how they drift during training. The implicit learning rate adaptation that comes from re-scaling invariance — where the effective learning rate adapts based on the weight norm — is the mechanism that prevents gradient explosion.

Re-centering invariance (absorbing constant shifts) provides a theoretical nicety but doesn't solve the core problem of activation scale. In practice, activations in deep Transformers don't develop large DC offsets that would require centering — the dominant failure mode is scale explosion, which RMSNorm handles.

### Computational Savings

RMSNorm eliminates:
- One reduction operation (computing the mean across $d$ dimensions)
- One element-wise subtraction (subtracting the mean)
- The learnable bias parameter $\beta$

On GPU kernels, each reduction requires a synchronization across threads within a warp/block. Eliminating one reduction per normalization layer, applied at every layer of every token in every training step, compounds to **7-64% wall-clock speedup** depending on model size and architecture (larger speedups for larger hidden dimensions where the reduction cost is more significant).

RMSNorm is now the default in LLaMA, Mistral, PaLM, GPT-NeoX, OLMo 2, Gemma 3, and essentially every modern LLM.

---

## 4. Pre-Norm vs. Post-Norm: Where You Normalize Changes Everything

The original Transformer (Vaswani et al., 2017) placed LayerNorm **after** the residual addition — the **post-norm** configuration:

$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$

Most modern LLMs use **pre-norm**, placing normalization **before** the sublayer:

$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

This seemingly trivial rearrangement has profound consequences for training dynamics.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Pre-Norm vs. Post-Norm Placement</div>
<div style="display:flex; gap:48px; justify-content:center; flex-wrap:wrap; font-family:monospace; font-size:12px;">
<div style="text-align:center;">
<div style="color:#e94560; font-weight:bold; margin-bottom:8px; font-family:sans-serif; font-size:13px;">POST-NORM</div>
<div style="color:#e0e0e0;">x<sub>l</sub> ──┬──────────────┐</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│</div>
<div style="color:#e94560;">&nbsp;&nbsp;Sublayer(Attn/FFN)&nbsp;│</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│</div>
<div style="color:#e94560; font-weight:bold;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ ←────────────┘</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</div>
<div style="color:#e94560;">&nbsp;&nbsp;<span style="background:#2a1a3e; padding:2px 8px; border-radius:4px;">LayerNorm</span></div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</div>
<div style="color:#e0e0e0;">x<sub>l+1</sub></div>
</div>
<div style="text-align:center;">
<div style="color:#e94560; font-weight:bold; margin-bottom:8px; font-family:sans-serif; font-size:13px;">PRE-NORM</div>
<div style="color:#e0e0e0;">x<sub>l</sub> ──┬──────────────┐</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│</div>
<div style="color:#e94560;">&nbsp;&nbsp;<span style="background:#2a1a3e; padding:2px 8px; border-radius:4px;">RMSNorm</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│</div>
<div style="color:#e94560;">&nbsp;&nbsp;Sublayer(Attn/FFN)&nbsp;│</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│</div>
<div style="color:#e94560; font-weight:bold;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ ←────────────┘</div>
<div style="color:#e0e0e0;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</div>
<div style="color:#e0e0e0;">x<sub>l+1</sub></div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">Pre-norm: residual path is unobstructed identity. Post-norm: norm sits on the residual path, distorting gradient flow.</div>
</div>

[Deep Dive: Pre-Norm Stability Proof Sketch](excerpts/pre-norm-stability.md) — walk through the mean field theory argument: why post-norm gradients are O(d*L) at initialization while pre-norm gradients are O(1).

[Interactive: Pre-Norm vs Post-Norm Placement Comparison](figures/norm-placement.html) — animated forward pass through both configurations, with side-by-side diagrams and simulated training loss curves showing stability differences.

### Why Pre-Norm Trains More Stably

Xiong et al. (2020) ([[pre-norm-vs-post-norm|paper]]) proved this rigorously using mean field theory. The core argument:

**Post-norm:** The LayerNorm sits on the main residual path. At initialization, the expected gradient magnitude near the output layer is $O(d \cdot L)$ — it grows with both hidden dimension $d$ and depth $L$. This means that without warmup, the first gradient steps are disproportionately large for the output-side layers, causing training instability. The learning rate warmup that every original Transformer required was a patch for this gradient imbalance.

**Pre-norm:** The normalization is inside the branch, not on the main residual highway. The residual connection provides an unobstructed identity path from input to output. At initialization, gradients flow through this identity path with $O(1)$ magnitude — independent of depth. Each sublayer contributes an additive gradient term, but the identity path dominates at initialization, keeping the overall gradient well-behaved.

The practical consequence: pre-norm Transformers **don't need learning rate warmup** and are more robust to hyperparameter choices. This was a major breakthrough for scaling — fewer knobs to tune means faster iteration on architectures.

### The Representational Cost of Pre-Norm

Pre-norm is not strictly better. There's a tradeoff:

In pre-norm, the residual path is an identity. Each sublayer's contribution passes through a normalization before being added to the residual stream, which constrains its magnitude. This means each layer can only make a "small" additive update to the residual stream. Post-norm applies normalization *after* the addition, allowing each layer's contribution to have larger relative impact before being normalized.

Empirically, post-norm models can achieve slightly better final performance when training is successful — but they're harder to train and more prone to divergence. For practical LLM training at scale, the stability of pre-norm wins.

### The Hybrid Approach (Gemma 3)

Gemma 3 ([[gemma-3|report]]) uses **both** pre-norm and post-norm — RMSNorm before and after attention/FFN sublayers. This attempts to capture post-norm's representational benefit while using pre-norm for stability. The Raschka comparison ([[raschka-llm-architecture-comparison|blog]]) notes this as a distinctive design choice among current architectures.

### OLMo 2's Post-Norm Revival

Interestingly, OLMo 2 ([[olmo-2|report]]) moved to post-norm placement (RMSNorm after attention/FFN) combined with QK-norm and improved initialization. This suggests post-norm's instability can be overcome with complementary stability techniques, potentially recovering its representational advantage. The OLMo 2 team reported that post-norm + QK-norm improved training stability over their previous OLMo 1 architecture.

---

## 5. QK-Norm: Taming Attention Logit Growth

[Deep Dive: QK-Norm Gradient Analysis](excerpts/qk-norm-gradient-analysis.md) — the entropy collapse cascade, how QK-norm bounds logits and self-regulates gradients, and comparison with logit soft-capping.

As Transformers scale to billions of parameters and train on trillions of tokens, a subtle failure mode emerges: **attention logit growth**. The dot products $q_i^T k_j$ that determine attention weights can grow unboundedly during training, causing attention entropy to collapse — the model increasingly attends to a single token, losing the ability to distribute attention across relevant context.

### The Problem

Attention scores are computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The $\sqrt{d_k}$ scaling prevents explosion at initialization, but during training, the norms of $Q$ and $K$ vectors can grow, making $q_i^T k_j$ increasingly large in magnitude. When attention logits are large, softmax saturates — it approaches a one-hot distribution. This is **attention entropy collapse**: the model loses its ability to attend broadly and becomes stuck attending to a few tokens.

This manifests as **loss spikes** during training — sudden jumps in loss that can destabilize or ruin long training runs. For a 6-trillion-token OLMo 2 training run, even rare instabilities are intolerable.

### The QK-Norm Solution

QK-norm applies normalization to queries and keys before the dot product:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\text{Norm}(Q) \cdot \text{Norm}(K)^T}{\sqrt{d_k}}\right)V$$

where $\text{Norm}$ is typically RMSNorm or L2 normalization applied per-head. This bounds the magnitude of $q_i^T k_j$ because both vectors are normalized to unit (or bounded) scale. The attention logits can no longer grow unboundedly regardless of how training evolves.

**OLMo 2** ([[olmo-2|report]]) uses QK-norm (RMSNorm on queries and keys before RoPE application) as a core stability mechanism. Combined with Z-loss regularization (which penalizes large logits in the output layer), QK-norm was one of the key changes from OLMo 1 that enabled stable training across their 7B, 13B, and 32B models on multi-trillion-token runs.

**Gemma 3** ([[gemma-3|report]]) replaced the logit soft-capping mechanism from Gemma 2 with QK-norm, noting it was both simpler and more effective. Soft-capping applies a $\tanh$ ceiling to attention logits ($\text{logits} \leftarrow \text{cap} \cdot \tanh(\text{logits} / \text{cap})$), which prevents explosion but introduces a nonlinearity that complicates gradient flow. QK-norm achieves the same stability goal through normalization, which is a more principled operation.

### Why QK-Norm Doesn't Hurt Expressiveness

A natural concern: doesn't normalizing Q and K discard magnitude information? Yes — but attention scores only need to encode *relative* preferences (which keys a query should attend to), not absolute magnitudes. The softmax already discards absolute scale information. QK-norm makes this explicit by normalizing before the dot product rather than relying on the softmax to handle growing logits.

---

## 6. The Residual Stream as Information Highway

[Interactive: Residual Stream Information Flow](figures/residual-stream.html) — interactive diagram showing how attention and FFN layers read from and write to the residual stream, with gradient flow visualization and layer count slider.

The most powerful conceptual framework for understanding Transformers is the **residual stream** view, developed in the mechanistic interpretability literature (Elhage et al., "A Mathematical Framework for Transformer Circuits", 2021).

### The Core Idea

Consider a Transformer with $L$ layers. The residual stream is the vector that flows through the skip connections:

$$x_0 \xrightarrow{+f_1(x_0)} x_1 \xrightarrow{+f_2(x_1)} x_2 \xrightarrow{\cdots} x_L$$

where $x_{l+1} = x_l + f_{l+1}(x_l)$ and $f_l$ is the sublayer function (attention or FFN). Unrolling:

$$x_L = x_0 + \sum_{l=1}^{L} f_l(x_{l-1})$$

The final representation is the embedding $x_0$ plus the sum of all layer contributions. This is not just notation — it's the actual computation. Each layer **reads from** the residual stream (via the norm + projection into Q/K/V or FFN input) and **writes to** the residual stream (via the output projection added back). The residual stream is a shared communication bus.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:12px; font-family:sans-serif; font-weight:bold;">The Residual Stream: Layers Read and Write</div>
<div style="font-family:monospace; font-size:12px; color:#e0e0e0; line-height:1.8; padding:0 20px;">
<span style="color:#e94560; font-weight:bold;">x<sub>0</sub></span> (embedding)
→ <span style="color:#4ecdc4;">+ f<sub>attn,1</sub></span> (<span style="color:#888;">Attn 1 reads x<sub>0</sub>, writes delta</span>)
→ <span style="color:#4ecdc4;">+ f<sub>ffn,1</sub></span>&nbsp; (<span style="color:#888;">FFN 1 reads x<sub>1</sub>, writes delta</span>)
→ &nbsp;. . .
→ <span style="color:#4ecdc4;">+ f<sub>attn,L</sub></span> (<span style="color:#888;">Attn L reads x<sub>2L-1</sub>, writes delta</span>)
→ <span style="color:#4ecdc4;">+ f<sub>ffn,L</sub></span>&nbsp; (<span style="color:#888;">FFN L reads x<sub>2L</sub>, writes delta</span>)
→ <span style="color:#e94560; font-weight:bold;">x<sub>L</sub> = x<sub>0</sub> + &Sigma; f<sub>l</sub></span>
</div>
<div style="color:#888; font-size:11px; margin-top:10px;">Each layer reads the current stream state, computes a delta, and adds it back. The final output is the embedding plus all deltas.</div>
</div>

### Why This Framework Matters

**Gradient flow.** The gradient of the loss with respect to $x_l$ is:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{k=l+1}^{L}\left(I + \frac{\partial f_k}{\partial x_{k-1}}\right)$$

The identity matrix $I$ in each factor ensures a direct gradient path through the residual connections — even if $\frac{\partial f_k}{\partial x_{k-1}}$ is poorly conditioned, the $I$ term preserves signal. This is He et al.'s ([[resnet|paper]]) core ResNet insight.

**Superposition.** The $d$-dimensional stream must encode information from all layers simultaneously. Each layer's additive write means features can interfere, leading to the **superposition hypothesis** — models encode more features than dimensions allow by using nearly-orthogonal directions.

**Layer ablation.** Additive contributions mean removing a single layer causes graceful degradation in pre-norm Transformers. Post-norm models show catastrophic collapse from single-layer removal because the norm on the residual path creates tighter inter-layer coupling.

---

## 7. Initialization and Its Interaction with Normalization

At initialization, each sublayer $f_l$ is random. If $\text{Var}(f_l) \approx \text{Var}(x_l)$, then after $L$ layers the variance grows by $O(L)$ — activations explode. Two solutions:

**Scaled initialization (GPT-2/3):** Scale each sublayer's output projection by $\frac{1}{\sqrt{2L}}$, ensuring each contribution has variance $O(1/L)$. After $L$ layers, total added variance is $O(1)$. The factor of 2 accounts for two sublayers (attention + FFN) per block.

**Pre-norm's forgiveness:** Pre-norm applies normalization before the sublayer, so each sublayer always receives standardized-scale inputs regardless of upstream drift. This makes pre-norm far less sensitive to initialization — another reason it dominates.

**OLMo 2's** ([[olmo-2|report]]) **approach:** Improved initialization preserving activation/gradient scale, combined with QK-norm and Z-loss, enabled post-norm placement while maintaining stability.

The practical map:
- **Pre-norm + RMSNorm** = modern default (LLaMA, Mistral, GPT-NeoX). Easy to train, robust.
- **Post-norm + standard init** = needs warmup, can diverge. Gradients are $O(d \cdot L)$ at init.
- **Post-norm + scaled init + QK-norm + Z-loss** = OLMo 2 recipe. Trainable, potentially better final performance.

---

## Core Insights from the Literature

### Insight 1: Only the scaling component of normalization is essential

**Paper:** Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) ([[rmsnorm|paper]])

LayerNorm has two components: mean-centering (re-centering invariance) and variance normalization (re-scaling invariance). RMSNorm drops mean-centering entirely and performs just as well, with 7-64% speedup. This reveals that the essential function of normalization in Transformers is **controlling activation scale**, not correcting distribution shape. The mean of activations across hidden dimensions is simply not a source of instability that needs active correction. **Guideline:** When designing normalization for new architectures, start with the simplest operation that controls scale. Add complexity only if you have evidence that scale control alone is insufficient.

### Insight 2: Norm placement determines whether warmup is a necessity or a crutch

**Paper:** Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) ([[pre-norm-vs-post-norm|paper]])

The learning rate warmup that every early Transformer used was not a generic training trick — it was a specific patch for the gradient imbalance caused by post-norm placement. Pre-norm makes warmup unnecessary by ensuring $O(1)$ gradient magnitude at initialization. This paper turned norm placement from an implementation detail into a principled architectural choice with provable consequences. **Guideline:** If your training requires fragile warmup schedules, check whether the instability is caused by norm placement before adding complexity. Pre-norm eliminates the root cause rather than patching the symptom.

### Insight 3: Skip connections solve a deeper problem than vanishing gradients

**Paper:** He et al., "Deep Residual Learning for Image Recognition" (2015) ([[resnet|paper]])

The standard explanation for skip connections is "they prevent vanishing gradients." This is true but incomplete. He et al. observed a **degradation problem** where deeper *plain* networks had *higher training error* than shallower ones — not just higher test error (which would be overfitting) but higher training error. This means the deeper networks couldn't even express the identity function effectively, despite having strictly more capacity. Skip connections solve this by making the identity the default: each layer learns $f(x) = 0$ at initialization (a residual of zero), and the identity flows through for free. **Guideline:** When stacking more layers improves theory but hurts practice, the likely cause is optimization landscape issues, not capacity. Residual connections fix the optimization without constraining the capacity.

### Insight 4: Attention stability requires explicit normalization at scale

**Papers:** OLMo 2 Technical Report (AI2, 2025) ([[olmo-2|report]]); Gemma 3 Technical Report (Google DeepMind, 2025) ([[gemma-3|report]])

Both OLMo 2 and Gemma 3 independently converged on QK-norm as a solution to attention logit growth. OLMo 2 uses it alongside Z-loss; Gemma 3 uses it to replace soft-capping. The convergence of two independent teams on the same solution — normalizing Q and K before the dot product — strongly suggests this is not a trick but a necessary component for stable training at the multi-trillion-token scale. **Guideline:** For any model training beyond ~1T tokens, QK-norm should be considered a default rather than an optional stability measure. The cost is negligible; the risk of loss spikes without it is real.

---

## Key Takeaways

1. **Normalization axis determines applicability.** BatchNorm (across batch) fails for variable-length sequences and autoregressive generation. LayerNorm and RMSNorm (across features) work because they're independent of batch size and sequence length.

2. **RMSNorm is the modern default.** Dropping mean-centering loses re-centering invariance but preserves the re-scaling invariance that actually prevents instability. Simpler, faster, and empirically equivalent — used by LLaMA, Mistral, PaLM, OLMo 2, Gemma 3.

3. **Pre-norm makes deep Transformers trainable without warmup.** Placing normalization before the sublayer keeps the residual path as a clean identity, giving $O(1)$ gradient magnitude at initialization regardless of depth.

4. **Post-norm has a representational advantage** but requires complementary techniques (QK-norm, Z-loss, scaled initialization) to train stably. OLMo 2 demonstrates this is achievable.

5. **QK-norm prevents attention entropy collapse at scale.** Normalizing queries and keys before the dot product bounds attention logits, preventing the softmax saturation that causes loss spikes during multi-trillion-token training.

6. **The residual stream is a shared communication bus**, not just a gradient trick. Each layer reads from and writes to this bus additively. This view explains gradient flow, layer ablation behavior, and feature superposition.

7. **Initialization and normalization are coupled.** Scaled initialization ($1/\sqrt{2L}$) controls variance growth across depth; normalization controls it at each layer. The right combination depends on norm placement: pre-norm is forgiving, post-norm is sensitive.

---

## References

- Ba, Kiros & Hinton, "Layer Normalization" (2016) — [[layer-norm|paper]]
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) — [[rmsnorm|paper]]
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) — [[pre-norm-vs-post-norm|paper]]
- He et al., "Deep Residual Learning for Image Recognition" (2015) — [[resnet|paper]]
- OLMo 2 Technical Report, AI2 (2025) — [[olmo-2|report]]
- Gemma 3 Technical Report, Google DeepMind (2025) — [[gemma-3|report]]
- Raschka, "The Big LLM Architecture Comparison" (2025) — [[raschka-llm-architecture-comparison|blog]]
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021)
- Vaswani et al., "Attention Is All You Need" (2017)
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (2015)
