# Chapter 3: The Original Transformer

<!-- scope: Vaswani et al. (2017) deep dive — encoder-decoder, self-attention, cross-attention, residual stream, positional encoding, training innovations
     deps: [[ch-02]]
     see-also: [[ch-04]], [[ch-06]], [[ch-08]], [[ch-09]]
-->

## Overview

"Attention Is All You Need" (Vaswani et al., 2017 — [[attention-is-all-you-need|paper]]) is the single most consequential architecture paper in modern deep learning. It replaced recurrence and convolution with pure attention, achieving state-of-the-art translation (28.4 BLEU on WMT EN-DE, 41.8 on EN-FR) while training in 3.5 days on 8 P100 GPUs — a fraction of the cost of prior models. Every production LLM descends from this design.

But the paper's lasting contribution is not just the architecture. It established a set of compositional building blocks — multi-head attention, position-wise feed-forward networks, residual connections, layer normalization — that remain nearly unchanged in 2026 models. Understanding this paper means understanding the skeleton that GPT, LLaMA, Gemini, and Claude are all built on. Equally important is understanding what the original paper got *wrong*, because those mistakes define the trajectory of every subsequent architecture paper you will study.

This chapter goes deep on the mechanical details, the design decisions and their rationales, and the places where subsequent work diverged from the 2017 blueprint.

---

## 1. Why "Attention Is All You Need"

The title is a claim about what you can *remove*. Prior sequence models (LSTMs, GRUs, ConvS2S) processed tokens sequentially or with limited receptive fields. The key bottleneck was **sequential computation**: an RNN must process position $t$ before position $t+1$, making training time $O(T)$ in sequence length. Convolutions improved parallelism but required stacking layers to achieve long-range dependencies (receptive field grows linearly or logarithmically with depth).

Self-attention connects every position to every other position in a single layer — $O(1)$ sequential operations for any dependency distance. The cost is $O(T^2)$ memory and compute per layer, but for the sequence lengths of 2017 (typically $T < 512$), this was acceptable. The parallelism advantage was decisive: the base Transformer trained in ~12 hours vs. days or weeks for comparable RNN models.

**The architectural bet:** Replace the $O(T)$ sequential bottleneck with $O(T^2)$ all-pairs attention, and bet that GPU parallelism makes $T^2$ cheaper in wall-clock time than $T$ sequential steps. This bet paid off spectacularly at 2017 sequence lengths. The $O(T^2)$ cost only became a problem later, motivating the efficient attention work you'll study in [[ch-08]].

---

## 2. The Encoder-Decoder Architecture

[Interactive: Full Transformer Architecture](figures/transformer-architecture.html)

The original Transformer is an encoder-decoder model for sequence-to-sequence tasks (machine translation). This is **not** the architecture used by modern LLMs — GPT and its descendants use decoder-only ([[ch-04]]) — but understanding the full encoder-decoder design clarifies *why* each component exists.

<div style="background:#1a1a2e; border-radius:12px; padding:28px; margin:20px 0; font-family:sans-serif; color:#e0e0e0;">
<div style="text-align:center; font-weight:bold; font-size:16px; margin-bottom:20px; color:#e94560;">The Transformer — Encoder-Decoder Architecture</div>

<div style="display:flex; gap:40px; justify-content:center; flex-wrap:wrap;">

<!-- Encoder -->
<div style="background:#16213e; border-radius:10px; padding:20px; min-width:240px; border:2px solid #0f3460;">
<div style="text-align:center; font-weight:bold; color:#e94560; margin-bottom:16px; font-size:15px;">ENCODER (×6)</div>

<div style="background:#0f3460; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Add & Norm</div>
</div>

<div style="background:#1a1a3e; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center; border:1px dashed #e94560;">
<div style="color:#eee; font-size:13px;">Feed-Forward Network</div>
<div style="color:#888; font-size:11px;">d_model=512 → d_ff=2048 → d_model=512</div>
</div>

<div style="background:#0f3460; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Add & Norm</div>
</div>

<div style="background:#1a1a3e; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center; border:1px dashed #e94560;">
<div style="color:#eee; font-size:13px;">Multi-Head Self-Attention</div>
<div style="color:#888; font-size:11px;">h=8, d_k=d_v=64</div>
</div>

<div style="background:#0a2040; border-radius:8px; padding:10px; text-align:center;">
<div style="color:#aaa; font-size:12px;">Input Embedding + Positional Encoding</div>
</div>
</div>

<!-- Decoder -->
<div style="background:#16213e; border-radius:10px; padding:20px; min-width:240px; border:2px solid #0f3460;">
<div style="text-align:center; font-weight:bold; color:#e94560; margin-bottom:16px; font-size:15px;">DECODER (×6)</div>

<div style="background:#0f3460; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Add & Norm</div>
</div>

<div style="background:#1a1a3e; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center; border:1px dashed #e94560;">
<div style="color:#eee; font-size:13px;">Feed-Forward Network</div>
<div style="color:#888; font-size:11px;">d_model=512 → d_ff=2048 → d_model=512</div>
</div>

<div style="background:#0f3460; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Add & Norm</div>
</div>

<div style="background:#1a1a3e; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center; border:1px dashed #4ecdc4;">
<div style="color:#4ecdc4; font-size:13px;">Multi-Head Cross-Attention</div>
<div style="color:#888; font-size:11px;">Q from decoder, K/V from encoder</div>
</div>

<div style="background:#0f3460; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Add & Norm</div>
</div>

<div style="background:#1a1a3e; border-radius:8px; padding:12px; margin-bottom:8px; text-align:center; border:1px dashed #e94560;">
<div style="color:#eee; font-size:13px;">Masked Multi-Head Self-Attention</div>
<div style="color:#888; font-size:11px;">causal mask: attend only to past</div>
</div>

<div style="background:#0a2040; border-radius:8px; padding:10px; text-align:center;">
<div style="color:#aaa; font-size:12px;">Output Embedding + Positional Encoding</div>
</div>
</div>
</div>

<div style="text-align:center; margin-top:16px; color:#888; font-size:12px;">
Encoder: bidirectional self-attention. Decoder: causal self-attention + cross-attention to encoder output.
<br>Each sub-layer is wrapped in a residual connection: output = LayerNorm(x + SubLayer(x)).
</div>
</div>

### Three Types of Attention

The Transformer uses **the same attention mechanism** in three different configurations:

1. **Encoder self-attention:** Every source position attends to every other source position. Bidirectional — no masking. This builds a contextualized representation of the input sequence.

2. **Decoder masked self-attention:** Every target position attends only to earlier target positions. A causal mask sets future positions to $-\infty$ before softmax, preventing information leakage. This is the autoregressive constraint from [[ch-01]].

3. **Encoder-decoder cross-attention:** Queries come from the decoder; keys and values come from the encoder's final output. This is how the decoder "reads" the source sequence. The encoder output is computed once, then reused at every decoder layer and every decoding step — an important efficiency property.

The unification of these three roles under a single mechanism (scaled dot-product attention with different masking and input routing) is one of the paper's most elegant contributions.

---

## 3. Scaled Dot-Product Attention: The Core Computation

The attention function computes a weighted sum of values, where weights are determined by query-key compatibility:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q \in \mathbb{R}^{T_q \times d_k}$, $K \in \mathbb{R}^{T_k \times d_k}$, $V \in \mathbb{R}^{T_k \times d_v}$.

### Why Scale by $\sqrt{d_k}$?

This is not cosmetic. Without scaling, dot products between query and key vectors grow in magnitude proportional to $d_k$. For $d_k = 64$, the expected magnitude of a dot product between two random unit-variance vectors is $\sqrt{64} = 8$. When dot products are large, softmax saturates — most of its output mass concentrates on one or two positions, gradients vanish for the non-attended positions, and learning stalls.

The paper is explicit: "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."

**The $1/\sqrt{d_k}$ factor normalizes dot products to unit variance**, keeping softmax in its responsive regime throughout training. With $d_k = 64$ and the scaling factor $1/8$, attention scores remain in a range where softmax gradients are large enough for effective learning.

### Multi-Head Attention

Rather than computing one attention function with $d_{\text{model}} = 512$ dimensions, the paper projects Q, K, V into $h = 8$ separate $d_k = 64$-dimensional subspaces, computes attention independently in each, concatenates, and projects back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

where $W_i^Q \in \mathbb{R}^{512 \times 64}$, $W_i^K \in \mathbb{R}^{512 \times 64}$, $W_i^V \in \mathbb{R}^{512 \times 64}$, and $W^O \in \mathbb{R}^{512 \times 512}$.

**Why multiple heads?** A single attention head computes one set of attention weights — one "view" of which positions are relevant. Multiple heads allow the model to attend to different aspects simultaneously: head 1 might track syntactic dependencies while head 3 tracks coreference. The paper's Table 2 ablation shows that reducing from $h=8$ to $h=1$ drops BLEU by 0.9, and even $h=4$ underperforms $h=8$.

**Computational cost is identical to single-head attention.** With $h = 8$ heads of dimension $d_k = 64$, the total dimension is $8 \times 64 = 512 = d_{\text{model}}$. The same number of parameters and FLOPs, just organized differently. This is a free lunch in expressiveness at no cost in compute.

---

## 4. The Residual Stream View

[Deep Dive: The Residual Stream — A Deep Dive](excerpts/residual-stream-deep-dive.md)

The Transformer's skip connections are not just a training trick — they define the fundamental information flow architecture.

### From ResNet to Transformers

He et al. (2015) ([[resnet|paper]]) showed that residual connections solve the degradation problem: plain deep networks get *worse* with more layers because gradients vanish or explode. The fix: instead of learning $H(x)$, each layer learns the residual $F(x) = H(x) - x$, and the skip connection adds back the identity:

$$y = x + F(x)$$

If $F(x) = 0$ (the layer learns nothing), the identity passes through unchanged. This means adding layers can never make the network *worse* than a shallower version — a crucial property for depth scaling. The Transformer applies this to every sub-layer: both the attention block and the FFN block are wrapped in residual connections.

### The "Residual Stream" Interpretation

A powerful way to understand Transformers is the **residual stream** model (Elhage et al., 2021). Think of the 512-dimensional vector at each token position as a "stream" flowing through all layers. Each attention head and FFN sub-layer *reads from* this stream and *writes an additive update back to it*:

$$x_\ell = x_{\ell-1} + \text{Attn}_\ell(x_{\ell-1}) + \text{FFN}_\ell(x_{\ell-1} + \text{Attn}_\ell(x_{\ell-1}))$$

<div style="background:#1a1a2e; border-radius:12px; padding:28px; margin:20px 0; font-family:sans-serif; color:#e0e0e0;">
<div style="text-align:center; font-weight:bold; font-size:16px; margin-bottom:24px; color:#e94560;">The Residual Stream View</div>

<div style="display:flex; align-items:center; justify-content:center; gap:0px;">

<!-- Input -->
<div style="text-align:center;">
<div style="background:#0a2040; padding:10px 16px; border-radius:8px; font-size:12px; color:#aaa;">Input<br>Embed</div>
</div>

<!-- Stream line -->
<div style="width:40px; height:4px; background:linear-gradient(to right, #0f3460, #e94560);"></div>

<!-- Layer 1 Attn -->
<div style="text-align:center;">
<div style="background:#1a1a3e; padding:8px 12px; border-radius:8px; font-size:11px; color:#e94560; border:1px solid #e94560; margin-bottom:4px;">Attn₁</div>
<div style="color:#666; font-size:18px;">+</div>
</div>

<div style="width:30px; height:4px; background:#e94560;"></div>

<!-- Layer 1 FFN -->
<div style="text-align:center;">
<div style="background:#1a1a3e; padding:8px 12px; border-radius:8px; font-size:11px; color:#4ecdc4; border:1px solid #4ecdc4; margin-bottom:4px;">FFN₁</div>
<div style="color:#666; font-size:18px;">+</div>
</div>

<div style="width:30px; height:4px; background:#e94560;"></div>

<!-- Layer 2 Attn -->
<div style="text-align:center;">
<div style="background:#1a1a3e; padding:8px 12px; border-radius:8px; font-size:11px; color:#e94560; border:1px solid #e94560; margin-bottom:4px;">Attn₂</div>
<div style="color:#666; font-size:18px;">+</div>
</div>

<div style="width:30px; height:4px; background:#e94560;"></div>

<!-- Layer 2 FFN -->
<div style="text-align:center;">
<div style="background:#1a1a3e; padding:8px 12px; border-radius:8px; font-size:11px; color:#4ecdc4; border:1px solid #4ecdc4; margin-bottom:4px;">FFN₂</div>
<div style="color:#666; font-size:18px;">+</div>
</div>

<div style="width:30px; height:4px; background:#e94560;"></div>

<!-- Dots -->
<div style="color:#888; font-size:18px; padding:0 8px;">···</div>

<div style="width:30px; height:4px; background:#e94560;"></div>

<!-- Layer 6 Attn -->
<div style="text-align:center;">
<div style="background:#1a1a3e; padding:8px 12px; border-radius:8px; font-size:11px; color:#e94560; border:1px solid #e94560; margin-bottom:4px;">Attn₆</div>
<div style="color:#666; font-size:18px;">+</div>
</div>

<div style="width:30px; height:4px; background:#e94560;"></div>

<!-- Layer 6 FFN -->
<div style="text-align:center;">
<div style="background:#1a1a3e; padding:8px 12px; border-radius:8px; font-size:11px; color:#4ecdc4; border:1px solid #4ecdc4; margin-bottom:4px;">FFN₆</div>
<div style="color:#666; font-size:18px;">+</div>
</div>

<div style="width:40px; height:4px; background:linear-gradient(to right, #e94560, #0f3460);"></div>

<!-- Output -->
<div style="text-align:center;">
<div style="background:#0a2040; padding:10px 16px; border-radius:8px; font-size:12px; color:#aaa;">Output<br>Head</div>
</div>
</div>

<div style="text-align:center; margin-top:20px;">
<div style="color:#e94560; font-size:13px; font-weight:bold;">━━━━━━━━━━━━━━━ The Residual Stream ━━━━━━━━━━━━━━━</div>
<div style="color:#888; font-size:12px; margin-top:8px;">
Each sub-layer reads from the stream and writes an additive update.<br>
<span style="color:#e94560;">Attention</span> = inter-position communication. <span style="color:#4ecdc4;">FFN</span> = per-position computation.<br>
The stream carries information from embedding to output; layers are read/write operations on it.
</div>
</div>
</div>

**Why this matters for interpretability and architecture research:**

1. **Composition through addition.** Because updates are additive, information written by layer 1's attention can be read by layer 4's FFN without being "overwritten" — it's still in the stream. This is what makes features like induction heads possible ([[ch-06]]).

2. **Gradient flow is direct.** The gradient from the output flows straight back through the skip connections to every layer. Layer 5 gets gradients that are nearly as strong as layer 6. This is why Transformers can be deep without vanishing gradients — the identity path provides a gradient superhighway.

3. **Layers are parallel readers/writers, not a pipeline.** Unlike a traditional pipeline where stage $n$ fully processes input before stage $n+1$, Transformer layers incrementally refine the residual stream. Early layers might write syntactic information; later layers read that and write semantic features. The stream accumulates everything.

---

## 5. Positional Encoding: The Sinusoidal Solution

[Deep Dive: Sinusoidal Encoding Frequency Analysis](excerpts/sinusoidal-encoding-frequency-analysis.md) | [Interactive: Positional Encoding Visualizer](figures/positional-encoding-visualizer.html) | [Interactive: Sinusoidal Positional Encoding](figures/sinusoidal-encoding.html)

Attention is permutation-equivariant: $\text{Attention}(\pi(X)) = \pi(\text{Attention}(X))$ for any permutation $\pi$. Without positional encoding, the model treats "the cat sat on the mat" identically to "mat the on sat cat the." Position information must be explicitly injected.

### The Sinusoidal Encoding

The paper adds a fixed (non-learned) vector to each token embedding:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the token position and $i$ is the dimension index. Each dimension oscillates at a different frequency — dimension 0 oscillates rapidly (period $2\pi \approx 6.28$ positions), while dimension 510 oscillates extremely slowly (period $2\pi \times 10000 \approx 62{,}832$ positions).

<div style="background:#1a1a2e; border-radius:12px; padding:28px; margin:20px 0; font-family:sans-serif; color:#e0e0e0;">
<div style="text-align:center; font-weight:bold; font-size:16px; margin-bottom:20px; color:#e94560;">Sinusoidal Positional Encoding — Frequency Spectrum</div>

<div style="display:flex; flex-direction:column; gap:12px; max-width:600px; margin:0 auto;">

<div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="min-width:100px; text-align:right; font-size:12px; color:#888;">dim 0–1<br><span style="color:#e94560;">λ = 2π</span></div>
<div style="flex:1; height:24px; background:repeating-linear-gradient(90deg, #e94560 0px, #e94560 3px, #1a1a2e 3px, #1a1a2e 6px); border-radius:4px; opacity:0.9;"></div>
<div style="min-width:80px; font-size:11px; color:#666;">high freq</div>
</div>
</div>

<div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="min-width:100px; text-align:right; font-size:12px; color:#888;">dim 128–129<br><span style="color:#e94560;">λ = 56</span></div>
<div style="flex:1; height:24px; background:repeating-linear-gradient(90deg, #e94560 0px, #e94560 10px, #1a1a2e 10px, #1a1a2e 20px); border-radius:4px; opacity:0.7;"></div>
<div style="min-width:80px; font-size:11px; color:#666;">mid freq</div>
</div>
</div>

<div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="min-width:100px; text-align:right; font-size:12px; color:#888;">dim 256–257<br><span style="color:#e94560;">λ = 628</span></div>
<div style="flex:1; height:24px; background:repeating-linear-gradient(90deg, #e94560 0px, #e94560 30px, #1a1a2e 30px, #1a1a2e 60px); border-radius:4px; opacity:0.5;"></div>
<div style="min-width:80px; font-size:11px; color:#666;">low freq</div>
</div>
</div>

<div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="min-width:100px; text-align:right; font-size:12px; color:#888;">dim 510–511<br><span style="color:#e94560;">λ = 62,832</span></div>
<div style="flex:1; height:24px; background:linear-gradient(90deg, #e94560, #1a1a2e); border-radius:4px; opacity:0.3;"></div>
<div style="min-width:80px; font-size:11px; color:#666;">near-DC</div>
</div>
</div>

</div>

<div style="text-align:center; margin-top:16px; color:#888; font-size:12px;">
Each pair of dimensions (sin, cos) encodes position at a unique frequency.<br>
High-frequency dims distinguish nearby positions; low-frequency dims encode global position.<br>
The full spectrum creates a unique "fingerprint" for every position.
</div>
</div>

### Why Sinusoidal?

**Relative position through linear projection.** The paper's key insight: for any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$. This is because $\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$ — the encoding at position $pos + k$ is a rotation of the encoding at position $pos$ by a fixed angle. The model can therefore learn to attend to relative positions (e.g., "the token 3 positions back") through linear transformations in the attention weights.

**Extrapolation to unseen lengths.** Because the encoding is a deterministic function of position, the model can in principle handle sequences longer than those seen during training. The paper notes: "it would allow the model to extrapolate to sequence lengths longer than the ones encountered during training." In practice, this extrapolation is limited — a problem that RoPE ([[ch-09]]) and ALiBi later addressed more effectively.

**No additional parameters.** Unlike learned positional embeddings (which GPT-2 and BERT used), sinusoidal encodings add zero trainable parameters. The paper's Table 3 footnote shows that learned embeddings produced "nearly identical results" — so the choice was about simplicity, not performance.

### What Subsequent Work Changed

Sinusoidal positional encoding was the first thing to be replaced. The problems:

- **Additive encoding pollutes the residual stream.** Adding position to the token embedding means every downstream computation mixes content and position in ways the model must learn to disentangle.
- **Absolute positions are wasteful.** Most linguistic structure depends on *relative* position (how far apart two tokens are), not absolute position (which slot a token occupies).
- **Extrapolation fails in practice.** Despite the theoretical argument, models trained on sequences of length 512 do not generalize well to length 2048 with sinusoidal encodings.

RoPE (Su et al., 2021, [[ch-09]]) solved these by rotating query and key vectors rather than adding to embeddings — encoding relative position directly in the attention score computation, not in the residual stream.

---

## 6. Layer Normalization and Its Placement

[Deep Dive: Layer Normalization Placement — Post-LN vs. Pre-LN](excerpts/layer-norm-placement.md)

### Why Normalize?

Ba et al. (2016) ([[layer-norm|paper]]) introduced layer normalization to stabilize hidden state dynamics. Unlike batch normalization (which normalizes across the batch dimension), layer normalization normalizes across the feature dimension of a single example:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu$ and $\sigma$ are computed over the $d_{\text{model}} = 512$ features for each position independently. The learnable parameters $\gamma$ (gain) and $\beta$ (bias) allow the network to recover any scale and shift after normalization.

**Why layer norm and not batch norm?** Batch normalization computes statistics over the batch dimension, which means (a) it depends on batch size and (b) it's ill-defined for variable-length sequences where different samples contribute different positions. Layer normalization computes statistics per-sample, per-position — it works identically at training and test time, and naturally handles variable-length inputs.

### Post-Norm: The Original Placement

The 2017 paper places layer normalization **after** the residual addition:

$$x_{\ell+1} = \text{LayerNorm}(x_\ell + \text{SubLayer}(x_\ell))$$

This is called **Post-LN**. It normalizes the full residual stream, keeping activations well-scaled. But it has a critical flaw that Xiong et al. (2020) ([[pre-norm-vs-post-norm|paper]]) diagnosed.

### The Warmup Mystery — Solved by Pre-Norm

The original Transformer required a carefully tuned learning rate warmup: linearly increase the learning rate for 4000 steps, then decay proportionally to $1/\sqrt{\text{step}}$:

$$lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

Without warmup, training diverged. The paper treated this as a hyperparameter choice. Xiong et al. (2020) proved *why* it was necessary: in Post-LN Transformers, the expected gradient magnitude near the output layer is disproportionately large at initialization. Large learning rates on large gradients cause instability, so you must start with a small learning rate (warmup) until the parameters settle.

**Pre-LN** moves layer normalization inside the residual block, before each sub-layer:

$$x_{\ell+1} = x_\ell + \text{SubLayer}(\text{LayerNorm}(x_\ell))$$

Xiong et al. proved (via mean field theory) that Pre-LN produces well-behaved gradients at initialization — eliminating the need for warmup entirely. Pre-LN Transformers train stably from the start with a constant or simple decaying learning rate schedule.

**Almost every modern LLM uses Pre-LN or a variant** (GPT-2 onward, LLaMA's RMSNorm Pre-LN). This is one of the most impactful "quiet" corrections to the original Transformer — it doesn't change the architecture's capability, but it makes training dramatically easier.

---

## 7. The Feed-Forward Sub-Layer

Each Transformer layer has two sub-layers: attention and a position-wise feed-forward network (FFN). The FFN is often overlooked, but it contains **two-thirds of the model's parameters**.

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

where $W_1 \in \mathbb{R}^{512 \times 2048}$, $W_2 \in \mathbb{R}^{2048 \times 512}$. The inner dimension $d_{ff} = 2048 = 4 \times d_{\text{model}}$ is a design choice — the paper found this 4:1 ratio worked well.

### What the FFN Does

In the residual stream view, attention handles **inter-position communication** (moving information between token positions) while the FFN handles **per-position computation** (transforming the information at each position independently).

Mechanistically, research has shown that FFN layers act as **key-value memories** (Geva et al., 2021): each row of $W_1$ is a "key" that matches certain input patterns, and the corresponding column of $W_2$ is a "value" that gets written to the residual stream when the key fires. The ReLU activation provides sparsity — only a fraction of the 2048 intermediate neurons activate for any given input, selecting which "memories" to read.

**Parameter count comparison** for one encoder layer:
- Multi-head attention: $4 \times 512^2 = 1{,}048{,}576$ params (Q, K, V, O projections)
- Feed-forward: $2 \times 512 \times 2048 = 2{,}097{,}152$ params
- FFN has exactly **2x** the parameters of attention

This ratio matters for architecture research. Scaling the FFN (increasing $d_{ff}$) adds capacity for "knowledge storage" without affecting attention patterns. This is one motivation for the Mixture of Experts approach ([[ch-06]]), which makes the FFN conditional — using a sparse subset of a much larger FFN for each token.

---

## 8. Training Innovations

The paper introduced or adopted several training techniques that became standard practice. Understanding *why* each was needed reveals the fragility of training the original design.

### Learning Rate Schedule (Warmup + Inverse Square Root)

As discussed in Section 6, warmup was a bandage for Post-LN's gradient instability. The specific schedule peaks at step 4000 with $lr \approx 7 \times 10^{-4}$ (for $d_{\text{model}} = 512$), then decays. This schedule became the "Noam scheduler" in the community and is still used when training Post-LN models.

### Label Smoothing ($\epsilon_{ls} = 0.1$)

Instead of training against hard targets (probability 1 for the correct token, 0 for everything else), label smoothing distributes $\epsilon = 0.1$ of the probability mass uniformly across all vocabulary tokens:

$$P_{smooth}(k) = (1 - \epsilon)\,\mathbb{1}_{k=y} + \frac{\epsilon}{|V|}$$

**Why this matters:** Hard targets push the model to produce infinitely confident logits (pushing the correct token's logit to $+\infty$). This causes (a) loss explosion as logits grow, (b) poor calibration, and (c) reduced generalization. Label smoothing caps confidence at $1 - \epsilon = 0.9$, acting as implicit regularization.

The paper reports that label smoothing "hurt perplexity, as the model learns to be more unsure, but improved accuracy and BLEU score." This is a critical insight: perplexity rewards confidence, but actual translation quality rewards calibration. The metrics disagree, and you should trust the downstream metric.

### Dropout ($p = 0.1$)

Applied in three places: (1) after each sub-layer output, before the residual addition; (2) to attention weights after softmax; (3) to the sum of token embeddings and positional encodings. The paper uses $p = 0.1$ for the base model and $p = 0.3$ for the big model — more regularization for more parameters.

### The Optimizer: Adam with Non-Standard $\beta_2$

The paper uses Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$. The standard $\beta_2$ is 0.999; the lower value here makes the second moment estimate adapt faster, which helps in the early stages of training where the loss landscape changes rapidly. This choice is often overlooked but can matter significantly for training stability.

---

## 9. What the Paper Got Right — and What Changed

This section is essential for an architecture research perspective. The Transformer paper established a blueprint; subsequent work kept the skeleton and replaced individual bones.

### What Survived Unchanged

| Component | Status in 2026 |
|-----------|---------------|
| Scaled dot-product attention | Universal. The $QK^T/\sqrt{d_k}$ formula is in every Transformer variant. |
| Multi-head attention | Universal, though head counts have grown (LLaMA 2 70B: 64 heads). |
| Residual connections | Universal. The residual stream is the backbone of every LLM. |
| FFN sub-layer per block | Universal, though activation functions changed (GELU, SiLU/Swish, GeGLU). |
| The 4:1 FFN ratio ($d_{ff}/d_{\text{model}}$) | Still common, though some models use $8/3$ ratio with gated FFN. |

### What Changed

| Original (2017) | Modern Replacement | Why |
|---|---|---|
| Encoder-decoder | Decoder-only ([[ch-04]]) | Single architecture handles all tasks via prompting |
| Post-LN | Pre-LN / Pre-RMSNorm | Training stability without warmup |
| Sinusoidal positional encoding | RoPE ([[ch-09]]) / ALiBi | Relative position, better length extrapolation |
| ReLU in FFN | SiLU/Swish, GeGLU | Smoother gradients, gated linear units improve quality |
| Learned absolute position (BERT/GPT-2) | RoPE | Rotary position encodes relative position in attention scores |
| Dense FFN | Mixture of Experts ([[ch-06]]) | Conditional computation for parameter efficiency |
| 6 layers | 32–80+ layers | Scaling laws showed deeper = better given compute budget |
| $d_{\text{model}} = 512$ | 4096–8192 | Wider residual streams for more capacity |
| Adam $\beta_2 = 0.98$ | AdamW, varied schedules | Weight decay decoupled, cosine schedules common |

The most consequential change was the shift from encoder-decoder to decoder-only. The original Transformer needed cross-attention because translation requires reading a source sequence and generating a target. GPT showed that a decoder-only model with causal masking could handle translation, summarization, QA, and generation — all as conditional language modeling. The entire encoder and cross-attention mechanism was dropped, simplifying the architecture and enabling unified pretraining.

---

## Core Insights from the Literature

### Insight 1: The scaling factor $1/\sqrt{d_k}$ is not a normalization — it is a variance control mechanism
**Paper:** Vaswani et al. (2017) ([[attention-is-all-you-need|paper]])

The scaling prevents softmax saturation by keeping the variance of dot products at $O(1)$ regardless of $d_k$. Without it, increasing $d_k$ (which you must do for wider models) pushes attention toward hard one-hot distributions where gradients vanish. This is a specific instance of a general principle: anywhere you compute dot products of high-dimensional vectors followed by softmax, you must control the input scale. **Guideline:** When designing any attention variant, always verify that the attention logits have bounded variance at initialization. If you change $d_k$, the scaling factor must change accordingly.

### Insight 2: Post-LN's warmup requirement was a symptom, not a design choice
**Paper:** Xiong et al. (2020) ([[pre-norm-vs-post-norm|paper]]), "On Layer Normalization in the Transformer Architecture"

The original paper treated warmup as a hyperparameter to tune. Xiong et al. proved it was a necessary patch for a gradient pathology: Post-LN produces gradients near the output layer with expected magnitude $O(d \cdot L)$ at initialization, where $L$ is depth. Pre-LN keeps gradients at $O(1)$ regardless of depth. **Guideline:** When a training recipe requires an unusual stabilization trick (warmup, gradient clipping, reduced learning rate for certain layers), ask whether it is compensating for an architectural defect. The fix is often a structural change, not a better hyperparameter.

### Insight 3: The residual stream is the actual architecture; attention and FFN are peripheral processors
**Paper:** He et al. (2015) ([[resnet|paper]]), Elhage et al. (2021)

The skip connections are not auxiliary — they define the model's information backbone. Without skip connections, a 6-layer Transformer is a 6-step pipeline where each step must preserve all previously computed information in its output. With skip connections, each layer only needs to compute a useful *delta*. This is why Transformers scale to 100+ layers: each layer is free to be a small, specialized read/write operation rather than a complete information bottleneck. **Guideline:** When analyzing or designing Transformer variants, think in terms of what each layer reads from and writes to the residual stream, not in terms of layer-by-layer processing.

### Insight 4: The FFN is not an afterthought — it stores the model's factual knowledge
**Paper:** Geva et al. (2021), "Transformer Feed-Forward Layers Are Key-Value Memories"

The attention mechanism gets most of the research attention, but the FFN sub-layer contains 2/3 of the parameters and functions as an associative memory. Each intermediate neuron in the FFN acts as a detector for specific input patterns, and its output column writes a corresponding "value" to the residual stream. This explains why model editing techniques (ROME, MEMIT) target FFN weights, and why Mixture of Experts ([[ch-06]]) focuses on scaling the FFN: you are scaling the model's memory bank. **Guideline:** When considering where capacity should be added to a Transformer, recognize that attention capacity (more heads, larger $d_k$) improves *routing* of information, while FFN capacity improves *storage* of knowledge.

---

## Key Takeaways

1. **The Transformer trades sequential computation for parallel all-pairs attention.** $O(T^2)$ compute per layer, but $O(1)$ sequential operations — a decisive win at 2017 sequence lengths. The $T^2$ cost became the dominant research problem afterward ([[ch-08]]).

2. **Three attention configurations, one mechanism.** Encoder self-attention (bidirectional), decoder self-attention (causal), and cross-attention (decoder queries encoder) are the same scaled dot-product attention with different masks and input routing.

3. **The residual stream is the architecture's backbone.** Attention and FFN sub-layers are additive read/write operations on a persistent information stream. This enables depth scaling, compositional features, and interpretability.

4. **Sinusoidal positional encoding was a correct first attempt with known limitations.** It proved position injection was necessary but was replaced by RoPE's relative-position approach in virtually all modern LLMs.

5. **Post-LN placement was the original paper's most consequential mistake.** It caused the gradient pathology that required warmup, and fixing it (Pre-LN) was one of the first and most important corrections.

6. **The FFN sub-layer holds 2/3 of parameters and functions as knowledge storage.** Attention routes information; FFN stores and transforms it. This split informs every modern scaling and efficiency decision.

7. **The paper's lasting legacy is the component vocabulary.** Multi-head attention, residual connections, layer normalization, position-wise FFN — these building blocks, not the specific encoder-decoder configuration, are what define the Transformer family in 2026.

---

## References

- [[attention-is-all-you-need|Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762. — paper]]
- [[resnet|He, K., Zhang, X., Ren, S., & Sun, J. "Deep Residual Learning for Image Recognition." CVPR 2016. arXiv:1512.03385. — paper]]
- [[layer-norm|Ba, J.L., Kiros, J.R., & Hinton, G.E. "Layer Normalization." arXiv:1607.06450, 2016. — paper]]
- [[pre-norm-vs-post-norm|Xiong, R., Yang, Y., He, D., et al. "On Layer Normalization in the Transformer Architecture." ICML 2020. arXiv:2002.04745. — paper]]
- [[alammar-illustrated-transformer|Alammar, J. "The Illustrated Transformer." jalammar.github.io/illustrated-transformer/, 2018. — blog]]
- Elhage, N., Nanda, N., Olsson, C., et al. "A Mathematical Framework for Transformer Circuits." Anthropic, 2021.
- Geva, M., Schuster, R., Berant, J., & Levy, O. "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864, 2021.
