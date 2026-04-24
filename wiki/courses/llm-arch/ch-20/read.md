# Chapter 20: Case Study — Gemma 3

<!-- scope: Gemma 3 architecture deep dive — local/global interleaving, SoftCap→QK-norm, SigLIP + Pan-and-Scan, hybrid normalization, scaling 1B-27B
     deps: [[ch-07]], [[ch-16]], [[ch-17]]
     see-also: [[ch-18]], [[ch-24]]
-->

## Overview

Gemma 3 is Google DeepMind's 2025 open-weight model family spanning 1B to 27B parameters. Where LLaMA 3 ([[ch-18]]) chose conservative architecture and massive scale (405B dense), Gemma 3 takes the opposite bet: aggressive architectural optimization to extract maximum capability from a modest parameter budget. The 27B model is dense — no MoE routing, no sparse activation — yet it matches Gemini 1.5 Pro across benchmarks and outranks Llama 3.1 405B on Chatbot Arena (Elo 1338 vs 1269).

The architectural thesis is that most transformer layers do not need global attention. Gemma 3 interleaves 5 local sliding-window layers for every 1 global full-attention layer, compressing KV cache from ~60% to under 15% of model memory. This is not a novel idea in isolation — Mistral 7B ([[ch-07]]) introduced sliding window attention, and [[ch-16]] covered long-context strategies. What Gemma 3 contributes is the specific ratio (5:1), the dual RoPE frequency scheme that makes it work, and the empirical evidence that this aggressive interleaving costs almost nothing in quality.

This chapter examines the four pillars of Gemma 3's architecture: the local/global attention interleaving pattern, the evolution from SoftCap logit capping to QK-norm, the SigLIP + Pan-and-Scan multimodal pipeline, and the hybrid normalization strategy. We close by analyzing how these choices scale across the 1B-27B range and contrast the design philosophy with LLaMA's.

---

## 1. Architecture Inventory

Before dissecting individual innovations, here is the complete architectural specification across all four model sizes:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Gemma 3 Model Family</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Component</th>
<th style="text-align:right; padding:8px;">1B</th>
<th style="text-align:right; padding:8px;">4B</th>
<th style="text-align:right; padding:8px;">12B</th>
<th style="text-align:right; padding:8px;">27B</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Total Parameters</td>
<td style="text-align:right; padding:8px;">1B</td>
<td style="text-align:right; padding:8px;">4B</td>
<td style="text-align:right; padding:8px;">12B</td>
<td style="text-align:right; padding:8px;">27B</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Vision Encoder</td>
<td style="text-align:right; padding:8px; color:#888;">none</td>
<td style="text-align:right; padding:8px;">417M</td>
<td style="text-align:right; padding:8px;">417M</td>
<td style="text-align:right; padding:8px;">417M</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Context Length</td>
<td style="text-align:right; padding:8px;">32K</td>
<td style="text-align:right; padding:8px;">128K</td>
<td style="text-align:right; padding:8px;">128K</td>
<td style="text-align:right; padding:8px;">128K</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Local:Global Ratio</td>
<td style="text-align:right; padding:8px; color:#4ecdc4;">5:1</td>
<td style="text-align:right; padding:8px; color:#4ecdc4;">5:1</td>
<td style="text-align:right; padding:8px; color:#4ecdc4;">5:1</td>
<td style="text-align:right; padding:8px; color:#4ecdc4;">5:1</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Local Window Size</td>
<td style="text-align:right; padding:8px;">1,024</td>
<td style="text-align:right; padding:8px;">1,024</td>
<td style="text-align:right; padding:8px;">1,024</td>
<td style="text-align:right; padding:8px;">1,024</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Attention Type</td>
<td style="text-align:right; padding:8px;">GQA</td>
<td style="text-align:right; padding:8px;">GQA</td>
<td style="text-align:right; padding:8px;">GQA</td>
<td style="text-align:right; padding:8px;">GQA</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Vocabulary Size</td>
<td style="text-align:right; padding:8px;" colspan="4">262,144 (256K)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Normalization</td>
<td style="text-align:right; padding:8px;" colspan="4">Pre-norm + Post-norm RMSNorm</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Activation</td>
<td style="text-align:right; padding:8px;" colspan="4">GeGLU</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Positional Encoding</td>
<td style="text-align:right; padding:8px;" colspan="4">RoPE (dual frequency)</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
The 5:1 ratio, 1024-token window, and hybrid normalization are constant across all sizes — these are not tuned per model.
</div>
</div>

Three things stand out immediately. First, the 5:1 local-to-global ratio is invariant across sizes — Google settled on this ratio through ablation and applied it uniformly. Second, the local window is just 1,024 tokens, down from Gemma 2's 4,096 and Mistral 7B's 4,096. Third, the vision encoder is fixed at 417M parameters (SigLIP 400M variant) regardless of LLM backbone size. Each of these choices has deep architectural implications that we will unpack.

---

## 2. Local/Global Attention Interleaving

This is Gemma 3's defining architectural contribution. The pattern is simple: layers 0-4 use sliding window attention with a 1,024-token window, layer 5 uses full global attention, layers 6-10 use sliding window, layer 11 uses global, and so on. In a model with $L$ layers, exactly $\lfloor L/6 \rfloor$ layers have global attention.

### Why 5:1?

The intuition is that global attention is expensive but redundant at every layer. Consider what happens in a 32-layer model during a 128K-token forward pass:

- **All-global (standard):** Each of 32 layers attends to all 128K tokens. KV cache stores keys and values for every position at every layer.
- **5:1 interleaving:** Only ~5 layers attend to all 128K tokens. The other ~27 layers each cache only 1,024 tokens.

The KV cache reduction is dramatic. For a layer with GQA ($G$ KV-head groups, head dimension $d_k$), the cache per layer is:

$$\text{Cache}_\text{global} = 2 \times G \times d_k \times S \quad \text{(full sequence } S \text{)}$$
$$\text{Cache}_\text{local} = 2 \times G \times d_k \times W \quad \text{(window } W = 1024 \text{)}$$

With $S = 128{,}000$ and $W = 1{,}024$, a local layer caches $125\times$ fewer entries than a global layer. Over a stack with 5 local layers per global layer, the total KV cache drops to roughly:

$$\text{Total} \approx \frac{5 \times W + 1 \times S}{6 \times S} \approx \frac{5120 + 128000}{768000} \approx 17\%$$

of the all-global baseline. The Gemma 3 report states "less than 15% of model memory" — consistent with this calculation once you account for other memory consumers.

See [figures/local-global-interleaving.html](figures/local-global-interleaving.html) for an interactive visualization of this pattern.

### The Dual RoPE Frequency Scheme

A subtle but critical detail: local and global layers use **different RoPE base frequencies**.

- **Global layers:** $\theta_\text{base} = 1{,}000{,}000$ (1M)
- **Local layers:** $\theta_\text{base} = 10{,}000$ (10K)

Recall from [[ch-06]] that RoPE encodes position by rotating query and key vectors. The base frequency $\theta$ controls how quickly the rotation angle grows with position:

$$\theta_i = \theta_\text{base}^{-2i/d}$$

A higher base frequency produces slower rotation, allowing the model to distinguish positions farther apart without the rotation wrapping around. Global layers need 1M base because they attend across 128K tokens — they need fine-grained positional resolution across the full context. Local layers only attend to 1,024 tokens, so 10K base is sufficient and provides *higher angular resolution* for nearby positions.

This is not merely a performance optimization. Using 1M base for local layers would waste representational capacity: the rotation angles within a 1,024-token window would be nearly indistinguishable. Conversely, using 10K base for global layers would cause position aliasing at long distances. The dual-frequency design matches the positional encoding to each layer's actual receptive field.

### Information Flow Through the Stack

A natural concern: if 5 out of 6 layers can only see 1,024 tokens, how does information from distant positions propagate? The mechanism is the **residual stream** (see [[ch-09]]). Each global layer reads the full context and writes information into the residual stream. The subsequent 5 local layers can then access that information — not by directly attending to distant tokens, but by reading what the previous global layer deposited.

In a 30-layer model with 5:1 interleaving, global layers appear at positions 5, 11, 17, 23, 29. A token at position 100,000 in the sequence has its information written into the residual stream at layer 5 (the first global layer). By layer 6, every token in the model has access to a representation that includes information from position 100,000, even though layers 6-10 are local. The next global layer (11) can further refine and redistribute long-range information.

This is the same principle behind Mistral 7B's claim that the "theoretical receptive field" of SWA spans $W \times L$ tokens ([[ch-07]]). But Gemma 3 makes the tradeoff much more aggressive: Mistral used SWA at every layer with $W = 4096$; Gemma 3 uses $W = 1024$ at 5/6 of layers and reserves true global attention for the remainder.

### Comparison with Mistral's Approach

| Design Choice | Mistral 7B | Gemma 3 |
|--------------|-----------|---------|
| Sliding window size | 4,096 | 1,024 |
| Global layers | 0 (pure SWA) | Every 6th layer |
| RoPE frequency | Single (10K) | Dual (10K local, 1M global) |
| Context length | 8K (trained) | 128K |
| Information propagation | Residual stream only | Residual stream + periodic global refresh |

Mistral's pure-SWA approach relies entirely on the residual stream for long-range information flow. Gemma 3's hybrid ensures that every 6 layers, the model gets an explicit "refresh" of global context. This is why Gemma 3 can support 128K context while Mistral 7B was trained on only 8K — the periodic global layers provide reliable anchoring points for long-range dependencies that residual-stream-only propagation cannot guarantee.

---

## 3. From SoftCap to QK-Norm: Stabilizing Attention Logits

### The SoftCap Mechanism (Gemma 2)

Gemma 2 introduced **logit soft-capping**, a technique to prevent attention logits from growing unboundedly during training. The mechanism applies a $\tanh$ squash to the raw attention scores:

$$\text{SoftCap}(x, c) = c \cdot \tanh\!\left(\frac{x}{c}\right)$$

where $c$ is the cap value. This bounds the logits to the range $(-c, c)$ regardless of the actual score magnitudes. Gemma 2 applied soft-capping at two points:

1. **Attention logits:** After computing $QK^\top / \sqrt{d_k}$, before softmax. Cap value $c_\text{attn} = 50$.
2. **Final logits:** Before the output softmax over the vocabulary. Cap value $c_\text{final} = 30$.

See [figures/softcap-vs-qknorm.html](figures/softcap-vs-qknorm.html) for an interactive comparison of both approaches.

### Why SoftCap Was Needed

The problem SoftCap addresses is **attention logit growth** during training. As models train, the norms of query and key vectors can grow, causing $QK^\top$ scores to become very large. Large logits create two issues:

1. **Numerical instability:** Softmax of very large values approaches one-hot, causing gradient saturation.
2. **Entropy collapse:** When one attention score dominates, the attention pattern collapses to attending to a single position, discarding information from all others.

Traditional gradient clipping addresses exploding gradients but not the logits themselves. SoftCap directly bounds the logit values, preventing both overflow and entropy collapse regardless of gradient magnitude.

### The SoftCap Problem

SoftCap worked but introduced complications:

- **Incompatibility with Flash Attention:** The $\tanh$ nonlinearity inside the attention computation breaks Flash Attention's tiling algorithm ([[ch-07]]). Flash Attention assumes the attention computation is $\text{softmax}(QK^\top/\sqrt{d_k}) \cdot V$ — inserting a nonlinearity between the QK product and softmax requires custom kernel modifications.
- **Gradient distortion:** The $\tanh$ squash compresses gradients for large logits, which can slow learning for attention patterns that genuinely need strong focus on specific positions.
- **Additional hyperparameters:** The cap values ($c_\text{attn}$, $c_\text{final}$) require tuning and may need different values at different scales.

### QK-Norm: The Gemma 3 Solution

Gemma 3 replaces SoftCap with **QK-norm** — applying RMSNorm to query and key vectors before computing attention scores:

$$Q' = \text{RMSNorm}(Q), \quad K' = \text{RMSNorm}(K)$$
$$\text{Attention} = \text{softmax}\!\left(\frac{Q' K'^\top}{\sqrt{d_k}}\right) V$$

This addresses the same problem (logit growth) but through a different mechanism. Instead of bounding the *output* of $QK^\top$, QK-norm bounds the *inputs*. After RMSNorm, query and key vectors have unit RMS norm, so the dot product $Q'K'^\top$ is bounded by the Cauchy-Schwarz inequality:

$$|q_i' \cdot k_j'| \leq \|q_i'\| \cdot \|k_j'\| = 1$$

The raw attention scores are thus naturally bounded in $[-d_k, d_k]$ (after summing over the head dimension), preventing the unbounded growth that SoftCap was designed to contain.

### Why QK-Norm is Architecturally Superior

1. **Flash Attention compatible.** RMSNorm is applied *before* the attention computation, so the attention kernel itself remains standard $\text{softmax}(QK^\top / \sqrt{d_k}) V$. No custom kernels needed.
2. **No additional hyperparameters.** There are no cap values to tune — the normalization is parameter-free (beyond the learnable scale factor in RMSNorm, which is already present elsewhere in the model).
3. **Consistent with the normalization stack.** Gemma 3 already uses RMSNorm extensively (pre-norm + post-norm). QK-norm is the same primitive applied at a new location, keeping the architecture conceptually uniform.
4. **Independent validation.** OLMo 2 ([[ch-24]]) independently adopted QK-norm and reported improved training stability, providing cross-organization confirmation that this is the right approach.

The trade here was clean: SoftCap solved the right problem but in a way that fought the hardware (Flash Attention) and added complexity. QK-norm solves the same problem while working *with* the inference stack.

---

## 4. SigLIP + Pan-and-Scan: Multimodal Vision

Gemma 3 (4B and above) is natively multimodal, combining a text decoder with a SigLIP vision encoder. The vision pipeline has two notable components: the encoder itself and the Pan-and-Scan input strategy.

### SigLIP Vision Encoder

SigLIP (Sigmoid Loss for Language-Image Pre-training) is a contrastive vision-language model that replaces the standard softmax-based contrastive loss (as in CLIP) with a sigmoid-based binary classification loss. The architectural relevance for Gemma 3:

- **Model:** SigLIP 400M variant (ViT architecture), processing images at 896x896 resolution
- **Output:** The encoder produces a grid of patch embeddings, which are average-pooled down to **256 vision tokens** per image
- **Frozen across sizes:** The same 417M-parameter vision encoder serves the 4B, 12B, and 27B models

The 256-token budget is aggressive — LLaVA-style models typically use 576 tokens per image. The tokens are projected through a learned linear layer into the text decoder's embedding space and interleaved with text tokens.

### Pan-and-Scan: Adaptive Image Decomposition

Standard vision-language models resize all images to a fixed resolution (e.g., 896x896) before encoding. This distorts aspect ratios (a panoramic image squashed to a square) and loses resolution (a document image downsampled). Pan-and-Scan addresses both by decomposing images into multiple views at inference time:

1. **Analyze the image aspect ratio and content** — determine whether the image benefits from cropping
2. **Extract multiple sub-images** — select crops that preserve aspect ratio and focus on content-dense regions
3. **Encode each sub-image independently** — each crop produces its own 256-token representation
4. **Concatenate representations** — all crop tokens are fed to the text decoder along with the full-image tokens

Pan-and-Scan is an **inference-time** strategy — the model architecture does not change. Higher resolution would increase compute quadratically (ViT's patch attention is $O(n^2)$). Pan-and-Scan increases compute only linearly in the number of crops, and the crop count can be dynamically chosen per image.

### Architectural Integration

The multimodal design is deliberate: **keep the vision encoder fixed; push complexity into the text decoder.** The 417M SigLIP encoder is small relative to the text backbone (25.6B non-embedding at 27B) and identical across all model sizes. Larger text decoders get better vision performance not from better visual features, but from better *reasoning about* the same features. This contrasts with Flamingo or PaLI, which scale both components.

---

## 5. Hybrid Normalization: Pre-Norm + Post-Norm

Most modern transformers use **pre-norm** — applying layer normalization before the attention and FFN sub-layers ([[ch-09]]). Gemma 3 uses both pre-norm *and* post-norm RMSNorm, placing normalization on both sides of each sub-layer:

```
x_out = RMSNorm_post(Attention(RMSNorm_pre(x)) + x)
x_out = RMSNorm_post(FFN(RMSNorm_pre(x)) + x)
```

### Why Both?

- **Pre-norm** normalizes the input to each sub-layer, preventing *input* activations from growing unboundedly. This is why pre-norm became the default after GPT-2.
- **Post-norm** normalizes the *output* (after the residual connection), preventing the residual stream itself from drifting in scale across layers.

Using both is especially important for Gemma 3 because the interleaved attention pattern creates **heterogeneous layers** — global layers process 128K tokens while local layers process 1,024. Post-norm keeps the residual stream consistent regardless of which attention pattern was used. At 46 layers (the 27B model), many additive contributions would cause unbounded residual growth without post-norm.

### Comparison with Other Normalization Strategies

| Model | Pre-Norm | Post-Norm | QK-Norm | Result |
|-------|----------|-----------|---------|--------|
| LLaMA 3 | RMSNorm | No | No | Stable with GC |
| OLMo 2 | No | RMSNorm | RMSNorm | Improved stability |
| Gemma 3 | RMSNorm | RMSNorm | RMSNorm | Most conservative |
| GPT-2 (original) | LayerNorm | No | No | Required careful LR |

Gemma 3 is the most aggressive normalizer in the current landscape. Every possible normalization point is active: pre-norm, post-norm, and QK-norm. This reflects a design philosophy of over-constraining activations rather than relying on gradient clipping or learning rate tuning to prevent instability.

Raschka ([[raschka-llm-architecture-comparison|blog]]) categorizes this as the "hybrid" normalization strategy, noting that Gemma 3's approach ensures no component of the forward pass produces unbounded activations. The cost is additional computation (three RMSNorm applications per sub-layer instead of one), but RMSNorm is cheap relative to attention and FFN operations — roughly 0.1% of per-layer FLOPs.

---

## 6. Knowledge Distillation

A distinguishing feature of the Gemma 3 family is that **all models are trained with knowledge distillation** from larger teacher models. The distillation setup:

- Sample 256 logits per token, weighted by the teacher's probability distribution
- The loss combines standard next-token prediction with a KL divergence term against the teacher's distribution
- This applies at all sizes: the 1B, 4B, 12B, and 27B models all learn from a larger teacher

The architectural implication is that Gemma 3 models are not independently trained — their learned representations are shaped by the teacher. This is why the 4B model is competitive with Gemma 2's 27B model (a 7x size difference): the 4B model effectively inherits representational structure from a much larger model, compressed into its smaller parameter budget.

Distillation interacts with the 5:1 interleaving in an important way. The teacher presumably uses a different (possibly all-global) attention pattern. The student must learn to reproduce the teacher's output distribution using the 5:1 interleaved architecture. This means the distillation signal implicitly teaches the model *how to use the interleaved pattern effectively* — which information to cache locally and which to propagate through the residual stream for the next global layer.

---

## 7. Quantization-Aware Training

Gemma 3 includes QAT as a first-class architectural concern, not an afterthought:

- **Process:** Fine-tune each model for ~5,000 steps with simulated quantization noise
- **Supported formats:** Per-channel INT4, per-block INT4, switched FP8
- **Result:** The 27B model in INT4 requires only 14.1 GB (vs 54 GB in BF16) — a 3.8x compression

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Gemma 3 27B: Memory Footprint After Quantization</div>
<div style="display:flex; gap:16px; align-items:flex-end; justify-content:center; flex-wrap:wrap;">
<div style="text-align:center;">
<div style="background:#e94560; width:60px; height:216px; border-radius:8px 8px 0 0; display:flex; flex-direction:column; justify-content:center; align-items:center;">
<div style="color:#fff; font-weight:bold; font-size:18px;">54</div>
<div style="color:#fff; font-size:10px;">GB</div>
</div>
<div style="background:#16213e; padding:6px; border-radius:0 0 8px 8px; width:60px;">
<div style="color:#e94560; font-weight:bold; font-size:11px;">BF16</div>
</div>
</div>
<div style="text-align:center;">
<div style="background:#ffd93d; width:60px; height:108px; border-radius:8px 8px 0 0; display:flex; flex-direction:column; justify-content:center; align-items:center; margin-top:108px;">
<div style="color:#1a1a2e; font-weight:bold; font-size:18px;">27</div>
<div style="color:#1a1a2e; font-size:10px;">GB</div>
</div>
<div style="background:#16213e; padding:6px; border-radius:0 0 8px 8px; width:60px;">
<div style="color:#ffd93d; font-weight:bold; font-size:11px;">FP8</div>
</div>
</div>
<div style="text-align:center;">
<div style="background:#4ecdc4; width:60px; height:56px; border-radius:8px 8px 0 0; display:flex; flex-direction:column; justify-content:center; align-items:center; margin-top:160px;">
<div style="color:#1a1a2e; font-weight:bold; font-size:18px;">14.1</div>
<div style="color:#1a1a2e; font-size:10px;">GB</div>
</div>
<div style="background:#16213e; padding:6px; border-radius:0 0 8px 8px; width:60px;">
<div style="color:#4ecdc4; font-weight:bold; font-size:11px;">INT4</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
INT4 fits the full 27B model on a single 16GB GPU. Combined with 5:1 interleaving, the KV cache is also minimal.
</div>
</div>

The 14.1 GB INT4 footprint is particularly significant because it means the 27B model fits on a single consumer GPU (16 GB VRAM). Combined with the 5:1 interleaving that minimizes KV cache, this makes Gemma 3 27B deployable on hardware that couldn't run models of comparable capability before.

---

## 8. Scaling Analysis: 1B to 27B

### What Stays Constant

The following architectural choices are **invariant** across all four sizes:

- 5:1 local-to-global attention ratio
- 1,024-token local window
- Dual RoPE frequency (1M global, 10K local)
- Pre-norm + post-norm + QK-norm
- GeGLU activation
- 256K vocabulary
- SigLIP 417M vision encoder (4B+)

This uniformity is a design statement: these are not hyperparameters to tune per size but structural decisions validated through ablation at one scale and applied universally.

### What Scales

The parameters that change across sizes are the standard ones: layer count, hidden dimension, number of attention heads, and FFN intermediate dimension. Importantly, the GQA group count and the attention interleaving pattern do not change — suggesting Google found these to be size-independent optima.

### Training Compute

| Model | Training Tokens | TPU Chips |
|-------|----------------|-----------|
| 1B | 2T | 512 TPUv5e |
| 4B | 4T | 2,048 TPUv5e |
| 12B | 12T | 6,144 TPUv4 |
| 27B | 14T | 6,144 TPUv5p |

The token-to-parameter ratios are high: 2000:1 for the 1B model, 519:1 for the 27B model. These ratios exceed Chinchilla-optimal ([[ch-10]]) by a large margin, reflecting the post-Chinchilla consensus (pioneered by LLaMA) that inference-optimal training overshoots compute-optimal training — you train on more data than the loss curve suggests because inference cost dominates total lifetime cost.

---

## 9. Gemma 3 vs LLaMA 3: Two Philosophies

These two model families represent fundamentally different design philosophies for open-weight LLMs:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Design Philosophy Comparison</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Dimension</th>
<th style="text-align:left; padding:8px; color:#4ecdc4;">Gemma 3</th>
<th style="text-align:left; padding:8px; color:#e94560;">LLaMA 3</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Max size</td>
<td style="padding:8px;">27B (dense)</td>
<td style="padding:8px;">405B (dense)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Attention</td>
<td style="padding:8px;">5:1 local/global + GQA</td>
<td style="padding:8px;">All-global GQA</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Normalization</td>
<td style="padding:8px;">Pre + Post + QK-norm</td>
<td style="padding:8px;">Pre-norm only</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Training strategy</td>
<td style="padding:8px;">Distillation from teacher</td>
<td style="padding:8px;">Self-supervised only</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Vision</td>
<td style="padding:8px;">Native (SigLIP + Pan-and-Scan)</td>
<td style="padding:8px;">Separate model (added post-hoc)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Design bet</td>
<td style="padding:8px;">Precision per FLOP</td>
<td style="padding:8px;">Scale per FLOP</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Quantization</td>
<td style="padding:8px;">QAT built-in</td>
<td style="padding:8px;">PTQ (post-training)</td>
</tr>
</tbody>
</table>
</div>

**LLaMA's philosophy** is conservative architecture at massive scale: use well-understood components (all-global GQA, pre-norm RMSNorm, SwiGLU) and pour in parameters and data. This minimizes architectural risk and maximizes reproducibility. The 405B model is a known architecture scaled up, not a novel one.

**Gemma 3's philosophy** is architectural precision at modest scale: use every technique available (interleaved attention, dual RoPE, hybrid normalization, QK-norm, distillation, QAT) to extract maximum capability from 27B parameters. This increases architectural complexity but enables deployment on consumer hardware.

The results are striking. Gemma 3 27B with INT4 quantization (14.1 GB) achieves Chatbot Arena Elo of 1338, placing it ahead of the 405B LLaMA 3.1 model (Elo 1269) that requires over 200 GB in FP16. The capability-per-memory ratio favors Gemma 3 by roughly an order of magnitude.

This does not mean Gemma 3 is "better" than LLaMA 3. It means they optimize for different deployment scenarios. LLaMA 3 405B is designed for data-center deployment where memory is abundant and raw capability is the priority. Gemma 3 27B is designed for edge deployment, on-device inference, and cost-sensitive serving where memory is the binding constraint.

---

## Core Insights from the Literature

### Insight 1: Not every layer needs global attention
**Source:** Gemma 3 Technical Report ([[gemma-3|report]])

The 5:1 local/global ratio demonstrates that the information-carrying capacity of global attention is highly redundant across consecutive layers. Five local layers operating on residual-stream representations enriched by a single global layer capture most of the benefit. The report's ablation shows minimal perplexity degradation despite the substantial reduction. **Guideline:** When designing attention patterns, start by asking "which layers actually need global context?" rather than assuming all do. The answer is likely "fewer than you think" — and the KV cache savings compound multiplicatively with GQA.

### Insight 2: Stabilize logits at the source, not the output
**Source:** Gemma 3 Technical Report, OLMo 2 ([[ch-24]])

Gemma 2's SoftCap and Gemma 3's QK-norm both prevent attention logit explosion, but QK-norm does it upstream (normalizing Q and K inputs) rather than downstream (clamping QK output). The upstream approach is compatible with optimized attention kernels, adds no hyperparameters, and uses the same primitive (RMSNorm) already present elsewhere. **Guideline:** Prefer pre-computation normalization over post-computation clamping. If your attention logits are growing, normalize the queries and keys rather than bounding their product.

### Insight 3: Fixed vision encoder is sufficient across scales
**Source:** Gemma 3 Technical Report ([[gemma-3|report]])

Using the same 417M SigLIP encoder for models from 4B to 27B is a strong claim: visual feature extraction quality does not need to scale with LLM backbone size. The implication is that vision understanding is bottlenecked by the text decoder's ability to *reason about* visual features, not by the quality of the features themselves. **Guideline:** For vision-language models, consider keeping the vision encoder fixed and scaling only the text backbone. The marginal capability gain from a larger vision encoder may not justify the additional serving cost.

### Insight 4: Architectural complexity can substitute for parameter count
**Source:** Gemma 3 vs LLaMA 3 comparison, Raschka ([[raschka-llm-architecture-comparison|blog]])

Gemma 3 27B competes with LLaMA 3.1 405B — a 15x parameter disadvantage — by using more sophisticated architecture (interleaved attention, distillation, QAT, hybrid normalization) plus aggressive quantization. This suggests that the "simple architecture, more parameters" approach has diminishing returns, and that the next frontier of efficiency is architectural innovation at fixed parameter count. **Guideline:** Evaluate model capability on a capability-per-memory-GB basis, not just on raw benchmark scores. The serving cost over the model's lifetime typically dominates the training cost.

---

## Key Takeaways

1. **5:1 local/global interleaving reduces KV cache to ~15% of all-global baselines** while maintaining quality. The key enablers are dual RoPE frequencies (1M for global, 10K for local) and residual-stream information propagation between global layers.

2. **QK-norm supersedes SoftCap** for attention logit stabilization. It solves the same problem (unbounded logit growth) while remaining compatible with Flash Attention and adding no hyperparameters. Both OLMo 2 and Gemma 3 independently converged on this approach.

3. **Pan-and-Scan decomposes images into adaptive crops**, increasing effective visual resolution without changing the model architecture. Combined with a fixed 417M SigLIP encoder, this gives Gemma 3 competitive vision understanding at low parameter cost.

4. **Hybrid normalization (pre-norm + post-norm + QK-norm)** is the most aggressive normalization strategy in current models. It constrains activations at every stage, which is especially important for heterogeneous architectures mixing local and global attention.

5. **All architectural choices are scale-invariant.** The 5:1 ratio, window size, normalization strategy, and dual RoPE frequencies are identical from 1B to 27B, suggesting these are robust optima rather than size-specific tuning.

6. **Distillation enables smaller models to punch above their weight.** The 4B model matching Gemma 2's 27B demonstrates that representational structure from a larger teacher compresses effectively into smaller architectures, especially when the student architecture (5:1 interleaving) forces efficient information use.

7. **Gemma 3 represents the "precision over scale" philosophy** — the opposite of LLaMA's approach. Both are valid, optimizing for different deployment constraints. The field is bifurcating: data-center models that maximize raw capability vs edge models that maximize capability per memory byte.

---

## References

- [[gemma-3|Google DeepMind, "Gemma 3 Technical Report" (2025) (report)]] — primary source
- [[mistral-7b|Jiang et al., "Mistral 7B" (2023) (report)]] — sliding window attention, rolling buffer KV cache
- [[raschka-llm-architecture-comparison|Raschka, "The Big LLM Architecture Comparison" (2026) (blog)]] — cross-model normalization and attention taxonomy
- [[raschka-attention-variants|Raschka, "A Visual Guide to Attention Variants in Modern LLMs" (2026) (blog)]] — SWA + GQA interaction analysis
- [[ch-07|Chapter 7: Attention Variants]] — MQA, GQA, MLA, Flash Attention, SWA foundations
- [[ch-09|Chapter 9: Normalization and Residual Connections]] — pre-norm vs post-norm, RMSNorm, QK-norm
- [[ch-16|Chapter 16: Long Context]] — RoPE scaling, length generalization
- [[ch-17|Chapter 17: Multimodal Architectures]] — SigLIP, vision-language integration
- [[ch-18|Chapter 18: Case Study — LLaMA 3 and Llama 4]] — contrasting design philosophy
- [[ch-24|Chapter 24: Case Study — OLMo 2]] — independent QK-norm validation
