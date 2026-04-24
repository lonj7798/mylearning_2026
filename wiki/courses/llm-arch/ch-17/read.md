# Chapter 17: Multimodal Architectures

<!-- scope: ViT patch embedding, early vs late fusion, cross-modal attention, SigLIP contrastive pretraining, Pan-and-Scan, native vs bolted-on multimodality
     deps: [[ch-04]]
     see-also: [[ch-20]], [[ch-18]]
-->

## Overview

Every frontier LLM now accepts images — and some accept audio, video, and structured data alongside text. But "multimodal" is not a single architecture. It is a design space with sharp tradeoffs between training cost, cross-modal reasoning depth, and serving efficiency. The decisions made at the architecture level — how modalities are tokenized, where they are fused, and whether the model is trained jointly or stitched together post-hoc — determine the ceiling of what the model can learn about the relationship between what it reads and what it sees.

This chapter traces that design space from its origin. The Vision Transformer (ViT, [[vit|paper]]) established that images can be tokenized as patches and processed with standard self-attention, eliminating the need for separate vision architectures. From there, the question became: *how do you combine a vision encoder with a language model?* The answer splits into two fundamentally different philosophies — **late fusion** (bolt a vision encoder onto a frozen LLM via adapters) and **early fusion** (interleave vision and text tokens in a unified backbone from the start). Gemma 3 ([[gemma-3|report]]) and Llama 4 ([[llama-4|report]]) represent the current state of the art on opposite sides of this divide, and comparing them reveals what each approach gains and sacrifices.

Along the way, we cover the contrastive pretraining objective (SigLIP/CLIP) that teaches vision encoders to produce language-aligned representations, the cross-modal attention mechanisms that let language tokens attend to visual features, and the practical engineering of handling variable-resolution images (Pan-and-Scan). The progression is not merely historical — it reflects a deepening understanding of where the real architectural constraints lie in making a model that genuinely *understands* images rather than merely captioning them.

**Prerequisite:** [[ch-04]] (decoder-only architecture, causal masking). Understanding how the language backbone works is essential before studying how vision tokens enter it.

---

## 1. Vision Transformer: Patches as Tokens

Before ViT, applying Transformers to vision meant hybrid architectures — convolutional feature extractors feeding into attention layers, or attention modules inserted within CNN pipelines. Dosovitskiy et al. ([[vit|paper]], 2020) asked a simpler question: what if you just tokenize the image directly and use a standard Transformer?

### Patch Embedding

An image of resolution $H \times W$ with $C$ channels is divided into a grid of non-overlapping patches, each of size $P \times P$. Each patch is flattened into a vector of dimension $P^2 \cdot C$ and linearly projected to the model's hidden dimension $d$:

$$x_p^{(i)} = \text{flatten}(\text{patch}_i) \cdot E + e_{\text{pos}}^{(i)}, \qquad E \in \mathbb{R}^{(P^2 C) \times d}$$

For a standard configuration — 224x224 image, 16x16 patches, 3 color channels — this produces $(224/16)^2 = 196$ tokens, each projected from $16^2 \times 3 = 768$ raw dimensions into the model dimension. A special `[CLS]` token is prepended, and learnable position embeddings are added.

The insight is almost embarrassingly simple: **an image is a sequence of patch tokens, just as a sentence is a sequence of word tokens.** No convolutions, no pooling hierarchies, no domain-specific inductive biases. Standard multi-head self-attention operates over the patch sequence identically to how it operates over text.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">ViT: Image to Token Sequence</div>
<div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap; justify-content:center;">
<div style="text-align:center;">
<div style="width:112px; height:112px; background:linear-gradient(135deg, #e94560, #0f3460); border-radius:4px; display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:repeat(3,1fr); gap:2px; padding:3px; box-sizing:border-box;">
<div style="background:rgba(255,255,255,0.15); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.25); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.1); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.2); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.3); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.15); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.1); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.2); border-radius:2px;"></div>
<div style="background:rgba(255,255,255,0.25); border-radius:2px;"></div>
</div>
<div style="color:#888; font-size:11px; margin-top:6px;">224x224 image<br>split into 14x14 grid<br>(P=16)</div>
</div>
<div style="color:#e94560; font-size:24px;">&#8594;</div>
<div style="text-align:center;">
<div style="display:flex; gap:3px; flex-wrap:wrap; max-width:200px; justify-content:center;">
<div style="width:24px; height:24px; background:#4ecdc4; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#1a1a2e; font-size:8px; font-weight:bold;">CLS</div>
<div style="width:24px; height:24px; background:#0f3460; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#e0e0e0; font-size:8px;">P1</div>
<div style="width:24px; height:24px; background:#0f3460; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#e0e0e0; font-size:8px;">P2</div>
<div style="width:24px; height:24px; background:#0f3460; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#e0e0e0; font-size:8px;">P3</div>
<div style="width:24px; height:24px; background:#0f3460; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#e0e0e0; font-size:8px;">...</div>
<div style="width:24px; height:24px; background:#0f3460; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#e0e0e0; font-size:7px;">P196</div>
</div>
<div style="color:#888; font-size:11px; margin-top:6px;">197 tokens<br>(1 CLS + 196 patches)<br>each dim = d</div>
</div>
<div style="color:#e94560; font-size:24px;">&#8594;</div>
<div style="text-align:center;">
<div style="background:#16213e; border:2px solid #e94560; border-radius:8px; padding:16px 12px; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">Standard</div>
<div style="color:#e94560; font-weight:bold; font-size:12px;">Transformer</div>
<div style="color:#888; font-size:10px; margin-top:4px;">L layers of<br>self-attention<br>+ FFN</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
Each patch is flattened (P^2 x C = 768 dims), linearly projected to d, and given a learnable position embedding.
</div>
</div>

### Why It Works — and When It Doesn't

ViT revealed a critical insight about Transformers versus CNNs: **convolutions encode a locality inductive bias** (nearby pixels are more related), while Transformers have no such bias — they learn spatial relationships entirely from data. This means:

- **With limited data, CNNs win.** On ImageNet alone (1.3M images), ViT-Base underperforms a comparable ResNet. The locality bias of convolutions gives CNNs a head start that small datasets cannot overcome.
- **With sufficient data, ViT wins decisively.** Pre-trained on JFT-300M (300M images), ViT-Large achieves 87.76% top-1 on ImageNet, surpassing the best CNNs while requiring substantially less compute to train.

This parallels the broader Transformer story in NLP: the architecture is a general-purpose sequence processor that trades inductive biases for data efficiency. Given enough data, learning the structure is better than assuming it.

### Why This Matters for Multimodal Models

ViT's real contribution is not to image classification — it is to **architectural unification**. Once images are patch tokens processed by standard Transformers, a multimodal model becomes a question of how to combine two token sequences (text and image patches) rather than how to bridge two fundamentally different architectures (CNN + Transformer). Every multimodal LLM discussed in this chapter builds on this foundation.

---

## 2. Contrastive Pretraining: Aligning Vision and Language

A vision encoder produces visual features, but those features live in their own representation space. A language model's embedding space is different. Before you can feed vision features into an LLM, you need a bridge — a way to ensure that the vector representing "a photo of a dog" in vision space is close to the vector representing the word "dog" in language space.

### CLIP and the Contrastive Objective

CLIP (Contrastive Language-Image Pretraining, Radford et al., 2021) trains a vision encoder and a text encoder jointly on image-text pairs. The objective is simple: given a batch of $N$ (image, text) pairs, maximize the cosine similarity of the $N$ correct pairs while minimizing it for the $N^2 - N$ incorrect pairs:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j) / \tau)}$$

where $v_i$ and $t_i$ are the normalized embeddings of image $i$ and text $i$, and $\tau$ is a learned temperature parameter.

This produces an image encoder whose representation space is *aligned* with language — images of dogs cluster near the text "dog," images of cats cluster near "cat," and so on. The alignment is semantic, not pixel-level.

### SigLIP: Sigmoid Loss for Better Scaling

SigLIP (Zhai et al., 2023) replaces CLIP's softmax-based contrastive loss with a sigmoid loss applied independently to each (image, text) pair:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma\!\left(y_{ij} \cdot z_{ij}\right), \qquad z_{ij} = (v_i^\top t_j) \cdot e^\tau + b$$

where $y_{ij} = +1$ for matched pairs and $y_{ij} = -1$ for unmatched pairs. The crucial difference: **SigLIP does not require a global softmax normalization across the batch.** In CLIP's softmax loss, computing the denominator requires all-to-all communication across devices when training with large batch sizes distributed across many GPUs/TPUs. SigLIP's sigmoid loss is decomposable — each pair's loss can be computed independently, enabling much larger effective batch sizes without communication bottlenecks.

Gemma 3 uses a **SigLIP 400M** vision encoder at 896x896 resolution. This encoder produces dense visual features that are already semantically aligned with language, making the projection into the LLM's embedding space much easier than projecting from an unaligned vision encoder.

### Why Contrastive Pretraining Matters Architecturally

The contrastive objective does not produce a multimodal model itself — it produces an *aligned vision encoder* that can be composed with a language model. This decomposition is the foundation of the late-fusion approach: train the vision encoder separately (with contrastive pretraining), train the LLM separately (with next-token prediction), then connect them with a lightweight adapter. The quality of the contrastive alignment directly determines how much work the adapter must do.

---

## 3. Early Fusion vs. Late Fusion

The central architectural question in multimodal LLMs is *where* vision and text representations are combined. This decision has cascading consequences for training cost, cross-modal reasoning depth, and serving flexibility.

### Late Fusion: Adapter-Based Composition

Late fusion trains the vision encoder and language model independently, then connects them through a projection layer or adapter module. The language model's weights are typically frozen or only lightly fine-tuned.

**Llama 3's approach ([[llama-3|report]]):** Vision capabilities are added post-hoc through an adapter-based architecture. An image encoder processes the image, a projection layer maps visual features into the LLM's embedding dimension, and these projected features are inserted into the token sequence. The language backbone remains frozen during multimodal adaptation. This is the definition of "bolted-on" multimodality.

**The adapter** is typically a small MLP (2-3 layers) or a cross-attention module:

$$h_{\text{visual}} = \text{Adapter}(\text{VisionEncoder}(\text{image})) \in \mathbb{R}^{N_v \times d_\text{model}}$$

where $N_v$ is the number of visual tokens and $d_\text{model}$ is the LLM's hidden dimension. These visual tokens are concatenated with text tokens and fed into the frozen LLM.

**Advantages:**
- Train vision and language independently; compose at the end
- The LLM's language capabilities are preserved without catastrophic forgetting
- Can swap vision encoders without retraining the LLM
- Much cheaper — only the adapter weights are trained

**Disadvantages:**
- Cross-modal interaction is limited to whatever the frozen LLM can extract from pre-projected visual tokens
- The LLM never learns to jointly represent vision and language — it treats visual tokens as a foreign language it was not trained on
- Performance ceiling on tasks requiring deep visual reasoning (spatial relationships, counting, OCR)

### Early Fusion: Joint Backbone

Early fusion interleaves vision and text tokens within a unified Transformer backbone from the start of pretraining. The model learns to process both modalities jointly.

**Llama 4's approach ([[llama-4|report]]):** Text and vision tokens are processed jointly in a unified backbone. A MetaCLIP-based vision encoder produces visual tokens that are projected and interleaved with text tokens *before* the first Transformer layer. Critically, the model is pretrained on multimodal data — it sees images and text together during the core pretraining phase, not just during a later adaptation stage.

**Advantages:**
- The model develops genuine cross-modal representations — it can learn that spatial prepositions ("above," "behind") relate to pixel-level spatial relationships
- Deeper integration enables harder multimodal reasoning
- No adapter bottleneck; the full model capacity is available for cross-modal computation

**Disadvantages:**
- Requires multimodal data during pretraining (expensive, harder to curate)
- Cannot easily swap the vision encoder post-hoc
- Risk of forgetting or degrading text-only capabilities if multimodal data is not balanced carefully
- Significantly more expensive to train from scratch

See [Early vs. Late Fusion Comparison](figures/early-vs-late-fusion.html) for an interactive diagram contrasting these architectures.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Architecture Comparison: Late Fusion vs. Early Fusion</div>
<div style="display:flex; gap:40px; flex-wrap:wrap; justify-content:center;">
<!-- Late Fusion -->
<div style="text-align:center; max-width:260px;">
<div style="color:#ffd93d; font-weight:bold; font-size:13px; margin-bottom:12px;">Late Fusion (Llama 3)</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
<div style="display:flex; gap:12px;">
<div style="background:#0f3460; padding:10px 14px; border-radius:6px; color:#4ecdc4; font-size:11px; font-weight:bold;">Vision Encoder<br><span style="color:#888; font-weight:normal;">(frozen)</span></div>
<div style="background:#0f3460; padding:10px 14px; border-radius:6px; color:#e94560; font-size:11px; font-weight:bold;">Text Tokens</div>
</div>
<div style="color:#ffd93d; font-size:14px;">&#8595;</div>
<div style="background:#ffd93d; padding:6px 20px; border-radius:6px; color:#1a1a2e; font-size:11px; font-weight:bold;">Adapter / Projection</div>
<div style="color:#ffd93d; font-size:14px;">&#8595;</div>
<div style="background:#16213e; border:2px solid #ffd93d; padding:12px 20px; border-radius:8px; color:#ffd93d; font-size:12px; font-weight:bold;">
LLM Backbone<br><span style="color:#888; font-size:10px; font-weight:normal;">(frozen or lightly tuned)</span>
</div>
</div>
<div style="color:#888; font-size:10px; margin-top:8px;">Vision and language trained<br>separately, composed later</div>
</div>
<!-- Early Fusion -->
<div style="text-align:center; max-width:260px;">
<div style="color:#4ecdc4; font-weight:bold; font-size:13px; margin-bottom:12px;">Early Fusion (Llama 4)</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
<div style="display:flex; gap:12px;">
<div style="background:#0f3460; padding:10px 14px; border-radius:6px; color:#4ecdc4; font-size:11px; font-weight:bold;">Vision Encoder<br><span style="color:#888; font-weight:normal;">(MetaCLIP)</span></div>
<div style="background:#0f3460; padding:10px 14px; border-radius:6px; color:#e94560; font-size:11px; font-weight:bold;">Text Tokens</div>
</div>
<div style="color:#4ecdc4; font-size:14px;">&#8595; interleave &#8595;</div>
<div style="background:#16213e; border:2px solid #4ecdc4; padding:12px 20px; border-radius:8px; color:#4ecdc4; font-size:12px; font-weight:bold;">
Unified Backbone<br><span style="color:#888; font-size:10px; font-weight:normal;">(jointly pretrained on multimodal data)</span>
</div>
</div>
<div style="color:#888; font-size:10px; margin-top:8px;">Vision and language tokens<br>processed together from layer 1</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
Late fusion preserves text-only quality cheaply. Early fusion enables deeper cross-modal reasoning at higher training cost.
</div>
</div>

### The Llama 3 to Llama 4 Transition

The Llama family provides a clean case study of this tradeoff. Llama 3 ([[llama-3|report]]) used "compositional multimodal integration" — vision, speech, and tool-use capabilities were added post-hoc through adapter-based approaches, keeping the 405B language backbone frozen or lightly fine-tuned. This was pragmatic: the 405B model was the most expensive dense model Meta had ever trained, and retraining it for multimodality was prohibitively costly.

Llama 4 ([[llama-4|report]]) committed to early fusion. The decision was enabled by two factors: (1) the switch to MoE architecture meant per-token compute was lower despite larger total parameter count, making multimodal pretraining feasible; (2) Meta had accumulated substantially more multimodal training data (30T+ tokens including image and video data). The result: Scout and Maverick outperform Llama 3 on multimodal benchmarks by a wide margin, particularly on tasks requiring spatial reasoning and detailed image understanding.

---

## 4. Cross-Modal Attention: How Language Sees Images

Once visual tokens are projected into the LLM's embedding space, the Transformer's self-attention mechanism handles cross-modal interaction. But the *how* matters — different architectures give language tokens different levels of access to visual information.

### Full Cross-Attention (Dense)

In the simplest approach, visual tokens are concatenated with text tokens and processed through standard self-attention. Every text token can attend to every visual token and vice versa:

$$\text{Attention}(Q_{\text{text+vis}}, K_{\text{text+vis}}, V_{\text{text+vis}})$$

This is maximally expressive but expensive. If the image produces $N_v$ visual tokens and the text has $N_t$ tokens, the attention matrix is $(N_v + N_t)^2$. For 256 visual tokens and 4K text tokens, the visual tokens add roughly 12% to the attention computation — manageable. For 1024 visual tokens or multiple images, the cost scales quadratically.

### Interleaved Cross-Attention

An alternative inserts dedicated cross-attention layers between standard self-attention layers. In a cross-attention layer, the queries come from text tokens and the keys/values come from visual tokens:

$$Q = H_{\text{text}} W_Q, \quad K = H_{\text{visual}} W_K, \quad V = H_{\text{visual}} W_V$$

$$\text{CrossAttn}(H_{\text{text}}, H_{\text{visual}}) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

This is more parameter-efficient: only the cross-attention layers need to know about visual tokens, and these layers can be inserted at a subset of positions (e.g., every 4th layer). The text-only self-attention layers operate exactly as they would without vision, preserving the LLM's text capabilities.

### Gemma 3's Approach: Sparse Visual Attention

Gemma 3 ([[gemma-3|report]]) takes an aggressive compression approach: the SigLIP 400M encoder processes images at 896x896 resolution, and the output is average-pooled down to just **256 visual tokens** per image. These tokens are projected into the LLM's dimension and concatenated with text tokens.

The 5:1 local-to-global attention ratio interacts with vision tokens in an important way: visual tokens are part of the global context, so they are only attended to in global attention layers (every 6th layer). In local attention layers with a 1024-token sliding window, if the visual tokens fall outside the window, they receive zero attention. This means the model relies on information propagated through the residual stream across layers to combine visual and textual information — similar to how sliding window attention handles long-range text dependencies.

This design compresses the visual pathway aggressively: 256 tokens per image, attention only in global layers. Yet Gemma 3-27B achieves 64.9% on MMMU (a challenging multimodal reasoning benchmark), suggesting that the compression is sufficient for most visual understanding tasks.

---

## 5. Pan-and-Scan: Handling Variable-Resolution Images

Real images come in arbitrary aspect ratios and resolutions. The naive approach — resize everything to a fixed square (e.g., 224x224 or 896x896) — distorts aspect ratios and loses fine detail. Gemma 3 introduces **Pan-and-Scan** as an alternative.

### The Problem with Resize-and-Pad

When you resize a 1920x1080 widescreen image to 896x896, you have two bad options:

1. **Stretch** to fill the square: distorts all spatial relationships
2. **Pad** to maintain aspect ratio: wastes ~47% of your token budget on black padding pixels that carry no information

Both approaches also fail on high-resolution images where small but important details (text in a document, a street sign, a label) are lost when downscaled to the encoder's native resolution.

### Pan-and-Scan Algorithm

Pan-and-Scan decomposes the image into multiple crops:

1. **Analyze** the image's aspect ratio and resolution
2. **Select** a set of overlapping crops that tile the image, each at the encoder's native resolution
3. **Process** each crop independently through the vision encoder
4. **Pool** the resulting feature maps and concatenate them

The key constraint: the total number of visual tokens is kept constant (256 per image in Gemma 3). Pan-and-Scan does not increase the token budget — it spends the same budget more intelligently by ensuring each crop captures detail at the encoder's native resolution rather than forcing the entire image into a single downscaled view.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Pan-and-Scan vs. Resize-and-Pad</div>
<div style="display:flex; gap:32px; flex-wrap:wrap; justify-content:center;">
<!-- Resize and Pad -->
<div style="text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:12px; margin-bottom:8px;">Resize-and-Pad</div>
<div style="width:140px; height:140px; background:#16213e; border:2px solid #e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; margin:0 auto;">
<div style="width:130px; height:73px; background:linear-gradient(135deg, #e94560 0%, #0f3460 100%); border-radius:2px; display:flex; align-items:center; justify-content:center;">
<div style="color:#fff; font-size:9px; opacity:0.7;">16:9 image squished</div>
</div>
</div>
<div style="color:#e94560; font-size:10px; margin-top:6px;">~47% wasted on padding<br>fine detail lost to downscaling</div>
</div>
<!-- Pan and Scan -->
<div style="text-align:center;">
<div style="color:#4ecdc4; font-weight:bold; font-size:12px; margin-bottom:8px;">Pan-and-Scan</div>
<div style="width:200px; height:80px; background:#16213e; border:2px solid #4ecdc4; border-radius:4px; display:grid; grid-template-columns:repeat(3,1fr); gap:3px; padding:3px; margin:0 auto;">
<div style="background:linear-gradient(135deg, #4ecdc4 0%, #0f3460 100%); border-radius:2px; display:flex; align-items:center; justify-content:center;"><div style="color:#fff; font-size:8px; font-weight:bold;">Crop 1</div></div>
<div style="background:linear-gradient(135deg, #0f3460 0%, #4ecdc4 50%, #0f3460 100%); border-radius:2px; display:flex; align-items:center; justify-content:center;"><div style="color:#fff; font-size:8px; font-weight:bold;">Crop 2</div></div>
<div style="background:linear-gradient(135deg, #0f3460 0%, #4ecdc4 100%); border-radius:2px; display:flex; align-items:center; justify-content:center;"><div style="color:#fff; font-size:8px; font-weight:bold;">Crop 3</div></div>
</div>
<div style="color:#4ecdc4; font-size:10px; margin-top:6px;">Each crop at native resolution<br>detail preserved, no wasted tokens</div>
</div>
</div>
</div>

**Why this matters for architecture:** Pan-and-Scan is an inference-time technique — it does not change the model architecture or weights. But it changes the *input representation* in a way that significantly impacts performance on detail-sensitive tasks (document understanding, chart reading, fine-grained visual question answering). It demonstrates that input preprocessing is an architectural decision, not just an engineering detail.

---

## 6. Native Multimodal Training vs. Bolted-On

The early-vs-late fusion distinction is about architecture. Native-vs-bolted-on is about **training procedure** — and the two are related but not identical.

### What "Native" Means

A natively multimodal model sees multimodal data during its core pretraining phase. The model's representations are shaped by both text and visual input from the start, so its internal features learn to represent cross-modal relationships as first-class concepts.

Llama 4's report is explicit: "early fusion multimodality integrates text and vision tokens into a unified backbone from the start, rather than adding adapters post-hoc. Enables joint pre-training with unlabeled multimodal data." The training data includes over 30 trillion tokens of text, image, and video data.

### What "Bolted-On" Means

A bolted-on multimodal model adds vision capabilities after language pretraining. The language model is already fully trained; a vision encoder and adapter are attached and fine-tuned on multimodal data while the language backbone is frozen or minimally updated.

Llama 3's multimodal integration follows this pattern: "vision (via image encoder), speech, and tool-use capabilities added post-hoc through adapter-based approaches, keeping the language backbone frozen or lightly fine-tuned."

### The Architecture Implications

| Dimension | Bolted-On (Llama 3 style) | Native (Llama 4 style) |
|-----------|--------------------------|----------------------|
| Training cost | Low (adapter only) | High (full pretraining) |
| Text quality preservation | Guaranteed (backbone frozen) | Must balance with multimodal data |
| Cross-modal reasoning | Limited by adapter capacity | Full backbone capacity |
| Vision encoder swappability | Easy (adapter is thin interface) | Difficult (backbone expects specific features) |
| Data requirements | Small multimodal dataset | Massive multimodal corpus |
| Architectural complexity | Modular, composable | Monolithic, tightly coupled |

### Gemma 3: A Pragmatic Middle Ground

Gemma 3 ([[gemma-3|report]]) occupies an interesting middle position. Its SigLIP vision encoder is fixed (417M parameters, same across all model sizes from 4B to 27B). The vision tokens are projected and concatenated into the LLM's input sequence. But unlike pure late fusion, the full LLM is trained with multimodal data — it is not a frozen language backbone with an adapter bolted on.

This hybrid approach gets most of the benefits of both sides: the vision encoder is a proven, well-aligned component (SigLIP trained with contrastive pretraining), and the LLM has been exposed to multimodal data during training so its attention layers know how to extract information from visual tokens. The tradeoff is that the vision encoder cannot be easily upgraded without retraining the LLM, since the LLM's layers have adapted to the specific representation space of this particular SigLIP model.

---

## 7. Putting It Together: The Multimodal Pipeline

A complete multimodal LLM pipeline involves several components working in sequence. See [Multimodal Pipeline Diagram](figures/multimodal-pipeline.html) for the full interactive visualization.

### Step-by-Step Processing

1. **Image preprocessing:** Resize, crop (Pan-and-Scan), or pad the input image to the vision encoder's expected resolution
2. **Patch embedding:** Split the preprocessed image into patches, flatten and project each patch to the encoder's hidden dimension (ViT-style)
3. **Vision encoding:** Process patch tokens through the vision encoder (SigLIP, MetaCLIP, etc.) — produces dense visual features
4. **Projection:** Map visual features from the encoder's dimension to the LLM's dimension via a learned linear projection or adapter MLP
5. **Token interleaving:** Insert projected visual tokens into the text token sequence at the appropriate position (typically before the text query)
6. **LLM processing:** Standard autoregressive Transformer processes the combined sequence, with visual and text tokens attending to each other through self-attention (or cross-attention in interleaved designs)
7. **Text generation:** The LLM generates text tokens autoregressively, conditioned on both the visual and textual context

### Token Budget Arithmetic

The number of visual tokens directly impacts inference cost. Consider a 27B model processing one image with a 4K text prompt:

| Configuration | Visual Tokens | Total Tokens | Visual Overhead |
|---------------|--------------|-------------|----------------|
| Gemma 3 (SigLIP, pooled) | 256 | 4,352 | +6.2% |
| ViT-L/14 (no pooling) | 577 | 4,673 | +14.1% |
| High-res multi-crop (4 crops) | 1,024 | 5,120 | +25.0% |
| Naive ViT-H at 1024x1024 | 4,096 | 8,192 | +100.0% |

Gemma 3's aggressive pooling (256 tokens) is a deliberate engineering choice: it keeps the visual overhead under 7% of a typical prompt, ensuring that adding vision barely impacts serving latency or KV cache size.

---

## Core Insights from the Literature

### Insight 1: Images are just another token sequence
**Paper:** Dosovitskiy et al., "An Image is Worth 16x16 Words" ([[vit|paper]])

ViT's contribution is not a new architecture — it is the *absence* of one. By showing that a standard Transformer processes image patches as effectively as domain-specific CNNs (given sufficient data), it unified the sequence-processing paradigm across modalities. Every multimodal LLM inherits this: images become tokens, audio becomes tokens, video becomes tokens — the Transformer backbone does not need to know what modality it is processing. **Guideline:** To extend a Transformer to a new modality, find a natural tokenization (patches for images, frames for video, spectrograms for audio) and rely on scale rather than inductive biases.

### Insight 2: Contrastive pretraining bridges the representation gap
**Relevant to:** SigLIP, CLIP, MetaCLIP — used in [[gemma-3|report]], [[llama-4|report]]

The reason projection layers between vision encoders and LLMs work at all is that contrastive pretraining has already aligned visual and textual representations into a shared semantic space. Without this alignment, a linear projection from vision-space to language-space would be meaningless — there is no reason a priori that a CNN's internal features should be linearly related to a language model's embeddings. CLIP/SigLIP solves this by training vision and text encoders to agree on what concepts look like, making the downstream projection a fine-grained alignment rather than a wholesale translation. **Guideline:** Always use a contrastively pretrained vision encoder (SigLIP, CLIP, MetaCLIP) when building a multimodal LLM. Training a vision encoder from scratch alongside the LLM wastes massive compute on solving a problem that contrastive pretraining already handles.

### Insight 3: Early fusion costs more but enables deeper reasoning
**Source:** Llama 4 Technical Report ([[llama-4|report]])

Llama 4's explicit shift from Llama 3's late fusion to early fusion, and the corresponding improvement in multimodal benchmarks, demonstrates that the architectural choice of *where* modalities are fused has direct consequences for what the model can learn. Late fusion limits cross-modal reasoning to what a frozen backbone can extract from pre-projected visual tokens — it is a ceiling on cross-modal capability. Early fusion removes this ceiling at the cost of requiring multimodal data during pretraining. **Guideline:** Choose late fusion when multimodal capability is secondary and text quality must be preserved at minimal cost. Choose early fusion when multimodal reasoning quality is the primary objective and you can afford the training investment.

### Insight 4: Token budget compression is an architectural decision, not just optimization
**Source:** Gemma 3 Technical Report ([[gemma-3|report]])

Gemma 3 compresses images to 256 tokens via SigLIP average pooling and restricts visual attention to global layers (every 6th layer). This is not just serving optimization — it is an architectural statement about how much visual information the model needs. The fact that 256 tokens achieves 64.9% MMMU suggests that for most visual understanding tasks, the bottleneck is not visual token count but the quality of visual-textual alignment and the model's reasoning capabilities. **Guideline:** Start with aggressive visual token compression (256-512 tokens). Only increase token count when benchmark analysis shows that fine-grained visual detail (OCR, small object detection, chart reading) is the limiting factor — and consider Pan-and-Scan before increasing the per-image token budget.

### Insight 5: Input preprocessing (Pan-and-Scan) can substitute for architectural complexity
**Source:** Gemma 3 Technical Report ([[gemma-3|report]])

Pan-and-Scan achieves better high-resolution understanding than increasing the vision encoder's native resolution or token count. This is a general principle: preprocessing the input to match the model's strengths is often cheaper and more effective than changing the model to handle arbitrary inputs. The same principle applies to text (chunking long documents), audio (splitting into windows), and other modalities. **Guideline:** Before increasing model capacity to handle a new input format, first consider whether smart preprocessing can present the input in a form the model already handles well.

---

## Key Takeaways

1. **ViT unified the architecture.** Once images are patch tokens, multimodal models are a question of token interleaving, not architectural bridging. The Transformer processes images and text identically — the modality is encoded in the tokenization, not the architecture.

2. **Contrastive pretraining (CLIP/SigLIP) is the bridge.** Vision encoders trained with contrastive objectives produce representations already aligned with language, making the projection into an LLM's space a fine-grained alignment rather than a translation between unrelated spaces.

3. **Early fusion vs. late fusion is the central architectural tradeoff.** Late fusion (Llama 3) preserves text quality cheaply but limits cross-modal reasoning. Early fusion (Llama 4) enables deeper multimodal understanding but requires multimodal data during pretraining and risks degrading text-only capabilities.

4. **Token budget compression determines serving cost.** Gemma 3's 256 visual tokens per image keep visual overhead under 7% of a typical prompt. This is an architectural decision with direct cost implications — more visual tokens means larger KV cache, more attention computation, and slower serving.

5. **Pan-and-Scan is preprocessing-as-architecture.** Smart input decomposition (multiple crops at native resolution) achieves better detail preservation than brute-force resolution increases, without changing the model's architecture or increasing the token budget.

6. **Native multimodal training produces stronger cross-modal reasoning than bolted-on approaches.** The gap between Llama 3 and Llama 4 on multimodal benchmarks demonstrates that exposure to multimodal data during pretraining — not just during fine-tuning — is necessary for the model to develop deep cross-modal representations.

7. **The modular (bolted-on) approach has legitimate advantages.** When text quality is paramount and multimodal capability is secondary, a frozen LLM with a pluggable vision adapter is the right choice. It is cheaper, more predictable, and allows independent upgrades to either component.

---

## References

- [[vit|Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020) (paper)]] — Vision Transformer, patch embedding
- [[gemma-3|Google DeepMind, "Gemma 3 Technical Report" (2025) (report)]] — SigLIP vision encoder, Pan-and-Scan, 5:1 local/global attention
- [[llama-4|Meta AI, "Llama 4: The Beginning of a New Era of Natively Multimodal AI" (2025) (report)]] — early fusion, MetaCLIP, MoE + multimodality
- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]] — late fusion, adapter-based multimodality
- Radford et al., "Learning Transferable Visual Models from Natural Language Supervision" (2021) — CLIP contrastive pretraining
- Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (2023) — SigLIP, scaling contrastive learning
