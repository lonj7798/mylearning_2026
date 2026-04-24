# Excerpt: Gemma 3 Vision Architecture

<!-- source: [[gemma-3|report]] — Google DeepMind, 2025 -->

## Vision Encoder: SigLIP 400M

Gemma 3 uses a fixed SigLIP 400M vision encoder across all model sizes (4B, 12B, 27B):

| Component | Specification |
|-----------|---------------|
| Encoder | SigLIP 400M |
| Input resolution | 896 x 896 |
| Encoder parameters | 417M |
| Output tokens (before pooling) | $(896/14)^2 = 4{,}096$ |
| Output tokens (after pooling) | 256 |
| Pooling method | Average pooling |

The SigLIP encoder is contrastively pretrained with a sigmoid loss (not softmax), enabling larger batch sizes and better scaling. Its 417M parameters are the same regardless of the language model size — the vision component is deliberately lightweight relative to the LLM backbone.

### Why a Fixed Encoder Across Sizes?

Using the same 417M encoder for 4B, 12B, and 27B models is a pragmatic tradeoff:

- **Pro:** Simplifies the architecture, keeps total parameter count manageable, allows sharing pretrained encoder weights
- **Con:** The vision encoder may become a bottleneck for larger models — a 417M encoder feeding a 27B LLM means the visual representation is compressed through a much smaller model than the language representation
- **Empirical evidence:** 64.9% on MMMU with the 27B model suggests the bottleneck is not yet binding for most visual reasoning tasks

## Aggressive Token Compression

The 4,096 encoder output tokens are average-pooled down to **256 tokens** before projection into the LLM's dimension. This 16x compression is far more aggressive than most multimodal models.

**Why this works:** The SigLIP encoder's contrastive pretraining ensures that each output token already encodes high-level semantic information, not raw pixel features. Average pooling over semantically meaningful features preserves most task-relevant information. The lost information is primarily fine-grained spatial detail — pixel-level positions, exact character shapes for OCR, precise object boundaries.

**When it fails:** Tasks requiring fine-grained visual detail (small text recognition, counting small objects, reading charts with dense data) may hit the ceiling imposed by 256-token compression. Pan-and-Scan partially mitigates this by processing multiple native-resolution crops.

## Pan-and-Scan

Pan-and-Scan is an inference-time technique for handling variable aspect ratios and high-resolution inputs:

1. Analyze the input image's aspect ratio
2. Determine optimal crop locations based on content
3. Extract multiple crops, each at the encoder's native 896x896 resolution
4. Process each crop independently through SigLIP
5. Pool and combine the resulting features into the same 256-token budget

The key property: **total visual token count stays constant at 256.** Pan-and-Scan does not increase serving cost — it spends the same token budget more intelligently. Each crop captures detail at native resolution rather than forcing the entire image through a single downscaled view.

### Pan-and-Scan vs. Resize-and-Pad

For a 1920x1080 (16:9) image:

- **Resize-and-Pad:** Downscale to 504x896 within 896x896 frame. 43.8% of pixels are padding. Fine detail lost to downscaling.
- **Pan-and-Scan:** Extract 2-3 overlapping crops at 896x896 from the original resolution. Each crop processes a region at full detail. No wasted tokens on padding.

## Interaction with 5:1 Local/Global Attention

Gemma 3's attention architecture has direct implications for vision:

- **Local attention layers** (5 out of 6): 1024-token sliding window with RoPE frequency 10K. If visual tokens fall outside the current window, they receive zero attention.
- **Global attention layers** (1 out of 6): Full context with RoPE frequency 1M. Visual tokens are visible to all text tokens.

This means visual information is **only directly accessible in every 6th layer.** For the remaining 5 layers, cross-modal interaction depends on information already propagated into the residual stream by previous global layers. This is aggressive sparsity — but it works, likely because:

1. A single global attention pass propagates visual information into the text representations' residual stream
2. Subsequent local layers can refine this information through text-to-text attention
3. The next global layer reinforces and updates the visual grounding

The 5:1 ratio was chosen to minimize KV cache overhead: only global layers cache across the full context. Local layers cache only 1,024 positions. This reduces KV cache memory from ~60% to under 15% of model memory — a critical optimization for 128K context support.

## Architectural Position: Pragmatic Middle Ground

Gemma 3 is neither pure late fusion nor pure early fusion:

- **Unlike late fusion (Llama 3):** The LLM is trained with multimodal data, not frozen during multimodal adaptation. The attention layers learn to extract information from visual tokens.
- **Unlike early fusion (Llama 4):** The vision encoder is a separate, fixed component. Visual tokens are projected and concatenated, not deeply interleaved from layer 1.

This middle-ground position gets most benefits of both approaches while keeping the vision pipeline simple and the serving cost low. The 27B model is competitive with much larger models on multimodal benchmarks while remaining servable on consumer hardware with quantization.
