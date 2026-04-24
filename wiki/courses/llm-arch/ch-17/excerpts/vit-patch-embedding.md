# Excerpt: ViT Patch Embedding

<!-- source: [[vit|paper]] — Dosovitskiy et al., 2020 -->

## The Core Idea

The Vision Transformer's fundamental contribution is reducing image recognition to sequence modeling. An image of resolution $H \times W$ with $C$ channels is split into a grid of non-overlapping $P \times P$ patches. Each patch is flattened into a vector and linearly projected to the Transformer's hidden dimension:

$$\mathbf{z}_0 = [x_{\text{class}};\; x_p^1 E;\; x_p^2 E;\; \ldots;\; x_p^N E] + E_{\text{pos}}$$

where:
- $x_p^i \in \mathbb{R}^{P^2 \cdot C}$ is the flattened $i$-th patch
- $E \in \mathbb{R}^{(P^2 C) \times D}$ is the patch embedding projection
- $x_{\text{class}}$ is a learnable `[CLS]` token prepended to the sequence
- $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are learnable position embeddings
- $N = HW/P^2$ is the number of patches

For the standard ViT-Base/16 configuration: $P=16$, input $224 \times 224 \times 3$, yielding $N = 196$ tokens, each projected from $16^2 \times 3 = 768$ raw dimensions.

## Why Patches, Not Pixels?

Processing individual pixels as tokens would produce $224^2 = 50{,}176$ tokens — an $O(N^2)$ attention matrix of 2.5 billion entries. Patches reduce the sequence length by $P^2$ (256x for $P=16$), making self-attention computationally feasible.

This is the same compression principle as BPE tokenization in NLP: merge low-level units (characters/pixels) into higher-level tokens (subwords/patches) to keep sequence lengths manageable for attention.

## The Scale Dependence

ViT's key finding: **performance depends critically on pretraining data scale.**

| Model | Pretrained on | ImageNet top-1 |
|-------|--------------|----------------|
| ResNet-152 (BiT) | ImageNet-21K (14M) | 84.7% |
| ViT-L/16 | ImageNet-21K (14M) | 76.5% |
| ViT-L/16 | JFT-300M (300M) | 87.8% |

With only 14M images, ViT-L *underperforms* a comparable ResNet by 8 points. With 300M images, it surpasses the ResNet by 3 points. The locality inductive bias of convolutions gives CNNs a head start on small datasets that Transformers need more data to overcome.

This parallels the GPT scaling story in NLP: the Transformer's advantage is not in architecture-specific inductive biases but in its capacity to learn arbitrary patterns given sufficient data.

## Position Embeddings: 2D vs. 1D

ViT uses standard 1D learnable position embeddings — the same approach as GPT. The authors tested 2D position embeddings (encoding row and column separately) and found **no significant improvement.** This suggests that the model learns spatial structure from the data rather than requiring it to be encoded architecturally.

This result is significant for multimodal architectures: it means vision tokens can use the same position embedding scheme as text tokens, simplifying the unified backbone.

## Architectural Implications for Multimodal Models

ViT's legacy is not image classification — it is **modality-agnostic sequence processing.** Once images are patch tokens, a multimodal model is just a question of how to interleave two types of tokens in a single Transformer. This directly enables:

1. **Shared backbones:** Same Transformer processes text and image tokens (Gemma 3, Llama 4)
2. **Contrastive pretraining:** ViT produces features that can be aligned with text via CLIP/SigLIP
3. **Extension to other modalities:** Audio spectrograms, video frames, and point clouds can all be "patched" using the same principle

The title says it: an image is worth 16x16 words. Literally — it becomes a sequence of tokens, processed by the same architecture that processes words.
