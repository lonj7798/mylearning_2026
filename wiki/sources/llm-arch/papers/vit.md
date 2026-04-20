# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2010.11929
- **Core Insight:** Transformers work for images with minimal modification; patches as tokens is surprisingly effective.
- **Guideline:** To apply Transformers to non-text modalities, find a natural way to tokenize the input (e.g., image patches, audio frames). The architecture transfers with minimal modification when combined with sufficient pretraining data.
- **Relevant chapters:** Vision Transformers, Multimodal models, Tokenization, Transfer learning

## Abstract
While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

## Key Contributions
- Demonstrated that a pure Transformer, with no convolutions at all, can achieve state-of-the-art image classification when pretrained at scale
- Introduced the simple but powerful idea of splitting images into fixed-size patches and treating each patch as a token (analogous to words in NLP)
- Showed that ViT requires less compute to train than comparable CNNs when pretrained on large datasets (JFT-300M), challenging the assumption that vision needs inductive biases like convolutions
- Revealed a critical finding: ViT underperforms CNNs on smaller datasets but excels with sufficient pretraining data, highlighting the importance of scale for Transformers
- Unified the architecture across vision and language, opening the door to multimodal models that share the same Transformer backbone

## Why This Paper Matters
ViT proved that the Transformer is not just an NLP architecture -- it is a general-purpose sequence processor. By showing that images can be tokenized as patches and processed with standard self-attention, it eliminated the need for domain-specific architectures in vision. This insight directly enabled multimodal models (GPT-4V, Claude's vision, Gemini) that process text and images through the same Transformer backbone. The "patches as tokens" idea has since been extended to video, audio, and other modalities.
