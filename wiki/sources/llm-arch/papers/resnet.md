# Deep Residual Learning for Image Recognition
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year:** 2015
- **URL:** https://arxiv.org/abs/1512.03385
- **Core Insight:** Skip connections enable training very deep networks; foundation for the Transformer's residual stream.
- **Guideline:** When stacking many layers, use residual (skip) connections so each layer only needs to learn the delta from its input, preventing gradient degradation and enabling much deeper architectures.
- **Relevant chapters:** Residual connections, Transformer architecture, Training deep networks, Gradient flow

## Abstract
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

## Key Contributions
- Introduced the residual learning framework: instead of learning H(x), layers learn F(x) = H(x) - x, with the identity mapping provided by a skip connection
- Demonstrated that residual networks can be trained effectively at depths of 152+ layers, far beyond what was previously possible
- Showed that deeper residual networks consistently improve accuracy, resolving the degradation problem where adding layers to plain networks actually hurt performance
- Won first place across multiple tracks at ILSVRC and COCO 2015 competitions, establishing a new standard for deep learning architectures
- Provided both empirical evidence and theoretical motivation for why identity shortcuts enable gradient flow through very deep networks

## Why This Paper Matters
The residual connection is one of the most important architectural innovations in deep learning history. Every Transformer layer uses residual connections -- the "residual stream" interpretation of Transformers (where information flows through skip connections and each layer reads from / writes to this stream) comes directly from this paper. Without skip connections, training the 96-layer GPT models or 100+ layer networks used in modern LLMs would be impossible due to vanishing gradients.
