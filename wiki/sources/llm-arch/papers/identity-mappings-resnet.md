<!-- scope: improved residual connections via identity mappings — why pre-norm works
     deps: [[resnet]]
     see-also: [[pre-norm-vs-post-norm]], [[layer-norm]], [[rmsnorm]]
-->

# Identity Mappings in Deep Residual Networks
- **Core Insight:** Using pure identity shortcuts (no gating, scaling, or convolution on the skip path) and moving normalization+activation before the weight layers (pre-activation) enables training of 1000+ layer networks and produces cleaner gradient flow.
- **Guideline:** Keep the residual shortcut as a pure identity mapping; place normalization before (not after) each sub-layer -- this is the theoretical basis for pre-norm Transformers.
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year:** 2016
- **URL:** https://arxiv.org/abs/1603.05027
- **Relevant chapters:** Residual connections, pre-norm vs post-norm, normalization placement, gradient flow, deep network training

## Abstract
Deep residual networks have emerged as a family of extremely deep architectures showing compelling accuracy and nice convergence behaviors. In this paper, we analyze the propagation formulations behind the residual building blocks, which suggest that the forward and backward signals can be directly propagated from one block to any other block, when using identity mappings as the skip connections and after-addition activation. A series of ablation experiments support the importance of these identity mappings. This motivates us to propose a new residual unit, which makes training easier and improves generalization. We report improved results using a 1001-layer ResNet on CIFAR-10 (4.62% test error) and CIFAR-100, and a 200-layer ResNet on ImageNet.

## Key Contributions
- Proved mathematically that when the shortcut is a pure identity mapping, any layer's output can be expressed as the sum of the input plus a residual function -- and the gradient decomposes into a direct term (identity) plus a residual term, preventing vanishing gradients regardless of depth
- Demonstrated through ablation that any modification to the shortcut path (scaling, gating, 1x1 convolution) degrades performance, especially at extreme depth -- the shortcut must be pure identity
- Proposed the "pre-activation" residual unit: BN -> ReLU -> Conv -> BN -> ReLU -> Conv, where the input to each block has already been normalized, versus the original "post-activation" design
- Pre-activation design reduces overfitting and enables training of 1001-layer networks on CIFAR-10, achieving 4.62% test error
- This paper is the direct theoretical ancestor of pre-norm Transformers (which place LayerNorm before attention/FFN rather than after)

## Key Figures/Tables to Study
- **Figure 1** (Original vs. proposed residual unit): Side-by-side comparison of post-activation (original) vs pre-activation (proposed). The key visual: in pre-activation, BN+ReLU come before the weight layer, and the addition feeds directly into the next unit without intervening nonlinearity.
- **Figure 2** (Shortcut path experiments): Tests gated shortcuts, scaling shortcuts, 1x1 conv shortcuts, and dropout on shortcuts. All degrade performance relative to identity. This is the ablation that proves identity is necessary.
- **Figure 4** (Training curves, 1001 layers): Pre-activation allows 1001 layers to converge smoothly; post-activation shows optimization difficulties.
- **Table 2** (CIFAR results): 1001-layer pre-activation ResNet achieves 4.62% on CIFAR-10, improving over the 164-layer post-activation version.

## Architecture Details
- **Original (post-activation) residual unit:** x_{l+1} = f(x_l + F(x_l, W_l)) where f is ReLU applied after addition. The nonlinearity f after addition means the skip path is not pure identity.
- **Proposed (pre-activation) residual unit:** x_{l+1} = x_l + F(BN(ReLU(x_l)), W_l). The key change: BN and ReLU are moved inside the residual function, before the weight layers. The addition output goes directly to the next unit.
- **Gradient analysis:** With identity shortcuts, gradient of loss w.r.t. any layer x_l is: dL/dx_l = dL/dx_L * (1 + d/dx_l * sum of residuals). The "1" term means the gradient always has a direct path that doesn't vanish, regardless of the depth of the network.
- **Why pre-activation matters for Transformers:** The pre-activation ResNet insight directly motivates pre-norm Transformers (Pre-LN): place LayerNorm before attention and FFN sub-layers, keeping the residual stream as a clean sum. Post-norm Transformers (original Vaswani) place LayerNorm after addition, which corresponds to the original ResNet design that this paper shows is suboptimal.
- **Connection to modern LLMs:** Every modern LLM (LLaMA, DeepSeek, Gemma, Qwen, etc.) uses pre-norm placement, validating this paper's core finding. The residual stream in a modern Transformer is a direct information highway exactly as this paper prescribes.
- **Depth achieved:** 1001 layers (CIFAR), 200 layers (ImageNet) -- previously impractical even with the original ResNet skip connections
- **Publication venue:** ECCV 2016
