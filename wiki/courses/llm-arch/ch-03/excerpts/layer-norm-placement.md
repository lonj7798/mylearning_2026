<!-- scope: layer normalization placement analysis — Post-LN vs Pre-LN, parent: [[ch-03]] -->

# Layer Normalization Placement: Post-LN vs. Pre-LN

This excerpt provides a detailed analysis of why the original Transformer's Post-LN placement required warmup and how Pre-LN fixed it. We work through the gradient analysis that explains the difference and cover modern variants including RMSNorm.

---

## 1. Layer Normalization Mechanics

LayerNorm normalizes the features across the $d_{\text{model}}$ dimension for each position independently:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ (mean over features)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$ (variance over features)
- $\gamma, \beta \in \mathbb{R}^d$ are learnable gain and bias
- $\epsilon \approx 10^{-5}$ prevents division by zero

**Why per-position?** Unlike batch normalization, LayerNorm computes statistics for a single position in a single example. This makes it:
- Independent of batch size (works with batch size 1)
- Well-defined for variable-length sequences
- Identical at training and inference time (no running statistics to maintain)

---

## 2. Post-LN: The Original Placement

The 2017 Transformer normalizes **after** the residual addition:

$$x_{\ell+1} = \text{LN}(x_\ell + \text{SubLayer}(x_\ell))$$

Expanding through $L$ layers (ignoring FFN for clarity):

$$x_L = \text{LN}(\text{LN}(\ldots \text{LN}(x_0 + \text{Attn}_1(x_0)) \ldots + \text{Attn}_{L-1}) + \text{Attn}_L)$$

Each LayerNorm rescales its input to unit variance. This means the output of every layer has approximately the same scale, regardless of depth. But the gradient behavior is problematic.

---

## 3. Xiong et al.'s Gradient Analysis

Xiong et al. (2020) proved that Post-LN creates a gradient magnitude imbalance at initialization.

**Setup:** Consider the gradient of the loss $\mathcal{L}$ with respect to parameters at layer $\ell$. Using the chain rule through the residual + LayerNorm:

$$\frac{\partial \mathcal{L}}{\partial \theta_\ell} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{m=\ell+1}^{L} \frac{\partial x_m}{\partial x_{m-1}} \cdot \frac{\partial x_\ell}{\partial \theta_\ell}$$

The key is the product $\prod_{m} \frac{\partial x_m}{\partial x_{m-1}}$. For Post-LN, each factor involves the Jacobian of LayerNorm applied to a sum. At initialization (before training), the sub-layer outputs are small (near zero due to random initialization), so $x_\ell + \text{SubLayer}(x_\ell) \approx x_\ell$, and the LayerNorm Jacobian is approximately an identity-like rescaling.

**The problem emerges during early training.** As the sub-layer outputs grow, the LayerNorm applied to $x_\ell + \text{SubLayer}(x_\ell)$ creates a gradient path that amplifies gradients near the output layer and shrinks them near the input. Xiong et al. showed:

$$\mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial \theta_\ell}\right\|\right] \propto O\left(\frac{1}{\sqrt{\ell}}\right) \quad \text{(Post-LN)}$$

Gradients at early layers are smaller than at late layers. At initialization, the ratio can be as large as $O(L)$ between the first and last layers. This makes the global learning rate a poor fit: a rate that is good for early layers is too large for late layers, and vice versa.

**Warmup is the patch:** Starting with a small learning rate (warmup) prevents the large gradients at late layers from destabilizing training. As training progresses and the model settles, the gradient imbalance diminishes, and the learning rate can increase.

---

## 4. Pre-LN: The Fix

Pre-LN moves normalization **before** each sub-layer, inside the residual block:

$$x_{\ell+1} = x_\ell + \text{SubLayer}(\text{LN}(x_\ell))$$

Expanding the gradient through the skip connection:

$$\frac{\partial x_{\ell+1}}{\partial x_\ell} = \mathbf{I} + \frac{\partial \text{SubLayer}(\text{LN}(x_\ell))}{\partial x_\ell}$$

The identity $\mathbf{I}$ from the skip connection ensures that gradients have a direct path from output to input. The LayerNorm is now inside the sub-layer path, not on the main gradient highway.

Xiong et al. proved:

$$\mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial \theta_\ell}\right\|\right] \propto O(1) \quad \text{(Pre-LN)}$$

Gradient magnitudes are approximately constant across layers. No warmup needed. The model trains stably from step 1 with a constant or simple decaying learning rate.

---

## 5. Empirical Comparison

| Property | Post-LN | Pre-LN |
|---|---|---|
| Warmup required | Yes (4000 steps in original paper) | No |
| Gradient magnitude across layers | Imbalanced ($O(1/\sqrt{\ell})$) | Balanced ($O(1)$) |
| Training stability | Fragile without careful hyperparameters | Robust |
| Final model quality | Slightly higher ceiling (with tuning) | Slightly lower ceiling |
| Adoption (2026) | Rarely used alone | Universal (with RMSNorm) |

The "slightly higher ceiling" of Post-LN is debated. Some evidence suggests that when Post-LN is carefully tuned (with warmup, gradient clipping, etc.), it can produce marginally better models. But the practical advantage of Pre-LN's robustness is overwhelming -- most researchers prefer a model that trains reliably over one that might train 0.1% better with perfect hyperparameters.

---

## 6. RMSNorm: The Modern Variant

Zhang & Sennrich (2019) proposed Root Mean Square Layer Normalization, which simplifies LayerNorm by removing the mean centering:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}$$

**Advantages over LayerNorm:**
- Fewer operations (no mean subtraction, no bias term $\beta$)
- ~10% faster on GPU (reduced memory reads)
- Empirically equivalent quality

**Why removing the mean works:** The mean-centering step in LayerNorm removes the DC component of the activation vector. In practice, this component carries little task-relevant information in deep Transformers -- the useful information is in the relative magnitudes and directions, not the offset. RMSNorm preserves these while being computationally cheaper.

LLaMA, Gemma, Mistral, Qwen, and most modern LLMs use Pre-RMSNorm -- the Pre-LN placement with RMSNorm instead of full LayerNorm.

---

## 7. The Architecture Research Takeaway

Post-LN's warmup requirement is a case study in how training instability can masquerade as a hyperparameter tuning problem. The original Transformer paper presented warmup as a design choice ("we use warmup over the first ... training steps"). Xiong et al. revealed it was a necessary patch for an architectural defect.

**General principle:** When a training recipe requires an unusual stabilization trick (warmup, gradient clipping at specific layers, learning rate schedules per layer group), ask: is this compensating for an architectural problem? If the trick can be eliminated by changing where a normalization layer sits, the structural fix is almost always better. Tricks add fragility; structural fixes add robustness.

*Source: [[attention-is-all-you-need|paper]], Xiong et al. (2020), Zhang & Sennrich (2019)*
