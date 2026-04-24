<!-- scope: pre-norm vs post-norm stability analysis, parent: [[ch-09]] -->

# Pre-Norm vs Post-Norm: Training Stability Analysis

Why does moving LayerNorm from after the residual addition to before the sublayer eliminate the need for learning rate warmup? This excerpt traces the gradient flow argument from Xiong et al. (2020), explains the representational cost hypothesis, and connects QK-norm as a complementary technique.

---

## The Core Question

The original Transformer (Vaswani et al., 2017) placed normalization **after** the residual addition (post-norm). Training required learning rate warmup — linearly increasing the learning rate from near-zero for thousands of steps. Without warmup, training diverged.

GPT-2 moved normalization **before** the sublayer (pre-norm). Warmup became optional. Why?

---

## Post-Norm: Why Gradients Explode at Initialization

In post-norm, the forward recurrence is:

$$x_{l+1} = \text{LN}(x_l + f_l(x_l))$$

LayerNorm sits directly on the residual path. Its Jacobian:

$$J_{\text{LN}} = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{\hat{x}\hat{x}^T}{d}\right)\text{diag}(\gamma)$$

This is **not** the identity — it projects out the mean direction and rescales by $1/\sigma$. Each layer's gradient passes through this Jacobian, and the distortion accumulates.

Xiong et al. ([[pre-norm-vs-post-norm|paper]]) proved via mean field theory that at initialization:

- **Output-layer gradients:** $\mathbb{E}[\|\nabla_{\theta_L}\mathcal{L}\|] = O(d)$
- **Earlier-layer gradients:** magnitude decays moving toward the input, with ratio between output and input layers growing with $L$

For a 96-layer model at $d = 12{,}288$, the gradient magnitude near the output can be orders of magnitude larger than near the input. A single learning rate cannot serve both ends — too large for the output causes divergence, too small for the input causes stagnation. **Warmup is the patch**: start tiny (safe for all layers), then increase as gradients stabilize.

---

## Pre-Norm: Why Gradients Stay O(1)

In pre-norm, the recurrence is:

$$x_{l+1} = x_l + f_l(\text{LN}(x_l))$$

The residual path is an **unobstructed identity**. LayerNorm is inside the branch, not on the highway. The gradient:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{k=l+1}^{L}(I + J_k)$$

where $J_k = \frac{\partial f_k(\text{LN}(x_k))}{\partial x_k}$ is the branch Jacobian. At initialization with small random weights, $\|J_k\| \ll 1$, so the identity $I$ dominates each factor. The product stays $\approx I$ regardless of depth.

**Result:** $\mathbb{E}[\|\nabla_{\theta_l}\mathcal{L}\|] = O(1)$ for all layers $l$, independent of $L$. A constant learning rate works from step 1.

---

## The Representational Cost of Pre-Norm

Pre-norm is not strictly better. There is a tradeoff:

In pre-norm, each sublayer's contribution is normalized before being added to the stream. This constrains each layer to make a "small" additive update. Post-norm normalizes *after* addition, allowing each layer's contribution to have larger relative impact before renormalization.

**Empirical signal:** post-norm models can achieve slightly better final perplexity when training succeeds. The hypothesis is that post-norm's non-uniform gradient magnitudes allow deeper layers to learn richer transformations, whereas pre-norm's uniform gradients impose a more democratic (but potentially limiting) layer contribution structure.

This is why post-norm hasn't been fully abandoned — it potentially offers a quality ceiling that pre-norm doesn't reach.

---

## Recovering Post-Norm: The OLMo 2 Recipe

OLMo 2 ([[olmo-2|report]]) revived post-norm by combining it with three stability techniques:

1. **QK-norm:** RMSNorm applied to queries and keys before the dot product. Bounds attention logits regardless of training dynamics, preventing the entropy collapse that causes loss spikes.

2. **Z-loss:** Regularization term $\lambda \cdot \log^2 Z$ on the output softmax partition function, penalizing large logits in the final layer. Prevents the output distribution from becoming too peaked.

3. **Improved initialization:** Scale output projections to preserve activation and gradient norms across depth.

The combination stabilized training across 7B, 13B, and 32B models on multi-trillion-token runs. The key insight: post-norm's instability has multiple sources (gradient imbalance, attention logit growth, output logit growth), and each needs a targeted fix.

---

## QK-Norm as Complementary Technique

QK-norm addresses a failure mode orthogonal to norm placement: **attention logit growth**.

During long training runs, $\|q\|$ and $\|k\|$ can grow unboundedly, making $q^T k$ increasingly large. Softmax saturates toward one-hot, collapsing attention entropy. This manifests as sudden loss spikes.

$$\text{Attention} = \text{softmax}\left(\frac{\text{Norm}(Q) \cdot \text{Norm}(K)^T}{\sqrt{d_k}}\right)V$$

QK-norm bounds the attention logits by construction. Both OLMo 2 and Gemma 3 ([[gemma-3|report]]) converged on this independently — Gemma 3 replaced soft-capping ($\text{cap} \cdot \tanh(\text{logits}/\text{cap})$) with QK-norm, finding it simpler and more effective.

**Why QK-norm doesn't hurt:** attention scores encode *relative* preferences, not absolute magnitudes. Softmax already discards absolute scale. QK-norm makes this explicit.

---

## Interaction with Scaled Initialization

GPT-2/3's output projection scaling by $\frac{1}{\sqrt{2L}}$ makes each sublayer's contribution have variance $O(1/L)$. After $L$ layers, total added variance is $O(1)$.

- **Pre-norm + scaled init:** Identity path dominates even more; extremely stable but each layer's contribution is heavily damped in early training.
- **Post-norm + scaled init:** Partially compensates for gradient imbalance by reducing branch Jacobian magnitude. Helps but doesn't eliminate the need for warmup.

---

## Decision Map

| Configuration | Warmup needed? | Stability | Final quality | Used by |
|---------------|---------------|-----------|---------------|---------|
| Pre-norm + RMSNorm | No | Excellent | Good | LLaMA, Mistral, GPT-NeoX |
| Post-norm + standard init | Yes (critical) | Fragile | Potentially best | Original Transformer |
| Pre+Post-norm (sandwich) | No | Excellent | Good+ | Gemma 3 |
| Post-norm + QK-norm + Z-loss | Minimal | Good | Good+ | OLMo 2 |

---

**References:** [[pre-norm-vs-post-norm|Xiong et al. (2020)]], [[olmo-2|AI2, OLMo 2 (2025)]], [[gemma-3|Google DeepMind, Gemma 3 (2025)]]
