# Excerpt: From SoftCap to QK-Norm

<!-- source: [[gemma-3|report]], [[ch-09]], [[ch-24]] -->

## The Problem: Attention Logit Growth

During training, the norms of Q and K vectors grow over time. The dot product $q_i \cdot k_j$ grows proportionally to $\|q_i\| \cdot \|k_j\|$, which can become very large. This causes:

1. **Softmax saturation:** When one score dominates, $\text{softmax}$ approaches a one-hot distribution. Gradients through the saturated softmax vanish.
2. **Entropy collapse:** The attention pattern locks onto a single position, discarding all other context. The model loses its ability to aggregate information.
3. **Numerical overflow:** FP16/BF16 have limited dynamic range. Logits exceeding ~65K (FP16) or ~3.4e38 (BF16) cause NaN.

Standard gradient clipping does not prevent this. Clipping bounds the gradient *norm*, not the *activation magnitude*. Logits can grow slowly over many steps, staying below the clipping threshold at each step but accumulating to problematic levels.

## Gemma 2's Solution: SoftCap

Apply a $\tanh$-based clamp after computing the raw scores:

$$\text{SoftCap}(x, c) = c \cdot \tanh\!\left(\frac{x}{c}\right)$$

**Properties:**
- Output is strictly in $(-c, c)$
- Near-linear for $|x| \ll c$: $\tanh(x/c) \approx x/c$, so $\text{SoftCap}(x, c) \approx x$
- Saturates for $|x| \gg c$: approaches $\pm c$
- Smooth and differentiable everywhere

**Gemma 2 used two cap values:**
- Attention logits: $c_\text{attn} = 50$
- Final output logits: $c_\text{final} = 30$

This worked. Training stability improved. But it created engineering friction.

## Why SoftCap Conflicted with Flash Attention

Flash Attention's tiling algorithm assumes the attention computation follows this exact pattern:

```
for each tile of K:
    S_block = Q_tile @ K_tile^T / sqrt(d)    # tile of scores
    update running max and sum for online softmax
    O_tile += softmax_tile(S_block) @ V_tile   # accumulate output
```

SoftCap inserts a nonlinearity between the score computation and the softmax:

```
S_block = Q_tile @ K_tile^T / sqrt(d)
S_block = c * tanh(S_block / c)               # <-- breaks tiling
update running max and sum for online softmax
```

The $\tanh$ is element-wise and does not break mathematical correctness. But it:
- Requires additional SRAM reads/writes per tile
- Prevents fused kernel optimizations that combine score computation with softmax
- Requires maintaining custom forks of Flash Attention

Both the JAX/XLA and CUDA implementations of Flash Attention needed modifications for Gemma 2. This maintenance burden was a direct motivation for finding an alternative.

## Gemma 3's Solution: QK-Norm

Normalize Q and K *before* computing scores:

$$Q' = \text{RMSNorm}(Q), \quad K' = \text{RMSNorm}(K)$$

After normalization, each vector has unit RMS norm. The dot product of unit-norm vectors satisfies:

$$|q'_i \cdot k'_j| \leq \|q'_i\| \cdot \|k'_j\| = 1$$

Summing over the head dimension $d_k$ gives raw scores bounded in $[-d_k, d_k]$. After the $1/\sqrt{d_k}$ scaling, the final logits are bounded in $[-\sqrt{d_k}, \sqrt{d_k}]$. For $d_k = 128$: logits in $[-11.3, 11.3]$.

**Key advantage:** The attention kernel itself is unchanged. RMSNorm is applied before entering the kernel. Standard Flash Attention works unmodified.

## The RMSNorm Computation

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

where $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$ and $\gamma$ is a learnable per-dimension scale.

The learnable scale $\gamma$ means the vectors are not *exactly* unit norm after normalization --- $\gamma$ allows the model to learn a per-dimension weighting. But the norm is controlled and bounded, preventing the unbounded growth that SoftCap was designed to contain.

## OLMo 2's Independent Validation

OLMo 2 adopted QK-norm independently of Gemma 3, arriving at the same architectural choice from a different starting point. OLMo 2 used post-norm (not pre-norm) and added QK-norm to stabilize training. Their ablations confirmed that QK-norm improved training stability without quality degradation.

Two teams at different organizations converging on the same solution is strong evidence that QK-norm is the right approach to the logit growth problem.

## Summary: SoftCap vs QK-Norm

| Property | SoftCap | QK-Norm |
|----------|---------|---------|
| Where applied | After QK^T | Before QK^T |
| Bound mechanism | tanh clamp | Cauchy-Schwarz |
| Flash Attention | Incompatible (custom kernels) | Fully compatible |
| Hyperparameters | c_attn, c_final | None (learnable gamma) |
| Gradient flow | Compressed at saturation | Unaffected |
| Independent validation | Gemma 2 only | OLMo 2 + Gemma 3 |
