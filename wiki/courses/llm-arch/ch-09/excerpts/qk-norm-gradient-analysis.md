<!-- scope: QK-norm gradient analysis, parent: [[ch-09]] -->

# QK-Norm Gradient Analysis

QK-norm -- normalizing queries and keys before the dot product -- prevents attention entropy collapse during long training runs. This excerpt analyzes the problem mathematically, shows how QK-norm bounds attention logits, and examines the gradient flow implications.

---

## The Attention Logit Growth Problem

### Standard Attention Logits

In standard attention, the logits (pre-softmax scores) for query $q_i$ and key $k_j$ are:

$$s_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$$

The $\sqrt{d_k}$ scaling ensures that at initialization (when $q$ and $k$ are random with unit variance), $\text{Var}(s_{ij}) \approx 1$. But during training, the norms $\|q_i\|$ and $\|k_j\|$ can grow unboundedly.

### Why Norms Grow

The Q and K projections are $q = W_Q x$ and $k = W_K x$, where $x$ comes from the residual stream (post-normalization in pre-norm architectures). While RMSNorm bounds $\|x\|$, the weight matrices $W_Q$ and $W_K$ can grow during training. After many gradient updates:

$$\|q_i\| \propto \|W_Q\| \cdot \|x\|$$

If $\|W_Q\|$ grows (which it does under standard SGD/Adam without explicit weight decay on these projections, or when weight decay is insufficient), then $\|q_i\|$ grows proportionally.

### The Entropy Collapse Cascade

When logit magnitudes grow, the softmax distribution becomes increasingly peaked:

1. **Large logits** $\Rightarrow$ softmax approaches one-hot (attending to a single token)
2. **One-hot attention** $\Rightarrow$ low entropy, losing the ability to aggregate information from multiple positions
3. **Low entropy attention** $\Rightarrow$ gradient sparsity (most attention weights are near zero, contributing near-zero gradients)
4. **Gradient sparsity** $\Rightarrow$ further concentration of updates on the few active paths, reinforcing the collapse

This manifests as **loss spikes** during training: sudden jumps in loss when attention heads collapse. For trillion-token training runs (OLMo 2, Gemma 3), even rare instabilities are unacceptable.

### Quantifying the Instability

The entropy of the attention distribution for query $i$ is:

$$H_i = -\sum_j p_{ij} \log p_{ij}, \qquad p_{ij} = \text{softmax}(s_{ij})$$

At initialization with unit-scale logits, $H_i \approx \log N$ (near-uniform attention). As logit magnitude $M = \max_{ij} |s_{ij}|$ grows:

$$H_i \to 0 \qquad \text{as } M \to \infty$$

The rate of entropy decrease depends on the logit gap: if the largest logit exceeds the second-largest by $\Delta$, the attention entropy is bounded by $H_i \lesssim \log(1 + (N-1) \cdot e^{-\Delta})$.

---

## QK-Norm: The Solution

QK-norm applies normalization to queries and keys before the dot product:

$$s_{ij} = \frac{\text{Norm}(q_i)^T \cdot \text{Norm}(k_j)}{\sqrt{d_k}}$$

where Norm is typically RMSNorm or L2 normalization, applied per-head.

### Bounded Logit Magnitudes

With L2 normalization ($\text{Norm}(v) = v / \|v\|$):

$$|s_{ij}| = \frac{|\hat{q}_i^T \hat{k}_j|}{\sqrt{d_k}} \le \frac{\|\hat{q}_i\| \cdot \|\hat{k}_j\|}{\sqrt{d_k}} = \frac{1}{\sqrt{d_k}}$$

The logits are bounded by $\frac{1}{\sqrt{d_k}}$ regardless of training dynamics. With $d_k = 128$: $|s_{ij}| \le 0.088$.

This is extremely conservative. In practice, RMSNorm is used instead of hard L2 normalization, which allows some variation in magnitude:

$$\text{RMSNorm}(q) = \gamma \odot \frac{q}{\text{RMS}(q)}$$

The learnable scale $\gamma$ provides flexibility, but the key property is maintained: the norm of the query/key is controlled by $\gamma$, which is a small per-dimension parameter, not the large weight matrix $W_Q$. Attention logit growth is decoupled from weight matrix growth.

---

## Gradient Flow Analysis

### Without QK-Norm

The gradient of the loss with respect to $W_Q$ includes a term:

$$\frac{\partial \mathcal{L}}{\partial W_Q} \ni \frac{\partial \mathcal{L}}{\partial s_{ij}} \cdot \frac{\partial s_{ij}}{\partial q_i} \cdot \frac{\partial q_i}{\partial W_Q}$$

where $\frac{\partial s_{ij}}{\partial q_i} = \frac{k_j}{\sqrt{d_k}}$. As $\|k_j\|$ grows, this gradient grows proportionally, creating a feedback loop: large keys produce large gradients on queries, which produce larger queries, which produce larger logits.

### With QK-Norm

The gradient now includes the Jacobian of the normalization:

$$\frac{\partial s_{ij}}{\partial q_i} = \frac{1}{\sqrt{d_k}} \cdot \hat{k}_j \cdot J_{\text{Norm}}(q_i)$$

For L2 normalization, the Jacobian is:

$$J_{\text{L2}}(q) = \frac{1}{\|q\|}\left(I - \hat{q}\hat{q}^T\right)$$

This has two effects:

1. **Magnitude suppression:** The $\frac{1}{\|q\|}$ factor means that as $\|q\|$ grows, the gradient decreases inversely. This creates natural **self-regulation**: large norms receive smaller gradients, preventing further growth.

2. **Direction preservation:** The $(I - \hat{q}\hat{q}^T)$ term projects the gradient onto the hyperplane orthogonal to $q$. This means gradients can change the *direction* of $q$ but not its *magnitude* (magnitude changes are suppressed by the $\frac{1}{\|q\|}$ factor). Direction is what matters for attention patterns; magnitude is what causes instability.

### The Effective Learning Rate

Without QK-norm, the effective learning rate for updating $q$'s contribution to logits is:

$$\eta_{\text{eff}} \propto \eta \cdot \|k\|$$

This grows with training as $\|k\|$ grows -- an uncontrolled effective learning rate.

With QK-norm:

$$\eta_{\text{eff}} \propto \eta \cdot \frac{1}{\|q\|}$$

This *decreases* as $\|q\|$ grows, providing implicit learning rate decay for the attention logit computation. The system is self-stabilizing.

---

## Expressiveness Concerns

### Does QK-Norm Hurt Attention Quality?

Normalizing Q and K discards magnitude information. But attention scores only need to encode **relative** preferences -- which keys a query should attend to more -- not absolute magnitudes. The softmax already discards absolute scale (it's invariant to adding a constant to all logits). QK-norm makes this explicit by normalizing before the dot product.

Empirically, both OLMo 2 ([[olmo-2|report]]) and Gemma 3 ([[gemma-3|report]]) found that QK-norm did not degrade model quality on any measured benchmark. The stability gains (no loss spikes, robust training across scales) far outweighed any potential expressiveness loss.

### The RoPE Interaction

An important implementation detail: when using RoPE (Rotary Position Embeddings), QK-norm should be applied **before** RoPE. RoPE applies a position-dependent rotation to queries and keys, which preserves norms (rotations are orthogonal). If QK-norm is applied after RoPE, the normalization undoes the position-dependent scaling that RoPE may introduce in some implementations.

OLMo 2's implementation: RMSNorm on Q and K, then apply RoPE. This ensures:
1. Q and K have bounded norms entering the dot product
2. RoPE rotations don't affect norms (orthogonal transformation)
3. Position information is fully preserved

---

## QK-Norm vs. Logit Soft-Capping

Gemma 2 used an alternative approach: **soft-capping** attention logits:

$$s_{ij} \leftarrow \text{cap} \cdot \tanh\left(\frac{s_{ij}}{\text{cap}}\right)$$

This bounds logits to $[-\text{cap}, +\text{cap}]$ after computation. Gemma 3 replaced this with QK-norm, noting several advantages:

| Property | Soft-Capping | QK-Norm |
|----------|-------------|---------|
| Gradient flow | $\tanh$ saturates, killing gradients at extremes | Smooth gradients throughout |
| Bound tightness | Fixed cap value (hyperparameter) | Adaptive via learned $\gamma$ |
| Computational cost | Extra element-wise $\tanh$ | Extra norm (amortized into existing norm ops) |
| Flash Attention compatibility | Requires custom kernel modifications | Compatible with standard Flash Attention |

The Flash Attention compatibility point is significant: soft-capping applies a nonlinearity inside the attention kernel, which complicates the online softmax algorithm. QK-norm applies normalization outside the kernel (to Q and K before they enter attention), making it compatible with standard Flash Attention implementations.

---

## Practical Recommendations

1. **For any training run exceeding ~1T tokens, use QK-norm.** The cost is negligible; the risk of loss spikes without it is real.

2. **Use RMSNorm (not L2-norm) for Q and K.** The learnable $\gamma$ provides flexibility while maintaining bounded logits.

3. **Apply QK-norm before RoPE.** This preserves position information while controlling norms.

4. **Consider combining with Z-loss** (OLMo 2's approach). Z-loss regularizes the output logits of the language model head, complementing QK-norm's regularization of attention logits.

---

## References

- [[olmo-2|AI2, "OLMo 2 Technical Report" (2025) (report)]]
- [[gemma-3|Google DeepMind, "Gemma 3 Technical Report" (2025) (report)]]
- [[rmsnorm|Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) (paper)]]
- [[pre-norm-vs-post-norm|Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) (paper)]]
