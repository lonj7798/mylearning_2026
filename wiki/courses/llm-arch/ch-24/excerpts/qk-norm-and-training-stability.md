# Excerpt: QK-Norm and Training Stability in OLMo 2

<!-- source: [[olmo-2|report]], [[ch-09]] -->

## The Problem OLMo 1 Faced

OLMo 1 suffered from loss spikes during training — sudden jumps in the training loss that required manual restarts or rollbacks. These are expensive at scale: on a 1,280-GPU cluster, a single loss spike that forces a rollback to a checkpoint 1,000 steps earlier wastes hours of compute.

The root cause is **attention logit growth**. During the dot-product attention computation $\text{softmax}(QK^T / \sqrt{d_k})$, the magnitudes of query and key vectors tend to increase during training. As $\|q\| \cdot \|k\|$ grows, the pre-softmax scores become large, the softmax saturates (one entry dominates), and gradients through the saturated softmax vanish. The model effectively stops learning from that attention head, and the resulting instability propagates through the network as a loss spike.

The $1/\sqrt{d_k}$ scaling factor is a constant set at initialization. It was calibrated assuming query and key vectors have unit variance — an assumption that becomes invalid after billions of training steps.

## The QK-Norm Solution

OLMo 2 applies RMSNorm to query and key vectors per-head, before the RoPE rotation:

$$\hat{q}_h = \frac{q_h}{\text{RMS}(q_h)} \cdot \gamma_q, \quad \hat{k}_h = \frac{k_h}{\text{RMS}(k_h)} \cdot \gamma_k$$

where $\gamma_q, \gamma_k$ are learnable scale parameters (from RMSNorm). After normalization, the vectors have bounded norm regardless of how training evolves. The dot product $\hat{q}_h^T \hat{k}_h$ is therefore bounded, and the softmax cannot saturate.

### Bounding the Attention Logits: A Concrete Calculation

Without QK-norm, the attention logit for position $i$ attending to position $j$ is:

$$s_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$$

The magnitude of $q_i^T k_j$ is bounded by $\|q_i\| \cdot \|k_j\|$ (Cauchy-Schwarz). If query and key norms drift to, say, 50 during training (which is common in long runs without normalization), then $|s_{ij}|$ can reach $50 \times 50 / \sqrt{128} \approx 221$. A softmax with inputs of magnitude 200+ is numerically degenerate — one entry rounds to 1.0 and all others to 0.0 in any practical floating-point format.

With QK-norm, the RMSNorm operation ensures $\|\hat{q}_h\| \approx \sqrt{d_k}$ and $\|\hat{k}_h\| \approx \sqrt{d_k}$ (the learnable $\gamma$ can adjust this, but initializes at 1). The dot product $\hat{q}_h^T \hat{k}_h$ is then bounded by approximately $d_k$, and after the $1/\sqrt{d_k}$ scaling, the logit stays in $[-\sqrt{d_k}, \sqrt{d_k}]$. For $d_k = 128$, that is $[-11.3, 11.3]$ — well within the numerically stable regime for softmax.

### Why QK-Norm Does Not Reduce Expressiveness

A natural concern: normalizing Q and K discards magnitude information. Does this hurt the model?

The answer is no, for a specific reason: **attention scores only encode relative preferences**. The softmax operation already discards absolute magnitude information — it depends only on the *differences* between logits ($e^{s_i - s_j}$), not their absolute values. QK-norm makes this explicit by standardizing the scale *before* the dot product, rather than relying on the softmax to compensate for unbounded inputs.

The learnable $\gamma$ parameters in the RMSNorm give the model control over the *overall* scale of attention logits (how peaked or flat the attention distribution should be), while the dot product structure controls the *pattern* (which keys each query should attend to). This separation of scale and pattern is arguably cleaner than the standard approach where both are entangled.

### Why Before RoPE?

RoPE applies a position-dependent rotation $R_\theta$ to queries and keys. Rotation preserves norms ($\|R_\theta x\| = \|x\|$), so normalizing before or after RoPE gives the same norm bound. Normalizing *before* RoPE is cleaner because the normalization operates on the raw linear projection output — no interaction with positional encoding to reason about.

### The Z-Loss Complement

QK-norm stabilizes the *interior* of the model (attention logits). But instability can also arise at the *output* — the logits over the vocabulary. Z-loss adds a regularization penalty:

$$\mathcal{L}_Z = \alpha \cdot \log^2\!\left(\sum_j e^{z_j}\right)$$

This grows quadratically when the log-sum-exp of output logits becomes large, gently pushing the model away from extreme output distributions. The coefficient $\alpha \sim 10^{-4}$ is small enough to not interfere with normal training.

Together, QK-norm and Z-loss cover the two primary failure modes: attention entropy collapse (interior) and logit explosion (output).

## Evidence from the OLMo 2 Report

The OLMo 2 team published loss curves comparing training with and without QK-norm. The results are unambiguous:

- **Without QK-norm:** Loss spikes appear at irregular intervals during training, requiring checkpointing and rollback. The spikes are unpredictable — they do not occur at fixed intervals or training steps, making them impossible to anticipate and pre-empt.
- **With QK-norm:** Stable loss curves across the entire training run (4-6T tokens depending on model size). No loss spikes, no rollbacks required.

The stability mechanisms also interact with norm placement. Combined with Z-loss and improved initialization, QK-norm enabled OLMo 2 to use **post-norm** placement (RMSNorm after attention and FFN, rather than before). As [[ch-09]] discussed, post-norm is generally harder to train because the residual path is not a clean identity — each sublayer's output passes through normalization before being added back. Without stability mechanisms, this amplifies gradient variance and causes divergence. With QK-norm + Z-loss + scaled initialization, the instability is controlled, and post-norm's potential representational advantage (the normalization operates on the full sublayer output, not the input) becomes accessible.

The OLMo 2 ablations show that post-norm does *not* work stably without these complementary mechanisms. This is an important caveat: post-norm is not universally better, it is better *when the stability prerequisites are met*.

## Broader Adoption

QK-norm is not unique to OLMo 2. Gemma 3 ([[gemma-3|report]]) replaced its previous logit soft-capping mechanism with QK-norm, finding it simpler and more effective. Soft-capping applies $\tanh$ to bound logits, which works but introduces a nonlinearity into the attention computation that complicates gradient flow. QK-norm achieves the same goal through normalization — a linear-scale operation that is better understood and has cleaner gradients.

The convergence of independent teams (AI2, Google DeepMind) on the same solution is strong evidence that QK-norm addresses a genuine, scale-dependent problem. It is not a quirk of one model's training setup or data distribution — it is a structural property of dot-product attention that becomes critical as training scale increases.

## Alternatives to QK-Norm

QK-norm is not the only approach to bounding attention logits. Understanding the alternatives clarifies why QK-norm has become the preferred solution:

- **Logit soft-capping** (Gemma 2): Applies $\text{logits} \leftarrow \text{cap} \cdot \tanh(\text{logits} / \text{cap})$ to bound attention scores. Effective, but the $\tanh$ nonlinearity compresses gradients near the cap boundary and complicates Flash Attention integration (the tiling algorithm must accommodate the nonlinearity). Gemma 3 dropped soft-capping in favor of QK-norm for this reason.

- **Clipped attention** (ad hoc): Some implementations clip attention logits to a fixed range after computation. This is simpler than QK-norm but creates a hard gradient discontinuity at the clip boundary — the gradient is exactly zero for any logit that hits the clip threshold, which can cause optimization artifacts.

- **Temperature scaling**: Dividing logits by a learned temperature parameter can prevent extreme values, but the temperature interacts with the $1/\sqrt{d_k}$ scaling and is hard to tune independently per head.

QK-norm avoids all these issues: it operates before the dot product (so no interaction with Flash Attention tiling), it uses smooth normalization (no gradient discontinuities), and it is applied per-head (allowing head-specific scale through learnable $\gamma$).

## Key Takeaway

The lesson from QK-norm is that **scale creates failure modes that are invisible at small scale**. At 1B parameters or 100B training tokens, attention logit growth may never cause problems. At 32B parameters and 6T tokens, it is a near-certainty. Stability mechanisms like QK-norm are insurance policies — cheap to implement, negligible compute overhead, and essential at scale.
