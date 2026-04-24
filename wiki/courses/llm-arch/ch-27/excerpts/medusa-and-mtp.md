# Excerpt: Medusa Heads and Multi-Token Prediction

<!-- sources: Cai et al. "Medusa" (2024), [[deepseek-v3|report]] — DeepSeek-V3 MTP -->

## The Draft Model Problem

Standard speculative decoding requires maintaining and serving a separate draft model alongside the target model. This doubles deployment complexity: two sets of weights in GPU memory, two KV caches, orchestration logic for draft-verify cycles. For production serving systems already operating near memory capacity, the overhead can be prohibitive.

Two approaches eliminate the separate draft model by integrating multi-token prediction directly into the target model.

## Medusa: Bolt-On Prediction Heads

Medusa adds K lightweight MLP heads to the target model, each predicting a future token position:

$$\text{Head}_k(h_t) = W_k^{(2)} \cdot \text{SiLU}(W_k^{(1)} \cdot h_t + h_t)$$

Each head is a single-hidden-layer network with a residual connection, taking the final hidden state $h_t$ as input. Head $k$ predicts the token at position $t+k+1$. The heads collectively add roughly 0.5% additional parameters to the model.

**Key limitation:** Each Medusa head predicts future tokens using only $h_t$ — the hidden state at the current position. It has no information about what tokens will actually appear between position $t+1$ and its prediction target $t+k+1$. This is fundamentally less informed than autoregressive prediction, where each token conditions on all previous tokens.

### Tree-Structured Verification

Medusa compensates for per-head uncertainty by generating multiple candidates per head (top-$s$) and arranging them in a tree. With $s=3$ candidates per head and $K=4$ heads, there are up to $3^4 = 81$ candidate paths. A custom tree attention mask allows the target model to score all paths in a single forward pass.

The tree structure means that if the best candidate at position $t+2$ is rejected, an alternative candidate might be accepted, preserving work done for positions $t+3$ and beyond. This recoverability is what makes Medusa competitive despite weaker per-position accuracy.

### Training Variants

**Medusa-1 (freeze backbone):** Train only the heads on the model's output distributions. Fast (hours), preserves base quality perfectly, but heads are limited by the features in $h_t$.

**Medusa-2 (joint fine-tune):** Fine-tune backbone and heads together. The backbone learns to encode features useful for multi-token prediction. Higher acceptance rates, but requires careful fine-tuning to avoid degrading base quality.

## DeepSeek-V3 MTP: Training-Time Integration

DeepSeek-V3 takes the opposite approach: integrate multi-token prediction into the pre-training objective from the start, using $D$ sequential MTP modules that maintain a **causal chain**.

Each MTP module receives:
1. The main model's hidden state $h_t$ at position $t$
2. The embedding of the token predicted by the previous module (or the ground-truth token during training)

$$h_t^{(k)} = \text{MTPModule}_k(\text{concat}(h_t^{(k-1)}, \text{embed}(\hat{x}_{t+k})))$$

The total training loss is:
$$\mathcal{L} = \mathcal{L}_\text{NTP} + \lambda \sum_{k=1}^{D} \mathcal{L}_\text{MTP}^{(k)}$$

with DeepSeek-V3 using $\lambda = 0.3$.

### Why the Causal Chain Matters

The critical difference from Medusa: MTP module $k$ receives the embeddings of tokens $t+1$ through $t+k-1$, making its prediction of token $t+k$ significantly more informed. During training (with teacher forcing), these are ground-truth tokens. During inference, they are the predictions of preceding MTP modules — less accurate, but still much more informative than Medusa's dependence on $h_t$ alone.

### The Training Quality Bonus

A counterintuitive result: MTP improves model quality even when MTP modules are discarded at inference. DeepSeek-V3 ablations show consistent benchmark improvements from the MTP training objective alone.

The explanation: predicting future tokens forces the model to develop richer representations at each position. To predict token $t+3$, the model must capture not just the immediate next token but the trajectory of the sequence — a form of implicit planning that strengthens the features available for standard next-token prediction.

## Comparison

| Dimension | Medusa | DeepSeek MTP |
|-----------|--------|--------------|
| When added | Post-training | Pre-training |
| Draft information | Only $h_t$ | Causal chain with intervening tokens |
| Extra params | ~0.5% (heads only) | MTP modules (larger) |
| Training cost | Hours (head-only) | Integrated in pre-training |
| Base model quality | Unchanged (Medusa-1) | Improved |
| Deployment | Single model + heads | Single model + modules |
| Reported speedup | 1.5-2.5x | 1.8x |

## EAGLE: A Middle Ground

EAGLE (Li et al., 2024) occupies an interesting middle position. Instead of training separate heads (Medusa) or full MTP modules (DeepSeek), EAGLE reuses the target model's own early layers as a lightweight draft mechanism. A small autoregressive head operates on the target model's feature representations, drafting tokens with awareness of the model's internal state.

EAGLE achieves 2-3x speedup — comparable to external draft models — with only ~2% additional parameters. It requires a few days of training but produces stronger drafts than Medusa because it operates on richer feature representations rather than just the final hidden state.

## The Design Space Going Forward

The trend is clear: the field is moving from post-hoc bolt-ons (external draft models, Medusa heads) toward training-time integration (MTP), where the model is designed from the start for efficient multi-token generation.

The key insight connecting all three approaches: **the quality of the draft is bounded by the information available to the drafter**. Medusa has only $h_t$. EAGLE has the target model's features. MTP has a causal chain of intervening token embeddings. External draft models have their own complete forward pass. More information yields better drafts, but at increasing cost — and the art is finding the sweet spot for your deployment constraints.
