# Training Deep Nets with Sublinear Memory Cost
<!-- slug: gradient-checkpointing-chen · type: paper · source: https://arxiv.org/abs/1604.06174 -->

**Core Insight.** Full backpropagation through an n-layer network requires O(n) activation memory; by checkpointing only √n evenly spaced activations and recomputing the rest on demand during the backward pass, memory drops to O(√n) at the cost of exactly one additional forward pass per mini-batch (≈33% more compute).

**Guideline.** Apply gradient checkpointing (activation checkpointing) whenever per-GPU activation memory exceeds budget. The √n scheme checkpoints one activation every √n layers; all intermediate activations in each √n-layer segment are recomputed during backward using the stored boundary activation. Budget the overhead as ~33% increase in compute time, not 2×.

## Technical Details

- **Standard backprop memory:** O(n) — all intermediate feature maps stored for the chain rule.
- **Checkpointed memory:** O(√n) — only √n boundary activations retained; each of the √n segments recomputes locally during backward.
- **Compute cost:** "only the computational cost of an extra forward pass per mini-batch" — one additional forward, not one per layer.
- **Extreme variant:** O(log n) memory at O(n log n) compute cost by applying the scheme recursively (checkpoint at log-spaced intervals, recompute segments which themselves use checkpointing).
- **Empirical result:** 1,000-layer ResNet on ImageNet — activation memory reduced from **48 GB → 7 GB** (6.8×) with only **30% runtime overhead**.
- **Algorithm basis:** Computation graph analysis identifies in-place and memory-sharing opportunities; the √n checkpoint placement minimizes peak memory for a fixed recompute budget.
- **Modern adoption:** PyTorch's `torch.utils.checkpoint.checkpoint()` and Hugging Face `gradient_checkpointing_enable()` are direct descendants; JAX's `jax.checkpoint` / `jax.remat` applies the same pattern. All major frameworks expose this as a one-line toggle.
- **Training-memory angle:** Activation checkpointing converts the activation bucket from O(n·s·b·h) to O(√n · s·b·h) — enabling models and batch sizes that would otherwise OOM. The 33% compute overhead is the key practitioner number: it is why "selective recompute" (Korthikanti 2022, [[selective-recompute-korthikanti]]) targets expensive-to-store but cheap-to-recompute activations to get most of the memory saving with far less than 33% overhead.

## Citation
Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin. "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174, 2016. https://arxiv.org/abs/1604.06174
