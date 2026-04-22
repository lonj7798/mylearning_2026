<!-- scope: gradient norm clipping for exploding gradients; foundational stability tool
     deps: [[adam]]
     see-also: [[mixed-precision]], [[lr-schedules]], [[ppo]]
-->

# On the difficulty of training Recurrent Neural Networks (Gradient Clipping)
- **Core Insight:** Exploding gradients can be tamed by rescaling the entire gradient vector whenever its global L2 norm exceeds a threshold, preserving direction while bounding step size.
- **Guideline:** Always clip the global gradient norm (typical threshold = 1.0 for LLM pretraining, 0.5–1.0 for RL) — never per-parameter clip-by-value, which destroys the gradient direction.
- **Authors:** Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
- **Year:** 2013 (ICML)
- **URL:** https://arxiv.org/abs/1211.5063
- **Relevant topics:** optimization stability, RNN/Transformer training, RL rollouts, FSDP/DDP gradient sync

## Abstract
The paper studies the well-known difficulty of training RNNs caused by vanishing and exploding gradients. It frames the exploding-gradient problem from analytical, geometric, and dynamical-systems viewpoints, showing that the recurrent Jacobian's spectral radius governs whether gradients blow up. To handle the exploding case the authors propose a simple, effective heuristic — gradient norm clipping — and a regularization term to combat vanishing. Their experiments on character-level language modeling and synthetic temporal-dependency tasks show that clipping alone removes a major source of training instability, allowing much larger learning rates without divergence.

## Key Contributions
- Formal characterization of the exploding-gradient problem in terms of the recurrent Jacobian's largest singular value.
- The **global-norm gradient clipping** heuristic: rescale all gradients by `threshold / ||g||` whenever `||g|| > threshold`.
- A regularizer that pushes back against vanishing gradients by encouraging unit-norm propagation of error signals.
- Empirical evidence that clipping makes RNN training robust to learning rate; the same trick later proved essential for Transformers and large-batch optimization.
- Geometric intuition (the "wall in error surface") explaining why a single bad gradient can destroy hours of progress.

## Key Figures/Tables to Study
- **Figure 1** (Error surface with a "cliff"): visual intuition for why a normal gradient step lands the parameters far from the manifold; clipping limits the leap.
- **Figure 6** (Effect of clipping vs. no clipping): the canonical demonstration that clipping enables higher learning rates without divergence.

## Technical Details
**Global-norm clipping** (the standard choice):
```
g_norm = sqrt(sum_i ||g_i||^2)            # over all parameter tensors
if g_norm > c:
    g_i <- g_i * (c / g_norm)             # uniform rescale, direction preserved
```
The key property: **direction is preserved**, only magnitude is bounded. Contrast with two inferior alternatives:
- **Clip-by-value** (`g_i <- clip(g_i, -c, c)` element-wise): distorts the descent direction; rarely used.
- **Per-tensor norm clip** (PyTorch `clip_grad_norm_(p, c)` looped per parameter): biases optimizer toward small tensors; almost never desired.

**Modern LLM defaults**:
- Pretraining (GPT/Llama/Qwen lineage): `max_grad_norm = 1.0`.
- SFT: typically `1.0`; sometimes `0.5` for noisy synthetic data.
- RL (PPO/GRPO): `0.5–1.0` on the policy gradient; reward spikes during rollout can produce 10x norm bursts that clipping absorbs.

**Distributed-training pitfall (FSDP / ZeRO-3)**: the global norm must be computed across **all shards** before scaling. Naively calling `clip_grad_norm_` on local shards under-counts the norm, leading to inconsistent scaling and silent divergence. Use `torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_` or the equivalent reduce-then-scale pattern. Same issue exists with DeepSpeed's ZeRO; both frameworks ship a correct utility you should call instead of writing your own.

**Mixed-precision interaction**: when loss-scaling (fp16), gradients are scaled by `S`. You must **unscale before clipping**, otherwise the threshold is meaningless. PyTorch's `GradScaler.unscale_(optimizer)` exists for this.

**Common pitfalls**:
- Clipping threshold too low (e.g. 0.1) → optimizer never makes a real step on hard examples; loss plateaus.
- Forgetting to clip after gradient accumulation → the accumulated gradient has different statistics from per-microbatch gradients; you must clip on the accumulated tensor.
- Tracking `pre-clip grad_norm` is one of the most informative training metrics: a sudden 100x spike usually predicts an imminent loss-spike or NaN.

## Connections
- **PPO / GRPO rollouts**: reward outliers and length variance produce extreme advantages; gradient clipping is the second line of defense after PPO's ratio clip and KL regularization. Most failures of RL fine-tuning trace back to either an unclipped advantage or an unclipped grad norm.
- **[[mixed-precision]]**: ordering is `unscale → clip → step`. Get this wrong and the run silently produces garbage.
- **[[adam]] / AdamW**: Adam's adaptive scaling does *not* protect against exploding gradients in early training; clipping is still required.
- **Loss spikes in pretraining**: the standard Llama-3 / OLMo-2 mitigation stack is: (1) global-norm clip 1.0, (2) skip-step on loss-spike, (3) embedding-norm monitoring. Clipping alone is necessary but not sufficient at 70B+ scale.
- **Karpathy's recipe** ([[karpathy-training-neural-net-recipe]]) lists "monitor and clip the gradient norm" as a non-negotiable.
