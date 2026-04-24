---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "On the difficulty of training Recurrent Neural Networks (Pascanu, Mikolov & Bengio 2013)"
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
---

# Excerpt: Gradient clipping — taming exploding gradients

**Authors:** Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
**Year:** 2013
**Venue:** ICML 2013
**URL:** https://arxiv.org/abs/1211.5063
**arXiv ID:** 1211.5063
**Raw-data source:** [[raw-data/gradient-clipping]]

---

## The phenomenon the paper names

Pascanu et al. formalise what practitioners had long observed: training deep recurrent networks periodically produces a gradient whose norm is orders of magnitude larger than normal — a single `||g|| ≈ 10^6` update can destroy thousands of steps of progress. Figure 1 of the paper visualises this as the loss surface having a "wall" or "cliff": the network parameters are on a near-flat plateau, a tiny region has an enormous slope, and a standard gradient step leaps off the cliff to somewhere random.

The formal characterisation they give is in terms of the recurrent Jacobian:

```math
\frac{\partial \mathcal{L}_T}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=t+1}^{T} \underbrace{\frac{\partial h_k}{\partial h_{k-1}}}_{J_k} \frac{\partial h_t}{\partial \theta}
```

If the largest singular value of the Jacobian `J_k` is `σ_max > 1`, the product telescopes geometrically and `||∂L/∂θ||` grows as `σ_max^T`. For RNNs with `T=100` and `σ_max=1.1`, that is a `10^4` multiplier. Transformers do not have the same deep recurrent Jacobian, but they have analogous amplifiers: the residual stream variance grows with depth (see [[excerpts/weight-init]]), attention softmax produces heavy-tailed gradients on rare tokens, and MoE routing decisions can send near-zero probability gradients through expert parameters.

---

## The heuristic (Algorithm 1 of the paper)

```math
\begin{aligned}
\|g\|_2 &= \sqrt{\sum_i \|g_i\|_2^{\,2}} \quad \text{(global L2 norm over all parameter tensors)} \\
g_i &\leftarrow
\begin{cases}
g_i & \text{if } \|g\|_2 \le c \\
g_i \cdot \dfrac{c}{\|g\|_2} & \text{if } \|g\|_2 > c
\end{cases}
\end{aligned}
```

The critical property, in one sentence: **direction is preserved, magnitude is bounded.** The rescaling is uniform across all parameter tensors, so every coordinate of every tensor is scaled by the same factor `c / ||g||`. The descent direction — the entire `g/||g||` unit vector — is untouched.

---

## Why global-norm is the right quantity (derivation)

Consider two ways to cap a gradient:

**Option A — element-wise clip by value**: `g_i ← clip(g_i, -c, c)` elementwise. This changes the *direction* of the gradient because different coordinates get clipped by different amounts. The clipped vector is no longer a descent direction for `L` in general.

**Option B — per-tensor norm clip**: `g_i ← g_i · min(1, c/||g_i||_2)` for each tensor separately. This biases the optimizer: tensors with small norm never get clipped; tensors with large norm do. Under distributed training, a single "hot" tensor (e.g. an attention output projection) gets penalised while LayerNorm weights sail through. The optimizer's effective learning rate becomes non-uniform across tensor types.

**Option C — global-norm clip** (the paper's proposal): compute `||g||` over the concatenation of all parameter tensors; rescale uniformly. The update direction is `g / ||g||` regardless of how close to or past `c` the norm is. This is the only option that preserves the *geometry* of the step.

The derivation is worth internalising: gradient descent with step size `η` on norm-clipped gradient is equivalent to gradient descent with an **adaptive step size** `η · min(1, c/||g||)`. When `||g|| ≤ c`, you take a normal step; when `||g||` spikes, you take a step of fixed magnitude `η · c` in the same direction. Clipping is step-size damping in disguise.

Figure 6 of Pascanu et al. is the canonical plot: train loss vs. training time for RNNs, with and without clipping, at several learning rates. Unclipped runs diverge at learning rates that work fine with clipping. The geometric intuition (Figure 1, "walls in error surface"): a single encounter with a cliff is enough to kill the run, and clipping acts as a safety leash.

---

## Modern LLM defaults

| Setting | Max grad norm |
|---|---|
| Pretrain (GPT, Llama, Qwen, OLMo) | **1.0** |
| SFT, standard data | 1.0 |
| SFT, noisy synthetic data | 0.5 |
| RL (PPO / GRPO) | 0.5–1.0 |

Notice: almost every frontier lab uses the same `1.0` default. This is not a coincidence — at `max_grad_norm = 1.0`, a healthy run clips ~5–15% of steps, which turns out to be the right amount to absorb loss-spike precursors without penalising typical steps. Lowering to `0.1` clips essentially every step, starving the optimizer; raising to `10` clips almost nothing, leaving you unprotected.

**The pre-clip gradient norm is the single most informative training metric.** Log it every step. A healthy 70B pretraining run has `pre-clip_grad_norm ≈ 0.3` with occasional spikes to 1–3. A run about to NaN shows `pre-clip_grad_norm` climbing from ~1 to ~10 to ~100 over ~50 steps before the crash; clipping masks the symptom in post-clip metrics but the pre-clip series shows the disease.

---

## Interaction with FSDP / ZeRO-3 — a dangerous pitfall

Under full sharded data parallel (FSDP) or DeepSpeed ZeRO-3, each GPU holds only a shard of the parameters — and correspondingly a shard of the gradients. A naive call to `torch.nn.utils.clip_grad_norm_(params, 1.0)` inside each rank computes the *local* shard's norm, not the global norm. Because each shard has `||g_shard|| < ||g_global||`, the local check `||g_shard|| > 1` is much less likely to trigger than the global check — you effectively clip less than intended, sometimes not at all.

The fix (reduce-then-scale):

```math
\|g\|_{\text{global}}^{\,2} = \sum_{r=1}^{R} \|g^{(r)}\|_2^{\,2} \quad \text{(all-reduce across ranks)}
```

followed by uniform scaling by `c / ||g||_global` on every rank. PyTorch's `torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_` implements this, and DeepSpeed's engine does too — **never hand-roll it**. Silent under-clipping has been blamed for several published loss-spike incidents.

---

## Interaction with mixed-precision and gradient accumulation

Under fp16 loss scaling (see [[excerpts/mixed-precision]]), gradients are multiplied by the loss scale `S`. Clipping on scaled gradients means your effective threshold is `S · c`, not `c` — you almost never clip. **Unscale before clip**: PyTorch provides `GradScaler.unscale_(optimizer)` for exactly this. Ordering is non-negotiable:

```
unscale → clip → optimizer.step
```

Swap any two and the run is silently broken.

Gradient accumulation: when accumulating over `K` microbatches before a step, the accumulated gradient is the *sum* over microbatches (or mean, depending on implementation). Its statistics differ from per-microbatch gradients — typically smaller norm per-element because noise cancels. You must clip on the *accumulated* tensor, not inside the accumulation loop. A common mistake is clipping each microbatch's gradient; that is wrong because the microbatch gradient is not what the optimizer sees.

---

## What clipping does NOT solve

Clipping is a symptom-damper, not a fix for pathological dynamics. At 70B+ scale, frontier labs stack clipping with:

1. **Skip-step on loss spike**: if train loss at step `t` is > k× its EMA, discard the gradient entirely (OLMo-2, Llama-3 techniques).
2. **Embedding-norm monitoring**: the first sign of a collapsing model is embeddings drifting to very large or very small norm; a circuit-breaker triggers earlier than gradient norm.
3. **Router z-loss in MoE**: forces routing logits to stay centred at zero to prevent expert-assignment blow-ups.
4. **Correct init with residual scaling** (see [[excerpts/weight-init]]): prevents the buildup that causes the spikes in the first place.

Clipping alone is necessary but not sufficient at 70B+. The 2013 paper's contribution was to prove it is always *part* of the answer.

---

## Common pitfalls

- **Threshold too low (`0.1`)**: optimizer never makes a meaningful step on difficult examples; loss plateaus at a bad value.
- **Per-tensor instead of global norm**: biases the optimizer, subtle perplexity penalty.
- **Clipping after optimizer.step**: completely useless; the step has already happened.
- **Clipping on local FSDP shards**: silent under-clipping, eventual divergence at scale.
- **Not logging pre-clip grad norm**: flying blind on training health.

---

## Connections

- [[excerpts/adam]] — AdamW's adaptive per-parameter scaling does not prevent exploding gradients at the global level; clipping sits on top of AdamW, not instead of it.
- [[excerpts/mixed-precision]] — `unscale → clip → step` is the mandatory ordering. Breaking the order is a popular silent-failure mode.
- [[excerpts/lr-schedules]] — warmup and clipping solve overlapping problems: warmup prevents the early-training spike, clipping handles the rare mid-training cliff. Both are needed.
- [[excerpts/weight-init]] — clipping masks bad init; the symptom is abnormally high grad norm in the first 100 steps.
- [[ch-01]] — parent chapter for training fundamentals.
