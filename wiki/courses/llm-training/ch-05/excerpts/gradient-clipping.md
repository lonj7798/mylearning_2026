---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/gradient-clipping.md
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
---

# Excerpt: Global-Norm Gradient Clipping under FSDP Sharding

**Paper:** On the difficulty of training Recurrent Neural Networks
**Authors:** Pascanu, Mikolov, Bengio (ICML 2013)
**arXiv:** 1211.5063

**This excerpt focuses on the correctness requirement that forces you to use `FSDP.clip_grad_norm_` instead of `torch.nn.utils.clip_grad_norm_`.** The single-GPU clip semantics are covered in ch-01; this is the number-one silent-divergence bug in hand-rolled distributed training loops.

---

## The clip rule — what the paper actually says

Quoted from Technical Details:

> "`g_norm = sqrt(sum_i ||g_i||^2)` — over all parameter tensors. `if g_norm > c: g_i <- g_i * (c / g_norm)` — uniform rescale, direction preserved."

Mathematically:

```math
\|g\|_2 = \sqrt{\sum_{p \in \text{all params}} \|g_p\|_2^2} = \sqrt{\sum_p \sum_j g_{p,j}^2}
```

```math
g_p \leftarrow g_p \cdot \min\left(1, \frac{c}{\|g\|_2}\right)
```

The key property: **direction is preserved**, only magnitude is bounded. The paper's Figure 6 demonstrates that this one trick allows much higher learning rates without divergence.

**Notice:** "over all parameter tensors" is the load-bearing phrase for distributed. In a non-distributed setting, "all parameter tensors" means every tensor on this one GPU — trivial. Under FSDP, "all parameter tensors" means every tensor across every rank in the data-parallel group. The local computation on each rank is incomplete by itself.

---

## The distributed bug — why local clip under-counts by √N

Quoted directly from the Technical Details:

> "Distributed-training pitfall (FSDP / ZeRO-3): the global norm must be computed across **all shards** before scaling. Naively calling `clip_grad_norm_` on local shards under-counts the norm, leading to inconsistent scaling and silent divergence. Use `torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_` or the equivalent reduce-then-scale pattern."

**Derivation of the √N under-count.** Suppose the global gradient has per-parameter variance σ² and the parameters are sharded evenly across N ranks. Rank i holds P/N parameters. The local norm on rank i is:

```math
\|g^{(i)}\|_2 = \sqrt{\sum_{p \in \text{shard } i} \|g_p\|_2^2}
```

Assuming per-shard norms are roughly equal (true for uniformly random sharding):

```math
\|g^{(i)}\|_2 \approx \frac{\|g\|_2}{\sqrt{N}}
```

So `clip_grad_norm_(local_params, c)` fires only when `||g^{(i)}|| > c`, which happens when the *global* norm exceeds `c · √N`. Equivalently: the effective clip threshold is `c · √N` rather than `c`.

At N = 8: effective threshold is `2.83 · c`. With `c = 1.0`, the run never actually clips until the global norm exceeds 2.83 — which is exactly the regime where loss spikes start. The run **silently fails to clip** the events clipping was designed to catch.

**The fix:**

```python
# CORRECT — computes global norm via AllReduce on squared local norms
model.clip_grad_norm_(max_norm=1.0)

# WRONG — under-counts by √N
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

DeepSpeed's equivalent: `engine.clip_grad_norm_()` on the DeepSpeed engine object.

**Notice:** this is not a documentation footnote; it is a correctness contract. Frameworks that don't expose a distributed-aware clip API (some early Accelerate versions, hand-rolled FSDP wrappers) are shipping a silent bug.

---

## The internal AllReduce — what `FSDP.clip_grad_norm_` actually does

The correct implementation:

```python
# pseudo-code from torch/distributed/fsdp/fully_sharded_data_parallel.py
def clip_grad_norm_(self, max_norm: float) -> Tensor:
    local_norm_sq = torch.zeros(1, device=self.compute_device)
    for p in self.params:
        if p.grad is not None:
            local_norm_sq += p.grad.float().norm(2) ** 2

    # AllReduce across the data-parallel process group
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM, group=self.process_group)
    global_norm = local_norm_sq.sqrt()

    clip_coef = max_norm / (global_norm + 1e-6)
    if clip_coef < 1:
        for p in self.params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    return global_norm
```

Three critical details:

1. **Squared norms are summed** across ranks, then `sqrt` is taken — not norms summed directly. The formula `||g||_2 = sqrt(Σ_i ||g_i||_2²)` demands this order.
2. **One AllReduce** on a scalar per step. Negligible bandwidth (single fp32), but it is a synchronization point.
3. **The clip coefficient `max_norm / global_norm` is identical on every rank** (because both operands are global-reduced). Every rank applies the same scalar multiplier to its local shard — gradient direction is preserved globally.

**Notice:** if your training loop does gradient accumulation, the clip must fire on the *accumulated* gradient, not the per-micro-step gradient. Under FSDP, this means: accumulate gradients locally, ReduceScatter only at the end of accumulation (via `no_sync()` context on intermediate micro-steps), then clip on the fully-accumulated grad shard.

---

## TP / HYBRID_SHARD — the process-group subtlety

Under 2D parallelism (FSDP × TP), the clip must AllReduce over **both** the FSDP group and the TP group. TP shards split a single tensor across ranks (e.g., a row-parallel linear's weight is split across the TP group); the per-tensor norm contribution on any single rank is a fraction of the true tensor norm.

```math
\|W^{TP}\|_2 = \sqrt{\sum_{r \in \text{TP group}} \|W^{TP}_r\|_2^2}
```

So the correct global norm is:

```math
\|g\|_2 = \sqrt{\sum_{r_{fsdp}} \sum_{r_{tp}} \|g^{(r_{fsdp}, r_{tp})}\|_2^2}
```

torchtitan / Megatron handle this by AllReducing over the flat `world` process group, not the DP sub-group. Hand-rolled 2D parallelism that AllReduces only on the FSDP group under-counts by the TP factor — at TP=8 that's another √8 ≈ 2.83 on top of the FSDP √N under-count.

HYBRID_SHARD is similar: intra-node FULL_SHARD + inter-node REPLICATE. The clip must AllReduce over the intra-node shard group (to gather the local shard contributions) and then over the inter-node group (to gather the replicated norm contributions). FSDP's HYBRID_SHARD `clip_grad_norm_` handles both implicitly.

---

## Interaction with mixed-precision — the unscale-before-clip ordering

Quoted:

> "When loss-scaling (fp16), gradients are scaled by S. You must **unscale before clipping**, otherwise the threshold is meaningless. PyTorch's `GradScaler.unscale_(optimizer)` exists for this."

Under FSDP + fp16 (rare; bf16 preferred), the correct sequence is:

```
1. loss.backward()              # produces fp16 grads scaled by S
2. ShardedGradScaler.unscale_(optimizer)   # divides grads by S, casts to fp32
3. model.clip_grad_norm_(1.0)   # global-norm clip across all shards
4. ShardedGradScaler.step(optimizer)       # applies update
```

Inverting steps 2 and 3 makes the clip threshold `S` times too large; with typical `S = 2^15 ≈ 32768`, the clip never fires.

Under bf16: no loss scaling, so the sequence simplifies:

```
1. loss.backward()
2. model.clip_grad_norm_(1.0)
3. optimizer.step()
```

This is another reason bf16 is the 2025 default — one fewer correctness contract to get right.

See [[excerpts/mixed-precision]] for the full bf16 / fp16 comparison.

---

## Interaction with loss masking — token-count variance

From [[excerpts/loss-masking-prompt]]: under packed SFT, per-rank token counts `T_y^{(i)}` vary significantly. The per-rank loss `ℓ^{(i)} = (1 / T_y^{(i)}) Σ log π_θ(y_t)` produces gradients of variable magnitude:

```math
\|\nabla_\theta \ell^{(i)}\|_2 = \frac{1}{T_y^{(i)}} \left\| \sum_t \nabla_\theta \log \pi_\theta(y_t^{(i)}) \right\|_2
```

Ranks with small `T_y^{(i)}` produce larger local gradient norms. Summing squared local norms and taking sqrt (the correct global-norm formula) gives a global norm that is **not** equal to the gradient norm you'd get from a globally-averaged loss. The two agree only when `T_y^{(i)}` is constant across ranks.

**The fix.** Compute loss with `reduction="sum"` locally, AllReduce the sum and the token-count separately, then divide. This gives a globally-meaningful loss and a gradient whose norm matches the theoretical expectation. Then `FSDP.clip_grad_norm_` can be interpreted against the standard threshold of 1.0.

Without this fix: `FSDP.clip_grad_norm_` still computes the *true* global norm, but the norm's magnitude no longer corresponds to the standard `c = 1.0` regime. You'd need to tune the clip threshold per-dataset — a non-starter.

---

## Pre-clip grad norm as a training metric

Quoted:

> "Tracking `pre-clip grad_norm` is one of the most informative training metrics: a sudden 100x spike usually predicts an imminent loss-spike or NaN."

Under FSDP, `model.clip_grad_norm_` returns the pre-clip global norm — log this, not the per-rank local norm. The local norm is ~√N smaller and misses the √N factor of any spike.

**Notice:** In 70B+ pretraining, practitioners monitor not just grad norm but also the **skip-step rate** — if `FSDP.clip_grad_norm_` returns a norm > 10× the running EMA, skip the optimizer step entirely. This is the "loss-spike defense stack": clip (soft) + skip (hard) + embedding-norm monitoring. Llama-3 and OLMo-2 both document this pattern.

---

## Common pitfalls — replayed under FSDP

Quoted:

> "Clipping threshold too low (e.g. 0.1) → optimizer never makes a real step on hard examples; loss plateaus.
> Forgetting to clip after gradient accumulation → the accumulated gradient has different statistics from per-microbatch gradients; you must clip on the accumulated tensor."

Distributed translations:

- **Threshold too low under FSDP.** If you forget to switch from `torch.nn.utils.clip_grad_norm_` to `model.clip_grad_norm_`, you might over-compensate by setting `max_norm = 0.1` (thinking this matches your empirical grad-norm observations). Once you switch to the correct global-norm clip, the threshold of 0.1 is 10× too tight — loss plateaus.
- **Clip inside accum loop.** Under FSDP, calling `clip_grad_norm_` inside the accum loop triggers the AllReduce per micro-step. That's N - 1 extra synchronization points you don't need, and — worse — the intermediate grads have not yet been ReduceScatter'd (via `no_sync()` pattern), so the AllReduce sees partial gradients. Always clip once, after accumulation, outside the loop.

---

## Connections

- [[excerpts/fsdp-sft]] — FSDP ships `clip_grad_norm_` as a method on the FSDP-wrapped model.
- [[excerpts/mixed-precision]] — unscale-before-clip ordering; bf16 eliminates this concern.
- [[excerpts/adam]] — the clipped gradient is what AdamW's m, v update consumes.
- [[excerpts/loss-masking-prompt]] — token-count variance distorts per-rank local grad norms.
- [[excerpts/sequence-packing]] — packing amplifies token-count variance, amplifying the clip distortion.
- [[ch-05]] — synthesis and the silent-divergence bug list.
