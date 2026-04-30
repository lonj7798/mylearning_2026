---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/gradient-clipping.md
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
---

# Excerpt: Pascanu 2013 — gradient clipping as the pre-spike signal

**Source library:** `wiki/raw-data/llm-training/classics/gradient-clipping.md`
**Paper:** Pascanu, Mikolov, Bengio 2013, *"On the difficulty of training Recurrent Neural Networks"* (ICML).

---

## Why this source anchors ch-07

Pascanu's 2013 paper is the origin of the single most informative scalar in modern LLM training: `pre_clip_grad_norm`. The paper frames clipping as a fix for the exploding-gradient *problem*; 2025 practice treats the pre-clip value as a *signal* that pre-announces loss spikes, NaNs, and embedding-norm drift 1–5 steps ahead. Ch-07 relies on that predictive property in §1 (the liveness-assertion boundary), §2 (the diagnostic tree's top branch), and §5 (the FSDP global-norm pitfall as a hang precursor).

The source's key line for ch-07 is explicit:

> *"Tracking `pre-clip grad_norm` is one of the most informative training metrics: a sudden 100x spike usually predicts an imminent loss-spike or NaN."*

That sentence is why the pre-clip norm sits at the root of every diagnostic tree in the chapter.

---

## The formula — global-norm, direction preserving

From the source (Technical Details):

```
g_norm = sqrt(sum_i ||g_i||^2)            # over all parameter tensors
if g_norm > c:
    g_i <- g_i * (c / g_norm)             # uniform rescale, direction preserved
```

Notice: the rescale is *uniform*. Every parameter tensor shrinks by the same factor `c / g_norm`. Direction is preserved; only magnitude is bounded. Ch-07 §2's "plateau because clip threshold is too low" failure is the symmetric form of this: if `g_norm > c` for every step, the effective optimizer step size is fixed at `c / g_norm` regardless of the geometry of the loss, so the learning-rate schedule is meaningless and the run plateaus.

The source is explicit about two wrong alternatives:

- **Clip-by-value** (`g_i <- clip(g_i, -c, c)` element-wise): distorts descent direction. Appears in old Keras code; never use.
- **Per-tensor norm clip** (loop over parameters, clip each independently): biases the optimizer toward small tensors. PyTorch's `clip_grad_norm_(p, c)` called per-parameter silently does this if you pass `model.parameters()` inside a loop instead of as a single iterable.

Ch-07 §5's clip-every-step branch (`clipped_fraction == 1`) has a third variant specific to distributed training:

> *"Distributed-training pitfall (FSDP / ZeRO-3): the global norm must be computed across all shards before scaling. Naively calling `clip_grad_norm_` on local shards under-counts the norm, leading to inconsistent scaling and silent divergence."*

The fix: `torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_` or DeepSpeed's utility. Hand-rolled per-shard clipping reports a norm that is correct locally but a factor of `√N` off globally — the reported `clipped_fraction` is therefore misleading in exactly the way that hides the bug.

---

## Unscale before clip — the mixed-precision ordering

From the source:

> *"Mixed-precision interaction: when loss-scaling (fp16), gradients are scaled by `S`. You must unscale before clipping, otherwise the threshold is meaningless. PyTorch's `GradScaler.unscale_(optimizer)` exists for this."*

The canonical ordering is three lines, and all three must appear in this order:

```python
scaler.unscale_(optimizer)                                  # divide grads by S
grad_norm = model.clip_grad_norm_(max_norm=cfg.max_norm)    # compare to real c
scaler.step(optimizer)                                      # skip if any grad was inf/NaN
scaler.update()
```

Get this wrong and the clip threshold is off by `S ≈ 2^15`. The symptom is that `clipped_fraction` is structurally 0 or 1 rather than modulated — either every step's scaled norm is below the threshold (S = 2^15 makes `c = 1.0` correspond to `S·c = 32768` in scaled space; normal grads never exceed that) or every step is above it. Ch-07 §2's plateau branch includes this as the fp16-specific variant of dead-pipeline plateau.

Notice: bf16 has no scaler, so this entire class of bug vanishes under bf16. This is one of the *operational* reasons [[excerpts/mixed-precision]] argues bf16 is the 2025 default even though it has fewer mantissa bits than fp16.

---

## Modern defaults and the three-layer spike-mitigation stack

The source gives the canonical 2025 thresholds:

> *"Modern LLM defaults — Pretraining (GPT/Llama/Qwen lineage): `max_grad_norm = 1.0`. SFT: typically 1.0; sometimes 0.5 for noisy synthetic data. RL (PPO/GRPO): 0.5–1.0 on the policy gradient."*

These are the hyperparameters the diagnostic tree in ch-07 §2 assumes. A value like `max_grad_norm = 0.1` is the plateau cause; `max_grad_norm = 10.0` defeats the whole purpose (clipping will only trigger on explosion, not on the pre-spike signal).

The source's *"Connections"* section names the stack ch-07 §2 calls the "Llama-3 / OLMo-2 mitigation stack":

> *"Loss spikes in pretraining: the standard Llama-3 / OLMo-2 mitigation stack is: (1) global-norm clip 1.0, (2) skip-step on loss-spike, (3) embedding-norm monitoring. Clipping alone is necessary but not sufficient at 70B+ scale."*

Notice the ordering. Clip is the first line of defense but it is insufficient *by itself* at the 70B+ scale — the pre-clip norm was already past the threshold when the spike occurred, so the clip limited damage rather than preventing it. Skip-step (layer 2) adds the decision "if loss jumped past 5σ, discard the gradient entirely, don't even apply the clipped version." Embedding-norm monitoring (layer 3) runs slower (every checkpoint) and catches the cumulative drift that three clipped-but-large steps can produce before skip-step fires.

The persistence requirements of the three layers differ, which is ch-06's instrumentation topic: layer 1 is stateless, layer 2 needs ~200 steps of loss history, layer 3 needs multi-checkpoint embedding-norm series. Each layer is useless if its state is reset at resume.

---

## The gradient-accumulation subtlety

From the source:

> *"Forgetting to clip after gradient accumulation → the accumulated gradient has different statistics from per-microbatch gradients; you must clip on the accumulated tensor."*

This is a subtle but common bug. The correct ordering under accumulation:

```python
for micro_batch in accumulated:
    loss = model(**micro_batch).loss / grad_accum
    loss.backward()
    # NO CLIP HERE — grads are incomplete
grad_norm = model.clip_grad_norm_(cfg.max_norm)    # clip the sum
optimizer.step()
```

Clipping per microbatch would apply the threshold to each 1/N piece of the gradient, then those pre-clipped pieces accumulate, and the total may still exceed `c` — but now without any clip-event logged, because the accumulation is post-clip. Ch-07 §5's clip-every-step diagnostic is confused if clip is applied at the wrong point; `clipped_fraction` is a per-step metric that is only well-defined when the clip happens exactly once per optimizer step.

---

## The predictive-signal property — why it works

Pascanu's geometric framing explains why the pre-clip norm is predictive:

> *"Geometric intuition (the 'wall in error surface') explaining why a single bad gradient can destroy hours of progress."*

In that framing, the gradient of a loss landscape with a "cliff" explodes when the current iterate approaches the cliff face. The cliff is a function of the *data* (the current batch triggers it) but also of the *trajectory* — the iterate had to get close to the cliff first. The pre-clip norm is the dot product of the gradient with the cliff-normal; it grows several steps before the trajectory reaches the edge of the cliff, because the gradient is already pointing at increasingly large values as the iterate approaches.

Operationally this translates into the ch-07 §2 prediction: a 100× spike in `pre_clip_grad_norm` at step `k` predicts a loss spike at step `k+1` through `k+5`. The mitigation stack above buys you those 1–5 steps to skip, log, or roll back before the loss actually moves.

---

## What to take from Pascanu for ch-07

1. **`pre_clip_grad_norm` is the earliest signal of almost every non-hang failure.** Log it per step.
2. **Unscale → clip → step.** Under fp16, wrong ordering silently disables clipping; under bf16, the bug can't exist.
3. **FSDP needs the framework's global-norm utility.** Hand-rolled per-shard clipping produces a norm that is wrong by √N.
4. **Threshold too tight = plateau; threshold too loose = defeat the purpose.** Defaults are 1.0 for pretraining/SFT, 0.5–1.0 for RL.
5. **Clip is one layer of a three-layer stack.** Skip-step and embedding-norm monitoring are the other two. All three require persisted state across resumes (ch-06).

---

## Connections

- [[excerpts/mixed-precision]] — the unscale-before-clip ordering rule; bf16 as the ordering-bug eliminator.
- [[excerpts/adam]] — Adam's per-parameter scaling does not substitute for clipping; early training still needs it.
- [[excerpts/fsdp-sft]] — the global-norm utility is part of the FSDP surface; §5 hang branch has the distributed variant.
- [[excerpts/karpathy-training-neural-net-recipe]] — "monitor and clip the gradient norm" is listed as a non-negotiable.
- [[excerpts/olmo-2]] — provides the three-layer spike-mitigation stack that this source's Connections section names.
- [[ch-07]] — §1 (NaN prediction), §2 (diagnostic tree root), §5 (clipped_fraction diagnostic).
