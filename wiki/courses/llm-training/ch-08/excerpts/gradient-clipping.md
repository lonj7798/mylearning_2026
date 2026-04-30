---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/gradient-clipping.md
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
---

# Excerpt: Gradient Clipping — the ordering-line invariant ch-08 enforces

**Source library:** `wiki/raw-data/llm-training/classics/gradient-clipping.md`
**Paper:** Pascanu, Mikolov, Bengio 2013, "On the difficulty of training Recurrent Neural Networks"

---

## Why this source anchors ch-08 §5 and the ordering-line memo pick

Ch-08's third silent-failure line is *ordering*: the relative positions of `backward`, `clip`, `optimizer.step`, `scheduler.step`. This paper is where global-norm clipping was introduced as the direction-preserving fix, and it is the single reason `max_grad_norm = 1.0` survives as the universal default in 2025 LLM training. The lab's concept-mapping §5–§6 ties every instruction in the modern recipe back to a constraint in this paper.

---

## The attested clipping rule — what TRL / HF implement

From the source (lines 30-34):

```
g_norm = sqrt(sum_i ||g_i||^2)            # over all parameter tensors
if g_norm > c:
    g_i <- g_i * (c / g_norm)             # uniform rescale, direction preserved
```

> The key property: **direction is preserved**, only magnitude is bounded.

This is the invariant ch-08 cares about. `torch.nn.utils.clip_grad_norm_`, FSDP's `FullyShardedDataParallel.clip_grad_norm_`, and Accelerate's `accelerator.clip_grad_norm_` all implement exactly this. Any alternative — per-tensor norm clip, clip-by-value, local-shard clip — violates direction preservation and degrades training silently.

The lab's default `max_grad_norm=1.0` is the row below:

From the source (lines 39-42):

> - Pretraining (GPT/Llama/Qwen lineage): `max_grad_norm = 1.0`.
> - SFT: typically `1.0`; sometimes `0.5` for noisy synthetic data.
> - RL (PPO/GRPO): `0.5–1.0` on the policy gradient; reward spikes during rollout can produce 10x norm bursts that clipping absorbs.

Ch-08's lab stays at 1.0. If the learner's data mix is later revealed to be noisy synthetic, 0.5 is a justified one-line change.

---

## The FSDP pitfall — the one line ch-08's memo flags

From the source (line 44):

> **Distributed-training pitfall (FSDP / ZeRO-3)**: the global norm must be computed across **all shards** before scaling. Naively calling `clip_grad_norm_` on local shards under-counts the norm, leading to inconsistent scaling and silent divergence.

This is the attested silent-failure mode. Ch-08's trainer-map HTML pins `clip_grad_norm_` on the green band (ordering line) and the detail panel names this case:

> Per-tensor clip_grad_norm_(p, c) looped over params biases the optimizer toward small tensors. Also: custom code missing the FSDP dispatch under-counts the global norm and clips incorrectly.

Both variants of the bug are live hazards in 2025. A learner who writes a custom logging callback that calls `torch.nn.utils.clip_grad_norm_` "just to re-compute the norm for logging" triggers the second variant — the utility recomputes the local-shard norm, which is off by a √N factor from the global, and any downstream decision based on that log is wrong.

---

## The mixed-precision ordering clause — why `backward → clip → step` is load-bearing

From the source (line 46):

> **Mixed-precision interaction**: when loss-scaling (fp16), gradients are scaled by `S`. You must **unscale before clipping**, otherwise the threshold is meaningless. PyTorch's `GradScaler.unscale_(optimizer)` exists for this.

Under bf16 (the ch-08 default) there is no scaler and no unscale step — but the ordering still matters:

```
backward → clip → step → scheduler → zero_grad
```

Each transition encodes one thing that can silently break:

- `backward → clip` skipped → grad-norm never bounded; one outlier example produces a 100× step and the run diverges.
- `clip → step` swapped → clipping rescales a gradient that has already been applied; the clip is a no-op on this step, correct on the next, so the clip is *eventually* applied but lagged by one step — at high LR, one step of unclipped gradient is enough to diverge.
- `step → scheduler` swapped → the scheduler's `last_epoch` advances before `optimizer.step` consults `lr`, so step `k` runs at step-`(k+1)`'s LR. Ch-06 §5.3 shows this as cosine-off-by-one.
- `zero_grad` forgotten → gradients from step k leak into step k+1, effectively halving the gradient-accumulation denominator.

Ch-08's memo picks §1 lists the `scheduler.step()` order as a top-three candidate because it is the case where the failure is most subtle (first step's LR is wrong; curve looks fine).

---

## The accumulation clause — why clipping must be after grad-accum not inside it

From the source (line 50):

> Forgetting to clip after gradient accumulation → the accumulated gradient has different statistics from per-microbatch gradients; you must clip on the accumulated tensor.

HF `Trainer.training_step` implements this correctly: `loss.backward()` is called once per microbatch (accumulating into `param.grad`), and `clip_grad_norm_` is called once per optimizer step, after the last microbatch. The lab's config `gradient_accumulation_steps=16` is only safe because of this. A learner who writes a custom training loop and clips per-microbatch silently under-clips — the clip fires 16× with threshold 1.0, but the effective threshold on the accumulated gradient is `16 × 1.0 = 16`.

Ch-08 does not ask the learner to write a custom loop. This is exactly why.

---

## The `pre_clip_grad_norm` as diagnostic — the metric the memo must log

From the source (line 51):

> Tracking `pre-clip grad_norm` is one of the most informative training metrics: a sudden 100x spike usually predicts an imminent loss-spike or NaN.

Ch-08's §Deliverables memo §2 ("What you instrumented") mandates `pre_clip_grad_norm` as one of the four per-step metrics. This is not optional. The *post-clip* norm is trivially `min(raw, 1.0)` and carries no spike signal; the *pre-clip* norm is the earliest warning the trainer has.

HF `Trainer` logs `grad_norm` via `accelerator.clip_grad_norm_`'s return value, which is documented to be the pre-clip norm. If a learner writes a custom callback that reads `param.grad.norm()` *after* the clip call, they log the post-clip norm and the spike signal is lost. Ch-08 flags this in §4 (mixed precision) as "custom callback bug" and in §5 (clipping) as a direct source of diagnostic blindness.

---

## Connections

- [[excerpts/fsdp-sft]] — the FSDP sharded-clip dispatch contract.
- [[excerpts/mixed-precision]] — the `unscale → clip → step` ordering under fp16; under bf16 the unscale is absent but the rest holds.
- [[excerpts/karpathy-training-neural-net-recipe]] — "monitor and clip the gradient norm" as a non-negotiable from the 2019 recipe, unchanged in 2025.
- [[ch-01]] — concept-level intro to clipping.
- [[ch-06]] — grad-norm history as a checkpointed log; the 100× spike predictor loses value if the log resets on resume.
- [[ch-08]] — §5 (concept map), §6 (ordering), §Deliverables (memo must log pre-clip norm), figures/trainer-map.html ("clip_grad_norm_" node).
