---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/gradient-clipping.md
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
---

# Excerpt: Gradient Clipping — clip-norm history as the earliest spike-detection signal

**Source library:** `wiki/raw-data/llm-training/classics/gradient-clipping.md`
**Paper:** Pascanu, Mikolov, Bengio 2013, "On the Difficulty of Training Recurrent Neural Networks" (ICML)

---

## Why this source anchors ch-06 §4

Ch-06 §4's per-step instrumentation tier names one metric first: `pre_clip_grad_norm`. Ch-06 §1 lists grad-norm history as the instrumentation audit trail for the seven-item state list. Both choices trace to one sentence from this paper's "Common pitfalls" section:

> Tracking `pre-clip grad_norm` is one of the most informative training metrics: a sudden 100x spike usually predicts an imminent loss-spike or NaN.

This excerpt explains *why* — what the clip-norm actually measures, why pre-clip rather than post-clip, and how the log becomes a load-bearing piece of instrumentation that must survive resumes.

---

## The clipping operator — what it does, what it doesn't

From the source (lines 29-34):

```
# gradient-clipping.md, lines 30-34
g_norm = sqrt(sum_i ||g_i||^2)            # over all parameter tensors
if g_norm > c:
    g_i <- g_i * (c / g_norm)             # uniform rescale, direction preserved
```

Notice the **uniformity** of the rescale. Every parameter tensor is scaled by the same factor `c / g_norm`. The *direction* of the aggregate gradient (the unit vector in parameter space) is preserved exactly; only the *magnitude* is bounded by `c`.

This is the precise property that makes clipping composable with Adam/AdamW: Adam's per-parameter adaptive scaling operates on the clipped gradient, so the direction the optimizer follows is the same direction the model would have followed without clipping — just with a bounded step magnitude. If clipping *distorted* the direction (as element-wise clip-by-value does), Adam's direction estimate would be systematically biased toward the clipped axes.

The source's contrast with clip-by-value:

> **Clip-by-value** (`g_i <- clip(g_i, -c, c)` element-wise): distorts the descent direction; rarely used.

And per-tensor clipping:

> **Per-tensor norm clip** (PyTorch `clip_grad_norm_(p, c)` looped per parameter): biases optimizer toward small tensors; almost never desired.

The right call in every modern LLM is the *global* norm clip — one scalar computed across all parameters, one rescale factor applied uniformly.

---

## The cliff intuition — why one bad step destroys hours

The source's Figure 1 reference (line 25):

> **Figure 1** (Error surface with a "cliff"): visual intuition for why a normal gradient step lands the parameters far from the manifold; clipping limits the leap.

The cliff geometry is standard in the 2013 paper but worth spelling out. Near a loss cliff, the gradient magnitude explodes — the *same* direction, but suddenly 100× larger. A vanilla SGD step takes the parameters 100× farther than usual, landing somewhere with no training signal (often NaN or a region of the loss surface where the model has effectively forgotten its training).

Clipping at `c=1.0` caps the per-step motion in parameter space. The direction is right; the distance is bounded. You get one small step, the spike passes, the next step's gradient returns to normal. This is why LLM pretraining defaults are `max_grad_norm = 1.0`:

> Pretraining (GPT/Llama/Qwen lineage): `max_grad_norm = 1.0`.
> SFT: typically `1.0`; sometimes `0.5` for noisy synthetic data.
> RL (PPO/GRPO): `0.5–1.0` on the policy gradient; reward spikes during rollout can produce 10x norm bursts that clipping absorbs.

---

## The FSDP subtlety — why `clip_grad_norm_` must be sharding-aware

From the source (line 44):

> **Distributed-training pitfall (FSDP / ZeRO-3)**: the global norm must be computed across **all shards** before scaling. Naively calling `clip_grad_norm_` on local shards under-counts the norm, leading to inconsistent scaling and silent divergence. Use `torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_` or the equivalent reduce-then-scale pattern. Same issue exists with DeepSpeed's ZeRO; both frameworks ship a correct utility you should call instead of writing your own.

Under FSDP `FULL_SHARD`, each rank holds `|W|/N` of the parameters. Its local gradient norm is `||g_local||` which, if each shard's gradients are roughly isotropic, is about `||g_global|| / sqrt(N)`. Calling naive `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` per-rank under-counts the global norm by `sqrt(N)`, so each rank clips to a threshold that is effectively `sqrt(N)` too lax. The rescale factor each rank applies is *different* from its neighbors (different local norms) — and because FSDP's next step AllGathers the clipped params back, the aggregate model has been stretched in an inconsistent way.

`FSDP.clip_grad_norm_` does the right thing: each rank computes `||g_local||²`, an AllReduce sum gives `||g_global||`, each rank rescales by the *same* `c / ||g_global||`. Direction preserved globally, magnitude bounded globally.

For ch-06 this matters because **the value logged as `pre_clip_grad_norm` is this global norm**. If your logger calls `torch.nn.utils.clip_grad_norm_` (buggy) instead of `FSDP.clip_grad_norm_` (correct), your logs are `sqrt(N)`-underscaled and the "100× spike" threshold is miscalibrated. On resume to a different world size, the logged pre-clip norm shifts by `sqrt(N_old/N_new)` — a purely instrumentation artifact that looks like a training-regime change.

---

## The mixed-precision ordering — `unscale → clip → step`

From the source (line 46):

> **Mixed-precision interaction**: when loss-scaling (fp16), gradients are scaled by `S`. You must **unscale before clipping**, otherwise the threshold is meaningless. PyTorch's `GradScaler.unscale_(optimizer)` exists for this.

Cross-link: [[excerpts/mixed-precision]] expands on this. For ch-06's purposes the ordering rule is a resume-sensitive interaction: if scaler state is dropped on resume (§5.2) so `S` restarts at `2^15`, the unscale divides by the wrong factor, the pre-clip norm reads 8× smaller than it truly is, and the clip threshold is effectively 8× tighter. The log shows suspiciously-calm grad norms for ~2000 post-resume steps. If you resume-test against a "grad norm looks normal" heuristic, the scaler-state bug passes the test.

---

## Clip-norm as a pre-spike signal — the load-bearing claim

The source's one-liner again:

> a sudden 100x spike usually predicts an imminent loss-spike or NaN.

This is an *empirical* claim, not a theoretical one. The story is: a data point with pathological structure (a corrupt example, an unusually long sequence, a tokenization failure producing a very high-probability outlier token) produces a single forward pass with an anomalous loss, which produces gradients anomalously oriented and anomalously large. Clipping bounds the step size — so the immediate step does not blow up — but the optimizer's `m` and `v` accumulators absorb the anomaly. Over the next 1–5 steps the adaptive LR rises (because `v`'s denominator is briefly smaller than the gradient it must scale) and the model takes amplified steps in the anomalous direction. By step `spike_t + 5`, the loss spikes.

So the `pre_clip_grad_norm` log is a ~5-step *leading indicator*. That buys time for the spike-recovery playbook from [[excerpts/early-stopping-and-checkpointing]]:

> 1. Roll back to last clean checkpoint.
> 2. Skip the offending data shard (reorder).
> 3. Resume with same LR.

Without the pre-clip log, you only see the post-spike loss jump and you do not know which data shard to blame. With the log, the spike is timestamped and traceable: `pre_clip_grad_norm = 112` at step `k`, loss spike at step `k+4`; the batch at step `k` came from shard `s`; reorder shard `s` into the next cycle.

This is the payload of ch-06 §1's claim:

> **Grad-norm history is not ornamental.** [[gradient-clipping]] and [[olmo-2]] both make the point: a 100× `pre_clip_grad_norm` spike predicts a loss spike 1–5 steps ahead. That predictive signal only works if the log survives the resume. If your log ring-buffer resets at every resume, you lose the ability to distinguish "new instability" from "pre-existing drift I carried across the last crash."

---

## Why post-clip is useless

A short algebraic observation: post-clip norm is `min(pre_clip, c)`. At threshold `c = 1.0`, whenever there is any instability, the post-clip value is exactly `1.0` — a constant, carrying no information. The spike signature (the 100× value) is invisible.

This is why ch-06 §4 specifies `pre_clip_grad_norm` and why the code block (line 157) is:

```python
grad_norm = model.clip_grad_norm_(max_norm=1.0).item()   # pre-clip, global
```

`clip_grad_norm_` in PyTorch returns the pre-clip global norm as its return value. Logging `grad_norm` (the return value) captures the signal; logging the post-clip state of `.grad` would capture only the constant `1.0`.

---

## The OLMo 2 spike-mitigation stack — three layers

From the source (line 56):

> **Loss spikes in pretraining**: the standard Llama-3 / OLMo-2 mitigation stack is: (1) global-norm clip 1.0, (2) skip-step on loss-spike, (3) embedding-norm monitoring. Clipping alone is necessary but not sufficient at 70B+ scale.

Three layers of defense, each triggered by a different metric:

1. **Clip grad-norm at 1.0** — instantaneous, per-step, absorbs the spike into a bounded step.
2. **Skip-step on loss-spike** — triggered by `loss > running_mean * threshold` or `loss.isnan()`; discards the batch entirely.
3. **Embedding-norm monitoring** — `||W_embed||` drift over a window; catches slow-burn instability before it manifests as a loss spike.

Layer (2) requires a loss history buffer (N steps back). Layer (3) requires an embedding-norm history buffer (checkpoint-scale, rarer cadence). Both buffers are *log* state. If the log state does not survive a resume, the OLMo 2 stack has effectively been reset — and the first resumed spike may go undetected because the running mean has too few samples to call it an outlier.

Ch-06 §1's row "grad-norm / loss history" is the *union* of all three buffers. Persist it durably; reset it never.

---

## Karpathy's framing — why it is a "non-negotiable"

From the source (line 57):

> **Karpathy's recipe** ([[karpathy-training-neural-net-recipe]]) lists "monitor and clip the gradient norm" as a non-negotiable.

[[excerpts/karpathy-training-neural-net-recipe]] expands the "monitor and clip" rule. The operative word is **monitor**. Clipping without monitoring is a silent-safety-net — you are protected from crashes, but you do not learn *when* the protection fired. Monitoring without clipping is data without protection — you learn about spikes after they have damaged the run. Ch-06's instrumentation discipline demands both: clip to survive, monitor to diagnose.

---

## Connections

- [[excerpts/mixed-precision]] — `unscale → clip → step` ordering; scaler-state drop silently re-scales the clip threshold.
- [[excerpts/early-stopping-and-checkpointing]] — spike-recovery playbook that the `pre_clip_grad_norm` log makes possible.
- [[excerpts/fsdp-sft]] — `FSDP.clip_grad_norm_` is the correct all-shards norm; naive `torch.nn.utils` version is sharding-unaware.
- [[excerpts/olmo-2]] — the three-layer stack (clip + skip-step + embedding-norm) operationalized at 7B/13B/32B.
- [[excerpts/karpathy-training-neural-net-recipe]] — "monitor and clip" as non-negotiable.
- [[ch-06]] — §1 (grad-norm history row), §4 (per-step tier), §5.2 (scaler interaction).
