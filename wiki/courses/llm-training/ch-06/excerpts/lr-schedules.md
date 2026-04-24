---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/lr-schedules.md
source_url: https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/1608.03983 ; https://arxiv.org/abs/2404.06395
created_at: "2026-04-23"
---

# Excerpt: LR Schedules — scheduler-state resume and WSD fork-from-trunk semantics

**Source library:** `wiki/raw-data/llm-training/classics/lr-schedules.md`
**Heritage:** Vaswani 2017 (inverse-sqrt + warmup) → Loshchilov 2017 (cosine/SGDR) → Hu 2024 / DeepSeek-V3 / MiniCPM (WSD).

---

## Why this source anchors ch-06 §5.3 and §6

Ch-06 has two LR-schedule-critical sections:

1. §5.3 "LR-schedule off-by-one" — loading `scheduler.load_state_dict()` *after* the first `.step()` call silently advances the schedule past the saved point.
2. §6 "WSD stable-phase forkability" — the stable-phase checkpoint is the fork point from which multiple decay runs branch, each with different decay length.

This excerpt reconstructs the schedule shapes and the state each schedule persists, so you know exactly what `scheduler.state_dict()` is saving and why loading it in the wrong order breaks training.

---

## The four schedule families — reproduced

From the source (lines 31-56):

### 1. Linear warmup (always prepended)

```
lr(t) = peak_lr * t / warmup_steps          for t < warmup_steps
```

State: two ints — `t` (current step), `warmup_steps` (constant). Resume hazard: if `t` is not persisted, warmup restarts from zero on resume, clobbering peak LR for the next `warmup_steps`.

### 2. Cosine annealing (most common LLM default)

```
lr(t) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * (t - warmup) / (T - warmup)))
```

State: `t`, `warmup`, `T`, `peak_lr`, `min_lr`. The critical state is `t`. Cosine is smooth and derivative-bounded, so an off-by-one error in `t` produces a ~1/T fractional LR error — at T=100k, that is 0.001% instantaneous, but *systematic* across the rest of the run.

### 3. Inverse-square-root (Transformer original)

```
lr(t) = d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))
```

State: `t`, `warmup_steps`, `d_model`. The self-scaling with `d_model` is a nice property but the state is the same `t` — same off-by-one hazard.

### 4. WSD — Warmup-Stable-Decay

```
phase 1 (warmup):     lr ramps 0 → peak_lr      [0, warmup]
phase 2 (stable):     lr = peak_lr               [warmup, T - decay]
phase 3 (decay):      lr → min_lr                [T - decay, T]
                      shape: linear, cosine, or 1-sqrt
```

State: `t`, `warmup`, `T`, `decay`, `peak_lr`, `min_lr`, plus the decay shape enum. **The key advantage** (from the source, line 57):

> Decay phase is typically 10–20% of total tokens. Key advantage: the stable phase is *checkpoint-able* — you can fork off a 10%-decay run from any stable-phase checkpoint. This is how DeepSeek and MiniCPM produce many model variants from one pretraining trunk.

Notice: the stable phase is constant-LR. A trunk checkpoint saved during stable phase carries no LR-schedule history worth preserving (beyond `t` and the peak value) because replaying from the same `peak_lr` is trivial. This is why WSD is *checkpoint-friendly* in a way cosine is not — a cosine trunk checkpoint at step 60k of a 100k-step run is committed to the shape parameters of the original cosine; forking a different-length run requires rewriting `T` and accepting that the schedule shape has shifted at the fork point.

---

## The off-by-one — what actually happens

Ch-06 §5.3's description:

> You save at step 10,000 mid-cosine. You restart. If `lr_scheduler.step()` is called *before* the saved state is loaded, the scheduler advances to step 10,001 and the first optimizer step runs with step-10,001's LR — i.e. step 10,000 is skipped.

Trace through the PyTorch `LRScheduler.step()` semantics. At save time, `scheduler.last_epoch = 10000`, meaning "10000 `.step()` calls have been made; next LR is for step 10001." The `.state_dict()` captures `last_epoch = 10000`.

At load time, the canonical order is:

```python
# CORRECT
model, optimizer, scheduler = build()
scheduler.load_state_dict(state["sched"])   # last_epoch = 10000
# first training step:
optimizer.step()
scheduler.step()                             # advances to last_epoch = 10001, sets LR for step 10001
```

The buggy order:

```python
# WRONG — schedule advances before load
model, optimizer, scheduler = build()
scheduler.step()                             # advances to last_epoch = 1 (fresh scheduler)
scheduler.load_state_dict(state["sched"])   # restores last_epoch = 10000
optimizer.step()
scheduler.step()                             # advances to last_epoch = 10001
```

The buggy order *looks* correct because the final `last_epoch` is still 10001 — but a subtle variant appears when code sets `scheduler.last_epoch = saved_step` before `load_state_dict` and then advances:

```python
# WRONG — initial step call fires before load
scheduler.step()   # fresh scheduler: last_epoch was -1, now 0. LR for step 0 applied.
# ... model is now stepped once at warmup LR, not the saved cosine LR ...
scheduler.load_state_dict(...)
```

That first `optimizer.step()` after the errant `scheduler.step()` used the *wrong* LR. Over a 100k-step run, one step at the wrong LR is undetectable. But the source's line on WSD is instructive:

> Over a WSD decay phase that is 10–20% of the run ([[lr-schedules]]), one missed step at the start of decay shifts the entire decay shape by `1/decay_len` — still tiny but now *systematic*.

At the boundary `stable → decay`, the first few steps are the ones where the shape is most sensitive; a one-step shift propagates for the entire decay phase.

Ch-06 §7's defensive assertion:

```python
# ch-06/read.md, line 276
assert scheduler.last_epoch == state["step"], "LR scheduler off by one"
```

This is the *cheapest* possible guard. One integer comparison, no numerical noise. If it fires, you have the bug; if it passes, you do not. The absence of this assertion is the reason the bug ships to production.

---

## Why cosine-mismatch is small but non-zero

From the source:

> Empirical finding (Chinchilla, Hoffmann 2022): cosine length should match training length — over- or under-shooting cosine costs ~0.5–1% on validation perplexity.

This is a different failure mode — not a resume bug, but a *design* bug: you committed to a cosine with `T = 100k` then trained only 80k steps. The LR at step 80k is `min_lr + 0.5 * (peak - min) * (1 + cos(0.8 * pi))`, which is not `min_lr`. The model's last phase trained at a higher LR than cosine intended, never reaching the final flat-LR regime.

The resume variant: you save at step 80k intending to resume to 100k on a fresh cluster. The scheduler's state has `T = 100k`, `last_epoch = 80000`. Resume works fine — *if* the fresh cluster's `build()` also passes `T = 100k` to the scheduler constructor. If the new job's config says `T = 90k` (because "we're shortening the run"), `scheduler.load_state_dict()` restores `last_epoch = 80000` against the new `T = 90000`, and you now have 9 more steps of cosine before `min_lr`, compressed from the original 20k.

This is the **schedule-shape constant mismatch** — the scheduler's state-dict captures `last_epoch` but the shape constants (`T`, `peak_lr`, `min_lr`, `warmup`) are typically set at constructor time and *not* checked against the saved values. Best practice: check them explicitly on load.

---

## WSD fork-from-trunk — the operational mechanics

Ch-06 §6's claim:

> Every stable-phase checkpoint is a legitimate starting point for a fresh decay run — same trunk, different decay length, different downstream task.

What does "fork" mean mechanically?

1. Trunk run: WSD with `warmup=2000, stable_until=300000, decay_end=360000, decay_shape=linear`. At step 300000, the run transitions from stable (LR=`peak`) to decay (LR ramps down).
2. At step 300000 you save a checkpoint `trunk/step_00300000`. This is a *stable-phase endpoint*; its `scheduler.state_dict()` has `last_epoch=300000`, `T=360000` — committed to the original 60k-step decay.
3. You want to fork a *different* decay: 30k steps instead of 60k, cosine shape instead of linear. You cannot just load the saved scheduler — its `T` and shape are wrong.

The correct fork pattern:

```python
# Fork: new decay, 30k steps, cosine
model, optimizer, _ = build_from_saved_weights("trunk/step_00300000")
fork_scheduler = WSDScheduler(
    warmup=0,
    stable_until=0,            # already past stable; immediate decay
    decay_end=30000,           # 30k-step decay
    decay_shape="cosine",
    peak_lr=trunk_peak_lr,
    min_lr=0.0,
)
# Do NOT call fork_scheduler.load_state_dict() — the trunk scheduler's state is obsolete
# fork_scheduler.last_epoch = -1 (fresh scheduler, first step produces decay's step-0 LR)
```

Notice: the fork discards the trunk scheduler state entirely, keeps the model/optimizer state, and installs a fresh scheduler with a new shape. This is the ch-06 §6 "release artifact is the tree" claim operationalized — the trunk is shared, each decay child has its own scheduler identity.

For the ch-06 §7 `load_checkpoint` function, the trunk-checkpoint load path and the decay-fork load path diverge at the scheduler step:

```python
# Resume (continue same run)
scheduler.load_state_dict(state["sched"])       # keep trunk schedule

# Fork (new decay from trunk)
# deliberately skip scheduler.load_state_dict
# install fresh scheduler with new (T, shape)
```

The WSD source's one-line contrast with cosine:

> WSD's stable-phase checkpoints are the natural inputs for [[early-stopping-and-checkpointing]] / model averaging.

Cosine schedules produce *one* "best" point (the endpoint after full decay). WSD produces a family — each decay run from the trunk is an independent model, and averaging across multiple decay endpoints (the model-soup pattern from [[excerpts/early-stopping-and-checkpointing]]) is possible precisely because they share a trunk init.

---

## Warmup-is-required — the Adam bias-correction story

From the source (line 67):

> **Why warmup is essential with AdamW**: in the first few steps, `v_hat = (1-beta_2^t)^{-1} * v_t` is poorly estimated; the bias correction divides by a small number, inflating effective LR. Without warmup the first updates can NaN. Empirically: GPT-3 used 375M-token warmup; Llama-3 used 8000 steps; 0 warmup = divergence at 7B+.

For ch-06 resume semantics this creates one more hazard: a resume that *re-enters* warmup (because `last_epoch` was lost and defaults back to 0) runs a second warmup against an already-trained model. The LR ramps from 0 to peak over `warmup_steps`, which is the *correct* shape for a fresh run but catastrophic for a mid-training resume: the model sees a 2000-step window of near-zero LR, during which its optimizer state (`m`, `v`) drifts in directions set by a nearly-zero update magnitude. When peak LR returns, the first few full-LR steps are on a slightly stale optimizer state. Loss spike.

This is why ch-06 §5.3 puts the LR-schedule check alongside the data-iter and scaler checks: all three are "silent failure on resume from missing state" bugs, and they compound.

---

## The LR-schedule defaults table — for ch-06 instrumentation

From the source (lines 59-65):

| Setting | Pretrain | SFT | RL |
|---|---|---|---|
| Warmup steps | 2000–8000 | 100–500 (~3% of total) | 0–50 |
| Schedule | cosine or WSD | cosine or constant | constant |
| Peak LR | 3e-4 (1B) → 1.2e-4 (70B) → 8e-5 (405B) | 2e-5 (Llama-3 SFT) | 1e-6 to 1e-5 |
| Min LR | 0.1 × peak | 0.1 × peak | — |

Ch-06 §4's per-step logging tier includes `lr`. This table tells you the *expected* value at any step: if you log LR and see `3e-4` after a 70B resume where the peak should be `1.2e-4`, somebody copy-pasted a 1B config into the 70B run. The log is the audit trail; the table is the reference against which to audit.

For RL (ch-10, ch-11 in the outline) the schedule is often constant, so scheduler state is minimal and the off-by-one bug is moot. Ch-06's focus is pretraining and SFT, where the schedule is the main source of resume-sensitive LR state.

---

## Connections

- [[excerpts/early-stopping-and-checkpointing]] — WSD decay averaging needs checkpointed shape metadata, not just weights.
- [[excerpts/fsdp-sft]] — the SFT recipe's LR 1e-5 + cosine warmup 3% is the row one level down in the defaults table; DCP packages scheduler state alongside model state.
- [[excerpts/olmo-2]] / [[excerpts/olmo-3]] — OLMo 3's "Base → Dolmino → Longmino" is WSD-fork-from-trunk operationalized as a public model-flow.
- [[excerpts/llama-3]] — cosine decay with 3% warmup at 405B; scheduler state must persist across the 16K-GPU interruptions ([[excerpts/llama-3]] discusses the restart rate).
- [[excerpts/gradient-clipping]] — warmup is required partly because adaptive-optimizer bias correction produces early-step LR spikes; clipping is the second line of defense that also depends on resumed state.
- [[ch-06]] — §5.3 (off-by-one), §6 (WSD fork flywheel).
