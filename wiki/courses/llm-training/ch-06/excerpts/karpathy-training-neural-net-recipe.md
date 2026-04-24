---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md
source_url: https://karpathy.github.io/2019/04/25/recipe/
created_at: "2026-04-23"
---

# Excerpt: Karpathy's Recipe — "training fails silently, so instrument everything and verify resumes bit-exactly"

**Source library:** `wiki/raw-data/llm-training/blogs/karpathy-training-neural-net-recipe.md`
**Blog post:** Andrej Karpathy 2019, "A Recipe for Training Neural Networks."

---

## Why this source is ch-06's organizing axiom

Ch-06 opens with:

> The chapter's organizing principle is Karpathy's maxim: **training fails silently, so instrument everything and verify resumes bit-exactly** ([[karpathy-training-neural-net-recipe]]). Every design choice below is derived from that one rule.

Everything in ch-06 — the seven-item checkpoint state, the three instrumentation tiers, the cross-resume silent-failure catalog — is downstream of this maxim. This excerpt pulls out the specific recipe items that bind instrumentation discipline to checkpointing and resume.

---

## The leaky-abstraction framing

From the source (line 18):

> The framing: **"neural net training is a leaky abstraction"** — unlike standard SWE where libraries compose, ML components silently corrupt each other's assumptions.

Notice: Karpathy does not say "training is buggy." He says "training is a *leaky abstraction*." The distinction matters. A leaky abstraction is one where the abstraction barrier does not hold — you cannot treat an `Optimizer` object or a `DataLoader` object as a black box, because the internal state of each spills into the other. Adam's `v` accumulator's behavior depends on the gradient distribution, which depends on the loss distribution, which depends on the data order, which depends on the dataloader's seed, which depends on the RNG state you forgot to save.

For ch-06 this is the *reason* the seven-item checkpoint state is seven items and not one. You cannot encapsulate training state into "model weights" because the abstraction of "model" does not own Adam's state, the dataloader's state, the scheduler's state, or the RNG. They are all separate leaks that the checkpoint must catch.

---

## The six-step recipe — operational structure

From the source (lines 32-71):

### 1. Become one with the data
### 2. Set up the end-to-end training/evaluation skeleton + get dumb baselines
### 3. Overfit
### 4. Regularize
### 5. Tune
### 6. Squeeze out the juice

Steps 2 and 3 are the ones with direct ch-06 payoff.

**Step 2 — skeleton and baselines (the bit-exact resume test lives here):**

> Build the smallest end-to-end system: tiny model, dumb baseline (linear / constant / copy-input), full eval loop. Verify:
> - Initial loss ≈ expected (`ln(K)` for uniform classification; entropy of target distribution for regression).
> - `model.eval()` vs `model.train()` give same outputs when no dropout/BN is active.
> - The model can't improve by more than a known amount on the simplest proxy task.
> - **Fix random seed everywhere; turn off every non-essential feature.**

That last bullet — "fix random seed everywhere" — is the first piece of infrastructure a ch-06-compliant trainer needs. You cannot run the bit-exact resume test (§3) if the seed is not fixed. You cannot verify that `model.eval()` is deterministic if dropout masks are being drawn from an unseeded RNG.

**Step 3 — overfit a single batch:**

> Crank model capacity until you can **overfit one batch** to near-zero loss. If you cannot, the pipeline is broken — stop everything and debug.
> Quote: "if you can't overfit a single batch, you can't overfit the training set."

This is the *second* foundational sanity test. Ch-06's relationship: the single-batch overfit produces a training loss trajectory that is bit-determinable (same batch, same forward, same loss each iteration). If you save a checkpoint mid-overfit and resume, the trajectory should continue *exactly* — that is the strongest possible bit-exact resume test. Anything wrong with the checkpoint shows up as a divergent single-batch trajectory.

---

## The Famous-Maxims table — where checkpoint discipline is canonically named

From the source (lines 74-85):

| Maxim | What it protects against |
|---|---|
| "Neural net training fails silently." | Assuming green curves = correct code. |
| "Be paranoid about `model.train()` vs `model.eval()`." | BN/dropout active at eval. |
| "Init well." | Loss too high at step 0; bad convergence. |
| "Visualize just before the net." | Mis-normalized inputs; label mismatches. |
| "Generalize a special case." | Hard-to-debug loops; always write the `N=1` case first. |
| "Use backprop to chart dependencies." | Data leakage bugs; find them by setting one example's loss to zero and checking that no gradient flows to other examples. |
| "Monitor and clip gradient norms." | Silent divergence. |
| "Use a constant LR for sanity; schedule last." | Hyperparameter confounding. |
| "Adam `3e-4` is a safe default." | HP-sensitivity in early experiments. |
| "Don't be a hero." | Premature novelty; wasted months. |

Three rows deserve direct ch-06 mapping:

### "Neural net training fails silently" → every silent-failure mode in ch-06 §5

Row 1 is the *mission statement*. Every silent-failure mode in ch-06 §5 (data-iter desync, scaler-state drop, LR off-by-one, optimizer state partial load, embedding-norm drift) is a case study in silent failure. The reason they all require affirmative instrumentation (not just happy-path code) is this row.

### "Monitor and clip gradient norms" → ch-06 §4 per-step tier + [[excerpts/gradient-clipping]]

Row 7 is the direct source of the `pre_clip_grad_norm` rule. Ch-06 §4 formalizes it as a per-step log. [[excerpts/gradient-clipping]] makes it a spike-prediction signal. The Karpathy maxim is the *why* — without it, the clip just silently rescales and you never learn which step was the near-miss.

### "Use a constant LR for sanity; schedule last" → ch-06 §5.3 + [[excerpts/lr-schedules]]

Row 8 is why LR-schedule off-by-one bugs are so common: schedules are the *last* thing added, so the infrastructure around them (state persistence, resume ordering) is the *newest* code and the least-tested. Ch-06 §5.3 catches this with the `scheduler.last_epoch == saved_step` assertion — a bit of paranoia that would be overkill in an image-classification CNN but is mandatory in a 16K-GPU run where one missed step at a scheduler transition shifts the whole decay shape.

---

## The bit-exact resume rule — its explicit appearance

The source file does not use the phrase "bit-exact resume" verbatim, but it is implicit in the "fix random seed everywhere" maxim and the "neural net training fails silently" framing. Ch-06 §3 quotes Karpathy as saying:

> Karpathy's single most-quoted checkpoint rule ([[karpathy-training-neural-net-recipe]]): *"check that resume produces bit-exact loss."*

This is ch-06's own phrasing derived from Karpathy's broader methodology. The argument structure:

1. Training fails silently (maxim 1).
2. Silent failures can be caught only by strong-form checks, not curve inspection.
3. The strongest check of "did the checkpoint contain everything needed?" is bit-exact reproduction.
4. Therefore, after every checkpoint change, verify bit-exact resume.

This is the payload of ch-06 §3's code block:

```python
# ch-06/read.md, lines 110-117
torch.use_deterministic_algorithms(True)       # raises on non-det kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark      = False
# Per-rank seed from global seed + rank id
torch.manual_seed(cfg.seed + rank)
torch.cuda.manual_seed_all(cfg.seed + rank)
```

Every line is a Karpathy-derived defensive move. Deterministic algorithms? Because non-det kernels silently perturb the loss. cuDNN deterministic? Same. Per-rank seed = global + rank_id? Because a shared seed across ranks produces correlated RNG draws which correlate dropout masks which silently reduce effective model capacity.

---

## The silent-failure list — Karpathy's enumerated hazards vs ch-06's

The source mentions a "pitfalls list" in the original post (line 27, "Skim for these sections: 'Neural net training fails silently'"). Karpathy's pitfalls and ch-06's §5 catalog overlap:

| Karpathy pitfall | Ch-06 §5 analogue |
|---|---|
| Forgotten `.eval()` | Evaluation with dropout active → spurious loss metric; not a resume bug per se but an instrumentation bug. |
| Wrong data-loader ordering | §5.1 data-iter desync (the direct successor; same root cause at LLM scale). |
| Incorrect loss implementation | Invariant-break; bit-exact resume check catches this by definition. |
| Tokenizer mismatch | Not in §5 but critical for post-training (round-`k` tokenizer must match round-`k-1`'s). |
| Forgot to zero gradients | Partial-state bug; a resume that restores the gradient buffer but not the optimizer step counter produces correlated gradient residuals. |

Notice: Karpathy's pitfalls are per-run; ch-06's are cross-resume extensions of the same patterns. The abstraction leaks the same way across a resume as it does within a single run — the checkpoint is just another boundary where assumptions silently drift.

---

## Step 5 — "Tune LR schedule last"

From the source (line 65):

> - Tune LR schedule last — after architecture is fixed.

For ch-06 this is a *risk ordering* rule. The LR schedule is the most complex piece of training state (multiple hyperparameters, stateful step counter, multiple possible shapes). It should be added *last* to a stable pipeline — not because it is unimportant, but because adding it introduces the largest new surface for silent failure. If you add a cosine schedule on top of a debug pipeline that has not verified bit-exact resume, the schedule bugs will interact with other bugs and be unisolatable.

Applied to the ch-06 instrumentation tiers: the per-step log (loss, grad-norm, LR) should exist *before* you care about scheduled LR. When LR is constant, the `lr` log is a constant and uninteresting — but it is a regression test: on the day the schedule is added, the `lr` log changes, and that change is audit-able.

---

## Step 6 — "Ensemble and average"

From the source (lines 68-70):

> - Ensemble several runs (2% easy gain).
> - Leave training running longer than you think necessary; models keep improving slowly.
> - Review the 10 worst validation examples — they reveal systematic errors.

The "ensemble several runs" bullet predates SWA and Model Soups ([[excerpts/early-stopping-and-checkpointing]]) but captures the same instinct: one checkpoint is weaker than many averaged. This is the Karpathy-era precursor to the WSD-decay averaging pattern in ch-06 §6. The recipe of 2019 would have averaged *independent runs*; ch-06's recipe of 2025 averages *decay-phase checkpoints from one trunk*. Both rely on the same checkpointing infrastructure.

---

## What to take from Karpathy for ch-06

1. **"Fails silently" is the axiom.** Every piece of ch-06 infrastructure exists because loss curves cannot be trusted to surface bugs.
2. **"Fix random seed everywhere" is prerequisite, not optional.** Bit-exact resume is impossible without it.
3. **"Overfit a single batch" is the diagnostic before any serious experiment.** A checkpoint that cannot bit-exactly resume a single-batch overfit has a broken seven-item list.
4. **"Monitor and clip the gradient norm" is non-negotiable.** Clipping is safety; monitoring is diagnosis. You need both.
5. **"Schedule last" means LR schedule is the highest-risk addition.** Off-by-one resume bugs in scheduler state are the predictable failure mode.
6. **"Ensemble several runs" is the pre-SWA version of checkpoint averaging.** The infrastructure is the same; only the sources being averaged differ.

---

## The philosophical through-line

Karpathy's recipe is not a list of tricks; it is a *discipline*. Every ch-06 rule is a specific expression of that discipline at scale:

- Ch-06 §1 (seven-item state) = the Step-2 skeleton made persistent.
- Ch-06 §3 (bit-exact resume) = "fix random seed everywhere" enforced across process boundaries.
- Ch-06 §4 (instrumentation tiers) = "monitor and clip" generalized to every metric.
- Ch-06 §5 (silent-failure catalog) = "fails silently" applied to the cross-resume surface.
- Ch-06 §6 (checkpoint flywheel) = "ensemble several runs" productionized as the WSD-fork tree.
- Ch-06 §7 (reference save/load) = the code that makes the discipline executable.

This is why the source appears as the *closing* citation in ch-06's "Further reading" list. It is not a technical reference you consult once; it is the posture the entire chapter operates from.

---

## Connections

- [[excerpts/early-stopping-and-checkpointing]] — "ensemble several runs" is the 2019 precursor; the classical table is the 1998 precursor.
- [[excerpts/gradient-clipping]] — "monitor and clip" made explicit at LLM scale.
- [[excerpts/lr-schedules]] — "schedule last" is the risk-ordering rule behind LR-schedule off-by-one hazards.
- [[excerpts/mixed-precision]] — "start in fp32 for debugging, switch to mixed precision only after training is stable" (source line 82) is a specific recipe item.
- [[excerpts/fsdp-sft]] — the leaky-abstraction story gets worse at 70B; the DCP API is partly a response to the leakage.
- [[excerpts/olmo-2]] / [[excerpts/olmo-3]] / [[excerpts/llama-3]] — all three production flows are recipe-compliant at scale; each is a case study in what "fails silently" looks like on real clusters.
- [[ch-06]] — the entire chapter; this excerpt is the axiom every section reduces to.
