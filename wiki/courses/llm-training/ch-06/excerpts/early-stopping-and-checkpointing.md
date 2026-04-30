---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/early-stopping-and-checkpointing.md
source_url: https://arxiv.org/abs/1803.05407 ; https://arxiv.org/abs/2203.05482
created_at: "2026-04-23"
---

# Excerpt: Early Stopping, Checkpointing, SWA, Model Soups — what "state" means before FSDP

**Source library:** `wiki/raw-data/llm-training/classics/early-stopping-and-checkpointing.md`
**Heritage:** Prechelt 1998 (early stopping) → Izmailov 2018 (SWA) → Wortsman 2022 (Model Soups) → MiniCPM / DeepSeek 2024 (WSD-decay averaging).

---

## Why this source anchors ch-06

Ch-06 opens with a seven-item checkpoint-state table. The rows of that table did not appear in 2025 from nowhere; each row was *named* in a different decade by somebody who had just lost a run to it. This excerpt reconstructs the lineage so that when you see "optimizer state `m`, `v`, master fp32" you remember it was `m` and `v` first, long before DCP, long before FSDP, long before 70B.

---

## Classical early stopping — the first "save, keep the best, stop"

```
# early-stopping-and-checkpointing.md, lines 35-51
best_val = inf; patience_counter = 0; best_ckpt = None
for step in range(total_steps):
    train_step()
    if step % eval_every == 0:
        val = evaluate(val_set)
        if val < best_val - delta:
            best_val = val
            best_ckpt = save_checkpoint()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
return best_ckpt
```

The shape of this loop is 1998 vintage (Prechelt). Three things are worth noticing before we scale it up.

First, the function `save_checkpoint()` is abstract. In Prechelt's world it meant a NumPy pickle of weight matrices — no optimizer state, no RNG, no dataloader cursor, because stopping meant **ending the run**, not resuming it. The entire modern checkpoint-as-resume surface is an extension of this primitive the moment "stop" was replaced by "restart on a different machine after a node died."

Second, the condition `val < best_val - delta` requires a stable validation set. The source file's "Common pitfalls" section flags the practical cost: *"Computing val loss on a non-stratified sample → noisy curve; early-stop fires on noise. Use a fixed, large (≥100k tokens for LLM) val set."* In ch-06 §4 the per-eval tier inherits this constraint — the 1k-step cadence exists because you need enough tokens in the held-out set to produce a signal above the noise floor.

Third, the `patience` hyperparameter (typical values `3–5` evaluations) is the first appearance of a pattern that now pervades LLM training: **instrumentation drives decisions**. You do not stop because training time elapsed; you stop because a *logged metric* crossed a threshold. Ch-06 §1's "grad-norm / loss history" row — instrumentation as the audit trail for the other six rows — is the same idea pushed one layer up: the log is not decoration, it is the control signal.

---

## SWA — weight averaging as the first "checkpoint-as-artifact"

Izmailov 2018 broke the one-checkpoint-per-run assumption:

```
# early-stopping-and-checkpointing.md, lines 55-66
swa_weights = first_snapshot
n = 1
for snapshot in subsequent_snapshots:
    swa_weights = swa_weights + (snapshot - swa_weights) / (n + 1)   # running avg
    n += 1

# After SWA: re-compute BatchNorm statistics with one forward pass over training set
recompute_bn_statistics(swa_weights, train_set)
```

Notice: the checkpoint is now *multiple* files, averaged into a single final artifact. The paper's Figure 1 shows SWA landing in a *wider* basin than the SGD endpoint. The implication for ch-06: a checkpoint is no longer exclusively a crash-recovery backup. It is a *training product*. Ch-06 §6 ("WSD stable-phase forkability and the checkpoint flywheel") generalizes this — each trunk checkpoint is a legitimate starting point for a fresh decay run, and the release artifact is the tree, not the leaf.

The BN-recompute line is a historical artifact: SWA was an image-net / CNN technique. For LLMs with RMSNorm or LayerNorm, *"the BN-recompute step is unnecessary — RMSNorm/LayerNorm have no running statistics."* This is why LLM weight averaging (MiniCPM, DeepSeek) can be a literal `torch.mean` across `.pt` files; no forward pass needed. One less step that can go wrong.

---

## Model Soups — greedy, validation-gated averaging

Wortsman 2022's "greedy soup" (the variant that mattered):

```
# early-stopping-and-checkpointing.md, lines 70-77
sort fine-tuned models by val_acc descending
soup = models[0]
for m in models[1:]:
    candidate = average(soup, m)
    if val_acc(candidate) > val_acc(soup):
        soup = candidate
return soup
```

The source's one-line explanation — *"Works because fine-tuned models from a shared init occupy the same loss basin; weight-averaging stays in-basin"* — is the critical caveat. Soup-averaging models with different inits breaks them (the pitfalls section: *"Soup-averaging across *different* base architectures or *different* random inits → models in different basins; averaging breaks them."*).

For the ch-06 reader this is a concrete operational lesson: **fork semantics matter**. Ch-06 §6's checkpoint tree (`trunk/ → decay/ → post/`) is a branching structure where every child *must* inherit its parent's init. A `decay_from_300k/` run that secretly re-initializes embeddings (because somebody cleaned up the checkpoint layout) is a soup-break waiting to happen.

---

## WSD-decay averaging — the LLM-scale successor

The modern recipe (from the source file):

> **WSD-decay averaging (LLM pretrain default in 2024–2025)**:
> - Run pretraining with WSD schedule (warmup → stable → decay).
> - During the decay phase, save N checkpoints (every ~10% of decay tokens).
> - Final model = uniform average of the last K checkpoints.
> - DeepSeek-V3 and MiniCPM both report measurable val-loss reduction (~0.5%) from this trick alone.

Notice: WSD decay lasts 10–20% of total tokens. If decay is 40B tokens and you save every 10% of *decay*, that is 5 checkpoints over 40B tokens. At 70B model size, each checkpoint is ~1 TB. Five checkpoints is 5 TB of pure final-model averaging material, in addition to every trunk checkpoint you kept for spike-rollback. The storage math alone motivates sharded save (see [[excerpts/fsdp-sft]]).

A ~0.5% val-loss reduction sounds cosmetic. At frontier scale it is not. The source's framing: this is essentially free compute — the decay-phase checkpoints exist anyway for instrumentation; averaging them is one `torch.mean`. Compare against the cost of an additional decay run to chase the same 0.5%, which would be days of 1024-GPU time.

---

## The checkpointing table — direct lift

The exact table from the source file that became the spine of ch-06 §1:

| Item | Saved? | Why |
|---|---|---|
| Model weights (sharded) | Yes | obvious |
| Optimizer state (`m`, `v`, master fp32 weights) | **Yes** | restart needs Adam moments |
| LR scheduler state | Yes | step counter |
| RNG state (per rank) | Yes | deterministic data order |
| Dataloader iterator position | **Yes** | must not re-see same tokens |
| Loss-scaler state (fp16) | Yes | dynamic loss scale |
| Gradient accumulator | Optional | only mid-microbatch |

The bolded rows are the ones with silent-failure modes the source explicitly flags:

> Saving weights but not optimizer state → restart re-warms-up Adam moments → loss-spike on resume.
>
> Saving but not RNG / dataloader cursor → on resume, re-seeing the same tokens → spurious loss drop, then divergence.

Notice: both bolded-row failures **look like success** at the log level. The Adam re-warm produces a small loss spike that is often ascribed to "noise from the resume." The dataloader replay produces a loss *drop* (because the model has already been trained on those tokens), which is doubly misleading — the curve bends the *right way*. This is why ch-06 §3 insists on bit-exact resume as the audit: you cannot infer from the loss curve alone whether your checkpoint is complete. You need the bit-exact assertion.

---

## Spike recovery — the operational frame

From the source (lines 100-104):

> **Spike recovery**: if validation loss spikes mid-pretraining, the standard recipe is:
> 1. Roll back to last clean checkpoint.
> 2. Skip the offending data shard (reorder).
> 3. Resume with same LR.
> This costs ~1% throughput in expectation but saves runs at frontier scale.

Three things here.

One: "last clean checkpoint" presumes you have more than one. If your job saves every 1000 steps and keeps only the last, a spike at step `N-500` is already past rollback. Ch-06's implicit contract is that trunk checkpoints are *retained*, not rotated out — the 5 TB of WSD-decay checkpoints mentioned above is only one slice of the total disk cost.

Two: "Skip the offending data shard (reorder)" requires that the dataloader's per-shard mapping is inspectable and mutable on resume. This is the same API surface that enables per-rank `dataloader.state_dict()` — see [[excerpts/fsdp-sft]] for how DCP exposes this.

Three: "~1% throughput in expectation." This is the operational cost *with* working checkpointing. Without working checkpointing, a spike is a run-ending event. The difference between 1% overhead and "rewrite the last month of training" is what the seven-item list buys you.

---

## Cadence — the Llama-3 number in context

The source's one-line cadence note:

> PyTorch DCP (Distributed Checkpoint) is the standard for FSDP; Megatron has its own format. Cadence: every 30–60 minutes of compute (Llama-3 used ~1000-step cadence at 405B).

At 405B with Llama 3's published throughput (~400 TFLOPS/GPU × 16K GPUs), 1000 steps is roughly half an hour of wall time. The "30–60 minutes" recommendation is a function of two competing costs: checkpoint write time (~30 seconds with DCP, see [[excerpts/fsdp-sft]]) and expected-loss-on-failure (mean time to recovery × MTBF of the cluster). See [[excerpts/llama-3]] for the 419-interruption incident table that sets the MTBF side of this equation.

---

## Connections

- [[excerpts/fsdp-sft]] — the DCP API that makes the seven-item save feasible at 70B+.
- [[excerpts/mixed-precision]] — the ninth (loss-scaler) item that appears only under fp16.
- [[excerpts/lr-schedules]] — WSD's stable phase as the fork point; cosine's off-by-one trap on resume.
- [[excerpts/gradient-clipping]] — `pre_clip_grad_norm` as the pre-spike signal that early-stopping-era instrumentation anticipated.
- [[excerpts/olmo-2]] — the spike-recovery recipe operationalized at 7B/13B/32B.
- [[excerpts/olmo-3]] — WSD forkability pushed to its logical conclusion: the tree is the release.
- [[excerpts/llama-3]] — ~1000-step cadence at 16K-GPU scale; the cadence row of ch-06 §4.
- [[excerpts/karpathy-training-neural-net-recipe]] — the "overfit a single batch" + "bit-exact resume" discipline that binds instrumentation to checkpointing.
- [[ch-06]] — this excerpt is the classical backbone of §1 (seven-item table) and §6 (WSD fork flywheel).
