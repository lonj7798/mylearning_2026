<!-- chapter: ch-06
     track: foundations
     title: Checkpointing, Resume, Instrumentation
     sources: [[early-stopping-and-checkpointing]], [[fsdp-sft]], [[mixed-precision]], [[lr-schedules]], [[gradient-clipping]], [[olmo-2]], [[olmo-3]], [[llama-3]], [[karpathy-training-neural-net-recipe]]
     figures: figures/checkpoint-state.html
-->

# Chapter 6 — Checkpointing, Resume, Instrumentation

> **Core insight.** A checkpoint is not "the weights." It is the *entire* state needed to reproduce the next training step bit-for-bit: weights, optimizer moments, master copy, LR scheduler counter, per-rank RNG, data-iterator cursor, loss-scaler state, and the instrumentation history that lets you audit whether the resume actually worked. Drop any one element and the failure mode is silent — training continues, loss looks plausible, the run has quietly diverged from the one you meant to run.
>
> **Guideline.** Save sharded under FSDP using PyTorch's Distributed Checkpoint (DCP), never gather-to-rank-0 at 70B+ scale. Keep optimizer state in fp32 regardless of `param_dtype`. Checkpoint every ~1000 steps (Llama 3 cadence) and gate each cycle on a "resume-produces-bit-exact-loss" assertion — Karpathy's rule, and still the only reliable guard against data-iter and scaler-state desync.

---

## Why this chapter exists

The cheapest way to destroy a frontier training run is to mis-checkpoint it. The failure is never loud. You don't get a `CheckpointCorrupt` exception; you get a resume that starts at a slightly different loss than the one you saved at, a cosine schedule that is off by one step forever, a data loader that replays the same 500 batches you just trained on, or a loss-scaler that rewarms from `2^15` when it should have been at `2^18`. Each of these costs real tokens of wasted compute. At 1,024 H100s and OLMo 3's ~1 M GPU-hour budgets ([[olmo-3]]), a single mis-resumed week is the annual hardware bill of a small lab.

The previous five chapters built the training step. This chapter is about what happens *between* steps — the saved state, the reload path, and the logs you need to trust it. Three traditions converge here: classical early-stopping machinery ([[early-stopping-and-checkpointing]]) which taught us what "state" must mean; FSDP-era distributed checkpointing ([[fsdp-sft]]) which taught us how to save state that doesn't fit on one rank; and the WSD/soup lineage ([[lr-schedules]], [[early-stopping-and-checkpointing]]) which taught us that checkpoints are not just for recovery but for *forking* — the same trunk produces multiple instrumented decay runs.

The chapter's organizing principle is Karpathy's maxim: **training fails silently, so instrument everything and verify resumes bit-exactly** ([[karpathy-training-neural-net-recipe]]). Every design choice below is derived from that one rule.

---

## 1. What actually goes in a checkpoint

From [[early-stopping-and-checkpointing]], production checkpointing in 2025 is a seven-item list. Each item has a failure mode if it's missing. Hold this table in your head — it is the spine of the chapter.

| State component | Precision | Per-rank or global | What breaks if missing |
|---|---|---|---|
| Model weights | bf16 (shard) | sharded under FSDP | obvious; restart from zero |
| Optimizer 1st moment `m` | fp32 | sharded | Adam's momentum re-warms; loss spike on resume |
| Optimizer 2nd moment `v` | fp32 | sharded | adaptive denominator resets; effective LR wrong for ~20–1000 steps (see [[adam]]'s β₂ discussion in ch-01) |
| Master fp32 weights | fp32 | sharded | resume rounds bf16 → fp32 and loses 7-mantissa-bit progress; silent ~0.1% perplexity drift |
| LR scheduler state | int + float | global | cosine phase off by N steps; WSD decay fires at wrong token count |
| Per-rank RNG state | uint64[] | **per-rank** | dropout / label-smoothing / augmentation draws diverge; loss is non-reproducible |
| Data-iterator position | int | **per-rank** (global logical step) | **the silent killer** — see §5 |
| Loss-scaler state (fp16 only) | fp32 + int | global | scaler re-warms from `2^15`; first ~2000 resumed steps skip under-/overflow (see [[mixed-precision]]) |
| Step counter | int | global | every time-based decision (eval, save, decay onset) fires at wrong wall-time |
| Grad-norm / loss history | logs | global | you cannot tell whether the resume drifted; instrumentation is the *audit trail* for the seven items above |

Two items deserve expansion.

**Optimizer state is the majority of the checkpoint.** For AdamW on a 70B model in bf16 + fp32 optimizer ([[fsdp-sft]]): weights = 2 P = 140 GB, gradients transient, optimizer `(m, v, master)` = 12 P = 840 GB. Optimizer state is **6×** the weight size. A "weights-only" checkpoint is 14% of the real checkpoint. Skipping it is the number-one hobby-scale resume bug — [[early-stopping-and-checkpointing]] flags this explicitly: *"Saving weights but not optimizer state → restart re-warms-up Adam moments → loss-spike on resume."*

**Grad-norm history is not ornamental.** [[gradient-clipping]] and [[olmo-2]] both make the point: a 100× `pre_clip_grad_norm` spike predicts a loss spike 1–5 steps ahead. That predictive signal only works if the log survives the resume. If your log ring-buffer resets at every resume, you lose the ability to distinguish "new instability" from "pre-existing drift I carried across the last crash." Log durably. The OLMo 2 loss-spike mitigation stack — clip + skip-step + embedding-norm monitoring ([[olmo-2]]) — is a feedback loop on persisted metrics; resetting the metrics blinds the loop.

---

## 2. Sharded checkpointing under FSDP — DCP, `save_sharded` vs `save_full`, the rank-0 bug

Under FSDP FULL_SHARD ([[fsdp-sft]]), each rank owns a disjoint shard of `(weights, m, v, master)`. For `N = 8` ranks on a 70B model, each rank owns ~105 GB of state. Two naive save patterns both fail:

**Naive pattern A — gather full state to rank 0, save as one file.**
```python
# DO NOT USE AT SCALE
full_state = FSDP.state_dict(model)           # AllGather all params to rank 0
if rank == 0:
    torch.save(full_state, "ckpt.pt")
```
At 70B this materializes ~140 GB on rank 0 (weights only, not counting optimizer). Rank 0 OOMs before the first save. On smaller models it *works*, which is how the bug ships — ch-05 SFT-scale code is copied into 70B pretraining and explodes the first time a checkpoint is triggered.

**Naive pattern B — save local shards, one per rank, with no coordination.**
```python
torch.save(model.state_dict(), f"ckpt_rank_{rank}.pt")    # saves a FlatParameter shard
```
This saves *something* but the shard layout is implicit in the current FSDP wrap policy, the current world size, the current CUDA device mesh. Resume on a different `N` (node died; restart with 7 nodes instead of 8) silently loads wrong shards. This is the rank-0-bug's cousin: it saves successfully but cannot re-load robustly.

**The correct pattern — PyTorch Distributed Checkpoint (DCP).** DCP is FSDP's native sharded-save API. Each rank writes its own shard in parallel to the same directory; a metadata file describes the global layout; on load, DCP re-maps shards to the current world size.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)

# --- SAVE ---
opts = StateDictOptions(full_state_dict=False, cpu_offload=True)  # sharded, CPU-staged
model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
dcp.save(
    state_dict = {"model": model_sd, "optim": optim_sd,
                  "step":  step, "rng": torch.cuda.get_rng_state_all(),
                  "data":  data_loader.state_dict(),
                  "sched": lr_scheduler.state_dict(),
                  "scaler": scaler.state_dict() if scaler else None},
    checkpoint_id = f"ckpts/step_{step:08d}",
)

# --- LOAD (possibly with different world size) ---
state = {"model": model_sd, "optim": optim_sd, "step": 0, "rng": None,
         "data": None, "sched": None, "scaler": None}
dcp.load(state_dict=state, checkpoint_id=f"ckpts/step_{step:08d}")
set_state_dict(model, optimizer, model_state_dict=state["model"],
               optim_state_dict=state["optim"], options=opts)
```

Nanotron (Hugging Face) and Megatron ship their own analogous sharded formats. Use whichever ships with your framework; **do not** write your own. Every home-rolled FSDP checkpoint I have ever audited has had one of the two naive-pattern bugs.

The `full_state_dict=False` knob is the one that matters. Under the hood: DCP performs one round of ReduceScatter-like metadata exchange, then each rank does a direct-to-disk write of its shard. Save throughput scales with `N`, not with the rank-0 NIC. For a 70B checkpoint on 8 × 80 GB H100s with NVMe, total wall-clock is ~30 seconds; the naive-pattern A equivalent, if it didn't OOM, would take ~4 minutes bottlenecked on rank-0 IO.

---

## 3. Bit-exact resume vs approximate resume

Karpathy's single most-quoted checkpoint rule ([[karpathy-training-neural-net-recipe]]): *"check that resume produces bit-exact loss."* In operational terms: after saving at step `k`, restart the process, load step `k`, run one training step, compare the loss to what step `k+1` produced in the original run. If it differs by more than fp-rounding noise, something is missing from the checkpoint.

**Bit-exact resume** requires the full seven-item list in §1 plus deterministic ops:

```python
torch.use_deterministic_algorithms(True)       # raises on non-det kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark      = False
# Per-rank seed from global seed + rank id
torch.manual_seed(cfg.seed + rank)
torch.cuda.manual_seed_all(cfg.seed + rank)
```

The per-rank RNG state must be saved *after* the last step's random draws and restored *before* the next step's draws. Dropout, label smoothing, NEFTune noise ([[neftune]], ch-04), and any stochastic dataloader augmentation all consume RNG; forget to persist it and two resumes of the same checkpoint produce two different losses — even though the weights are identical. This is one of the more subtle bugs in the list because *both* runs look healthy.

**Approximate resume** drops determinism but preserves macro-statistics. The weights, optimizer state, scheduler, and data-iter are bit-identical; RNG and dropout masks are fresh. Loss after resume differs by ~1e-4 (fp rounding over the first few thousand tokens' dropout patterns) but the training trajectory is statistically indistinguishable from the original.

**When each matters.** The rule of thumb from the field:

| Regime | Required | Why |
|---|---|---|
| Debug / CI / repro of a published result | bit-exact | you want to prove the training path matches |
| Pretraining trunk (Llama-3, OLMo-2 ([[olmo-2]]), DeepSeek) | bit-exact | loss-spike rollback (§5) depends on deterministic replay to diagnose the spike |
| SFT / DPO at 8×H100 scale | approximate | determinism cost (~15% throughput from disabled cuDNN benchmark) > debugging value |
| RLVR / PPO rollouts | approximate | the policy is stochastic; bit-exactness is impossible by construction |

For fp16 training, the loss scaler's state is the ninth item and must be persisted. [[mixed-precision]]'s dynamic loss scaling maintains `S` that grows/halves based on inf/NaN detection; dropping it on resume means `S` re-warms from `2^15` and the first 2000 steps skip any underflows (inflated effective gradient). bf16 has no scaler — one of the reasons [[mixed-precision]] argues bf16 is the default for 2025 LLM training.

---

## 4. Instrumentation — what to log, at what cadence

You cannot debug what you don't log, and you cannot afford to log everything. The working cadence tiering, composed from [[olmo-2]] + [[gradient-clipping]] + [[early-stopping-and-checkpointing]]:

| Tier | Metrics | Cadence | Why |
|---|---|---|---|
| **Per-step** | training loss, LR, `pre_clip_grad_norm`, tokens/sec | every step | these are the heartbeat; a 100× grad-norm spike precedes loss spike 1–5 steps ([[gradient-clipping]]) |
| **Per-N-steps (~50)** | per-shard loss breakdown, entropy of output distribution, token-level KL to reference (DPO/RL), accumulated clip-events | N=50 | composes a rolling distribution; per-shard loss catches a poisoned data shard before it contaminates the run |
| **Per-eval (~1k steps)** | val-loss on fixed held-out set, capability benchmarks (MMLU, GSM8K for math-heavy), perplexity on canary set | ~1000 | early-stop ([[early-stopping-and-checkpointing]]) fires from this tier; keep val set ≥100k tokens or early-stop fires on noise |
| **Per-checkpoint (~1k–5k steps)** | full system: GPU util, NCCL bandwidth, disk usage, embedding-norm, activation-norm, weight-norm-per-layer | every checkpoint | the OLMo 2 spike-mitigation stack: clip + skip-step + embedding-norm monitoring ([[olmo-2]]) |
| **Per-run** | data-order hash, code commit, env capture, `pip freeze`, `nvidia-smi --query` | once | reproducibility audit trail; Karpathy's "fix random seed everywhere" ([[karpathy-training-neural-net-recipe]]) |

Two practitioner notes that are not in any paper but are in every production trainer.

**Log in fp32, always.** [[mixed-precision]] warns against logging loss in fp16 — the curve looks quantized-jaggy because fp16 has 10 mantissa bits. `loss.item()` upcasts implicitly; explicit `loss.detach().float().item()` documents intent.

**Grad-norm is logged `pre_clip`.** The quantity you want is the *raw* norm before clipping rescales it, because the clipped norm is trivially `min(raw, c)` and carries no spike information. In PyTorch, `clip_grad_norm_` returns the pre-clip norm — log its return value.

```python
# The canonical logging block
loss_f32 = loss.detach().float().item()
grad_norm = model.clip_grad_norm_(max_norm=1.0).item()   # pre-clip, global
optimizer.step()
logger.log_scalar("loss", loss_f32, step=step)
logger.log_scalar("grad_norm/pre_clip", grad_norm, step=step)
logger.log_scalar("lr", lr_scheduler.get_last_lr()[0], step=step)
logger.log_scalar("tokens_per_sec", n_tokens / (time.time() - t0), step=step)
```

---

## 5. Cross-resume silent-failure modes

This is the section the rest of the chapter exists to enable. Every bug here is something I have personally watched eat a training week.

**5.1 Data-iter desync.** The single most dangerous bug in the checkpoint surface. Each rank's dataloader samples from a sharded index; on resume, the *global logical step* must be restored to each rank, and each rank's local iterator must be fast-forwarded to where it was. The bugs:

- Save only rank-0's iterator state: other ranks restart at batch 0, re-see their first epoch's tokens, silently overweight them.
- Save all ranks but not the global step: iterators are restored, but the sampler re-seeds from the original RNG and produces a *new* permutation of the dataset; training replays already-seen batches under new labels.
- Save the iterator but not the curriculum/mix weights: OLMo 3's model-flow approach ([[olmo-3]]) stages data mixes — pretrain mix → mid-training mix (Dolmino) → long-context mix (Longmino). Resume without the current stage's mix pointer and you silently reverse-curriculum.

Symptom: loss drops ~0.01 for the first ~500 steps after resume, then slowly drifts upward as the model over-memorizes the replayed shard. By the time you notice, you've burned a checkpoint cycle.

Fix: save `dataloader.state_dict()` *per rank* (TorchData, Nanotron, and HF `datasets.IterableDataset` all support this), plus the global step, plus the active data-mix identifier. On resume, compare the hash of the next 100 samples against a known-good run — a five-line integration test that catches every variant of this bug.

**5.2 Scaler-state drop (fp16 only).** [[mixed-precision]]'s dynamic loss scaling maintains `S` that grows by 2 every 2000 successful steps. A resume that fails to persist `S` starts again at `2^15`; if your run had stabilized at `S = 2^18`, the first resumed gradients are 8× smaller than they should be, so the clip threshold is effectively 8× tighter and the optimizer makes 8× smaller real-scale steps until the scaler re-adapts. Loss looks fine; token-efficiency is 25% worse for 2000 steps. This is why bf16 won: no scaler, no bug.

**5.3 LR-schedule off-by-one.** You save at step 10,000 mid-cosine. You restart. If `lr_scheduler.step()` is called *before* the saved state is loaded, the scheduler advances to step 10,001 and the first optimizer step runs with step-10,001's LR — i.e. step 10,000 is skipped. Over a 100k-step cosine this is a 0.001% LR error, invisible. Over a WSD decay phase that is 10–20% of the run ([[lr-schedules]]), one missed step at the start of decay shifts the entire decay shape by `1/decay_len` — still tiny but now *systematic*. The fix: load scheduler state *before* the first `.step()` call, and assert `scheduler.last_epoch == saved_step`.

**5.4 Optimizer state partially loaded.** DCP is robust, hand-rolled saves are not. The common bug: save `m` and `v` but forget the master fp32 weights (they're a *separate* entry in AdamW's state under FSDP). Resume loads sharded bf16 weights, AdamW reconstructs master by upcasting bf16 → fp32, loses the 7 mantissa bits that had accumulated, and then every subsequent step compounds that error. Only catches: a bit-exact resume check (§3) or a weight-norm-per-layer log that suddenly jumps.

**5.5 Embedding-norm drift across resumes.** [[olmo-2]] explicitly monitors embedding norms as a pre-spike indicator. Each resume is an opportunity to introduce a small numerical hiccup (cast-round-cast) in the embedding table. Over 10 resumes at 70B, accumulated drift can be 0.1% of the L2 norm of the embedding matrix — enough to shift the softmax temperature noticeably. The defense: log `||W_embed||` per resume; alarm on delta > 1e-4 across a resume boundary.

---

## 6. WSD stable-phase forkability and the checkpoint flywheel

Checkpointing has a second purpose beyond crash recovery: **forking**. WSD schedules ([[lr-schedules]]) hold a constant LR for the bulk of training, then decay for the last 10–20% of tokens. Every stable-phase checkpoint is a legitimate starting point for a fresh decay run — same trunk, different decay length, different downstream task. MiniCPM and DeepSeek demonstrated that *averaging* the final K decay checkpoints produces a ~0.5% val-loss improvement over any single endpoint ([[early-stopping-and-checkpointing]], [[lr-schedules]]).

This only works if checkpoints are *instrumented*. You need to pick *which* checkpoint to decay from; that decision is a function of stable-phase val-loss, grad-norm stability, and the downstream eval gradient. The [[olmo-3]] "model flow" is this pattern generalized: Base → Mid-training (Dolmino) → Long-context (Longmino) → separate SFT/DPO/RLVR branches. Each arrow is a checkpoint fork with its own instrumentation gate. The release artifact is not the final weights; it's the tree.

The Llama 3 post-training flywheel ([[llama-3]]) is a higher-frequency version of the same idea. Six rounds of SFT → Rejection Sampling → DPO, each round starting from the prior round's best checkpoint, each round generating rejection-sampled outputs that become the next round's SFT data. Per round, the persisted artifacts are:

- round-`k` SFT checkpoint (the policy that generated the data)
- round-`k` reward model (the filter that scored the data)
- round-`k` rejection-sampled pool (the filtered data itself)
- round-`k` DPO preference batch

Miss any one and round `k+1` is not reproducible. The "checkpoint" is not a file; it is a bundle of (policy, scorer, data, preferences). Treat it as such in the filesystem layout.

Practical filesystem convention:
```
ckpts/
  trunk/
    step_00100000/    # WSD stable-phase; forkable
    step_00200000/
    step_00300000/
  decay/
    decay_from_300k/
      step_00350000/  # final model candidates
      step_00360000/
    decay_from_300k_20pct/
  post/
    r1_sft/     r1_rm/     r1_rs_pool/     r1_dpo/
    r2_sft/     r2_rm/     r2_rs_pool/     r2_dpo/
    ...
```

---

## 7. A drop-in reference — save, load, instrument

Combining §2, §3, §4 into the canonical training-loop skeleton. Everything here is production-shaped; names match PyTorch 2.5 APIs.

```python
# ------------------- 7.1 Save -------------------
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)

def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    data_loader, step, rng_state):
    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
    state = {
        "model":       model_sd,
        "optim":       optim_sd,                     # includes m, v, master fp32
        "sched":       scheduler.state_dict(),
        "scaler":      scaler.state_dict() if scaler else None,
        "data":        data_loader.state_dict(),     # per-rank iter state
        "step":        step,
        "rng/cpu":     torch.get_rng_state(),
        "rng/cuda":    torch.cuda.get_rng_state_all(),
        "rng/py":      random.getstate(),
        "rng/np":      np.random.get_state(),
        "code_sha":    os.environ.get("GIT_SHA", "unknown"),
    }
    dcp.save(state_dict=state, checkpoint_id=path)

# ------------------- 7.2 Load -------------------
def load_checkpoint(path, model, optimizer, scheduler, scaler, data_loader):
    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
    state = {"model": model_sd, "optim": optim_sd,
             "sched": None, "scaler": None, "data": None,
             "step": 0, "rng/cpu": None, "rng/cuda": None,
             "rng/py": None, "rng/np": None, "code_sha": None}
    dcp.load(state_dict=state, checkpoint_id=path)
    set_state_dict(model, optimizer,
                   model_state_dict=state["model"],
                   optim_state_dict=state["optim"], options=opts)
    scheduler.load_state_dict(state["sched"])
    if scaler and state["scaler"]: scaler.load_state_dict(state["scaler"])
    data_loader.load_state_dict(state["data"])
    torch.set_rng_state(state["rng/cpu"])
    torch.cuda.set_rng_state_all(state["rng/cuda"])
    random.setstate(state["rng/py"])
    np.random.set_state(state["rng/np"])
    # invariant: scheduler.last_epoch == state["step"]
    assert scheduler.last_epoch == state["step"], "LR scheduler off by one"
    return state["step"]

# ------------------- 7.3 Instrumentation logger -------------------
class TrainLogger:
    """Per-step lightweight; upcasts to fp32 for numerical sanity."""
    def __init__(self, sink):
        self.sink = sink    # e.g. wandb, tensorboard, jsonlines
        self.t0   = time.time()
        self.tokens_seen = 0

    def step(self, step, loss, grad_norm, lr, n_tokens):
        now = time.time()
        self.tokens_seen += n_tokens
        self.sink.log({
            "step":               step,
            "loss":               float(loss.detach().float().item()),
            "grad_norm/pre_clip": float(grad_norm),
            "lr":                 float(lr),
            "tokens/sec":         n_tokens / max(now - self.t0, 1e-6),
            "tokens_total":       self.tokens_seen,
        }, step=step)
        self.t0 = now

    def eval(self, step, val_loss, embed_norm, per_shard_loss):
        self.sink.log({
            "val/loss":     float(val_loss),
            "norm/embed":   float(embed_norm),
            **{f"loss/shard_{k}": float(v) for k, v in per_shard_loss.items()},
        }, step=step)

# ------------------- 7.4 Training step with the right ordering -------------------
for step in range(start_step, total_steps):
    batch = next(data_iter)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()

    grad_norm = model.clip_grad_norm_(max_norm=1.0)     # FSDP-aware; pre-clip norm
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    train_logger.step(step, loss, grad_norm,
                      scheduler.get_last_lr()[0],
                      n_tokens=batch["input_ids"].numel())

    if step % cfg.eval_every == 0:
        val_loss, embed_norm, per_shard = evaluate(model, val_loader)
        train_logger.eval(step, val_loss, embed_norm, per_shard)

    if step % cfg.save_every == 0 and step > 0:
        save_checkpoint(f"ckpts/step_{step:08d}",
                        model, optimizer, scheduler, scaler,
                        data_loader, step, rng_state=None)
```

This is the whole loop. Every item in the §1 table has a line here that reads or writes it, and every silent-failure mode in §5 is blocked by an assertion or a logged quantity.

---

## Connections and what's next

- **[[early-stopping-and-checkpointing]] / this chapter** — classical foundation; the seven-item state list.
- **[[fsdp-sft]] / ch-05** — FSDP mechanics; DCP is the save API.
- **[[mixed-precision]] / ch-02** — fp16 loss-scaler state; the ninth item for legacy runs.
- **[[lr-schedules]] / ch-03** — WSD stable-phase fork pattern; scheduler state's off-by-one trap.
- **[[gradient-clipping]] / ch-01** — `pre_clip_grad_norm` as the earliest warning signal.
- **[[olmo-2]] / [[olmo-3]] / [[llama-3]]** — three concrete production checkpoint pipelines; each is a flywheel, not a crash-recovery backup.
- **ch-07 (failure modes)** — the wider silent-failure catalog; §5 here is the checkpoint-specific subset.
- **ch-08 (lab)** — the mandatory unit test: save at step `k`, restart, verify bit-exact loss at step `k+1`.

## Further reading

- [[early-stopping-and-checkpointing]] — Prechelt 1998 / Izmailov 2018 / Wortsman 2022 / MiniCPM-DeepSeek WSD averaging; the canonical "what goes in a checkpoint" table is in the Technical Details section.
- [[fsdp-sft]] — Zhao 2023; FSDP memory formula and why sharded save is mandatory at 70B.
- [[mixed-precision]] — Micikevicius 2017; dynamic loss scaling + master-fp32 semantics.
- [[lr-schedules]] — WSD forkability and cosine-mismatch cost.
- [[olmo-2]] — loss-spike mitigation stack and the instrumentation that makes it possible.
- [[olmo-3]] — model-flow philosophy; checkpoints as public artifact, not recovery backup.
- [[llama-3]] — six-round post-training flywheel; what you must persist per round.
- [[karpathy-training-neural-net-recipe]] — "overfit a single batch" and "verify resume is bit-exact"; the rules the rest of this chapter enforces.

## Companion visualization

**[figures/checkpoint-state.html](figures/checkpoint-state.html)** — interactive diagram of the seven-item checkpoint state. Toggle each component "saved / not saved" and the page updates two live verdicts — "Bit-exact resume" and "Approximate resume" — plus a side panel that enumerates the silent-failure mode triggered by each omission (data-iter → replay divergence, scaler → 2000-step under-step, RNG → non-reproducible dropout, etc.). Use it to internalize which subset of state each resume-flavor actually requires, and why dropping "weights only" into a production pipeline is the single most expensive default.
