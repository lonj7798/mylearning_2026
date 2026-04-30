---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fsdp-sft.md
source_url: https://arxiv.org/abs/2304.11277
created_at: "2026-04-23"
---

# Excerpt: PyTorch FSDP — the sharded-state reality DCP exists to serialize

**Source library:** `wiki/raw-data/llm-training/papers/fsdp-sft.md`
**Paper:** Zhao et al. 2023, "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"

---

## Why this source anchors ch-06 §2

Ch-06 §2 tells you "never gather-to-rank-0 at 70B+; use DCP." This excerpt shows the *memory arithmetic* that forces the rule. Once you see the per-GPU breakdown in Table 1 (reproduced below), the gather-to-rank-0 bug stops being "a best-practices thing" and becomes "physically impossible without buying a new class of GPU."

---

## The memory formula — reproduced

The source file gives the canonical per-GPU memory breakdown (lines 29-41):

| Component | DDP | FSDP FULL_SHARD |
|-----------|-----|-----------------|
| Parameters (BF16) | 2P | 2P / N |
| Gradients (BF16) | 2P | 2P / N |
| Optim state (FP32) | 12P | 12P / N |
| Temporary AllGather buffer | 0 | 2P |
| Total steady | 16P | (16P / N) + 2P |

> For a 70B model on 8 GPUs: DDP = 1120 GB (impossible), FSDP = ~280 GB (≈ 35 GB per GPU, feasible on 80 GB cards).

Notice the row **"Optim state (FP32) 12P."** This is AdamW's first moment `m` (4P), second moment `v` (4P), and the master fp32 weight copy (4P). It is **6× the size of the bf16 weights**. Ch-06 §1's table row *"Optimizer state is the majority of the checkpoint"* is this same formula rearranged: if weights are 14%, optimizer is 86%. A "weights-only" checkpoint at 70B drops 840 GB of state.

Now project onto the save path. Under `FULL_SHARD`, each rank holds `(16P / 8) = 2P` of model state plus the transient `2P` AllGather buffer, totalling `4P ≈ 140 GB` peak — feasible on 80 GB cards because the peak is transient. But the *save* path, if you call `FSDP.state_dict()` with default `StateDictType.FULL_STATE_DICT`, tries to reconstruct the complete `16P = 1120 GB` on rank 0 alone. No 80 GB card survives this.

This is precisely the "naive pattern A" of ch-06 §2:

```python
# ch-06/read.md, lines 57-61 — DO NOT USE AT SCALE
full_state = FSDP.state_dict(model)           # AllGather all params to rank 0
if rank == 0:
    torch.save(full_state, "ckpt.pt")
```

The `FSDP.state_dict()` call is an AllGather across the 8 ranks. Rank 0's buffer balloons to 1120 GB. OOM before the `torch.save` even starts.

---

## The sharding-strategy table — what is actually being persisted

From the source (lines 50-56):

| Strategy | Shards P | Shards Grad | Shards Opt | Comm overhead |
|----------|----------|-------------|------------|---------------|
| `NO_SHARD` (DDP) | no | no | no | AllReduce grads |
| `SHARD_GRAD_OP` | no | yes | yes | ReduceScatter grads |
| `FULL_SHARD` | yes | yes | yes | AllGather + ReduceScatter |
| `HYBRID_SHARD` | intra-node | intra-node | intra-node | intra FULL, inter REPLICATE |

Checkpointing implications by strategy:

- Under `NO_SHARD`, every rank has the full `(P, grad, optim)` — saving is trivial and the whole "naive pattern" discussion is moot. This is the SFT-8B regime the ch-06 failure mode comes from: dev on `NO_SHARD`, code ships to 70B on `FULL_SHARD`, checkpoint path OOMs on first save.
- Under `SHARD_GRAD_OP` (ZeRO-2), weights are replicated, gradients and optimizer state are sharded. A rank-0 weight save is legal; the optimizer state is *not* available on rank 0 alone and must be gathered. Naive "save just the weights" still silently drops optimizer state (840 GB of `m`, `v`, master).
- Under `FULL_SHARD`, nothing is whole on any single rank. DCP is the only correct path.
- Under `HYBRID_SHARD`, intra-node is `FULL_SHARD`, across-node is `REPLICATE`. The shard layout is not "rank `i` owns shard `i`" but "node `j`'s 8 ranks collectively own the full sharded state, and node `k`'s 8 ranks own an identical copy." DCP handles this; hand-rolled saves do not.

---

## Wrapping policy — why granularity matters for save latency

From the source (line 59):

> `auto_wrap_policy = transformer_auto_wrap_policy({LlamaDecoderLayer})`
> Each transformer block becomes an FSDP unit → AllGather only per-block params, not the whole model.

Notice: the wrap policy determines the *shard unit*. If you wrap per-block (one FSDP unit per transformer layer), DCP saves at block granularity and the sharded layout is "rank `i` owns a slice of every block." If you wrap at coarser granularity (the whole model as one unit), DCP cannot re-distribute shards on resume with a different world size — the unit is atomic.

For ch-06 §2's "resume with a different `N` (node died; restart with 7 nodes instead of 8)" story, this means **the wrap policy is part of the checkpoint contract**. Change the wrap policy between save and load, and DCP's re-mapping breaks. In practice labs fix the wrap policy at the start of a training run and never touch it; if you need to change it, resume requires a full gather-and-reshard cycle (expensive, one-time).

---

## Mixed precision — the fp32 optimizer clause

From the source (lines 63-67):

```
MixedPrecision(param_dtype=torch.bfloat16,
               reduce_dtype=torch.bfloat16,
               buffer_dtype=torch.bfloat16)
```

> Keeps FP32 master weights in the optimizer, BF16 everywhere else — combined with ZeRO sharding this is the standard modern SFT setup.

This is the row that makes ch-06 §1's *"Master fp32 weights"* non-optional. Even though `param_dtype=bf16`, the optimizer's internal storage is fp32 — `m`, `v`, and a separate fp32 copy of each weight. This is why the optimizer state is `12P` not `2P`: three fp32 tensors per parameter, each 4 bytes.

If your checkpoint persists `model.state_dict()` (the bf16 params) but not `optimizer.state_dict()` (the fp32 master + moments), the master is silently reconstructed on resume by upcasting bf16 → fp32. Ch-06 §5.4 calls out the consequence: *"AdamW reconstructs master by upcasting bf16 → fp32, loses the 7 mantissa bits that had accumulated, and then every subsequent step compounds that error."* The 7 mantissa bits are the difference between bf16 (7) and fp32 (23); they are not recoverable.

This is also why [[excerpts/mixed-precision]] insists optimizer state must stay fp32 across resumes even if the active compute is bf16 or fp8.

---

## The typical SFT recipe — the values that ground the cadence

From the source (lines 70-81):

| Knob | Value |
|------|-------|
| Strategy | FULL_SHARD |
| Precision | BF16 params + BF16 reduce + FP32 optim master |
| Activation checkpointing | per transformer block |
| Micro-batch per GPU | 1 |
| Gradient accumulation | 16 |
| Packing | yes (see [[sequence-packing]]) |
| Max seq length | 4096 |
| Learning rate | 1e-5, cosine, warmup 3% |
| Optimizer | AdamW β = (0.9, 0.95) |

Apply ch-06's 1000-step cadence to this recipe. Micro-batch 1, grad-accum 16, across 8 GPUs, 4096 tokens/seq, packed. One optimizer step = 1 × 16 × 8 × 4096 = ~520K tokens. 1000 steps = ~520M tokens. At bf16 throughput of ~10K tokens/sec/GPU for 70B SFT on H100 (realistic), 8 GPUs produce 80K tok/s, so 520M tokens = ~1.8 hours. Checkpoint every ~1.8 hours at SFT scale.

At pretraining scale the numbers shift but the shape holds: the 1000-step number is a *time* constant (wall-clock between saves) expressed as a step count that happens to align with a given throughput. Ch-06 §4's per-checkpoint tier is synchronized to this cadence for a reason — the expensive metrics (GPU util, NCCL bandwidth, embedding norms) are computed *once per save* because that is when the pipeline already has a stop-the-world moment.

---

## The save path in DCP — what actually happens on disk

Ch-06 §2 shows the DCP save API:

```python
# ch-06/read.md, lines 78-88
opts = StateDictOptions(full_state_dict=False, cpu_offload=True)  # sharded, CPU-staged
model_sd, optim_sd = get_state_dict(model, optimizer, options=opts)
dcp.save(state_dict = {...}, checkpoint_id = f"ckpts/step_{step:08d}")
```

Two flags deserve expansion.

`full_state_dict=False` is the anti-naive-pattern-A guard. It tells `get_state_dict` to return *sharded* tensors (`DTensor` under the hood), each rank's slice only. No AllGather, no rank-0 materialization.

`cpu_offload=True` moves the sharded state from GPU to CPU before the DCP write. On 80 GB cards with the 35 GB steady-state footprint, you have ~45 GB headroom, but DCP's write buffer + Python/C++ overhead can still push OOM during save on large models. Offloading stages the tensor to pinned host memory first, then DCP writes from host. Save latency goes up by the PCIe copy time; OOM risk goes to zero.

The `checkpoint_id` is a *directory path*, not a file path. DCP writes one shard file per rank plus a `.metadata` file describing the global layout:

```
ckpts/step_00001000/
  .metadata            # DCP layout descriptor
  __0_0.distcp         # rank 0's shard
  __1_0.distcp         # rank 1's shard
  ...
  __7_0.distcp         # rank 7's shard
```

On load, each rank reads *its* shard in parallel. The `.metadata` file lets DCP re-map shards if the load-time world size differs from save-time — this is the "resume with 7 nodes instead of 8" scenario from ch-06. No gather, no rank-0 bottleneck, O(N) parallel IO.

---

## What FSDP does *not* save — the gap DCP fills

The source's FSDP paper describes the sharding of *compute state*: parameters, gradients, optimizer state. It says nothing about:

- Per-rank RNG state
- Dataloader iterator position
- LR scheduler counter
- Loss scaler state (fp16 only)
- Step counter

These live outside FSDP entirely. DCP's contribution is *also* saving them, in the same atomic checkpoint directory, co-located with the sharded compute state. Ch-06 §1's seven-item list is the superset; FSDP handles items 1–4, DCP's role is the *aggregation primitive* that packages all seven into one restartable unit.

This is why ch-06 §7 passes `data_loader.state_dict()` and `rng_state` *through* the `dcp.save(state_dict={...})` call, not as side-files. Side-files race against the DCP save: a crash between `dcp.save(...)` and `torch.save(rng_state)` produces a checkpoint that looks complete but has stale RNG. Putting everything inside the DCP state-dict makes the whole checkpoint atomic.

---

## Connections

- [[excerpts/early-stopping-and-checkpointing]] — the seven-item classical table FSDP+DCP operationalizes.
- [[excerpts/mixed-precision]] — the fp32 master + bf16 param split that makes optimizer state 86% of the checkpoint.
- [[excerpts/karpathy-training-neural-net-recipe]] — "overfit a single batch" runs at `NO_SHARD` scale where the FSDP checkpoint failure modes are invisible.
- [[excerpts/olmo-2]] / [[excerpts/olmo-3]] / [[excerpts/llama-3]] — production pipelines that run `FULL_SHARD` + DCP at the scales where rank-0 gather is fatal.
- [[ch-06]] — §2 (the DCP pattern), §5.4 (partial-optimizer-load failure), §7 (the reference save/load functions).
