<!-- qa-deep: ch-03 — parallelism-side activation memory (overflow from [[qa]], 120-line cap)
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-03 Q&A (deep) — parallelism-side activation memory

Overflow page for [[qa]] (which holds the checkpointing mechanism and its cost model).

---

### Q3 — what is Sequence Parallelism? Is it for longer context?

**Partly right, but two different techniques share the name — §4 is not the long-context one.**

**§4's SP (Megatron / [[selective-recompute-korthikanti]]) removes duplication that TP leaves behind.** Tensor Parallelism shards weight matrices across `t` GPUs, so attention/MLP activations are `1/t` each. But **LayerNorm and Dropout operate on the whole hidden vector**, so they cannot be sharded along the weight axis — every GPU in the TP group keeps a **full replica** (read.md L166). Eight TP ranks store the same LayerNorm tensor eight times.

The fix (read.md L168): **LayerNorm and Dropout are independent across sequence positions** (unlike attention), so shard them along the **sequence** axis — GPU0 takes tokens 0–511, GPU1 the next chunk, etc. Nothing is replicated any more.

```
per-layer activation coefficient
  without SP:  10 + 24/t + 5as/(ht)     <- the 10 is NOT divided by t (the replica)
  with SP:          34/t + 5as/(ht)     <- 10 and 24 merge and divide by t

  t = 8, linear part only:   13  ->  4.25   = 3.06x
```

**Caveat:** 3.06× is the **linear term only** — the `5as/(ht)` attention term is identical in both, so the total is 1.36× at s=4096 and just **1.05× at s=32768** (Q6).

**Communication cost is zero** (read.md L175): TP's all-reduce becomes `AllGather + ReduceScatter`, which is exactly what an all-reduce already decomposes into — same bandwidth. Free memory is rare; usually savings cost communication.

The long-context technique is a different axis entirely:

| | Sequence Parallelism (§4) | Context Parallelism / Ring Attention |
|---|---|---|
| shards | only LayerNorm/Dropout activations | the **whole sequence, attention included** |
| purpose | remove replication inside a TP group | make context too long for one GPU possible |
| attention | untouched (TP splits it by head) | attention itself is split (KV passed around a ring) |
| independence | **always paired with TP**, reuses its communicator | **its own parallel axis** (`world_size = TP×PP×CP×DP`) |
| extra comms | **none** | yes (KV ring) |
| where | ch-03 §4, [[ch-07]] | [[ch-06]], [[ch-07]] |

**The intuition wasn't wrong**, though: the term SP deletes is `10·s·b·h`, proportional to `s`. But it isn't what *enables* long context, and Q6 shows its share shrinks as `s` grows.

**boson:** GDN linear-attention is hard-asserted to **CP=1**, so the right column is unavailable — long context must come from TP+SP and checkpointing ([[ch-07]]/[[ch-09]]).

**One line:** §4's sequence parallelism shards the LayerNorm/Dropout regions that tensor parallelism has to replicate — dividing the *entire* activation formula by `t` at zero communication cost — whereas the technique that actually enables long context is context parallelism / Ring Attention, a separate axis that shards attention itself. See read.md §4, [[ch-06]], [[ch-07]].

---

### Q5 — is Sequence Parallelism about small models that fit on one GPU?

**The opposite.** SP only appears when the model does **not** fit on one GPU. read.md L177: it *"requires that tensor parallelism communicators already exist"* — SP presupposes TP, and TP exists for exactly one reason: the model is too big for a single device. Fits on one GPU → no TP → no SP.

**The confusion is upstream, in TP: "if it's sharded, why is anything replicated?"** Because **TP shards *weights*, while the activations at region boundaries stay replicated.** One MLP at `t = 2`:

```
  GPU0                              GPU1
  x [B,T,h]  <---- full replica --->  x [B,T,h]     <- same tensor, twice
    | @ W_up left half                 | @ W_up right half
  [B,T,2h]                          [B,T,2h]        <- sharded (1/t)
    | @ W_down top half               | @ W_down bottom half
  partial [B,T,h]                   partial [B,T,h]
    +----------- all-reduce -----------+
  y [B,T,h]  <---- full replica --->  y [B,T,h]     <- replicated again
```

GPU0 needs **all** of `x` to compute `x @ W_up^(left)` — the columns were split, not `x`. **LayerNorm and Dropout sit on those replicated `[B,T,h]` boundaries**, so at `t=8` the same 33.55 MB tensor is stored **8 times**. That is the waste §4 targets.

**SP shards the boundary along tokens (T):** each rank keeps `x[:, 0:T/2, :]`, runs LayerNorm locally (valid — it is per-position), **AllGathers** to enter the TP region, and **ReduceScatters** on exit instead of all-reduce. The full `x` exists only transiently inside the TP region; what is *stored* for backward is the half. Since `all-reduce = ReduceScatter + AllGather`, bandwidth is unchanged.

| situation | TP | SP |
|---|---|---|
| model fits on one GPU | not needed | **not applicable** — no replicas exist |
| model does not fit | required | **turn it on too** — it is free |

**boson:** the 27B full-FT static floor is `16 B × 27B = 432 GB` ([[ch-01]] §2) — far past one A100-40GB, so this is unavoidably TP territory and SP is the default companion.

**One line:** SP is a by-product of TP, not a small-model technique — TP splits weights but replicates the activations at region boundaries `t` times, and SP shards those boundaries along the token axis to delete the last duplication; with no TP there are no replicas to delete. See read.md §4, Q3.

---

## Continued in [[qa-deep-2]]

- **Q6** — SP vs selective recomputation: different terms of the same formula, and why SP alone collapses at long sequence
