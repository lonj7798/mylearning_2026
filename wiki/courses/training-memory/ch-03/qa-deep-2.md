<!-- qa-deep-2: ch-03 — combining the activation levers (overflow from [[qa-deep]])
     companion to [[read]] · append-only across cycles · kernels/index only -->

# ch-03 Q&A (deep 2) — combining the activation levers

Third page for ch-03. [[qa]] holds the checkpointing mechanism; [[qa-deep]] holds
sequence parallelism; this page holds how the levers compose.

---

### Q6 — is SP completely different from Selective Recomputation? And if SP is on, are the others redundant?

**Different category, yes:**

| | Selective Recomputation | Sequence Parallelism |
|---|---|---|
| mechanism | discard, then **recompute** | **delete cross-GPU duplicates** |
| price | +4 % compute | **none** |
| works on a single GPU? | ✅ | ❌ meaningless without TP |

**Redundant, no — they attack different *terms of the same formula*.** The per-layer coefficient always splits:

```
coefficient = A + B
              |   +-- 5as/(ht)  attention score matrix, quadratic in s
              +------ 10 + 24/t  everything else, linear in s
```

**SP only shrinks `A`** (`10 + 24/t → 34/t`, read.md L172). **Selective recomputation deletes `B` outright** (read.md L160: *"the quadratic term vanishes entirely"*). Neither can touch the other's term.

**s = 4096, h = 4096, a = 32, t = 8, L = 80** (`sbh` = 16.78 MB, `5as/(ht)` = 20):

| combination | A | B | coeff | per layer | ×80 |
|---|---|---|---|---|---|
| TP only | 13 | 20 | 33 | 554 MB | **44.3 GB** |
| TP + **SP** | 4.25 | 20 | 24.25 | 407 MB | **32.6 GB** |
| TP + **selective** | 13 | **0** | 13 | 218 MB | **17.4 GB** |
| TP + SP + selective | 4.25 | **0** | 4.25 | 71 MB | **5.7 GB** |

1.36× alone, 2.54× alone, **7.8× together** — neither substitutes for the other. (The combined `sbh·34/t` is composed from read.md L157 + L172, not quoted directly.)

**At long sequence SP alone nearly stops working.** At `s = 32,768` the term `5as/(ht)` jumps 20 → 160:

| combination | coeff | per layer | vs TP only |
|---|---|---|---|
| TP only | 173 | 23.2 GB | — |
| TP + **SP** | 164.25 | 22.0 GB | **1.05×** |
| TP + SP + selective | 4.25 | 0.57 GB | **41×** |

The `s²` term swallows everything and SP cannot reach it — so at long context selective recomputation is the decisive lever, not the optional one. Also, SP's divisor `t` is **capped**: TP is communication-heavy and must stay inside a node (NVLink), so `t ≤ 8` typically. Checkpointing has no such ceiling.

**One line:** SP owns the linear term and selective recomputation owns the `s²` term — complementary, not substitutable — and since the quadratic term dominates as sequence grows, SP alone buys 5 % at s=32k while adding selective recomputation buys 41×. See read.md §3/§4, Q3, Q5.

---

### Q7 — doesn't selective recomputation get risky at large scale, since recompute means slow cross-node traffic?

**Good instinct, wrong target: recomputation is entirely local and adds zero communication.**

Selective recomputation rebuilds one tensor — the `[b, a, s, s]` attention score matrix (read.md L137) — from `Q` and `K`, **which live on the same GPU**. TP shards attention **by head**, so a rank owning heads 0–3 holds all of their Q/K/V:

```
inside GPU0 only:  Q0,K0 (already held) --matmul--> QK^T --softmax--> scores
                                                          0 communication
```

Nothing is fetched from another rank. That is *why* the overhead is <4 % — pure FLOPs, no new collectives.

**Where the concern IS valid: full checkpointing over a TP region.** Re-running a whole block's forward in backward replays the collectives that forward performed — one attention all-reduce and one MLP all-reduce per block. Selective recomputation is designed precisely to avoid that; shift the target one slot and the worry is correct.

**Measurement points the other way** (read.md L148–152): validated at **530B on 2,240 A100s** — the largest scale in this chapter — selective recomputation raised MFU **42.1 % → 54.2 % (+29 % throughput) versus full recomputation**. Bigger scale made it *more* worthwhile, because memory pressure grows and the alternatives (more TP, or full recompute) cost more.

**The real communication risk is a different axis** — which parallel dimension crosses the node boundary:

| axis | traffic | placement |
|---|---|---|
| **TP (+SP)** | all-reduce **every layer** — the chatty one | **must stay inside a node** (NVLink ~600 GB/s) |
| PP | one activation per stage boundary | fine across nodes (small) |
| DP / ZeRO | gradient all-reduce once per step | fine across nodes (large but once) |

This is where `t ≤ 8` comes from: push TP past a node and every layer rides InfiniBand, collapsing throughput. **boson:** 8 GPUs per A100 node → `t ≤ 8`; beyond that add PP or ZeRO across nodes rather than growing TP. Formalized in [[ch-07]] as `world_size = TP × PP × CP × DP`.

**One line:** recompute reconstructs from inputs the GPU already holds, so it costs FLOPs and no communication — what actually adds cross-node traffic is pushing TP outside a node, and empirically selective recomputation becomes *more* favorable at scale (MFU 42→54 % at 530B/2,240 GPUs). See read.md §3, Q5, Q6, [[ch-07]].
