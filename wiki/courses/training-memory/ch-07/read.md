<!-- chapter: ch-07
     track: scaling
     title: Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP
     deps: [[ch-01]]
     sources: [[zero-memory-optimization]], [[megatron-tp-sp]], [[pipeline-parallelism-1f1b]], [[pytorch-fsdp]], [[deepspeed-moe-ep]]
-->

# Chapter 7 — Parallelism Taxonomy: DP / ZeRO / FSDP / TP / SP / PP / EP / CP

> **Core insight.** Every parallelism primitive is a partitioning decision over one axis of the training computation — batch (DP), optimizer state / gradient / weight (ZeRO), weight matrix columns/rows (TP), sequence positions (SP / CP), layer stages (PP), or expert parameters (EP). Each split trades communication overhead and synchronization complexity for a reduction in per-GPU memory. The identity `world_size = TP × PP × CP × DP` (where EP subdivides the DP-expert dimension) expresses the entire space; every production configuration is a point in this product.

> **Guideline.** Start with ZeRO-1 (free optimizer-state sharding) on any job with ≥4 DP ranks. Add TP=8 (one NVLink node) when a single ZeRO-2 rank still cannot hold even one layer's activations. Add PP to span multiple nodes. Add SP alongside TP ≥ 2 to recover the t× activation saving that TP alone misses at the normalization layers. Reserve EP for sparse MoE models where ZeRO-3's all-gather of all expert weights would itself OOM. Add CP only when sequence length exceeds what SP can handle within a single node.

---

## 1. What Every Primitive Shards — The Taxonomy Table

| Technique | What it splits | Per-GPU memory reduction | Communication cost per step |
|-----------|----------------|-------------------------|-----------------------------|
| **DP** (Data Parallel) | Batch | Nothing — full model replicated | All-reduce gradients: **2Ψ** |
| **ZeRO-1** | Optimizer states only | 4× Adam states at large N | Same as DDP: **2Ψ** |
| **ZeRO-2** | + Gradients | ~8× (states + grads) at large N | Same as DDP: **2Ψ** |
| **ZeRO-3 / FSDP** | + Parameters | **16Ψ/N** — linear in DP ranks | 1.5× DDP: AllGather + ReduceScatter = **3Ψ** |
| **TP** (Tensor Parallel) | Weight matrix columns/rows | 1/t weights + 1/t linear-op activations | 2 AllReduces per transformer layer (MLP + Attn) |
| **SP** (Sequence Parallel) | Non-TP activations (norm, dropout) along sequence | Full t× activation saving across entire layer | AllGather + ReduceScatter (replaces AllReduce — same BW) |
| **PP** (Pipeline Parallel) | Layers into stages | 1/p weights; activations only for local stage | P2P sends/receives at stage boundaries per microbatch |
| **EP** (Expert Parallel) | Expert weight matrices across MoE ranks | E/EP_size experts per GPU | 2 All-to-Alls per MoE layer (dispatch + combine) |
| **CP** (Context Parallel) | Sequence positions across CP ranks | Activations along sequence dim | Ring-AllReduce over KV blocks per attention step |

> **Interactive companion:** [figures/parallelism-sharding.html](figures/parallelism-sharding.html) — hover over each parallelism type to see which tensor axes are sharded, the communication pattern, and a live per-GPU memory estimate for a configurable model size.

---

## 2. Data Parallelism — The Replication Baseline

DP is the null case: every rank holds a complete copy of the model and processes a different mini-batch slice. The backward pass produces per-rank gradients that must be globally averaged via an **all-reduce** before the optimizer step. DDP (PyTorch DistributedDataParallel) overlaps this all-reduce with the backward pass bucket by bucket.

**Memory:** DP reduces nothing. For a 7.5B model in mixed precision the per-GPU footprint is the full `16Ψ` = 120 GB. At 80 GB A100s this is already infeasible for a single-node run without sharding.

**Communication:** One all-reduce per step transferring **2Ψ bytes** (bf16 gradients for all parameters). This is the baseline every other scheme is compared against.

DP alone is therefore a memory-neutral distribution scheme — it gets you data throughput scaling at zero memory benefit. Every other primitive in this chapter is about breaking the 16Ψ wall.

---

## 3. ZeRO-1/2/3: Partition the Optimizer Redundancy

ZeRO ([[zero-memory-optimization]], Rajbhandari et al. SC 2020) observes that DP wastes memory by replicating all three optimizer components on every rank — there is no mathematical requirement for each rank to hold all states; it only needs to apply its shard of the gradient to its shard of the parameters and broadcast the result.

### 3.1 The 16Ψ Decomposition

Mixed-precision AdamW requires exactly (from [[zero-memory-optimization]]):

```
2Ψ fp16 weights  +  2Ψ fp16 gradients  +  4Ψ fp32 master params
+  4Ψ fp32 momentum  +  4Ψ fp32 variance  =  16Ψ bytes/param
```

The `K=12` optimizer component (the last three terms: 4+4+4 = 12 B/param) is pure redundancy across DP ranks; each rank needs only its own 1/N slice to compute and apply the update.

### 3.2 Three Stages of Partitioning

**ZeRO-1 (Pos):** partition optimizer states only. Each rank stores `KΨ/Nd` of the optimizer state instead of `KΨ`, plus the full `4Ψ` weight + gradient block replicated.

- Per-GPU formula: `4Ψ + KΨ/Nd` — at Nd=64 with K=12: `(4 + 12/64)Ψ ≈ 4.19Ψ` per param.
- Communication: unchanged from DDP (**2Ψ** all-reduce). This stage is always worth enabling.

**ZeRO-2 (Pos+g):** additionally partition gradients. Gradients are reduce-scattered into their owner shard during backward instead of all-reduced.

- Per-GPU formula: `2Ψ + (2+K)Ψ/Nd`.
- Communication: still **2Ψ** (reduce-scatter has same volume as all-reduce).

**ZeRO-3 (Pos+g+p):** partition parameters as well. Every rank holds only `Ψ/Nd` of the weights; each layer's parameters must be all-gathered before that layer's forward and backward.

- Per-GPU formula: `16Ψ/Nd` — linear scaling with DP degree.
- Communication: **3Ψ** per step (AllGather Ψ before forward + AllGather Ψ before backward + ReduceScatter Ψ during backward) = **1.5× DDP**.

### 3.3 Concrete Numbers: 7.5B Model, Nd = 64

From [[zero-memory-optimization]]:

| Scheme | Formula | Per-GPU memory |
|--------|---------|----------------|
| DDP | 16Ψ | **120 GB** |
| ZeRO-1 | 4Ψ + 12Ψ/64 | **31 GB** |
| ZeRO-2 | 2Ψ + 14Ψ/64 | **17 GB** |
| ZeRO-3 | 16Ψ/64 | **1.9 GB** |

A 40 GB A100 cannot run DDP for this model at any batch size. ZeRO-3 brings the weight+optimizer footprint to under 2 GB — leaving nearly the entire GPU for activations and the attention kernel workspace.

**Critical caveat from [[zero-memory-optimization]]:** "activations and logit-buffer spike are not touched by ZeRO — those require separate treatment (activation checkpointing, sequence parallelism)." ZeRO eliminates state redundancy; it does not touch the forward-pass computation graph.

---

## 4. FSDP — ZeRO-3 as a PyTorch Primitive

PyTorch FSDP ([[pytorch-fsdp]], Zhao et al. VLDB 2023) implements ZeRO-3 natively, making it composable with TP, PP, and autograd through PyTorch's dispatcher.

### 4.1 The Per-Step Lifecycle

From [[pytorch-fsdp]], under `FULL_SHARD`:

```
1. AllGather parameters for current FSDP unit    → forward pass
2. Free gathered parameters immediately
3. AllGather parameters again                    → backward pass
4. ReduceScatter gradients into shard
5. Optimizer step on local shard only
```

Communication = 2 AllGathers (2Ψ) + 1 ReduceScatter (Ψ) = **3Ψ total** — identical to ZeRO-3.

### 4.2 The AllGather Transient Peak

FSDP wraps parameters into **FSDP units** (typically one transformer layer = one unit). During each unit's forward/backward, an AllGather buffer of size `≈ 2 × P_unit` is transiently allocated for the gathered full-precision parameters. The complete per-GPU memory formula is:

```
Memory = 16Ψ/N  +  2 × P_unit   (transient peak during largest unit's forward/backward)
```

The `2 × P_unit` buffer is the practical OOM risk under FSDP — it does not appear in the steady-state ZeRO-3 formula and must be budgeted separately. Wrapping at finer granularity (smaller FSDP units) reduces this buffer but increases the number of AllGather calls.

### 4.3 HYBRID_SHARD for Multi-Node

For jobs spanning multiple nodes, FSDP's `HYBRID_SHARD` mode performs `FULL_SHARD` within each NVLink-connected intra-node group (fast all-gather) and `REPLICATE` across nodes (avoids expensive inter-node AllGather). This trades partial memory savings for lower inter-node bandwidth consumption — the default for production multi-node training.

### 4.4 The clip_grad_norm_ Trap

From [[pytorch-fsdp]]: calling `torch.nn.utils.clip_grad_norm_` on local shards under FSDP "under-reports the global norm by √N and causes silent divergence." Always call `model.clip_grad_norm_(max_norm)` on the FSDP module itself, which performs a distributed norm computation across all ranks.

---

## 5. Tensor Parallelism — Split the Weight Matrix

Tensor parallelism ([[megatron-tp-sp]], Shoeybi et al. 2019) targets a different redundancy: within a single transformer layer, each GPU still holds the full weight matrix. TP shards that matrix across t GPUs (typically one NVLink node, t ≤ 8).

### 5.1 Column-Then-Row Split

For the MLP:

- First linear (up-projection): split **column-wise** — each rank computes a different subset of output features. No communication needed before this op (inputs are replicated).
- Second linear (down-projection): split **row-wise** — each rank holds a subset of input features. An **AllReduce** after this op sums the partial results.

Attention follows the same pattern: Q/K/V projections are column-split by attention heads; the output projection is row-split; one AllReduce synchronizes.

Total communication: **2 AllReduces per transformer layer** (one for MLP, one for attention). TP demands fast intra-node interconnect (NVLink) because these AllReduces are on the critical path — latency-bound, not throughput-bound at small t.

### 5.2 Activation Memory: The 10sbh Problem

TP reduces weight memory by 1/t and linear-op activation memory by 1/t. But the normalization layers (LayerNorm, Dropout) are **not split by TP** — they remain replicated on every rank. From [[megatron-tp-sp]], activation memory with TP=t and no SP:

```
activation memory per layer = sbh(10 + 24/t + 5as/ht) bytes
```

The `10sbh` term is the LayerNorm + Dropout region: replicated across all t ranks, unchanged by TP. For long sequences this term dominates, and TP alone achieves much less than t× activation reduction.

---

## 6. Sequence Parallelism — The 10sbh Fix

Sequence parallelism ([[megatron-tp-sp]], Korthikanti et al. 2022) shards the `10sbh` replicated activations along the **sequence dimension** across the same t TP ranks. This requires no extra bandwidth: SP replaces the AllReduce before/after the TP regions with an AllGather + ReduceScatter pair — same communication volume, but now the non-TP activations are also split 1/t.

With SP enabled alongside TP=t, activation memory becomes:

```
activation memory per layer = (sbh/t)(34 + 5as/h) bytes
```

This is a **true t× reduction** over TP alone. At t=8 (one node), SP delivers 8× activation savings across every layer, not just the linear-op regions.

From [[megatron-tp-sp]]: the 530B GPT-3 run on 2240 A100s achieved 54.2% MFU with SP + selective recompute vs. 42.1% without — a 29% throughput improvement attributable to the memory headroom SP unlocks.

**SP is only active when TP > 1.** It piggybacks on TP's intra-node communication groups; enabling SP at TP=1 is a no-op.

### 6.1 Selective Recomputation (the 5as²b/ht Term)

From [[megatron-tp-sp]], Korthikanti 2022: rather than checkpointing entire layers (30–40% FLOP overhead), recompute only the attention softmax/score region — the `5as²b/ht` bytes of low compute-density activations (attention weights and intermediate scores):

- Memory saving: ~5× reduction in activation footprint vs. storing everything
- FLOP overhead: < 2% additional computation
- vs. full-layer recompute: same memory benefit, 15–20× cheaper in FLOPs

The threshold for when selective recompute pays more than SP is model-dependent; at long sequences (a > h/5), the attention region dominates and selective recompute is usually combined with SP for maximum effect.

---

## 7. Pipeline Parallelism — Split the Layers

Pipeline parallelism ([[pipeline-parallelism-1f1b]]) is the only primitive that places different **layers** on different GPUs. Each pipeline stage p holds 1/p of the total layers, immediately dividing weight memory by p. Communication between stages is point-to-point tensor sends at stage boundaries.

### 7.1 The Bubble

Staging creates a pipeline startup and drain bubble. Both GPipe (all-forward-then-all-backward) and 1F1B (interleaved microbatch scheduling) share the same bubble fraction:

```
bubble fraction = (p - 1) / (m + p - 1)  ≈  (p - 1) / m  when m >> p
```

where m = number of microbatches (= total batch / microbatch size). To keep the bubble below 5%, the rule of thumb is **m ≥ 20 × (p - 1)**. A 4-stage pipeline requires at least 60 microbatches of gradient accumulation. This is the PP tax: you must accumulate enough microbatches to amortize the bubble, which itself affects batch-effective throughput.

### 7.2 GPipe vs. 1F1B: The Memory Difference

Both schedules have identical bubble fractions, but they differ critically in **activation memory**:

| Schedule | Activation peak per stage | Scales with |
|----------|--------------------------|-------------|
| GPipe | m microbatch activations (all-forward-then-backward) | O(m) — unbounded |
| 1F1B | p microbatch activations (at most p in flight) | O(p) — constant |

From [[pipeline-parallelism-1f1b]]: 1F1B "caps activation memory at p microbatches in flight (the pipeline depth) → memory scales O(p), independent of m." 1F1B's advantage over GPipe is **entirely in memory, not in bubble time**.

### 7.3 Interleaved 1F1B (Virtual Stages)

Megatron's interleaved schedule assigns each device v non-contiguous virtual stages instead of one contiguous stage. Bubble fraction drops to:

```
bubble fraction (interleaved) = (p - 1) / (v × m)
```

Cost: v additional P2P sends/receives per microbatch, and slightly higher per-rank activation peak (~v× the standard 1F1B within a stage). This is the production setting for large PP degree at Megatron scale.

---

## 8. Expert Parallelism — Sparse MoE at Scale

For Mixture-of-Experts models ([[deepspeed-moe-ep]]), expert parallelism shards the expert weight matrices rather than the dense parameters.

### 8.1 Memory Model

With E total experts and EP_size ranks:

```
expert weights per GPU = W_expert / EP_size
```

Each GPU holds E/EP_size expert weight matrices. Non-expert layers (attention, embedding, non-expert FFN) are replicated under DP or sharded under TP/PP as usual. For a 256-expert MoE: an EP_size=64 configuration gives 4 experts per GPU — dramatically smaller than the DP-replicated baseline.

**Why ZeRO-3 cannot replace EP for large MoE models:** ZeRO-3 all-gathers all parameters before each layer's forward pass. For a 256-expert dense model with experts representing 16× the dense parameter count, the ZeRO-3 all-gather would transiently load the full `W_expert` onto each GPU — immediately OOMing the job that EP was designed to solve.

### 8.2 All-to-All Routing — Two Collectives Per MoE Layer

Expert parallelism requires tokens to be physically routed to the GPU holding the target expert:

```
Forward MoE layer:
  1. Dispatch all-to-all: each GPU sends its assigned tokens to EP ranks with target experts
  2. Run expert FFN on received tokens (local computation)
  3. Combine all-to-all: each GPU receives expert outputs and aggregates by routing score
```

From [[deepspeed-moe-ep]], each all-to-all transfers `tokens × d_model` bytes per rank. Unlike all-reduce (fixed volume), the all-to-all volume **scales with sequence length × batch size** and can become the dominant communication bottleneck at long contexts.

**All-to-all buffer:** Each of the two collectives requires a transient buffer of size `tokens × d_model × 2 bytes` — these must be budgeted alongside the steady-state expert weight footprint.

### 8.3 Capacity Factor

```
capacity = capacity_factor × (tokens_per_batch / num_experts)
```

A capacity factor below 1.0 causes **token dropping** (tokens routed to over-loaded experts are silently discarded), which produces training instability. Setting capacity_factor ≥ 1.0 is the minimum safe threshold; common production values are 1.0–2.0.

---

## 9. Context Parallelism — Ring Attention for Long Sequences

Context parallelism shards the **sequence dimension** across CP ranks, with each rank responsible for a contiguous subsequence. Cross-rank attention requires a Ring Attention pattern: each rank passes its KV blocks around the ring so every rank eventually attends over all positions.

CP and SP address overlapping problems at different scales:
- **SP** operates within a single node (same TP group, NVLink), sharding normalization activations.
- **CP** operates across nodes (or large intra-node groups), sharding the sequence itself and the attention computation.

CP is activated only when sequence length exceeds what SP alone can handle at a given TP degree. From the identity: **world_size = TP × PP × CP × DP** — CP and TP occupy orthogonal dimensions; they can be composed.

---

## 10. The World-Size Identity and Composition Rules

All five non-EP dimensions compose multiplicatively:

```
world_size = TP × PP × CP × DP
```

EP subdivides the DP dimension: the EP ranks form a subset of the DP group's GPUs, so EP_size ≤ DP. The practical composition for a 64-GPU MoE job might be:

```
TP=8, PP=4, CP=1, DP=2  →  world_size = 64
EP=2  (subdivides DP=2)
```

**Order of application (heuristic):**
1. Set TP = GPUs per NVLink node (typically 8) — intra-node, high bandwidth.
2. Enable SP whenever TP ≥ 2.
3. Set PP = number of nodes needed to span layers — inter-node P2P.
4. Set DP = remaining dimension (DP = world_size / (TP × PP × CP)).
5. Apply ZeRO-3 / FSDP within the DP group.
6. If MoE: set EP ≤ DP to shard experts.
7. Add CP only if sequence length still exceeds per-GPU memory at the current TP×SP.

---

## 11. LoRA Node-Invariance vs. Full-FT World-Size Scaling

A critical asymmetry between full fine-tuning and LoRA:

**Full fine-tuning with ZeRO-3:** weight + optimizer state memory scales as `16Ψ/N` with DP degree N. Doubling DP (adding more GPUs) halves per-GPU state memory. The training job's memory footprint is world-size-sensitive — adding GPUs directly reduces per-GPU load.

**LoRA:** only the adapter parameters (rank r ≪ Ψ) accumulate gradients and optimizer states. The frozen base model still participates in forward/backward — each GPU holds the full frozen weights (2 B/param for inference) and computes the full activation graph to propagate gradients to the adapters.

From this: **LoRA activation memory is world-invariant.** Adding DP replicas does not reduce the per-GPU activation footprint, because:
1. The frozen base's activations must still be stored for the adapter backward pass.
2. DP adds batch copies, not sequence or layer splits.

The consequence: LoRA jobs that OOM on activations cannot be fixed with more DP ranks. The fix is activation checkpointing, shorter sequences, or reduced batch size — not scaling out.

Optimizer state and gradient memory do shrink under LoRA (only `r` rank matrices are trained), making the full-FT-like overhead from the base model optimizer irrelevant. But activations dominate at the sizes where LoRA is typically used (large base models, moderate batch sizes), so the node-invariance constraint is the binding one.

---

## 12. The Logit-Buffer Spike — Not Reduced by Any Parallelism

Revisiting [[ch-02]]'s loss-head spike: the logit buffer allocated at the forward-backward seam is:

```
spike = vocab_size × seq_len × batch_size × 2 bytes (bf16)
```

For a 32K-vocab model at seq=2048, batch=1: `32768 × 2048 × 1 × 2 ≈ 0.13 GB` — negligible. But for a 128K-vocab model at seq=4096, batch=4: `131072 × 4096 × 4 × 2 ≈ 4.3 GB` — and for production batch sizes with 256K vocab, the spike can reach 10–20 GB per step.

**Critically, this spike is not reduced by ZeRO, TP, PP, or SP.** From [[zero-memory-optimization]]: ZeRO does not touch activations or the logit buffer. TP splits the weight matrix of lm_head but the output logit tensor is re-gathered before the cross-entropy loss. The only lever is fused/chunked cross-entropy (Liger kernel, [[ch-02]]) which computes the loss in tiles, never materializing the full logit tensor.

OOM at `lm_head` during forward = the logit spike. The fix is chunked cross-entropy, not a parallelism change.

---

## Core Insights from the Literature

**1. ZeRO's 1.5× communication overhead is the minimum theoretical cost for eliminating parameter redundancy** ([[zero-memory-optimization]]). DDP needs 2Ψ (one all-reduce); ZeRO-3 needs 3Ψ (two all-gathers + one reduce-scatter). Any scheme that distributes parameters must pay at least one extra Ψ of communication vs. a scheme that replicates them. The paper frames this as "ZeRO achieves memory efficiency while retaining computational efficiency."

**2. SP is not a standalone technique — it only exists inside TP's communication group** ([[megatron-tp-sp]]). SP replaces TP's AllReduce with AllGather + ReduceScatter, a refactoring that opens the sequence dimension for sharding at zero additional bandwidth cost. This is why enabling SP is essentially free given TP > 1.

**3. 1F1B's innovation is entirely in memory, not in scheduling efficiency** ([[pipeline-parallelism-1f1b]]). Both GPipe and 1F1B achieve `(p-1)/m` bubble fraction. GPipe's flaw — O(m) activation memory — is fixed by 1F1B's rule of starting backward for microbatch k the moment its forward completes, capping in-flight activations at p.

**4. EP and ZeRO-3 are complementary, not redundant, for MoE** ([[deepspeed-moe-ep]]). ZeRO-3 shards parameters across DP ranks and all-gathers them per layer — fine for dense models. For sparse experts that are 4–16× the dense parameter count, the ZeRO-3 all-gather transient peak exceeds GPU memory. EP places experts permanently on their assigned GPU; ZeRO-3 then handles the remaining dense parameters. DeepSpeed-MoE combines both.

---

## Key Takeaways

- The 16Ψ formula decomposes into exactly `2Ψ + 2Ψ + 12Ψ` (weights, gradients, Adam states). ZeRO-3 partitions all three to reach `16Ψ/N` — activations and the logit spike are outside ZeRO's scope.
- FSDP is ZeRO-3 with one extra risk: the `2 × P_unit` AllGather transient buffer, invisible in the steady-state formula but the real OOM trigger.
- TP alone does not achieve t× activation reduction because LayerNorm/Dropout are replicated. SP fixes this by sharding those ops along the sequence. The correct formula pair is TP-only `sbh(10 + 24/t + 5as/ht)` vs SP+TP `(sbh/t)(34 + 5as/h)`.
- GPipe and 1F1B have identical bubble fractions; 1F1B's advantage is O(p) vs O(m) activation memory.
- EP is mandatory for large MoE (>~16 experts per GPU equivalent); two all-to-all collectives per MoE layer scale with sequence × batch, not model size.
- LoRA activation memory is node-invariant: more DP replicas do not help. The binding constraint for LoRA OOM is activation size, fixed only by checkpointing or sequence reduction.
- The logit-buffer spike is not reduced by any parallelism. Use chunked cross-entropy.
- The composition identity `world_size = TP × PP × CP × DP` is the design space. EP ≤ DP. Apply TP first (intra-node, highest bandwidth), then PP (inter-node layers), then fill DP.

---

## References

- Rajbhandari, S. et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC '20. https://arxiv.org/abs/1910.02054 ([[zero-memory-optimization]])
- Shoeybi, M. et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. https://arxiv.org/abs/1909.08053 ([[megatron-tp-sp]])
- Korthikanti, V. et al. (2022). Reducing Activation Recomputation in Large Transformer Models. MLSys 2023. https://arxiv.org/abs/2205.05198 ([[megatron-tp-sp]])
- Huang, Y. et al. (2019). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. NeurIPS 2019. https://arxiv.org/abs/1811.06965 ([[pipeline-parallelism-1f1b]])
- Narayanan, D. et al. (2021). Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC '21. https://arxiv.org/abs/2104.04473 ([[pipeline-parallelism-1f1b]])
- Zhao, Y. et al. (2023). PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. VLDB 2023. https://arxiv.org/abs/2304.11277 ([[pytorch-fsdp]])
- Lepikhin, D. et al. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. ICLR 2021. https://arxiv.org/abs/2006.16668 ([[deepspeed-moe-ep]])
- Rajbhandari, S. et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a ([[deepspeed-moe-ep]])
