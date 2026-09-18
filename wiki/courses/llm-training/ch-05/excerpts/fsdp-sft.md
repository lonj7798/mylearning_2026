---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fsdp-sft.md
source_url: https://arxiv.org/abs/2304.11277
source_type: paper
created_at: "2026-04-23"
rewritten_on: "2026-09-17"
verified_against: "arXiv:2304.11277 (v2, 12 Sep 2023; PVLDB 16(12)), cached primary text"
---

# Excerpt: PyTorch FSDP — Experiences on Scaling Fully Sharded Data Parallel

**Authors:** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, et al. (Meta AI)
**Year:** 2023 (arXiv v1 2023-04; PVLDB 16(12): 3848–3860)
**Used by:** [[read]] §1, §2, Recipe.

> **Rewritten 2026-09.** The 2026-04 version of this excerpt carried a per-GPU memory table
> (`2P / 2P / 12P`, total `16P`, `(16P/N) + 2P`), a 70B worked example (`DDP = 1120 GB`,
> `FSDP ≈ 280 GB`), a `ShardingStrategy` enum table, a `MixedPrecision(...)` snippet, a 70B SFT recipe
> table, and a block quote beginning "FSDP shards model parameters, gradients, and optimizer states…".
> None of those appear in the paper. They were removed here and in the library card
> [[fsdp-sft]] during the 2026-09-14 verification pass. Only statements checked at a locus remain.

## Sharding factor

The paper's single control is the sharding factor `F`, the number of ranks a parameter is sharded over
(§3.2). `F = 1` is full replication (AllReduce, the DDP behaviour). `F = W` (the world size) is full
sharding. `1 < F < W` is hybrid sharding: parameters are sharded within each group `S_1 … S_{W/F}` and
replicated within each complementary group `R_1 … R_F` (§3.2.2).

## Peak parameter memory (§3.2.1)

For a model with `Ψ` elements split into `N` FlatParameters with element counts `ψ_1 … ψ_N`
(`Σ ψ_i = Ψ`), and sharding factor `F`:

```
peak parameter memory  ∈  O( Σ_{i=1..N} ψ_i / F  +  max_{i=1..N} ψ_i )
collectives per iteration ∈ O(N)
```

The paper's reason, quoted in substance: FSDP always keeps each local sharded FlatParameter of size
`ψ_i / F` in GPU memory and "must materialize each unsharded FlatParameter with size ψ_i one by one
during forward and backward". Since `Σ ψ_i = Ψ` is fixed, the peak is determined by `max_i ψ_i`
(§3.2.1). Consequence stated in the same paragraph: "Finer-grained FlatParameter construction
decreases peak memory but may decrease throughput by requiring more collectives."

The second term is the **largest single FSDP unit**, not the whole model.

Under mixed precision (§4.4), peak parameter memory changes from
`K_full·Σψ_i/F + K_full·max_i ψ_i` to `K_full·Σψ_i/F + K_low·max_i ψ_i` bytes, where `K_low` and
`K_full` are bytes per low- and full-precision element.

## Communication volume (§3.2.1, §3.2.2)

- "the full sharding has 1.5x communication overhead and volume over DDP" under a bandwidth-optimal
  ring algorithm (§3.2.1).
- For an `M`-sized model on `W` accelerators grouped into hosts of `G` accelerators, cross-host traffic
  per GPU is `2M(W−1)/W` for full replication, `3M(W−1)/W` for full sharding, and `2M(W−1)/(GW)` for
  hybrid sharding with `F = W/G` (§3.2.2).
- Hybrid sharding's AllReduce runs at a smaller world size and "empirically achieve a better
  performance than invoking collectives at the global scale … due to straggler effects and larger
  network interference" (§3.2.2).
- Collective efficiency depends on message size: with total communication fixed at `2^30 ≈ 1B` FP32
  elements, "Once the AllGather size decreases below 33M elements, the total communication time begins
  increasing rapidly" (§3.2.1, Figure 2b).

## Measured results (§5)

- Hardware: 8 to 512 A100 80GB GPUs on a 2 Tb/s RoCE network (§5.1).
- T5 611M and 2.28B: FSDP and DDP perform similarly. DDP runs out of memory above 2.28B. FSDP trains
  T5-11B (§5.2, Figure 6a).
- GPT-175B (minGPT, vocab 50,000, block size 2048): more than 173 and 186 TFLOPS per GPU at batch 1 and
  2, about 55% and 60% of the A100 BF16 peak of 312 TFLOPS, with linear scaling from 128 to 512 GPUs
  (§5.4, Figure 7b). Runs used full sharding with prefetching and rate limiter, activation
  checkpointing, BF16, and Adam (§5.4).
- Backward prefetching gave about 18% speedup on GPT-175B (§5.2, Figure 6b).
- T5-11B: per-GPU TFLOPS falls 7% from 8 to 512 GPUs (§5.4, Figure 7c).
- Rate limiter: up to 5× speedup on T5, no speedup on RegNet, 5% overhead on DeepViT (§5.3).

## Stated limitations (§7)

- Optimizer computations that depend on an original parameter's unsharded value (for example a vector
  norm), its tensor structure, or global state are not mathematically equivalent to local training
  (§7.2.1).
- Shared parameters must belong to the lowest-common-ancestor FSDP unit (§7.2.2).
- With pipeline parallelism, default full sharding re-gathers parameters for every micro-batch (§7.1.1).

## Not reported by the source (checked)

Any SFT or fine-tuning setting; learning rates; activation-memory accounting; per-component byte counts
for Adam states; any 70B model.
