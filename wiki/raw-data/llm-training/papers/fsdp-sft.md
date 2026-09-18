<!-- scope: PyTorch FSDP systems paper (Meta, PVLDB 2023): FlatParameter sharding, sharding factor and hybrid sharding, communication overlap and prefetching, rate limiter, native mixed precision, and throughput/memory benchmarks up to GPT-175B on 512 A100s; contains no SFT recipe
     deps: []
     see-also: [[mixed-precision]], [[sequence-packing]], [[loss-masking-prompt]]
-->

# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
- **Core Insight:** FSDP keeps parameters sharded except for the one FSDP unit currently computing, so peak parameter memory is O(Σψ_i/F + max_i ψ_i) (§3.2.1); in the paper's runs DDP ran out of memory above a 2.28B T5 model, while FSDP trained T5-11B and GPT-175B, reaching more than 173 and 186 TFLOPS per GPU on the 175B model at batch sizes 1 and 2 (§5.2, §5.4).
- **Guideline:** When a model's memory under full replication is only slightly above device capacity, the paper describes hybrid sharding (sharding factor between 1 and the world size) as the better fit, because full sharding has 1.5× the communication volume of DDP with a ring algorithm (§3.2.1, §3.2.2); enable the rate limiter only after confirming allocator retries (`num_alloc_retries`), because it gave up to 5× speedup on T5 but 5% overhead on DeepViT (§5.3).
- **Authors:** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, et al. (18 authors, Meta AI)
- **Year:** 2023 (arXiv v1 2023-04; PVLDB 16(12): 3848–3860, 2023)
- **URL:** https://arxiv.org/abs/2304.11277
- **Source type:** paper
- **Relevant topics:** distributed data parallelism, parameter sharding, communication scheduling, mixed precision, GPU memory management

## Abstract
The paper presents PyTorch Fully Sharded Data Parallel (FSDP) as an industry-grade solution for large-model training. FSDP is co-designed with PyTorch's Tensor implementation, dispatcher, and CUDA caching allocator to give a non-intrusive user experience and high training efficiency, and it includes settings for different hardware configurations. The authors report that FSDP matches Distributed Data Parallel (DDP) performance on small models while supporting larger models with near-linear scalability in TFLOPS (abstract).

## Key Contributions
- Deferred initialization: the model is created on a fake device, initialization operations are recorded, and they are replayed unit by unit on the GPU, sharding each unit before the next (§3.1).
- FlatParameter: all parameters of an FSDP unit are flattened, concatenated, right-padded (at most F − 1 elements), and split into F equal shards, so AllGather and ReduceScatter need no extra copies (§3.2.1, §4.2).
- Sharding factor F: F = 1 is full replication with AllReduce; F = W (world size) is full sharding; 1 < F < W is hybrid sharding, with ReduceScatter inside sharding groups and AllReduce across replication groups (§3.2, Eq. 1).
- Communication scheduling: AllGather on a separate CUDA stream to overlap with computation, backward and forward prefetching, and gradient accumulation with or without communication (§3.3).
- Memory management: a rate limiter that allows at most two in-flight AllGathers (§3.4.2).
- Native mixed precision with independent dtypes for parameters, gradient reduction, and buffers, plus a sharded gradient scaler for FP16 (§4.4).

## Key Figures/Tables to Study
- Figure 1: algorithm overview (gather, compute, free per unit in forward and backward).
- Figure 2: AllGather time for even vs uneven inputs and for decreasing per-call size.
- Figures 3 and 4: full sharding and hybrid sharding on 16 GPUs.
- Figure 5: overlap of communication and computation, including prefetching.
- Figures 6–8: model scale vs DDP, backward prefetching, rate limiter, throughput and peak memory for DHEN, GPT-175B, and T5-11B. The paper has no numbered tables.

## Technical Details
**Memory.**
- FSDP memory is proportional to the size of the sharded model plus the largest fully materialized FSDP unit; optimizer states stay sharded for the whole training loop (§3).
- Peak parameter memory is O(Σ_{i=1..N} ψ_i/F + max_i ψ_i), and the number of collectives per iteration is O(N) (§3.2.1). Here N is the number of FlatParameters (FSDP units), ψ_i is the number of elements in FlatParameter i, and F is the sharding factor. Finer wrapping lowers peak memory and may lower throughput (§3.2.1).
- Mixed precision: forward and backward use low precision and the optimizer step uses full precision. FSDP lowers parameter peak memory from K_full·Σψ_i/F + K_full·max_i ψ_i to K_full·Σψ_i/F + K_low·max_i ψ_i bytes, where K_low and K_full are bytes per low- and full-precision element (§4.4).
- Gradient accumulation without communication keeps unsharded gradients on each rank, trading memory for less communication (§3.3.4).
- Sharding strategies that keep parameters unsharded after forward still shard gradients and optimizer states (§7.1.1). RAF (reshard after forward) lowers peak memory at higher communication cost; NRAF keeps unsharded parameters until backward finishes (§5.4).

**Communication.**
- Full sharding has 1.5× the communication overhead and volume of DDP under a bandwidth-optimal ring algorithm (§3.2.1).
- For an M-sized model, cross-host traffic per GPU is 2M(W−1)/W for full replication, 3M(W−1)/W for full sharding, and 2M(W−1)/(GW) for hybrid sharding with G accelerators per host (§3.2.2).
- Even input sizes (NCCL All-Gather Base) gave the highest efficiency. With total communication fixed at 2^30 ≈ 1B FP32 elements, total time rose rapidly once the per-AllGather size fell below 33M elements (§3.2.1, Figure 2).
- Backward prefetching issues the next AllGather before the current ReduceScatter, using the reverse of the recorded forward order (§3.3.2). On GPT-175B it gave about 18% speedup (§5.2, Figure 6b).
- The outermost unit's parameters are kept after forward so they are not freed and re-gathered at the start of backward (§3.3.1).

**Evaluation settings.** 8 to 512 A100 80GB GPUs with a 2Tb/s RoCE network (§5.1). Large-model runs used full sharding with prefetching and rate limiter, activation checkpointing, BF16, and Adam (§5.4): minGPT 175B (vocab 50000, block size 2048, batch 1 and 2 on 128–512 GPUs); T5-11B (sequence length 512, batch 8 and 16 on 8–512 GPUs); DHEN (768B sparse and 550M dense parameters, batch 1024) (§5.4).

**Results.**
- T5 611M and 2.28B: FSDP and DDP perform similarly; DDP runs out of memory above 2.28B; FSDP trains the 11B model, with higher TFLOPS under BF16 (§5.2, Figure 6a).
- GPT-175B: more than 173 and 186 TFLOPS per GPU at batch 1 and 2, about 55% and 60% of the A100 BF16 peak of 312 TFLOPS, with linear scaling from 128 to 512 GPUs (§5.4, Figure 7b). At 128 GPUs and batch 2, memory defragmentation made backward 85.56% of iteration latency vs about 67% normally (§5.4).
- T5-11B: per-GPU TFLOPS falls 7% from 8 to 512 GPUs (§5.4, Figure 7c).
- Rate limiter: up to 5× speedup on T5, no speedup on RegNet, 5% overhead on DeepViT (§5.3).

**Stated limitations.** Optimizer computations that depend on an original parameter's unsharded value (for example a vector norm), its tensor structure, or global state are not mathematically equivalent to local training (§7.2.1). Shared parameters must belong to the lowest-common-ancestor FSDP unit (§7.2.2). With pipeline parallelism, default full sharding re-gathers parameters for every micro-batch (§7.1.1).

**Recipe ledger: omitted.** The paper reports throughput and memory benchmarks (batch size, sequence length, precision, optimizer type) but no learning rate, data, token count, or fine-tuning run for any model (§5.1, §5.4 checked). It does not discuss SFT.

## Connections
- [[mixed-precision]]: FP16/BF16 training and loss scaling, which FSDP's native mixed precision (§4.4) builds on.
- [[sequence-packing]], [[loss-masking-prompt]]: SFT batch construction; not discussed in this paper.
- [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]]: SFT stacks; FSDP configuration values must come from those sources, since this paper gives none.
- ZeRO (Rajbhandari et al. 2020) and cross-replica sharding (Xu et al. 2020) are cited as the designs that inspired FSDP (§1, §6); this library has no card for them.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2304.11277 (arXiv v2, 12 Sep 2023; PVLDB 16(12))
- Corrections to the previous card version: "Communication volume: 2P per step vs DDP's 2P — same bandwidth" → full sharding has 1.5× DDP's communication volume with a ring algorithm (§3.2.1). "Figure 3: FULL_SHARD lifecycle" → Figure 3 shows FlatParameter sharding across 16 GPUs; the lifecycle is Figure 1. "Table 1: Throughput/memory vs DDP across 125M–175B" → the paper has no tables; the DDP comparison covers T5 611M–11.3B (Figure 6a) and 175B results are Figures 6b, 7b, 8b. "Section 4: Memory formula derivation" → formulas are in §3.2.1 and §4.4. Core insight "per-GPU memory from O(P) to O(P/N_GPUs) … SFT a 70B model with ≤ 8×80GB GPUs" → replaced with the paper's O(Σψ_i/F + max ψ_i) and measured results; the paper trains no 70B model. Abstract "one-line wrap", "tensor subclasses" → the abstract names the Tensor implementation, dispatcher, and CUDA caching allocator and makes no one-line claim.
- Removed as unsupported by the source: the per-GPU memory table (2P / 2P / 12P, 16P total, "(16P/N) + 2P") and the 70B example "DDP = 1120 GB, FSDP ≈ 280 GB (≈ 35 GB per GPU)", which is also internally inconsistent (the card's own formula gives 280 GB per GPU at N = 8); the ShardingStrategy table (`FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD`) and the ZeRO-2/ZeRO-3 mapping (these API names do not appear in the paper); "CPU offload of parameters/optimizer state" (CPU offloading is cited only as related work, §6); "torch.compile integration"; "replaces Fairscale FSDP / DeepSpeed ZeRO"; `transformer_auto_wrap_policy({LlamaDecoderLayer})` and the `MixedPrecision(...)` snippet (the paper names only the `auto_wrap_policy` argument, §4.1); "FP32 master weights … the standard modern SFT setup"; the "Typical SFT recipe (70B)" table (micro-batch 1, gradient accumulation 16, packing, max length 4096, LR 1e-5 cosine with 3% warmup, AdamW β = (0.9, 0.95)); the guideline "FULL_SHARD for SFT of models > 13B; SHARD_GRAD_OP when latency matters".
- Not reported by the source: any SFT or fine-tuning setting; learning rates; activation memory accounting; per-component byte counts for Adam states.
