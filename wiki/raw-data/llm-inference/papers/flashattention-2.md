<!-- scope: second-generation FlashAttention kernel focused on GPU parallelism and work partitioning
     deps: flashattention
     see-also: flashattention-3, flashdecoding, flashinfer
-->

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Core Insight:** Once HBM traffic is reduced, FlashAttention performance is limited by occupancy, non-matmul work, and warp/block partitioning.
- **Guideline:** Prefer FlashAttention-2 over v1 for modern transformer prefill/training paths unless a newer architecture-specific backend is available.
- **Authors:** Tri Dao
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.08691
- **Relevant topics:** attention kernels, GPU occupancy, sequence parallelism inside attention, MQA, GQA, work partitioning

## Abstract
FlashAttention-2 keeps the IO-aware exact-attention approach but attacks the gap between FlashAttention and optimized GEMM throughput. It reduces non-matmul FLOPs, parallelizes even a single head across multiple thread blocks, and repartitions work among warps to reduce shared-memory communication. The paper reports roughly 2x speedup over FlashAttention and much higher achieved peak FLOP utilization on A100.

## Key Contributions
- Removes avoidable rescaling and bound-checking work from the inner attention loop.
- Adds parallelism over sequence length for long sequences and small batch/head counts.
- Improves warp-level work partitioning so warps do less synchronization through shared memory.
- Supports head dimensions up to 256 and inference-relevant MQA/GQA patterns.
- Shows end-to-end training speedups for GPT-style models.

## Key Figures/Tables to Study
- Algorithm comparison with FlashAttention v1: shows where non-matmul operations are reduced.
- Work partitioning diagrams: explain split-K/split-Q choices and warp responsibilities.
- Throughput plots on A100/H100: useful for estimating when attention is near GEMM efficiency.
- End-to-end training table: demonstrates that kernel improvements survive beyond microbenchmarks.

## Technical Details
FlashAttention-2 preserves exactness and the blockwise online softmax, but it changes the mapping of attention work onto GPU resources. The first version parallelized mostly across batch and heads. FlashAttention-2 also partitions a single attention head across sequence blocks when needed, increasing occupancy for long sequences or small batch sizes.

Within a thread block, the implementation assigns warps to reduce shared-memory reads/writes and keeps more work in registers. The paper emphasizes that matmul operations are efficient on tensor cores, while softmax, masking, dropout, and rescaling need careful handling because they do not map as cleanly to tensor cores.

For inference libraries, the main inheritance is not just a faster prefill kernel but an operator API that handles packed QKV, variable-length sequences, causal masks, MQA, and GQA.

## Connections
- [[flashattention]] supplies the IO-aware tiling and online softmax base.
- [[flashattention-3]] moves the same design line to Hopper TMA, WGMMA, warp specialization, and FP8.
- [[flashdecoding]] solves the different decode-time problem where Q length is usually 1.
- [[flashinfer]] uses FlashAttention-family kernels as one backend inside a serving-oriented operator library.
