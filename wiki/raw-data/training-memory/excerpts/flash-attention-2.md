# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
<!-- slug: flash-attention-2 · type: paper · source: https://arxiv.org/abs/2307.08691 -->

**Core Insight.** FlashAttention-1 reached only 25–40% of theoretical GPU FLOPs/s because of poor warp-level work partitioning: different warps were racing to write shared memory results during the softmax rescaling step. FlashAttention-2 restructures the inner loop so each warp owns its output partition independently, nearly doubling utilization.

**Guideline.** On A100s, use FlashAttention-2 as the default attention kernel for all training runs — it delivers 50–73% MFU on attention layers vs. 25–40% for FA1, and the ~2× speedup compounds with long sequences where attention is the dominant cost. The memory footprint is the same O(N) as FA1; the gain is pure throughput.

## Technical Details
- **Three algorithmic improvements over FA1:**
  1. **Fewer non-matmul FLOPs**: FA1's outer loop iterates over rows of K/V for each Q tile, accumulating rescaled partial results with O(d) rescale operations per tile pair. FA2 reorders the loop (Q outer, KV inner) to batch rescaling, cutting the rescale FLOPs fraction. Non-matmul ops are ~16× slower per FLOP than matmul on A100 (no tensor core acceleration), so reducing them disproportionately lifts utilization.
  2. **Parallelism across sequence length**: FA1 could not parallelize within a single batch×head because causal masking made column strips depend on prior max values. FA2 assigns different thread blocks to disjoint row ranges of Q — each block is independent and runs in parallel, boosting occupancy especially for small batch/few-head configs.
  3. **Warp work partitioning**: FA1 split K/V across 4 warps in a "split-K" fashion that required inter-warp communication (shared memory writes/reads) for the softmax denominator. FA2 instead splits Q across warps — each warp produces a complete output row partition, eliminating shared memory synchronization for softmax.
- **Performance**: up to **225 TFLOPs/s per A100 GPU** (72% MFU) on GPT-style training. Roughly **2× speedup vs. FA1** in wall-clock attention time.
- **Memory**: identical O(N) activation footprint to FA1 — the recomputation-on-backward trick is unchanged. The speedup is purely compute utilization.
- **Training-memory angle:** same memory regime as FA1 (O(N) activations, no N×N score buffer). The significance for training budgets is throughput: with 2× faster attention at fixed memory, the model can process 2× more tokens per hour or use a larger sequence length within the same time budget. This indirectly improves memory-per-useful-computation efficiency.

## Citation
Tri Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. arXiv:2307.08691, July 2023. https://arxiv.org/abs/2307.08691
