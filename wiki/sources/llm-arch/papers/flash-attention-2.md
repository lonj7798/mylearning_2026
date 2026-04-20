<!-- scope: FlashAttention-2 — better warp partitioning for near-GEMM efficiency
     deps: [[flash-attention]]
     see-also: [[paged-attention]], [[ultra-scale-playbook]]
-->

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Core Insight:** Better work partitioning between GPU warps and reducing non-matmul FLOPs closes the gap between attention and optimized GEMM throughput.
- **Guideline:** Profile non-matmul overhead in your attention kernel; shifting work to tensor-core matmuls is the main lever after tiling.
- **Authors:** Tri Dao
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.08691
- **Relevant chapters:** attention mechanisms, GPU optimization, parallelism, training efficiency

## Abstract
Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation. The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length. FlashAttention exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4x compared to optimized baselines), with no approximation. However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2x speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization).

## Key Contributions
- Identifies that FlashAttention-1 reaches only 25-40% of theoretical max FLOPs/s due to suboptimal GPU work partitioning
- Reduces non-matmul FLOPs in the attention algorithm, shifting more computation to highly optimized GEMM operations
- Introduces parallelism across thread blocks for a single attention head, improving GPU occupancy even with few heads or small batch sizes
- Optimizes intra-block warp communication to minimize shared memory reads/writes
- Achieves 50-73% of theoretical max FLOPs/s on A100, approximately 2x faster than FlashAttention-1

## Architecture Details
- **Non-matmul FLOP reduction:** The original FlashAttention spends significant time on non-matmul operations (e.g., softmax rescaling, masking). FA-2 restructures the algorithm to minimize these, keeping the GPU tensor cores busy with matrix multiplications
- **Outer loop over Q blocks (not K/V):** FA-2 swaps the loop order so the outer loop iterates over Q blocks and the inner loop over K/V blocks. This avoids the need to rescale the output accumulator across different Q blocks, reducing non-matmul work
- **Sequence-level parallelism:** FA-1 parallelized across batch and heads only. FA-2 additionally splits the sequence dimension across thread blocks, so even a single attention head can utilize multiple SMs on the GPU
- **Warp-level partitioning:** Within each thread block (containing e.g., 4 warps), FA-2 partitions the Q tile across warps while sharing K/V. Each warp computes attention for its Q slice, then results are combined. This reduces shared memory traffic compared to FA-1's approach of splitting K/V across warps
- **Forward pass:** Achieves up to 230 TFLOPs/s on A100 for head dimension 128
- **Backward pass:** Also improved but remains slower than forward due to the need to recompute attention scores and handle more complex gradient flow
- **Causal masking optimization:** For causal (autoregressive) attention, FA-2 skips entire blocks where the causal mask would zero out all entries, saving roughly 50% of computation for the masked triangle
- **End-to-end training:** 225 TFLOPs/s per A100 GPU on GPT-style models, representing 72% model FLOPs utilization

## Tradeoffs Discussed
- The optimizations are A100-specific in terms of tuning; different GPU architectures (H100, etc.) may need re-tuning of block sizes and warp strategies
- Achieving near-GEMM efficiency means the remaining bottleneck shifts to other layers (FFN, communication), so attention is no longer the sole training bottleneck
- The increased parallelism across sequence blocks adds complexity to the reduction step where partial results must be combined
- Still computes exact attention (no approximation), so the O(N^2) FLOP count remains; the improvement is purely in hardware utilization
- Custom CUDA kernel complexity continues to grow, making maintenance and portability across GPU generations a concern
