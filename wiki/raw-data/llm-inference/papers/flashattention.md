<!-- scope: IO-aware exact attention kernel that avoids materializing the attention matrix
     deps: transformer-attention
     see-also: flashattention-2, flashattention-3, xformers-memory-efficient-attention
-->

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Core Insight:** Exact attention can be faster and use linear memory when the algorithm is designed around GPU HBM/SRAM traffic rather than only FLOP count.
- **Guideline:** Use FlashAttention-style tiled attention whenever long-context prefill or training attention would otherwise materialize the full scores/probabilities matrix.
- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2205.14135
- **Relevant topics:** attention kernels, IO awareness, tiling, online softmax, long context, block-sparse attention

## Abstract
The paper argues that many efficient-attention methods missed the main systems bottleneck: reads and writes between GPU high-bandwidth memory and on-chip SRAM. FlashAttention computes exact scaled dot-product attention by tiling Q, K, and V, streaming blocks through SRAM, and using an online softmax recurrence so the full attention matrix is never stored. The result keeps normal attention semantics while reducing memory from quadratic to linear in sequence length and improving wall-clock speed.

## Key Contributions
- Reframes attention optimization as an IO-complexity problem over the GPU memory hierarchy.
- Introduces a tiled exact-attention algorithm with numerically stable online softmax accumulation.
- Avoids writing the `N x N` score and probability matrices to HBM.
- Provides IO-complexity analysis and shows optimality for a range of SRAM sizes.
- Extends the idea to block-sparse FlashAttention.
- Demonstrates large speedups and lower memory on BERT, GPT-2, and long-range benchmarks.

## Key Figures/Tables to Study
- Algorithm 1: Forward pass with blockwise Q/K/V loading and softmax state updates.
- Figure 1: Memory hierarchy picture; useful for explaining why IO dominates.
- Table 1: HBM access comparison against standard attention.
- Long Range Arena results: shows that exact IO-aware attention enables longer contexts without approximate kernels.

## Technical Details
For a query block, the kernel iterates over key/value blocks, computes partial scores, updates per-row running max and normalization terms, and accumulates the output. The online softmax recurrence rescales previous partial outputs whenever a larger block max is found, preserving exact softmax over all keys.

The critical practical distinction is that FlashAttention stores only Q/K/V/O plus small row statistics, not the full score matrix. During prefill, where query length and key length are both large, this turns attention from a memory-capacity problem into a streaming kernel with high SRAM reuse.

Block-sparse FlashAttention applies the same IO-aware design to selected blocks. That variant is approximate because blocks are skipped, while dense FlashAttention is exact.

## Connections
- [[flashattention-2]] improves GPU occupancy and work partitioning while keeping the same exact-attention goal.
- [[flashattention-3]] adapts the family to Hopper asynchrony and FP8.
- [[xformers-memory-efficient-attention]] is a library-level operator family built around the same no-materialized-attention principle.
- [[flashdecoding]] changes the parallelization strategy for decode where query length is usually one.
