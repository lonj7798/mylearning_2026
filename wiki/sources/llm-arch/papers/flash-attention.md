<!-- scope: FlashAttention — IO-aware tiling for exact attention
     deps: [[attention-is-all-you-need]]
     see-also: [[flash-attention-2]], [[paged-attention]]
-->

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Core Insight:** Attention's real bottleneck is memory IO, not FLOPs; tiling for SRAM gives exact results faster than any approximation.
- **Guideline:** Always use FlashAttention (or its successors) for training and inference; never materialize the full N x N attention matrix in HBM.
- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2205.14135
- **Relevant chapters:** attention mechanisms, GPU optimization, memory hierarchy, long-context training

## Abstract
Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3x speedup on GPT-2 (seq. length 1K), and 2.4x speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

## Key Contributions
- Introduces the principle of IO-awareness for attention algorithms: accounting for data movement between GPU HBM and on-chip SRAM, not just FLOP count
- Proposes a tiling algorithm that computes exact (not approximate) attention while reducing HBM reads/writes from O(N^2) to O(N^2 d^2 / M), where M is SRAM size
- Proves the algorithm is IO-optimal for a range of SRAM sizes under standard assumptions
- Extends the approach to block-sparse attention patterns, enabling sub-quadratic approximate attention that is faster than all prior approximate methods
- Enables training with much longer context windows (16K-64K tokens), unlocking capabilities previously impossible for Transformers

## Architecture Details
- **Memory hierarchy insight:** Standard attention materializes the full N x N attention matrix in HBM. FlashAttention avoids this by never materializing the full matrix, instead computing attention in tiles that fit in SRAM
- **Tiling algorithm:** The Q, K, V matrices are divided into blocks. For each block of Q, the algorithm iterates over blocks of K and V, computing partial softmax and accumulating the weighted sum in SRAM. The online softmax trick (Milakov & Gimelshein, 2018) enables correct normalization without a full pass
- **Online softmax:** Maintains running max and sum-of-exponentials statistics, allowing softmax to be computed incrementally across K blocks without materializing the full attention score matrix
- **Recomputation in backward pass:** Instead of storing the O(N^2) attention matrix for backpropagation, FlashAttention recomputes it from Q, K, V blocks during the backward pass. This trades extra FLOPs for massive memory savings
- **Block-sparse extension:** Supports arbitrary sparsity patterns by skipping blocks of K, V that are masked out, achieving further speedup proportional to the sparsity ratio
- **IO complexity:** Standard attention requires O(N^2) HBM accesses; FlashAttention requires O(N^2 d^2 / M), which is significantly less when SRAM size M >> d^2
- **Memory usage:** O(N) instead of O(N^2), enabling much longer sequences within the same GPU memory budget

## Tradeoffs Discussed
- FlashAttention trades increased FLOPs (due to recomputation in the backward pass) for reduced memory IO, which is net positive because modern GPUs are memory-bandwidth-bound rather than compute-bound
- The tiling approach is hardware-specific: optimal block sizes depend on GPU SRAM capacity and memory bandwidth, requiring tuning per architecture
- The algorithm is more complex to implement than standard attention, requiring custom CUDA kernels rather than composing existing PyTorch/cuBLAS primitives
- Block-sparse variants require predefined sparsity patterns, which may not capture all useful attention structures
- At short sequence lengths (e.g., 512), the wall-clock improvement is modest (15%) because the memory bottleneck is less severe
