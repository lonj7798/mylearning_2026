<!-- scope: FlashAttention algorithm and GPU memory hierarchy
     deps: [[ch-03]]
     see-also: [[raschka-attention-variants]], [[hf-nanotron]]
-->

# ELI5: Flash Attention

- **Core Insight:** GPU memory hierarchy (SRAM vs HBM) determines attention's actual bottleneck.
- **Guideline:** Optimize for memory movement, not arithmetic operations.

- **Author:** Aleksa Gordic
- **URL:** https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
- **Relevant chapters:** Attention optimization, GPU memory hierarchy, efficient training

## Summary
A thorough, first-principles explanation of FlashAttention — the IO-aware attention algorithm that achieves 3x speedup on GPT-2 and reduces memory from O(N^2) to O(N) while computing exact (not approximate) attention. Explains the GPU memory hierarchy problem, kernel fusion, tiling, online softmax, and the complete forward/backward pass algorithms.

## Key Content

### The Hardware Problem: IO-Awareness

Modern GPUs have a fundamental imbalance: computational capacity (FLOPS) grows faster than memory bandwidth.

**GPU Memory Hierarchy (A100 example):**

| Component | Capacity | Bandwidth |
|-----------|----------|-----------|
| HBM (main VRAM) | 40-80GB | 1.5-2.0 TB/s |
| SRAM (on-chip) | 192KB per SM | ~19 TB/s |

SRAM is approximately 10x faster than HBM but 1000x smaller.

**Operation classification:**
- **Compute-bound:** Many operations per memory access (e.g., matrix multiplication)
- **Memory-bound:** Frequent memory access relative to computation (e.g., softmax, dropout, masking)

Standard attention is memory-bound because "it mostly consists of elementwise operations" with low arithmetic density.

### Why Vanilla Attention Is Inefficient

Traditional implementation ignores the memory hierarchy:
1. Load Q, K, V from HBM
2. Compute scores S = Q * K^T
3. **Write S back to HBM** (unnecessary round-trip)
4. Load S again to compute softmax
5. Write probabilities P to HBM
6. **Load P again** to compute output (another round-trip)

This causes redundant memory transfers between HBM and SRAM.

### FlashAttention's Two Core Ideas

**1. Kernel Fusion:** Combine multiple operations into one GPU kernel. Keep intermediate results in fast SRAM rather than writing to HBM. Eliminates redundant memory transfers.

**2. Tiling:** Break the NxN attention matrix into smaller blocks that fit in SRAM, processing them sequentially while maintaining exact results.

### The Softmax Challenge: Online Softmax

Standard softmax requires all N scores simultaneously:

softmax(x_i) = e^{x_i} / sum_j(e^{x_j})

The denominator couples all columns together, seemingly preventing block-wise computation.

**Solution: Recursive softmax combination.** Track two statistics per block:
- **m(x):** Row-wise maximum score (for numerical stability)
- **l(x):** Sum of exponential scores (the softmax denominator)

These allow seamless combination of partial results across blocks, making the final softmax computation exact.

### Forward Pass Algorithm

**Step 0:** Allocate Q, K, V in HBM.

**Steps 1-4:** Calculate block sizes:
- B_c = ceil(M / (4*d)) where M = SRAM capacity, d = head dimension
- Initialize output O with zeros, l with zeros, m with negative infinity

**Steps 5-7:** Nested loop structure:
- **Outer loop:** Iterate through key/value column blocks (j)
- **Inner loop:** Iterate through query row blocks (i)

**Step 6:** Load K_j and V_j blocks from HBM to SRAM.

**Step 8:** Load Q_i, O_i, l_i, m_i into SRAM.

**Step 9:** Compute S_ij = Q_i * K_j^T (the only place scores are computed — never stores full NxN matrix).

**Steps 10-11:** Partial softmax statistics:
- Compute m_tilde_ij (row-wise max of current block)
- Compute P_tilde_ij by normalizing: subtract max, apply exponential
- Compute l_tilde_ij (sum of P_tilde_ij rows)
- Merge with previous statistics

**Step 12 (most complex):** Update output accounting for all previously processed blocks:

O_i_new = diag(l_i) * O_i * e^{m_i - m_i_new} + e^{m_i_new - m_tilde_ij} * P_tilde_ij * V_j

The exponential terms cancel previous estimates and substitute the current maximum, guaranteeing exact results.

### Memory Complexity: O(N) Instead of O(N^2)

Standard attention allocates full NxN matrices (S and P). FlashAttention allocates:
- Q, K, V, O: (N x d) each
- l, m: (N) each
- Total: 4Nd + 2N ~ O(N) since d << N

Example: Sequence length 100,000 can now fit in 80GB rather than requiring impossibly high memory.

### Time Complexity

O(N^2 * d^2 / M) HBM accesses. While comparable to standard attention in theory, typical hardware parameters yield 9x fewer memory accesses in practice.

### Backward Pass: Recomputation Strategy

Instead of storing S and P matrices (O(N^2) memory), recompute them on-demand during backpropagation from blocks of Q, K, V in SRAM. Trade-off: backward pass is slightly slower but maintains O(N) memory throughout.

### Block-Sparse FlashAttention

Extends to sparse attention patterns. By applying block-form masking and skipping loads/stores for masked blocks, performance scales linearly with sparsity. If attention is 33% sparse, you get roughly 3x speedup.

### Scaling to Batches and Heads

A single batch element with one attention head runs as one thread block on one streaming multiprocessor (SM). With batch_size x num_heads thread blocks executing in parallel across available SMs, the algorithm naturally scales.

### Performance Results

| Task | Speedup |
|------|---------|
| BERT-large (512 tokens) | 15% faster |
| GPT-2 (1K tokens) | 3x faster |
| Long-range (1K-4K tokens) | 2.4x faster |

### Implementation Challenges

- Requires specialized CUDA kernels at low abstraction levels
- Hardware-specific (V100 unsupported in original implementation)
- Non-portable across GPU architectures
- Solution: Domain-specific languages like OpenAI's Triton provide intermediate abstraction

## Notable Insights
- FlashAttention computes EXACT attention, not an approximation — this distinguishes it from methods like Linformer or Performers that trade accuracy for efficiency.
- The key insight is that attention is memory-bound, not compute-bound. Optimization comes from reducing memory transfers, not reducing FLOPs.
- The online softmax trick is the mathematical breakthrough enabling tiling. Without it, you'd need all scores simultaneously for the denominator.
- The recomputation strategy for the backward pass is counterintuitive: it's faster to recompute attention scores than to store and retrieve them from HBM.
- FlashAttention could represent "trillions in computational savings globally" — the economic impact of a single algorithmic optimization in widely-deployed infrastructure.
