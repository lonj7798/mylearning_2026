<!-- scope: Flash Attention tiling algorithm detailed walkthrough, parent: [[ch-07]] -->

# Flash Attention Tiling Algorithm: A Detailed Walkthrough

The tiling algorithm is the heart of Flash Attention ([[flash-attention|paper]]). This excerpt walks through the complete forward pass, step by step, showing exactly how tiling + online softmax produces exact attention without ever materializing the N x N score matrix.

---

## The Problem Flash Attention Solves

Standard attention computes three intermediate results that each require O(N^2) HBM storage:

1. **Score matrix** $S = QK^T \in \mathbb{R}^{N \times N}$
2. **Probability matrix** $P = \text{softmax}(S) \in \mathbb{R}^{N \times N}$
3. **Output** $O = PV \in \mathbb{R}^{N \times d}$

Each of S and P requires a full round-trip to HBM. For sequence length 8K with FP16, each N x N matrix is 8192 x 8192 x 2 = 128 MB per head per layer. With 32 heads and 80 layers, that is 328 GB of intermediate storage -- impossible.

Flash Attention eliminates S and P entirely from HBM. Only Q, K, V (each N x d) and O (N x d) ever reside in HBM. All intermediate computation happens in SRAM tiles.

---

## Block Size Calculation

The algorithm begins by choosing block sizes that fit in SRAM. On an A100, each streaming multiprocessor (SM) has 192 KB of shared memory (SRAM). The block sizes are:

$$B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad B_r = \min\!\left(\left\lceil \frac{M}{4d} \right\rceil,\; d\right)$$

where $M$ is SRAM capacity and $d$ is the head dimension. The factor of 4 accounts for loading four matrices into SRAM simultaneously: a block of Q ($B_r \times d$), a block of K ($B_c \times d$), a block of V ($B_c \times d$), and the output accumulator ($B_r \times d$).

**Concrete example (A100, d = 128):**
- $M = 192\text{KB} = 192 \times 1024 / 2 = 98{,}304$ FP16 elements
- $B_c = \lceil 98{,}304 / 512 \rceil = 192$
- $B_r = \min(192, 128) = 128$

So we process Q in blocks of 128 rows and K/V in blocks of 192 rows. The number of column blocks is $T_c = \lceil N / B_c \rceil$ and row blocks is $T_r = \lceil N / B_r \rceil$.

---

## The Forward Pass Algorithm

### Initialization

```
Allocate O = zeros(N, d) in HBM       # output accumulator
Allocate l = zeros(N) in HBM          # softmax denominator (running sum)
Allocate m = -inf * ones(N) in HBM    # softmax numerator (running max)
```

### The Nested Loop

The outer loop iterates over K/V blocks (columns of the attention matrix). The inner loop iterates over Q blocks (rows).

```
for j = 1 to T_c:                         # outer: K/V blocks
    Load K_j, V_j from HBM to SRAM        # each B_c x d

    for i = 1 to T_r:                     # inner: Q blocks
        Load Q_i from HBM to SRAM         # B_r x d
        Load O_i, l_i, m_i from HBM       # current accumulators

        # Step 1: Compute local scores
        S_ij = Q_i @ K_j^T                # B_r x B_c, in SRAM

        # Step 2: Local softmax statistics
        m_tilde = rowmax(S_ij)            # B_r x 1
        P_tilde = exp(S_ij - m_tilde)     # B_r x B_c, numerically stable
        l_tilde = rowsum(P_tilde)         # B_r x 1

        # Step 3: Merge with running statistics
        m_new = max(m_i, m_tilde)
        l_new = l_i * exp(m_i - m_new) + l_tilde * exp(m_tilde - m_new)

        # Step 4: Update output accumulator
        O_i = diag(l_i)^{-1} * [
            diag(l_i) * O_i * exp(m_i - m_new)     # rescale old
          + exp(m_tilde - m_new) * P_tilde @ V_j    # add new contribution
        ]

        # Step 5: Update statistics
        l_i = l_new
        m_i = m_new

        Write O_i, l_i, m_i back to HBM
```

### Final Normalization

After all blocks have been processed:

$$O_i = \text{diag}(l_i)^{-1} \cdot O_i$$

This divides each row of O by its softmax denominator, completing the normalization.

---

## Why the Rescaling in Step 4 Is Correct

The key mathematical insight is the **online softmax** decomposition. Consider processing K/V in two chunks, A and B. Standard softmax over both is:

$$\text{softmax}([s_A, s_B])_i = \frac{e^{s_i}}{\sum_A e^{s_j} + \sum_B e^{s_j}}$$

After processing chunk A, we have partial results with max $m_A$ and denominator $l_A$. When chunk B arrives with max $m_B$, the new global max is $m_{new} = \max(m_A, m_B)$. The correction factors $e^{m_A - m_{new}}$ and $e^{m_B - m_{new}}$ rescale each chunk's contributions to the new global max.

This is Milakov & Gimelshein's (2018) online normalizer calculation. The critical property: after processing all chunks, the accumulated O and l give **exactly** the same result as computing softmax over all scores simultaneously. No approximation at any stage.

---

## IO Complexity Analysis

**Standard attention:**
- Reads: Q, K (for $S = QK^T$): $O(Nd)$ each, but S itself is $O(N^2)$ writes + reads
- Total HBM accesses: $\Theta(Nd + N^2)$

**Flash Attention:**
- Outer loop: $T_c$ iterations, each loading K_j, V_j ($B_c \times d$ each)
- Inner loop: $T_r$ iterations per outer, each loading/storing Q_i, O_i ($B_r \times d$ each)
- Total HBM accesses: $\Theta\!\left(\frac{N^2 d^2}{M}\right)$

Since $M \gg d^2$ on modern GPUs (SRAM ~100K elements, $d^2 = 16{,}384$), the ratio $M/d^2 \approx 6\text{x}$, so Flash Attention uses roughly $6\times$ fewer HBM accesses. In practice, the speedup is 2-4x because some overhead remains from kernel launch, synchronization, and the non-matmul operations.

---

## The Backward Pass: Recomputation Strategy

Standard backpropagation through attention requires the stored P matrix ($N \times N$). Flash Attention's backward pass **recomputes** P on-the-fly from Q, K, V blocks in SRAM ([[flash-attention-explained|blog]]).

This is counterintuitive: recomputation adds FLOPs. But since the recomputation happens entirely in fast SRAM (19 TB/s), while loading a stored P from HBM would cost bandwidth (2 TB/s), the recomputation is faster than storage-and-retrieval. The backward pass maintains $O(N)$ memory throughout.

**Stored for backward pass:** Only $O$, $l$, $m$ (all $O(N)$) plus Q, K, V. The $O(N^2)$ intermediates S and P are never stored.

---

## Flash Attention 2 Improvements

FA-2 ([[flash-attention-2|paper]]) made three changes that doubled throughput:

1. **Loop order swap:** Outer loop over Q blocks, inner over K/V. This avoids rescaling O across Q blocks (each Q block's output is independent), reducing non-matmul FLOPs.

2. **Sequence-level parallelism:** FA-1 parallelized only across batch x heads. FA-2 additionally splits the sequence dimension across thread blocks, improving GPU occupancy when batch x heads is small.

3. **Warp partitioning:** Q is partitioned across warps within a thread block (each warp handles a slice of Q rows), while K/V are shared. This reduces shared memory traffic compared to FA-1's approach of splitting K/V across warps.

Result: 50-73% of theoretical max FLOPs/s on A100 (up from 25-40%), achieving 225 TFLOPs/s (72% MFU).

---

## References

- [[flash-attention|Dao et al., "FlashAttention" (2022) (paper)]]
- [[flash-attention-2|Dao, "FlashAttention-2" (2023) (paper)]]
- [[flash-attention-explained|Gordic, "ELI5: Flash Attention" (2023) (blog)]]
- Milakov & Gimelshein, "Online Normalizer Calculation for Softmax" (2018)
