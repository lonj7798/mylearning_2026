<!-- scope: Online softmax derivation, parent: [[ch-07]] -->

# Online Softmax: The Mathematical Foundation of Flash Attention

The online softmax algorithm (Milakov & Gimelshein, 2018) is the mathematical enabler of Flash Attention's tiling approach. Without it, softmax would require all N scores to be available simultaneously, making tile-by-tile computation impossible. This excerpt derives the algorithm from first principles and proves its correctness.

---

## The Softmax Normalization Problem

Standard softmax over a vector $s = [s_1, \ldots, s_N]$:

$$\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{N} e^{s_j}}$$

The denominator $\ell = \sum_j e^{s_j}$ requires a complete pass over all $N$ scores. In standard attention, $s$ is a row of the $N \times N$ score matrix $S = QK^T$ -- one query's dot products with all keys. To compute softmax for this row, you need all $N$ scores. If the scores are computed tile-by-tile (only $B_c$ at a time), the denominator appears unknowable until all tiles are processed.

### Numerical Stability: The Safe Softmax

Before addressing the tiling problem, recall that naive softmax is numerically unstable. For large $s_i$, $e^{s_i}$ overflows. The standard fix subtracts the maximum:

$$\text{softmax}(s_i) = \frac{e^{s_i - m}}{\sum_{j=1}^{N} e^{s_j - m}}, \qquad m = \max_j s_j$$

This is mathematically identical (the $e^{-m}$ factors cancel) but avoids overflow since $s_i - m \le 0$ for all $i$.

The safe softmax needs **two** passes over the data: one to find $m$, one to compute the exponentials and their sum. With tiling, we want a **single-pass** algorithm that handles both incrementally.

---

## Derivation: Merging Two Blocks

Consider splitting the scores into two blocks, $s^{(1)} = [s_1, \ldots, s_k]$ and $s^{(2)} = [s_{k+1}, \ldots, s_N]$. After processing block 1, we have:

$$m^{(1)} = \max_{j \le k} s_j, \qquad \ell^{(1)} = \sum_{j=1}^{k} e^{s_j - m^{(1)}}$$

After processing block 2, we compute:

$$m^{(2)} = \max_{j > k} s_j, \qquad \ell^{(2)} = \sum_{j=k+1}^{N} e^{s_j - m^{(2)}}$$

The global maximum is:

$$m^{(\text{new})} = \max(m^{(1)}, m^{(2)})$$

Now we need the global denominator. The true denominator (with the global max for stability) is:

$$\ell^{(\text{true})} = \sum_{j=1}^{N} e^{s_j - m^{(\text{new})}}$$

Split this into the two blocks:

$$\ell^{(\text{true})} = \underbrace{\sum_{j=1}^{k} e^{s_j - m^{(\text{new})}}}_{A} + \underbrace{\sum_{j=k+1}^{N} e^{s_j - m^{(\text{new})}}}_{B}$$

For term A, factor out the correction between the old max and the new max:

$$A = \sum_{j=1}^{k} e^{s_j - m^{(1)}} \cdot e^{m^{(1)} - m^{(\text{new})}} = \ell^{(1)} \cdot e^{m^{(1)} - m^{(\text{new})}}$$

Similarly for term B:

$$B = \ell^{(2)} \cdot e^{m^{(2)} - m^{(\text{new})}}$$

Therefore:

$$\boxed{\ell^{(\text{new})} = \ell^{(1)} \cdot e^{m^{(1)} - m^{(\text{new})}} + \ell^{(2)} \cdot e^{m^{(2)} - m^{(\text{new})}}}$$

This is the online softmax merge formula. It combines partial denominators from two blocks using only local statistics ($m^{(i)}, \ell^{(i)}$) and the global max.

---

## Generalizing to K Blocks

The merge is associative. For $K$ blocks processed sequentially:

**After processing block $k$:**

$$m^{(\text{new})} = \max(m^{(\text{old})}, m^{(k)})$$

$$\ell^{(\text{new})} = \ell^{(\text{old})} \cdot e^{m^{(\text{old})} - m^{(\text{new})}} + \ell^{(k)} \cdot e^{m^{(k)} - m^{(\text{new})}}$$

Initialize with $m^{(0)} = -\infty$ and $\ell^{(0)} = 0$. After processing all $K$ blocks, $\ell^{(K)}$ equals the true softmax denominator $\sum_j e^{s_j - m^{(K)}}$, where $m^{(K)}$ is the global maximum.

**Proof sketch (by induction):**
- Base case: After block 1, $m = m^{(1)}$ and $\ell = \sum_{j \in \text{block 1}} e^{s_j - m^{(1)}}$. Correct by definition.
- Inductive step: Assume after blocks $1, \ldots, k-1$ we have the correct partial max $m^{(\text{old})}$ and partial sum $\ell^{(\text{old})} = \sum_{j \in \text{blocks } 1..k-1} e^{s_j - m^{(\text{old})}}$. After merging block $k$, the new sum is $\sum_{j \in \text{blocks } 1..k} e^{s_j - m^{(\text{new})}}$ by the derivation above.

---

## Extending to the Output Accumulator

Flash Attention doesn't just need the softmax denominator -- it needs the weighted sum $O = PV$ where $P = \text{softmax}(S)$. The output for query row $i$ is:

$$O_i = \sum_{j=1}^{N} \frac{e^{s_{ij} - m_i}}{\ell_i} \cdot V_j$$

We accumulate this tile-by-tile. After processing K/V block $k$, the partial output (before final normalization) is:

$$\tilde{O}_i^{(k)} = \sum_{j \in \text{blocks } 1..k} e^{s_{ij} - m_i^{(k)}} \cdot V_j$$

When the max changes from $m^{(\text{old})}$ to $m^{(\text{new})}$, the previously accumulated output must be rescaled:

$$\tilde{O}_i^{(\text{new})} = \tilde{O}_i^{(\text{old})} \cdot e^{m^{(\text{old})} - m^{(\text{new})}} + \sum_{j \in \text{block } k} e^{s_{ij} - m^{(\text{new})}} \cdot V_j$$

After all blocks, the final output is:

$$O_i = \frac{\tilde{O}_i^{(\text{final})}}{\ell_i^{(\text{final})}}$$

This is exact. Each rescaling adjusts the unnormalized accumulator to the current global max, and the final division by $\ell$ completes the softmax normalization.

---

## Why the Rescaling Factors Are Numerically Safe

A concern: do the correction factors $e^{m^{(\text{old})} - m^{(\text{new})}}$ cause numerical issues?

No, because $m^{(\text{new})} \ge m^{(\text{old})}$ always (the running max can only increase). Therefore:

$$m^{(\text{old})} - m^{(\text{new})} \le 0$$

So $e^{m^{(\text{old})} - m^{(\text{new})}} \in (0, 1]$. The correction factor is always a downscaling, never an upscaling. This prevents overflow. Underflow to zero is possible if the new max is much larger, but that's correct behavior: if a new block has much larger scores, the previous block's contributions should become negligible in the softmax.

---

## Computational Cost of Online Softmax

Per block merge:
- 1 max operation (element-wise, $O(B_c)$)
- 2 exponentiations ($e^{m^{(\text{old})} - m^{(\text{new})}}$ and $e^{m^{(k)} - m^{(\text{new})}}$)
- 2 multiply-adds for $\ell$ update
- 1 scalar multiply + 1 matrix-vector add for $\tilde{O}$ rescaling

These are all **non-matmul** operations -- they run on CUDA cores, not tensor cores. Flash Attention 2 ([[flash-attention-2|paper]]) restructured the algorithm specifically to minimize these operations, since tensor cores (for matmuls) are ~16x faster than CUDA cores (for element-wise ops) on A100.

---

## Connection to Attention Score Dropping

An elegant consequence of online softmax: if a block's maximum score $m^{(k)}$ is much smaller than the running max $m^{(\text{old})}$, then $e^{m^{(k)} - m^{(\text{new})}} \approx 0$, and the entire block contributes negligibly to both $\ell$ and $\tilde{O}$. This is the mathematical basis for Flash Attention's block-sparse extension: blocks whose scores are provably small can be skipped entirely without affecting the result.

For causal attention, Flash Attention 2 exploits this by skipping blocks that fall entirely below the causal mask diagonal, saving ~50% of computation.

---

## References

- Milakov & Gimelshein, "Online Normalizer Calculation for Softmax" (2018)
- [[flash-attention|Dao et al., "FlashAttention" (2022) (paper)]]
- [[flash-attention-2|Dao, "FlashAttention-2" (2023) (paper)]]
- [[flash-attention-explained|Gordic, "ELI5: Flash Attention" (2023) (blog)]]
