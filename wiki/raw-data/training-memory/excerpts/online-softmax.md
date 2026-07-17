# Online Normalizer Calculation for Softmax
<!-- slug: online-softmax · type: paper · source: https://arxiv.org/abs/1805.02867 -->

**Core Insight.** Softmax over long vectors can be computed in a single pass by maintaining two running scalars — a running maximum m and a running sum d — that are updated together as each element arrives, avoiding any second pass to rescale by the final normalizer.

**Guideline.** Fuse the one-pass online softmax recurrence into your attention kernel to eliminate the intermediate read of the full score row from HBM; this is the algorithmic primitive that makes FlashAttention's single-pass tiling numerically stable without storing the n-length attention row.

## Technical Details
- **Three-pass classical softmax**: (1) read all x to find max m; (2) read all x again to compute sum d = Σ exp(xᵢ−m); (3) read all x a third time to output exp(xᵢ−m)/d. Requires 3× memory bandwidth over the full score vector.
- **Two-pass online softmax**: first pass simultaneously tracks running max and accumulates unnormalized exponentials (with rescaling on max update); second pass divides by final sum. Stores only m and d across passes.
- **Single-pass online softmax recurrence** (the primitive FlashAttention inherits):
  - On element xₖ: `m_new = max(m_old, xₖ)`
  - Rescale: `d_new = exp(m_old − m_new) · d_old + exp(xₖ − m_new)`
  - Maintains numerical stability identically to the subtract-max trick at no extra cost.
- Measured speedup: Softmax alone accelerates up to **1.3×**; fused Softmax+TopK accelerates up to **5×** on GPU.
- Authors: Maxim Milakov and Natalia Gimelshein (NVIDIA).
- **Training-memory angle:** the one-pass recurrence is the key that lets FlashAttention tile an attention computation over blocks of Q and K without ever writing the full n×n score matrix to HBM. Without online softmax, each tile would need to read back previously written partial results to renormalize — requiring O(n²) HBM writes. With it, the running (m, d) scalars fit in registers and the attention output is accumulated in SRAM, so backward-pass activation storage for attention collapses from O(n²) to O(n).

## Citation
Maxim Milakov and Natalia Gimelshein. "Online normalizer calculation for softmax." arXiv:1805.02867, May 2018. https://arxiv.org/abs/1805.02867
