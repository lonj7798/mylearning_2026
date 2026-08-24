# Self-attention Does Not Need O(n²) Memory
<!-- slug: self-attention-no-n2-memory · type: paper · source: https://arxiv.org/abs/2112.05682 -->

**Core Insight.** Exact self-attention can be computed with O(1) extra memory per query by deferring the softmax normalization: accumulate weighted values and a running normalizer in a single outer-loop pass over K/V blocks, never materializing the full n×n score matrix.

**Guideline.** For very long sequences where the n×n attention matrix would OOM, use the streaming O(√n) variant (chunked over both queries and keys) — it is exact, numerically stable, and within a few percent of standard runtime; reserve O(log n) mode for settings that process one query at a time.

## Technical Details
- **Three memory variants** (exact, not approximate):
  - O(1) per-query memory: single query outer loop, streaming K/V inner loop; stores only a running sum of weighted values (dim d), a scalar normalizer, and the running maximum for stability.
  - O(log n) for full self-attention: requires one scalar index per query; the theoretical minimum for arbitrary self-attention.
  - O(√n) practical accelerator variant: chunks both Q and K at size √n, exploiting GPU/TPU tile parallelism while keeping tiles in SRAM.
- **Backward pass via checkpointing**: a naïve backward would store all intermediate attention scores, losing the memory advantage. The paper applies selective checkpointing over chunk-summarization functions — gradients are recomputed on-the-fly, never storing the n×n matrix during backprop.
- **Measured at n=16,384**: inference memory reduced **59×**, backpropagation memory reduced **32×** vs. standard attention. Runtime stays within a few percent of the baseline.
- Key claim: "device memory rather than compute capability is often the limiting factor on modern accelerators" — the n×n matrix is the bottleneck, not the n² FLOPs.
- **Training-memory angle:** eliminates the activation tensor that normally grows as O(n²·b·h) bytes (batch × heads × n² attention scores stored for the backward pass). Backward-pass memory for attention drops from O(n²) to O(n·d) — a qualitative regime change for long-sequence training; the √n variant holds the per-device activation footprint to O(√n) regardless of sequence length.

## Citation
Markus N. Rabe and Charles Staats. "Self-attention Does Not Need O(n²) Memory." arXiv:2112.05682, December 2021. https://arxiv.org/abs/2112.05682
