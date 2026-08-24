# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
<!-- slug: flash-attention-1 · type: paper · source: https://arxiv.org/abs/2205.14135 -->

**Core Insight.** Attention is slow not because of FLOPs but because of HBM bandwidth: the n×n score matrix is written to HBM and read back for softmax, then written again and read back for the weighted sum. FlashAttention tiles attention so the entire computation lives in SRAM, never writing the score matrix to HBM, while preserving exact outputs via the online softmax recurrence.

**Guideline.** Replace standard `torch.nn.MultiheadAttention` with FlashAttention (or enable it via `torch.nn.functional.scaled_dot_product_attention`) whenever sequence length exceeds ~512 tokens — the memory gain is O(n²) → O(n) for activations, and wall-clock training is 2–3× faster due to HBM read/write elimination, not FLOPs reduction.

## Technical Details
- **HBM read/write analysis**: standard attention performs Θ(Nd + N²) HBM accesses (write/read the N×N matrix S and P). FlashAttention reduces this to Θ(N²d²/M) HBM accesses where M is SRAM size (typically 20 MB on A100), because blocks of Q, K, V tiles are loaded once into SRAM and the output O is accumulated there.
- **Tiling**: divide Q into blocks of size Br = ⌈M/4d⌉, K/V into blocks of Bc = min(⌈M/4d⌉, d). Each block pair fits in SRAM; the online softmax recurrence (from Milakov & Gimelshein 2018) renormalizes the running output across tiles without writing intermediate results to HBM.
- **Memory complexity**: O(N) activations (stores only the output matrix and the per-row softmax statistics l, m — each of size N — rather than the N×N attention weights).
- **Backward pass recomputation**: instead of storing the N×N attention weight matrix P for the backward pass, FlashAttention recomputes S and P on-the-fly from the stored Q, K, V, and (l, m) statistics. This trades extra FLOPs for a massive activation memory reduction.
- **Measured speedups**: BERT-large (seq 512) 15% end-to-end over MLPerf 1.1; GPT-2 (seq 1K) 3×; Long-range Arena (seq 1K–4K) 2.4×. Block-sparse FA enables 64K-token sequences (first transformers to pass Path-256, 63.1% accuracy).
- **Training-memory angle:** the dominant activation-memory saving in training is eliminating the N×N score buffer. For sequence length 2048, head dim 64, batch 32, 32 heads: the score matrix would be 32·32·2048²·2 bytes = 8 GB per layer — dropped to near zero. The only activation memory stored for backward is O, l (N-vector), m (N-vector), and the original Q, K, V tensors.

## Citation
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. arXiv:2205.14135. https://arxiv.org/abs/2205.14135
