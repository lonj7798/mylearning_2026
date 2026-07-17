# Ring Attention with Blockwise Transformers for Near-Infinite Context
<!-- slug: ring-attention · type: paper · source: https://arxiv.org/abs/2310.01889 -->

**Core Insight.** When a single device cannot hold the KV cache for a very long sequence (even with FlashAttention's O(N) memory), you can shard the sequence across D devices in a ring — each device holds 1/D of the Q, K, and V tokens — and rotate KV blocks around the ring one step per attention "round," overlapping communication with blockwise attention compute so the total extra communication cost is zero.

**Guideline.** Use Ring Attention (context parallelism) when a sequence's per-device KV memory exceeds available HBM even after FlashAttention — the sequence length you can train scales linearly with the number of devices, with no approximation and no additional communication overhead because the KV rotation is pipelined behind the GEMM.

## Technical Details
- **Ring topology**: N devices arranged in a logical ring. Each device owns a contiguous slice of the sequence of length L/N. In each "round," each device (a) computes blockwise attention between its local Q slice and the currently held K/V slice, (b) sends its K/V slice to the next device in the ring while receiving the prior device's K/V slice.
- **Communication-computation overlap**: blockwise attention compute (the GEMM + softmax over the local K/V block) runs concurrently with the K/V ring-rotation communication. Since compute time ≥ communication time at typical head dimensions, the rotation adds zero latency to the critical path — total extra cost is "no additional communication and computation overheads."
- **Memory scaling**: per-device memory for attention is O(L/N · d) — constant in L for fixed N. Total achievable context length scales as L ∝ N (device count). The paper demonstrates "up to device count times longer" sequences than single-device memory-efficient Transformers, at "millions of tokens context size."
- **Exactness**: not approximate — the ring completes exactly N rounds so every query attends to every key in the full sequence; the blockwise softmax accumulation uses the same online normalization recurrence as FlashAttention.
- **Authors**: Hao Liu, Matei Zaharia, Pieter Abbeel.
- **Relation to Megatron/DeepSpeed context parallelism**: Liu et al.'s ring attention is the algorithmic foundation for what later production frameworks call "context parallelism" (CP) — Megatron-LM's `--context-parallel-size` flag implements the same ring-KV communication pattern.
- **Training-memory angle:** ring attention directly addresses the hard wall that FlashAttention does not solve — when L is so large that even O(L·d) per-device KV memory overflows HBM. It shards the activation memory (KV slices, Q slices, output O slices) across D devices, each holding 1/D of total attention activations. This enables training sequences that would be physically impossible on a single device regardless of kernel efficiency.

## Citation
Hao Liu, Matei Zaharia, Pieter Abbeel. "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024. arXiv:2310.01889, October 2023. https://arxiv.org/abs/2310.01889
