# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
<!-- slug: flash-attention-3 · type: paper · source: https://arxiv.org/abs/2407.08608 -->

**Core Insight.** Hopper H100 GPUs have a new asynchronous execution model (TMA + warp-group MMA instructions) that FlashAttention-2's synchronous kernel design cannot exploit. FA3 introduces producer-consumer warp specialization with a software pipeline, overlapping GEMM and softmax on separate warp groups simultaneously — plus FP8 support with incoherent processing to suppress quantization error.

**Guideline.** On H100s (Hopper), switch from FA2 to FA3 to reach 740 TFLOPs/s (75% utilization) in FP16 vs. FA2's ~370 TFLOPs/s; for FP8 training, FA3's block quantization + incoherent processing achieves 2.6× lower error than a naive FP8 attention baseline and approaches 1.2 PFLOPs/s.

## Technical Details
- **Three Hopper-specific techniques:**
  1. **Producer-consumer warp specialization (ping-pong pipeline)**: separate producer warps drive TMA (Tensor Memory Accelerator) async data loads into shared memory while consumer warps drive tensor-core GEMM. Ping-pong double-buffering ensures the consumer never stalls waiting for data — GEMM and softmax overlap with the next tile's load.
  2. **Interleaved block-wise GEMM + softmax**: within the consumer warp group, the two-GEMM attention sequence (S = QKᵀ, then P·V) is pipelined so that the softmax rescaling of partial P happens while the next GEMM stage is issued to tensor cores, hiding softmax latency.
  3. **FP8 block quantization + incoherent processing**: per-block quantization (small tiles within each attention tile, rather than per-tensor) bounds the dynamic range each quantization step must cover. Incoherent processing applies a random Hadamard rotation before quantization, spreading outliers uniformly so the FP8 range is used efficiently. Result: **2.6× lower numerical error** than baseline FP8 attention.
- **Measured performance on H100 SXM5:**
  - FP16: up to **740 TFLOPs/s**, 75% of theoretical peak (vs. FA2 ~370 TFLOPs/s, ~40%)
  - FP8: approaches **1.2 PFLOPs/s**
  - Wall-clock speedup: **1.5–2.0× vs. FA2** depending on head dimension and sequence length
- Authors: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
- **Training-memory angle:** FA3 does not change the O(N) activation memory regime established by FA1/2. Its memory significance is at the compute-memory bandwidth level: the TMA pipeline delivers tiles to SRAM faster and with less warp stall, so the on-chip SRAM is used more continuously — reducing the HBM-fetch stall time that was previously burning clock cycles while memory was in flight. For FP8 training, the 2× data-size reduction of FP8 vs. FP16 halves the HBM bandwidth consumed per attention tile, enabling longer sequences within the same HBM bandwidth budget.

## Citation
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608, July 2024. https://arxiv.org/abs/2407.08608
