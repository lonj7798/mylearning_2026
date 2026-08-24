# SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration
<!-- slug: sage-attention · type: paper · source: https://arxiv.org/abs/2410.02367 -->

**Core Insight.** Attention's two matrix multiplications (QKᵀ and P·V) can be quantized to INT8 and FP16 respectively — not naively, but after a per-channel smoothing that removes the token-invariant outlier bias in K — yielding 2.1× throughput over FlashAttention2 with negligible accuracy loss, plug-and-play at inference.

**Guideline.** Use SageAttention (or SageAttention2 for INT4) only for **inference** acceleration — neither paper addresses training; do not use in forward-backward training loops. For inference-only serving where FP16 attention is the throughput bottleneck, drop in `sageattn()` after validating end-to-end metric parity on your task.

## Technical Details
- **Quantization scheme (SageAttention v1, arxiv 2410.02367):**
  - QKᵀ matmul: quantize Q and K to **INT8**. INT8 matmul is 2× faster than FP8 matmul on RTX4090/RTX3090, and Table 3 in the paper shows INT8 accuracy dominates FP8 (E4M3/E5M2) for this operation.
  - P·V matmul: keep P and V in **FP16** with FP16 accumulators ("2x faster" than INT8 for this stage while preserving accuracy).
- **K-channel smoothing**: K exhibits "distinct channel-wise outliers" — each token's key = large shared bias + small token-wise signal. Apply `γ(K) = K − mean(K)` (mean across token dimension). This does not change attention output because `σ(q·(K−mean(K))ᵀ) = σ(q·Kᵀ)` (uniform bias cancels in softmax). Overhead: <0.2% of runtime.
- **Measured performance (RTX4090):**
  - OPS: 341 TOPS (52% of theoretical INT8 peak)
  - Speed: **2.1× vs. FlashAttention2**, **2.7× vs. xFormers**
  - Cosine similarity vs. full-precision: 1.0; Relative L1: 0.019
- **SageAttention2 (arxiv 2411.10958)**: extends to INT4 (per-warp quantization granularity) with Smooth Q and Smooth V centering tricks; achieves **3.1× vs. FA2** on RTX4090; explicitly **inference-only**, "plug-and-play without requiring retraining."
- **Training-memory angle:** SageAttention is **inference-only** — it has no backward-pass implementation and the authors explicitly position it as inference acceleration. It is included here as a contrast anchor: the INT8 weight-and-activation memory savings relevant to inference do not transfer to training, where gradients require full-precision accumulation. For training, the applicable technique is FA1/FA2/FA3's O(N) recomputation approach, not quantized kernels.

## Citation
SageAttention: Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen (Tsinghua University). "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration." ICLR 2025. arXiv:2410.02367, October 2024. https://arxiv.org/abs/2410.02367
SageAttention2: Jintao Zhang et al. arXiv:2411.10958, November 2024. https://arxiv.org/abs/2411.10958
