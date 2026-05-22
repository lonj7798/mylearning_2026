<!-- scope: Hopper-optimized FlashAttention using asynchrony, warp specialization, and FP8
     deps: flashattention-2
     see-also: flashinfer, cuda-graphs-inference
-->

# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
- **Core Insight:** Hopper GPUs need attention kernels that overlap data movement, matmul, and softmax instead of treating them as mostly sequential phases.
- **Guideline:** On H100/Hopper-class systems, use FlashAttention-3 or a backend derived from it when available, especially for FP8 or very high-throughput attention.
- **Authors:** Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2407.08608
- **Relevant topics:** Hopper GPUs, TMA, WGMMA, warp specialization, FP8 attention, low-precision accuracy

## Abstract
FlashAttention-3 updates FlashAttention for NVIDIA Hopper. The paper observes that FlashAttention-2 underuses H100 because it does not fully exploit asynchronous Tensor Memory Accelerator transfers, WGMMA tensor-core operations, or FP8 hardware. FlashAttention-3 overlaps data movement and compute with warp specialization, interleaves matmul and softmax, and adds FP8 techniques including block quantization and incoherent processing to improve accuracy.

## Key Contributions
- Uses warp specialization to overlap TMA data movement with tensor-core matmuls.
- Interleaves blockwise matmul and softmax to reduce stalls between phases.
- Adds FP8 attention with block quantization and accuracy-oriented processing.
- Reports about 1.5-2.0x speedup over FlashAttention-2 on H100 in FP16.
- Reports close to 1.2 PFLOP/s in FP8 and lower numerical error than a baseline FP8 attention.

## Key Figures/Tables to Study
- Pipeline/asynchrony diagrams: show producer/consumer warp roles and overlap.
- FP8 error comparison: important for teaching that low precision needs layout and scaling design.
- H100 throughput plots: connect architecture features to achieved utilization.
- Ablation tables: separate the value of asynchrony, interleaving, and FP8 processing.

## Technical Details
FlashAttention-3 keeps the exact-attention framing for FP16/BF16 while changing the GPU schedule. TMA moves tiles between HBM and shared memory asynchronously, WGMMA performs matrix multiply asynchronously, and specialized warps coordinate movement and compute.

The softmax remains a challenge because it is not a tensor-core matmul. The kernel interleaves softmax work with block matmuls so tensor cores are less often idle. For FP8, the paper uses block-level scaling and transformations aimed at reducing error from outlier features.

The method is hardware-specific: the main benefits depend on Hopper features, so older Ampere deployments generally use FlashAttention-2 or other backends.

## Connections
- [[flashattention-2]] is the direct baseline and supplies the prior work-partitioning improvements.
- [[flashinfer]] packages attention backends for serving workloads where Hopper support may be selected dynamically.
- [[cuda-graphs-inference]] is complementary: FlashAttention-3 speeds kernels, while CUDA graphs reduce launch overhead for stable shapes.
