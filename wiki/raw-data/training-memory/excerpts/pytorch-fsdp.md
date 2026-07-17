# PyTorch FSDP: Fully Sharded Data Parallel
<!-- slug: pytorch-fsdp · type: paper · source: https://arxiv.org/abs/2304.11277 -->

**Core Insight.** PyTorch FSDP is ZeRO-3 implemented natively in PyTorch: parameters are "flat-packed" per FSDP unit (one transformer layer = one unit), all-gathered before each forward/backward, then freed; gradients are reduce-scattered into their shard. This makes ZeRO-3 composable with TP, PP, and mixed precision through PyTorch's dispatcher.

**Guideline.** Wrap at the transformer-layer granularity: one `LlamaDecoderLayer` (or equivalent) per FSDP unit. Larger wrapping units reduce communication frequency but increase peak memory of the all-gather buffer (≈ 2P for the largest unit). Use `HYBRID_SHARD` for multi-node jobs: full-shard within a NVLink node, replicate across nodes.

## Technical Details

- **Per-step lifecycle under FULL_SHARD:** (1) AllGather parameters for current FSDP unit → (2) forward pass → (3) free gathered parameters immediately → (4) AllGather again for backward → (5) ReduceScatter gradients into shard → (6) optimizer step on local shard only.
- **Communication volume per step:** 2× AllGather (2Ψ) + 1× ReduceScatter (Ψ) = **3Ψ** vs. DDP's all-reduce = 2Ψ. The 1.5× overhead is the FSDP tax for N× memory savings.
- **Flat-parameter design:** Each FSDP unit's heterogeneous sub-tensors (e.g. QKV bias + weight) are concatenated into a single 1D flat tensor before sharding. This enables a single AllGather call per unit rather than per-tensor, and reduces CUDA kernel launch overhead.
- **Memory formula:** `16Ψ/N + 2P_unit` where N=DP degree, `2P_unit` is the AllGather buffer for the largest FSDP unit (held transiently during forward/backward of that unit).
- **HYBRID_SHARD:** `FULL_SHARD` within intra-node NVLink group (fast), `REPLICATE` across nodes (cheap). Trades some memory saving for lower inter-node communication cost; in practice the default for multi-node training on 8-GPU nodes.
- **clip_grad_norm_ correctness:** `model.clip_grad_norm_(max_norm)` must be called on the FSDP module — it performs a distributed norm computation. Calling `torch.nn.utils.clip_grad_norm_` on local shards under-reports the global norm by √N and causes silent divergence.
- **"Comparable performance to DDP while providing support for significantly larger models with near-linear scalability"** — Zhao et al. 2023.
- **Training-memory angle:** FSDP is the single most impactful lever for optimizer-state and weight memory in data-parallel training; it doesn't touch activations. Pair with activation checkpointing + SP for full memory coverage.

## Citation
Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., et al. (2023). PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. VLDB 2023. https://arxiv.org/abs/2304.11277
