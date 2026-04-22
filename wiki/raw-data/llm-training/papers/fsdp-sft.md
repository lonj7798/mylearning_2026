<!-- scope: PyTorch FSDP — fully-sharded data parallelism; model, gradients, and optimizer states sharded across GPUs
     deps: []
     see-also: [[sequence-packing]], [[loss-masking-prompt]]
-->

# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
- **Core Insight:** Sharding parameters, gradients, and optimizer states across the data-parallel group (instead of replicating them) reduces per-GPU memory from O(P) to O(P / N_GPUs) — the critical primitive that lets you SFT a 70B model with ≤ 8×80GB GPUs without model parallelism.
- **Guideline:** Default to `FULL_SHARD` for SFT of models > 13B; use `SHARD_GRAD_OP` (ZeRO-2 equivalent) when latency matters more than memory; combine with activation checkpointing and BF16 mixed precision for ~70B SFT.
- **Authors:** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, Shen Li
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.11277
- **Relevant topics:** distributed training, parameter sharding, ZeRO, memory-efficient SFT

## Abstract
PyTorch's Fully Sharded Data Parallel (FSDP) is an industry-grade solution for large-scale model training. FSDP shards model parameters, gradients, and optimizer states across the data-parallel group and reconstructs full parameters on demand via AllGather; gradients are reduced via ReduceScatter. Implemented natively in PyTorch via the dispatcher, tensor subclasses, and CUDA caching allocator, FSDP achieves near-linear scalability in TFLOPS with a non-intrusive user API (one-line wrap).

## Key Contributions
- Three sharding strategies: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (DDP).
- **Hybrid sharding** within-node + replicate across-node for cluster topologies.
- CPU offload of parameters/optimizer state for extreme cases.
- Native integration with mixed precision (BF16/FP16), activation checkpointing, and torch.compile.
- Replaces Fairscale FSDP / DeepSpeed ZeRO for PyTorch-native training.

## Key Figures/Tables to Study
- **Figure 3:** FULL_SHARD lifecycle — AllGather forward, ReduceScatter backward.
- **Table 1:** Throughput / memory vs DDP across 125M–175B.
- **Section 4:** Memory formula derivation.

## Technical Details

### Memory breakdown (per GPU)
Given P parameters, N GPUs, Adam optimizer:
| Component | DDP | FSDP FULL_SHARD |
|-----------|-----|-----------------|
| Parameters (BF16) | 2P | 2P / N |
| Gradients (BF16) | 2P | 2P / N |
| Optim state (FP32) | 12P | 12P / N |
| Temporary AllGather buffer | 0 | 2P |
| Total steady | 16P | (16P / N) + 2P |

For a 70B model on 8 GPUs: DDP = 1120 GB (impossible), FSDP = ~280 GB (≈ 35 GB per GPU, feasible on 80 GB cards).

### Communication
Per micro-batch forward:
- AllGather parameters per transformer block just before its forward (then free).
- Compute forward.
- Reverse process for backward + ReduceScatter gradients.
Communication volume: 2P per step vs DDP's 2P — same bandwidth, different latency profile (N smaller messages vs 1 big one).

### Sharding strategies
| Strategy | Shards P | Shards Grad | Shards Opt | Comm overhead |
|----------|----------|-------------|------------|---------------|
| `NO_SHARD` (DDP) | no | no | no | AllReduce grads |
| `SHARD_GRAD_OP` | no | yes | yes | ReduceScatter grads |
| `FULL_SHARD` | yes | yes | yes | AllGather + ReduceScatter |
| `HYBRID_SHARD` | intra-node | intra-node | intra-node | intra FULL, inter REPLICATE |

### Wrapping policy
`auto_wrap_policy = transformer_auto_wrap_policy({LlamaDecoderLayer})`
Each transformer block becomes an FSDP unit → AllGather only per-block params, not the whole model.

### Mixed precision
```
MixedPrecision(param_dtype=torch.bfloat16,
               reduce_dtype=torch.bfloat16,
               buffer_dtype=torch.bfloat16)
```
Keeps FP32 master weights in the optimizer, BF16 everywhere else — combined with ZeRO sharding this is the standard modern SFT setup.

### Typical SFT recipe (70B)
| Knob | Value |
|------|-------|
| Strategy | FULL_SHARD |
| Precision | BF16 params + BF16 reduce + FP32 optim master |
| Activation checkpointing | per transformer block |
| Micro-batch per GPU | 1 |
| Gradient accumulation | 16 |
| Packing | yes (see [[sequence-packing]]) |
| Max seq length | 4096 |
| Learning rate | 1e-5, cosine, warmup 3% |
| Optimizer | AdamW β = (0.9, 0.95) |

## Connections
- Throughput companion: [[sequence-packing]] — FSDP + packing is the standard 2025 SFT backbone.
- Loss definition: [[loss-masking-prompt]].
- HF recipes using FSDP: [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]].
- Alternative: DeepSpeed ZeRO-3 (same math, different implementation).
