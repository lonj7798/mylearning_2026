---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fsdp-sft.md
source_url: https://arxiv.org/abs/2304.11277
created_at: "2026-04-23"
---

# Excerpt: PyTorch FSDP — Experiences on Scaling Fully Sharded Data Parallel

**Paper:** PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
**Authors:** Zhao, Gu, Varma, Luo, Huang, Xu, Wright, Shojanazeri, Ott, Shleifer, Desmaison, Balioglu, Damania, Nguyen, Chauhan, Hao, Mathews, Li (Meta / PyTorch, 2023)
**arXiv:** 2304.11277

---

## What problem FSDP solves — the DDP failure mode

The source frames FSDP as an answer to a single limit: replicated data parallelism (DDP) holds an entire parameter set, gradient set, and optimizer state **on every rank**, which caps feasible model size at whatever fits on one GPU regardless of how many GPUs you own. Quoted:

> "FSDP shards model parameters, gradients, and optimizer states across the data-parallel group and reconstructs full parameters on demand via AllGather; gradients are reduced via ReduceScatter."

The sharding axis is the same process group DDP would use — same `N` ranks, same all-reduce bandwidth budget — but the memory footprint is divided by `N`. This is the reason FSDP (and its mathematical twin, DeepSpeed ZeRO-3) is the universal SFT backbone for ≥ 13B models in 2025.

**Notice:** FSDP's bandwidth budget is **not** the same as DDP's. The paper is explicit that the communication pattern changes: instead of one AllReduce at the end of backward, FSDP emits one AllGather in forward, one AllGather in backward, and one ReduceScatter in backward — **per transformer block**, not per step. Bandwidth volume is comparable (`2P` vs `2P` for DDP AllReduce), but the latency profile is `N` small messages rather than one big one.

---

## The memory formula — steady state and peak

The paper's Section 4 (Memory breakdown) enumerates each tensor class separately. The excerpt reproduces it (P = parameter count, N = data-parallel group size, AdamW assumed):

| Component                        | DDP  | FSDP `FULL_SHARD`   |
|----------------------------------|------|---------------------|
| Parameters (bf16)                | 2P   | 2P / N              |
| Gradients (bf16)                 | 2P   | 2P / N              |
| Optimizer state (fp32: m, v, θ*) | 12P  | 12P / N             |
| Temporary AllGather buffer       | 0    | ≈ 2P (largest unit) |
| **Steady-state total**           | 16P  | (16P / N) + 2P      |

Deriving the `12P` optimizer-state line: AdamW keeps `m_t` (fp32, 4P), `v_t` (fp32, 4P), and master weights `θ*` (fp32, 4P) → 12P bytes. See [[excerpts/adam]] for the per-parameter breakdown and [[excerpts/mixed-precision]] for why the master copy must stay in fp32.

Plug in P = 70B, N = 8 (single-node 8×80GB):

```math
\text{DDP:}\quad 16 \cdot 70\text{B} = 1120\ \text{GB/GPU}\ \text{(infeasible)}
```

```math
\text{FSDP:}\quad (16 \cdot 70\text{B} / 8) + 2 \cdot 70\text{B} = 140 + 140 = 280\ \text{GB/GPU}
```

Still > 80 GB; activation checkpointing plus gradient-accumulation (see `read.md` §6) must close the remaining gap. The paper's Table 1 empirically validates this — DDP refuses at ~7B, FSDP scales to 175B.

**Notice:** the `2P` temporary AllGather buffer is the size of the **largest FSDP unit**, not of the full model. This is why the wrapping policy matters: if the whole model is one unit, the buffer is `2P` of the model (defeating the memory win); if the unit is one transformer block, the buffer is `2 · (P / L)` where L is the block count.

---

## The three sharding strategies — a mathematical equivalence to ZeRO

Quoted from Key Contributions:

> "Three sharding strategies: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (DDP). Hybrid sharding within-node + replicate across-node for cluster topologies."

The table reproduced in the excerpt:

| Strategy         | Shards P     | Shards Grad  | Shards Opt   | Comm pattern                                       |
|------------------|--------------|--------------|--------------|----------------------------------------------------|
| `NO_SHARD`       | no           | no           | no           | AllReduce grads                                    |
| `SHARD_GRAD_OP`  | no           | yes          | yes          | ReduceScatter grads                                |
| `FULL_SHARD`     | yes          | yes          | yes          | AllGather params + ReduceScatter grads             |
| `HYBRID_SHARD`   | intra-node   | intra-node   | intra-node   | FULL_SHARD inside node, REPLICATE across nodes     |

```math
\text{AllReduce} \equiv \text{ReduceScatter} \circ \text{AllGather}
```

This identity is why `SHARD_GRAD_OP` can decompose DDP's AllReduce into the cheaper ReduceScatter (grads land already sharded, optimizer updates a shard, no post-step broadcast needed because parameters are still replicated). `FULL_SHARD` goes further by dropping the replicated parameter copy too — forcing an AllGather before each block's forward.

**Why HYBRID_SHARD is the pragmatic pick.** Single-node NVLink bandwidth (~600 GB/s per GPU) swamps cross-node IB (~25 GB/s per GPU) by 20×. HYBRID_SHARD pays FSDP's AllGather cost only on the fast intra-node link and reverts to DDP-style AllReduce across nodes — turning the cross-node cost back into one classical AllReduce per step rather than `L` AllGathers per step.

---

## The transformer-block wrapping policy

Quoted:

> "`auto_wrap_policy = transformer_auto_wrap_policy({LlamaDecoderLayer})`. Each transformer block becomes an FSDP unit → AllGather only per-block params, not the whole model."

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer},   # one unit per block
)
```

**Walkthrough of the forward pass under FULL_SHARD:**

1. Block 1's pre-forward hook fires. FSDP issues an AllGather across the N ranks to reconstruct Block 1's full parameters.
2. Block 1 runs forward using its gathered (full-rank) params.
3. Block 1's post-forward hook fires. The gathered copy is freed; only the local shard remains.
4. Block 2 repeats the cycle.

At any moment, **exactly one block's full parameters plus all blocks' shards** live on device. Peak activation memory = `(2P / N) · L + 2P / L` for params alone. This is the origin of the `+ 2P` term in the memory formula — it's the full-block buffer.

**Notice:** making the whole model a single FSDP unit defeats the scheme — the AllGather reconstructs every parameter at once, and the `+ 2P` term becomes `+ 2P` of the model. The auto-wrap policy by `transformer_layer_cls` is not optional at scale.

---

## BF16 MixedPrecision config — the three-dtype rule

Quoted:

```python
MixedPrecision(param_dtype=torch.bfloat16,
               reduce_dtype=torch.bfloat16,
               buffer_dtype=torch.bfloat16)
```

Three independent dtype axes — the paper's design ensures you can pick each one without accidentally casting across:

- `param_dtype` — weights live in bf16 after AllGather, forward matmuls are bf16.
- `reduce_dtype` — the ReduceScatter op runs in bf16. Setting `reduce_dtype=torch.float32` is sometimes used at extreme scale for numerically hot reductions, at 2× bandwidth cost.
- `buffer_dtype` — non-parameter buffers (e.g., RoPE frequency caches, attention masks) stored in bf16.

The optimizer state is **not** touched by `MixedPrecision` — FSDP keeps m, v, master weights in fp32 regardless. See [[excerpts/mixed-precision]] for why the master copy must survive the cast, and [[excerpts/adam]] for why bf16 `v_hat` underflows within ~100 steps.

**Notice:** a subtle production bug — setting `param_dtype=bf16` and `reduce_dtype=fp16` (or vice versa) creates a cast at every gradient reduction; numerical drift accumulates silently across thousands of steps. Pin both to bf16 unless you have a measured reason otherwise.

---

## Communication volume analysis — AllGather/ReduceScatter vs DDP

Per step, FULL_SHARD emits:

```math
\text{Comm}_{\text{FSDP}} = \underbrace{2P}_{\text{AllGather fwd}} + \underbrace{2P}_{\text{AllGather bwd}} + \underbrace{2P}_{\text{ReduceScatter grads}} = 6P
```

DDP's AllReduce decomposes as ReduceScatter + AllGather, each of volume `P` per rank:

```math
\text{Comm}_{\text{DDP}} = 2P
```

FSDP pays 3× the bandwidth for `N`× the memory. On NVLink this is a net throughput win only because the larger model fits at all. On Ethernet-only clusters the bandwidth bill dominates and HYBRID_SHARD (or full DDP with activation offload) is preferable.

The paper reports near-linear scalability in TFLOPS up to 175B — this is possible only because the AllGather is fully overlapped with the *previous* block's compute via CUDA streams. FSDP prefetches the next block's AllGather while the current block's forward is running. Misconfigured prefetch (`FORWARD_PREFETCH`) shows up as stalls exactly at block boundaries.

---

## The 70B SFT recipe — pinned

Quoted Table (Typical SFT recipe, 70B):

| Knob                     | Value                                                |
|--------------------------|------------------------------------------------------|
| Strategy                 | `FULL_SHARD`                                         |
| Precision                | bf16 params + bf16 reduce + fp32 optimizer master    |
| Activation checkpointing | per transformer block                                |
| Micro-batch per GPU      | 1                                                    |
| Gradient accumulation    | 16                                                   |
| Packing                  | yes (see [[excerpts/sequence-packing]])              |
| Max seq length           | 4096                                                 |
| Learning rate            | 1e-5, cosine, warmup 3%                              |
| Optimizer                | AdamW β=(0.9, 0.95)                                  |

Every field in this table is load-bearing for the memory-budget math:
- Micro-batch 1 + grad-accum 16 = effective batch 128 per DP group; accum trades step-wall-time for activation memory, keeping the per-step footprint within 80 GB.
- Packing doubles the useful tokens per batch (see [[excerpts/sequence-packing]]); without packing the activation budget dominates the `2P` buffer.
- bf16 reduce (not fp32) is what keeps the ReduceScatter bandwidth-bound instead of compute-bound on H100s.

---

## Connections

- [[excerpts/sequence-packing]] — packing reduces per-step memory so FSDP's sharding is effective.
- [[excerpts/loss-masking-prompt]] — label-masking interacts with FSDP's ReduceScatter on token-count normalization.
- [[excerpts/mixed-precision]] — `MixedPrecision(param_dtype, reduce_dtype)` ties directly to this paper's Table 1.
- [[excerpts/adam]] — the `12P` optimizer-state line is where AdamW's m, v, master weights live.
- [[excerpts/gradient-clipping]] — FSDP demands `model.clip_grad_norm_` for global-norm correctness.
- [[ch-05]] — synthesis and the 70B single-node recipe.
