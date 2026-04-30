<!-- chapter: ch-05
     track: foundations
     title: Distributed Training — FSDP, TP, PP, ZeRO
     sources: [[fsdp-sft]], [[sequence-packing]], [[loss-masking-prompt]], [[mixed-precision]], [[adam]], [[gradient-clipping]]
     figures: figures/fsdp-memory.html
-->

# Chapter 5 — Distributed Training: FSDP, TP, PP, ZeRO

> **Core insight.** No single parallelism primitive can train a 70B+ model end-to-end. Real training is **nD parallelism**: data sharding (FSDP / ZeRO-3) for memory, tensor parallelism for within-layer compute, pipeline parallelism to span nodes, plus expert parallelism for MoE. Each axis has a memory/bandwidth trade; getting them wrong is how training jobs hang.
>
> **Guideline.** For < 13B models: FSDP (FULL_SHARD) + bf16 + activation checkpointing + packed SFT is enough. For 13–70B: add within-node TP(=8) if NVLink exists; keep FSDP across nodes. For 200B+: 3D parallelism (DP × TP × PP) and expert parallelism for MoE. Do not write these primitives yourself — use verl / OpenRLHF / torchtitan / Megatron / DeepSpeed.

---

## Why this chapter exists

Distributed training is the single largest source of wasted frontier-lab compute. Not because the algorithms are novel — they aren't — but because the interactions between sharding, mixed precision, gradient clipping, and checkpointing create silent bugs that only show up at 64+ GPU scale. A naive clip-norm call on FSDP local shards under-counts the global norm by √N and silently diverges. A TP group that spans across nodes gets crushed by inter-node bandwidth. Activation checkpointing inside a TP group doubles the communication cost if you activate-checkpoint the wrong boundary.

This chapter gives you the correctness model — what goes where, which communication happens when, which framework call is the right one — so that when you sit down in front of a real training codebase, the primitives are already named in your head.

Primary source: [[fsdp-sft]] for the FSDP mechanics; [[sequence-packing]] and [[mixed-precision]] for the interactions.

---

## 1. The memory formula — why you can't DDP a 70B

Source: [[fsdp-sft]]. Per-GPU memory with `N` data-parallel ranks, `P` parameters, AdamW:

| Component | DDP (replicate) | ZeRO-2 (SHARD_GRAD_OP) | ZeRO-3 / FSDP (FULL_SHARD) |
|---|---|---|---|
| Parameters (bf16) | 2P | 2P | 2P / N |
| Gradients (bf16) | 2P | 2P / N | 2P / N |
| Optimizer state (fp32: m, v, master) | 12P | 12P / N | 12P / N |
| Temporary AllGather buffer | 0 | 0 | ≈ 2P (largest FSDP unit) |
| **Steady-state total** | **16P** | **(4P / N) + 12P** | **(16P / N) + 2P** |

Plug in 70B × 8 GPUs:

- DDP: 16 · 70 = 1120 GB per GPU. Infeasible.
- ZeRO-3 / FSDP: (16 · 70 / 8) + 2 · 70 ≈ 280 GB per GPU. **Still infeasible on 80 GB.**
- Add activation checkpointing (~2×+ activation memory saving) and ZeRO-3 offload is possible on 8 × 80 GB.

At 70B SFT, FSDP FULL_SHARD + activation-ckpt + bf16 + fp32 master is a hard *floor*, not a nicety. See `figures/fsdp-memory.html` for an interactive memory calculator that walks this out for any (P, N, precision) triple.

---

## 2. FSDP / ZeRO — the data-parallel sharding axis

FSDP and DeepSpeed ZeRO implement the same math; the implementations are independent. Three strategies matter:

```
NO_SHARD          ≡ DDP                → AllReduce grads
SHARD_GRAD_OP     ≡ ZeRO-2            → ReduceScatter grads; shard grads + opt state
FULL_SHARD        ≡ ZeRO-3            → AllGather params per-block + ReduceScatter grads
HYBRID_SHARD      ≡ intra-node FULL,   → FULL_SHARD within node, REPLICATE across nodes
                    inter-node REPL
```

**What happens per step under FULL_SHARD** ([[fsdp-sft]]):

1. **Forward.** Before each transformer block's forward, AllGather its sharded parameters into the full tensor; run the block; free the gathered copy.
2. **Backward.** Same AllGather for weights; ReduceScatter gradients into their shard.
3. **Optimizer step.** Each rank updates only its local shard of parameters and optimizer state.

Communication volume per step: 2P AllGather + 2P ReduceScatter = 4P. DDP's AllReduce is 2P. FSDP's extra bandwidth buys you N× memory saving.

**HYBRID_SHARD** is the pragmatic compromise. Within-node (8 GPUs, NVLink) do FULL_SHARD; across-node (slow Ethernet / IB) do REPLICATE. This turns the slow cross-node communication into a classic DDP AllReduce at the end of backward, while keeping FSDP's memory savings within the fast-NVLink node.

**Wrapping policy** — what a "sharding unit" is:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer},   # one unit per block
)
```

Each transformer block is an FSDP unit, so the AllGather fetches only that block's parameters. Making the whole model one unit would AllGather everything → you lose the memory savings.

### Mixed precision + FSDP — the correct incantation

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp import ShardingStrategy

bf16_mp = MixedPrecision(
    param_dtype   = torch.bfloat16,   # weights live in bf16 after AllGather
    reduce_dtype  = torch.bfloat16,   # gradient ReduceScatter in bf16
    buffer_dtype  = torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy = ShardingStrategy.FULL_SHARD,
    auto_wrap_policy  = auto_wrap_policy,
    mixed_precision   = bf16_mp,
    # activation checkpointing applied separately via apply_activation_checkpointing
)
```

Optimizer state (AdamW `m`, `v`, master weights) stays in fp32 — FSDP keeps them sharded in fp32 regardless of `param_dtype`. See [[mixed-precision]] for the cross-linked rule.

### The FSDP-specific gradient-clipping bug

From [[gradient-clipping]]: clip-norm must be computed **across all shards**. A naive `torch.nn.utils.clip_grad_norm_` on local parameters under-counts by a factor of √N. Use the FSDP method instead:

```python
model.clip_grad_norm_(max_norm=1.0)     # correct — computes global norm
```

DeepSpeed has an analogous `engine.clip_grad_norm_`. Do not write your own. This is the number-one silent divergence bug in home-rolled FSDP training loops.

---

## 3. Tensor Parallelism — splitting within a layer

FSDP shards whole parameters across ranks. **Tensor Parallelism (TP)** splits *within* a single parameter matrix across ranks. It's the right tool when a single layer's compute (not memory) is the bottleneck — 70B+ models, wide MLPs, long sequences.

The mechanic (Megatron-LM convention):

```
For Y = X · W where W is [d_in, d_out]:

  Column parallelism:  W → [W_1, W_2, ..., W_TP]   (split columns)
                       Y = X · W = concat(X · W_1, X · W_2, ...)
                       → no communication in forward, AllReduce in backward
  Row parallelism:     W^T → [W_1^T, W_2^T, ...]   (split rows)
                       Y = X · W, X itself is partitioned
                       → AllReduce in forward, no comm in backward
```

The attention sub-layer uses column-parallel `Q, K, V` projections followed by row-parallel `W_O` — so one AllReduce per attention block, one per MLP. Same for the `W_gate / W_up` + `W_down` MLP.

**TP is bandwidth-bound.** AllReduce happens inside every transformer block, twice. On NVLink-connected GPUs this is fine; across PCIe or Ethernet it tanks throughput. **Rule of thumb: TP must fit inside one node.** Typical choice: TP=8 on a single 8-GPU NVLink node.

**Composing TP × FSDP.** TP parallelises within a group of `TP` ranks; FSDP parallelises across the `N / TP` groups. This is "2D parallelism." Frameworks (Megatron, NeMo, torchtitan) handle this by setting up a 2D process group.

---

## 4. Pipeline Parallelism — splitting across depth

When a model can't fit in a single TP group (even at TP=8) or when cross-node communication kills TP, you slice the model *across depth*: layers 1–10 on node A, layers 11–20 on node B, and so on. This is **Pipeline Parallelism (PP)**.

The naive PP has the "bubble problem": while node A does the forward for micro-batch 1, nodes B/C/D are idle. **1F1B scheduling** (one-forward-one-backward) interleaves micro-batches to keep every node busy:

```
Time →
Stage 1:  F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
Stage 2:     F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
Stage 3:        F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
Stage 4:           F1 F2 F3 F4 F5 B1 B2 B3 B4 B5
```

The bubble fraction is roughly `(PP − 1) / (num_microbatches + PP − 1)`. With PP=8 and 64 micro-batches, bubble ≈ 10%. Interleaved 1F1B drops this further at the cost of activation memory.

PP trades latency for cross-node bandwidth — activations move between stages (small) instead of weights/grads (large). At frontier scale (DeepSeek-V3, Llama-3 405B), PP is how you span nodes.

---

## 5. Expert Parallelism — MoE-specific

For Mixture-of-Experts models (DeepSeek-V3, Mixtral, Qwen-MoE), each expert is a separate FFN that only processes a subset of tokens. **Expert Parallelism (EP)** places different experts on different ranks. Routing sends each token to its chosen expert rank via AllToAll.

EP's bottleneck is AllToAll bandwidth. At the DeepSeek-V3 scale (671B total, 37B active), expert parallelism is its own dimension layered on TP × PP × DP. DeepSeek developed custom AllToAll kernels (DualPipe) to hide this communication behind compute.

For a 2025 MoE training run, the parallelism recipe looks like `DP × TP × PP × EP`, where each axis is chosen to match a bandwidth tier of the cluster.

---

## 6. Activation checkpointing — the orthogonal memory knob

The distributed-memory formula in §1 budgets *parameter* memory. **Activation memory** (forward activations kept around for backward) can be larger. Two orders of magnitude smaller than parameters per token, but multiplied by sequence length × batch × layers.

Activation checkpointing re-runs the forward pass of a layer during backward instead of storing its activations. Memory saved: O(L · seqlen · hidden). Cost: 25–35% more compute.

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, CheckpointImpl,
)
apply_activation_checkpointing(
    model,
    check_fn = lambda m: isinstance(m, LlamaDecoderLayer),   # every block
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)
```

Modern default: checkpoint every transformer block. The NO_REENTRANT variant composes correctly with FSDP; the reentrant variant does not.

---

## 7. The typical 2025 SFT recipe — 70B on 8×80GB

From [[fsdp-sft]]:

| Knob | Value |
|---|---|
| Strategy | FSDP FULL_SHARD |
| Precision | bf16 params + bf16 reduce + fp32 optimizer master |
| Activation ckpt | per transformer block (NO_REENTRANT) |
| Micro-batch / GPU | 1 |
| Gradient accumulation | 16 (effective batch = 128) |
| Sequence packing | yes (see ch-04 / [[sequence-packing]]) |
| Max seq length | 4096 |
| Optimizer | AdamW β=(0.9, 0.95), wd=0.0 (SFT), ε=1e-8 |
| Learning rate | 1e-5, cosine, warmup 3% |
| Gradient clip | 1.0 via `model.clip_grad_norm_` |
| Chat template | tokenizer's `apply_chat_template` with `train_on_response_only=True` |

Pin the configuration. Every knob in that table has a failure mode documented in chapters 1–4.

---

## 8. Frameworks — the 2025 field guide

Rule of thumb: **don't hand-roll**. Pick one and understand its wrapping conventions.

| Framework | Sharding | TP | PP | MoE / EP | Best for |
|---|---|---|---|---|---|
| **PyTorch FSDP2** | FULL / HYBRID / NO_SHARD | via DeviceMesh | third-party | limited | research, SFT up to ~70B |
| **torchtitan** | FSDP2 + native TP + PP | yes | yes | yes | modern PyTorch reference for pretraining |
| **Megatron-LM** | DDP + ZeRO-1 | native TP | 1F1B PP | EP | Nvidia frontier pretraining |
| **DeepSpeed ZeRO-3** | same math as FSDP | via Megatron | yes | yes | legacy production, bf16/fp16 offload |
| **NeMo** | Megatron underneath | yes | yes | yes | Nvidia enterprise stack |
| **verl / OpenRLHF** | FSDP + TP | yes | limited | some | RL with actor + ref + RM + critic orchestration |
| **TRL** | Accelerate + FSDP | limited | no | no | single-node SFT / preference tuning |

Chapter 53–56 in the Infrastructure track will walk verl / OpenRLHF / TRL internals end-to-end.

---

## 9. Common silent-failure modes at scale

- **FSDP local clip.** Calling `clip_grad_norm_` instead of `model.clip_grad_norm_` under-counts the global norm by √N. Silent divergence.
- **Activation checkpointing inside a TP group with reentrant impl.** Double recompute cost; fix by using NO_REENTRANT.
- **Mixed fp16 / bf16 across shards.** `param_dtype=bf16` but `reduce_dtype=fp16` creates a type-cast at every gradient reduction; numerical drift.
- **TP group spanning nodes.** Inter-node AllReduce inside every block kills throughput. Constrain TP to a single NVLink node.
- **NCCL timeouts under skewed batches.** One rank's straggler (a particularly long sequence in the pack) delays the whole collective; timeout manifests as a hang. Diagnose with `py-spy dump --pid <rank>`.
- **Checkpointing the optimizer only on rank 0.** Under FSDP each rank owns a shard of the optimizer state; use the sharded-save API.
- **Data loader desync across resumes.** Each rank's sampler must resume from the same global step, not local. A resume bug that surfaces as "training starts at a different loss than expected."

---

## Connections and what's next

- **[[fsdp-sft]] / ch-05** — FSDP mechanics and memory formulas (this chapter).
- **[[sequence-packing]] / [[loss-masking-prompt]] / ch-04** — packing + masking are what make FSDP's per-step memory win real.
- **[[mixed-precision]] / ch-02** — `MixedPrecision(param_dtype, reduce_dtype)` selects the per-shard precision.
- **[[gradient-clipping]] / ch-01** — the distributed-clip-norm bug.
- **ch-06 (checkpointing + resume)** — sharded checkpoint save/load and bit-exact resume.
- **ch-07 (failure modes)** — the silent-failure list extends and debugs the checklist above.
- **ch-52–56 (infrastructure internals)** — verl / OpenRLHF / TRL source-level deep dives.

## Further reading

- [[fsdp-sft]] — Zhao 2023; PyTorch's industry-scale FSDP implementation.
- DeepSeek-V3 technical report ([[deepseek-v3]]) — frontier 3D + EP parallelism with DualPipe AllToAll hiding.
- Megatron-LM paper — tensor + pipeline parallelism foundations.
- Karpathy's "recipe" ([[karpathy-training-neural-net-recipe]]) — "scale up slowly; add parallelism after the single-GPU recipe works."

## Companion visualization

**[figures/fsdp-memory.html](figures/fsdp-memory.html)** — interactive per-GPU memory calculator across DDP / ZeRO-2 / ZeRO-3 / FSDP FULL_SHARD, with sliders for parameter count, GPU count, precision (bf16 / fp16 / fp32), and activation-checkpointing toggle. Use it to see why DDP gives up at ~7B and why FSDP + activation-ckpt is the floor at 70B.
