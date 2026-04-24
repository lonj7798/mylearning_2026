# ZeRO and FSDP: Communication Patterns in Detail

<!-- excerpt for ch-13, deep-dive on ZeRO stages and PyTorch FSDP implementation -->

## The Memory Arithmetic

In mixed precision training with Adam, the per-GPU memory for a model with $\Psi$ parameters breaks down as:

| Component | Precision | Bytes per Parameter |
|-----------|-----------|---------------------|
| BF16 parameters | BF16 | 2 |
| BF16 gradients | BF16 | 2 |
| FP32 master weights | FP32 | 4 |
| Adam momentum | FP32 | 4 |
| Adam variance | FP32 | 4 |
| **Total** | | **16** |

Optional: FP32 gradient accumulation adds another 4 bytes (nanotron does this for stability). Total: 20 bytes per parameter.

For a 70B model: $70 \times 10^9 \times 16 = 1{,}120$ GB — 14 H100 GPUs at 80 GB each just for static state, before any activations.

## ZeRO-1: Optimizer State Partitioning

**What changes:** The optimizer states (FP32 master weights + Adam momentum + Adam variance = $12\Psi$ bytes) are partitioned equally across $N_d$ data-parallel ranks.

**Training step:**
1. Forward pass: every rank has full BF16 parameters (identical). Different micro-batches.
2. Backward pass: every rank has full BF16 gradients (different values). **Reduce-scatter** gradients — each rank keeps only its $1/N_d$ shard.
3. Optimizer step: each rank updates only its $1/N_d$ of optimizer states and FP32 master weights.
4. **All-gather** the updated BF16 parameters — each rank sends its updated shard, all ranks get the full parameter set.

**Communication cost:** One reduce-scatter ($\Psi$ volume) + one all-gather ($\Psi$ volume) = $2\Psi$. This is **identical to vanilla DP's all-reduce** (which decomposes into reduce-scatter + all-gather internally). Zero overhead.

**Memory per GPU:** $2\Psi + 2\Psi + \frac{12\Psi}{N_d}$

At $N_d = 8$: $4\Psi + 1.5\Psi = 5.5\Psi$ bytes = 385 GB for 70B. Down from 1,120 GB — a 2.9x reduction.

## ZeRO-2: Adding Gradient Partitioning

**What changes:** Gradients are also sharded. Since each rank only needs $1/N_d$ of the gradients (for its optimizer shard), there's no point keeping the rest.

**Training step difference from ZeRO-1:** In the backward pass, the reduce-scatter directly produces gradient shards — each rank only stores $1/N_d$ of the gradients. The rest of the step is identical.

**Communication cost:** Still $2\Psi$ — the reduce-scatter and all-gather are the same operations.

**Memory per GPU:** $2\Psi + \frac{2\Psi + 12\Psi}{N_d}$

At $N_d = 8$: $2\Psi + 1.75\Psi = 3.5\Psi$ bytes = 245 GB for 70B. There is **no overhead vs ZeRO-1** — ZeRO-2 is strictly better.

## ZeRO-3 / FSDP: Full Parameter Sharding

**What changes:** Parameters themselves are sharded. Each rank stores only $1/N_d$ of the BF16 parameters and reconstructs full layers on-demand.

**Training step:**
1. Forward pass, layer $n$: **all-gather** parameters for layer $n$. Compute forward. **Discard** non-local parameter shards.
2. Backward pass, layer $n$: **all-gather** parameters again (needed for gradient computation). Compute backward. **Reduce-scatter** gradients. Discard non-local parameter shards.
3. Optimizer step: each rank updates its $1/N_d$ of everything.
4. **No final all-gather** needed — parameters will be gathered on-demand in the next forward pass.

**Communication cost:** $3\Psi$ total per step:
- Forward: $\Psi$ (one all-gather across all layers)
- Backward: $\Psi$ (one all-gather) + $\Psi$ (one reduce-scatter)

This is **50% more communication than ZeRO-2**. The additional $\Psi$ comes from re-gathering parameters in the backward pass (since we discarded them after forward).

**Memory per GPU:** $\frac{16\Psi}{N_d}$

At $N_d = 8$: $2\Psi = 140$ GB for 70B. Full sharding — memory scales inversely with GPU count.

## Prefetching: Hiding the 50% Overhead

The additional all-gather in ZeRO-3 can be overlapped with computation:

- **Forward prefetch:** While computing layer $n$'s forward pass, asynchronously all-gather layer $n+1$'s parameters. If the compute time exceeds the communication time, the all-gather is fully hidden.
- **Backward prefetch:** While computing layer $n$'s backward pass, all-gather layer $n-1$'s parameters.

The condition for full overlap: the compute time per layer must exceed the communication time per layer. This favors **large micro-batch sizes and long sequences** — more FLOPs per byte communicated.

When the condition is not met (small batch size, many GPUs), the all-gather sits on the critical path and ZeRO-3 is slower than ZeRO-2. This is why the Ultra-Scale Playbook recommends: "ZeRO-3 prefers large mbs and seq_len to hide comms."

## PyTorch FSDP: The Practical Implementation

PyTorch FSDP (Fully Sharded Data Parallel) is the production implementation of ZeRO-3. Key concepts:

**FSDP Unit:** The granularity of sharding. Typically one transformer layer. Parameters within a unit are flattened into a single contiguous buffer for efficient communication.

**Sharding strategy options:**
- `FULL_SHARD` — ZeRO-3: shard everything
- `SHARD_GRAD_OP` — ZeRO-2: shard gradients and optimizer states only
- `NO_SHARD` — vanilla DP: replicate everything

**FSDP2 (PyTorch 2.x):** Simplified API with per-parameter sharding (instead of per-unit), better composability with TP, and improved memory management. The key improvement: FSDP2 can overlap communication at parameter granularity rather than layer granularity, enabling finer-grained prefetching.

## When to Use Which Stage

| Scenario | Recommendation |
|----------|---------------|
| Model fits on 1 GPU | Vanilla DP (no ZeRO needed) |
| Optimizer doesn't fit | ZeRO-1 (free, no overhead) |
| Gradients also tight | ZeRO-2 (still no overhead vs DP) |
| Parameters don't fit | ZeRO-3 / FSDP (+50% comm, needs large batch) |
| Activations are bottleneck | ZeRO won't help — use TP or activation checkpointing |

DeepSeek V3 used ZeRO-1 + PP — enough optimizer sharding to fit, without ZeRO-3's communication overhead. This is a common production choice: use PP to distribute parameters across nodes, ZeRO-1/2 to handle optimizer state within each PP stage's DP group.
