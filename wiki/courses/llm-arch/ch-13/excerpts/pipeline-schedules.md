# Pipeline Parallelism Schedules: From Naive to Zero Bubble

<!-- excerpt for ch-13, deep-dive on pipeline scheduling strategies -->

## The Bubble Problem

Pipeline parallelism splits a model's $L$ layers across $p$ stages. Activations flow forward stage-by-stage, gradients flow backward. The fundamental inefficiency: while stage $k$ computes, stages $0$ through $k-1$ and $k+1$ through $p-1$ are idle.

Define:
- $t_f$ = time for one micro-batch's forward pass through one stage
- $t_b$ = time for one micro-batch's backward pass through one stage (typically $t_b \approx 2 \cdot t_f$)
- $p$ = number of pipeline stages
- $m$ = number of micro-batches

## Schedule 1: Naive (Single Micro-batch)

With one micro-batch ($m = 1$), the total time is $(p) \cdot t_f + (p) \cdot t_b$. The ideal time (perfect parallelism) is $t_f + t_b$. The bubble ratio:

$$r_{\text{bubble}} = \frac{(p - 1)(t_f + t_b)}{t_f + t_b} = p - 1$$

At $p = 8$: GPUs are idle **87.5%** of the time. Completely impractical.

## Schedule 2: All-Forward-All-Backward (AFAB)

Split the batch into $m$ micro-batches. Run all $m$ forward passes, then all $m$ backward passes. The bubble stays the same absolute size, but useful work increases:

$$r_{\text{bubble}} = \frac{p - 1}{m}$$

At $p = 8, m = 32$: bubble = **21.9%**. Much better.

**Problem:** AFAB must store activations for all $m$ micro-batches until the backward pass reaches them. Memory scales as $O(m)$ per stage. At $m = 32$, that is 32 sets of activations per stage — often exceeding GPU memory.

## Schedule 3: 1F1B (One-Forward-One-Backward)

The key insight: start backward passes as soon as the last stage finishes its first forward pass. In steady state, each stage alternates one forward and one backward pass.

**Three phases:**
1. **Warmup:** Stage $s$ runs $p - s$ forward passes to fill the pipeline
2. **Steady state:** Alternating 1 forward + 1 backward per step
3. **Cooldown:** Remaining backward passes drain the pipeline

**Bubble size:** Same as AFAB: $r_{\text{bubble}} = \frac{p-1}{m}$

**Memory advantage:** Each stage stores activations for at most $p$ micro-batches simultaneously (not $m$). Since $p \ll m$ in practice, this is a massive memory reduction.

1F1B is the industry-standard baseline. Most training frameworks (Megatron-LM, nanotron, PyTorch's PipelineSchedule) implement 1F1B as the default PP schedule.

## Schedule 4: Interleaved Stages

Instead of assigning contiguous layers to each GPU, assign $v$ non-contiguous "chunks." GPU 0 gets layers {1-4, 17-20}, GPU 1 gets layers {5-8, 21-24}, etc. Each micro-batch loops through each GPU $v$ times.

Each forward/backward pass through a single chunk takes $t_f/v$ and $t_b/v$ respectively. The bubble:

$$r_{\text{bubble}} = \frac{p - 1}{v \cdot m}$$

At $p = 8, v = 2, m = 16$: bubble = **21.9%** (same as AFAB with $m = 32$, but with half the micro-batches).

**Tradeoffs:**
- $v$ times more point-to-point communications (each micro-batch traverses each GPU $v$ times)
- More complex scheduling: must decide whether to prioritize depth-first (finish micro-batches quickly) or breadth-first (fill pipeline fully)
- Llama 3.1 uses interleaved 1F1B with tunable priority between depth-first and breadth-first

**Scheduling decision:** At any point, a GPU may have both forward work (earlier chunk of a later micro-batch) and backward work (later chunk of an earlier micro-batch) ready. Depth-first prioritizes the backward work (closing loops faster, reducing memory). Breadth-first prioritizes the forward work (filling the pipeline, reducing bubble). The optimal choice depends on the relative cost of memory vs bubble overhead.

## Schedule 5: Zero Bubble (B/W Decomposition)

The ZeroBubble paper observed that the backward pass through a matrix multiplication decomposes into two independent operations:

- **B (backward for inputs):** Computes $\frac{\partial L}{\partial x}$, needed by the previous layer's backward pass. Must happen in sequence.
- **W (backward for weights):** Computes $\frac{\partial L}{\partial W}$, needed only before the optimizer step. Can be scheduled flexibly.

Since W has no downstream dependency in the backward pipeline, it can fill bubble slots:

$$r_{\text{bubble}} \approx 0 \text{ (theoretically)}$$

In practice, the scheduling problem becomes: given the durations of F, B, and W for each stage, find a schedule that minimizes idle time. This is formulated as an integer linear program (ILP) and solved offline.

## DualPipe (DeepSeek V3)

DeepSeek V3 extended ZeroBubble with **DualPipe**: two micro-batch streams propagate from both ends of the pipeline simultaneously:

- Stream A flows forward from stage 0 to stage $p-1$
- Stream B flows forward from stage $p-1$ to stage 0

The two streams interleave on each GPU, filling each other's bubbles. Combined with the B/W decomposition, DualPipe achieves near-zero idle time.

The cost: DualPipe requires careful memory management (two concurrent streams double the activation memory), sophisticated scheduling (the ILP becomes harder with two streams), and bidirectional communication patterns. The DeepSeek V3 report notes they "achieved a near-zero all-to-all communication overhead" — implying the pipeline scheduling also overlaps expert-parallel communication.

## Practical Pipeline Sizing

From the Ultra-Scale Playbook benchmarks:

| Configuration | Throughput Loss vs Ideal | Notes |
|--------------|------------------------|-------|
| PP=2, 1 node | ~5% | Minimal overhead |
| PP=4, 1 node | ~10% | Acceptable |
| PP=8, 1 node | ~15% | Standard for 70B |
| PP=8, 2 nodes | ~14% | PP scales well cross-node |
| PP=16, 2 nodes | ~25% | Bubble starts dominating |

Compare to TP=16 across 2 nodes: **~43% loss**. Pipeline parallelism's lower bandwidth requirements make it the preferred strategy for cross-node parallelism.

**Interaction with batch size:** The bubble fraction $\frac{p-1}{v \cdot m}$ means PP prefers large numbers of micro-batches. Since $m = \frac{\text{gbs}}{N_{\text{DP}} \cdot \text{mbs}}$, this means PP prefers either large global batch sizes or small DP degrees. There is a direct tension between maximizing DP (for throughput) and maximizing $m$ (for small bubbles).
