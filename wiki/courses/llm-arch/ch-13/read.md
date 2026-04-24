# Chapter 13: Distributed Training

<!-- scope: data parallelism, tensor parallelism, pipeline parallelism, ZeRO/FSDP, expert parallelism — the 5D parallelism design space
     deps: [[ch-10]]
     see-also: [[ch-14]], [[ch-11]]
-->

## Overview

A single GPU cannot train a modern large language model. A Llama 3 405B model requires over 6 TB just for parameters, gradients, and optimizer states in mixed precision — roughly 75 H100 GPUs worth of memory before a single activation is stored. Even a 7B model, while it fits on one GPU, would take months to train without parallelizing across dozens or hundreds of devices.

Distributed training is therefore not optional infrastructure — it is an architectural constraint that shapes every design decision in modern LLMs. The number of attention heads must be divisible by the tensor parallelism degree. The number of layers must divide evenly across pipeline stages. The expert count in MoE models determines expert parallelism topology. Understanding distributed training is understanding *why* models are shaped the way they are.

This chapter covers the five parallelism dimensions that modern training stacks combine: **data parallelism** (replicate model, split data), **tensor parallelism** (split individual layers across GPUs), **pipeline parallelism** (split layers across GPUs with micro-batching), **ZeRO/FSDP** (shard optimizer states, gradients, and parameters), and **expert parallelism** (distribute MoE experts). The Ultra-Scale Playbook ([[ultra-scale-playbook|paper]]) ran over 4,000 experiments on up to 512 GPUs to benchmark these strategies — we draw heavily on their findings.

The core tension throughout is a three-way tradeoff: **memory** (hard limit — if it doesn't fit, training stops), **compute efficiency** (GPU utilization), and **communication overhead** (idle time waiting for data transfers). Every parallelism strategy trades one for another. Finding the optimal combination for a given model, cluster, and batch size is the central engineering challenge of large-scale training.

---

## 1. The Memory Budget: Why Parallelism Is Necessary

Before distributing anything, we need to understand what consumes GPU memory during training. Four components compete for the same VRAM:

**Parameters:** For a transformer with hidden dimension $h$, vocabulary $v$, and $L$ layers:

$$N = h \cdot v + L \cdot (12h^2 + 13h) + 2h$$

In mixed precision (BF16 compute + FP32 master weights), parameters consume $2N$ bytes for BF16 weights plus $4N$ bytes for the FP32 copy.

**Gradients:** $2N$ bytes in BF16 (or $6N$ if accumulating in FP32 for stability, as nanotron does).

**Optimizer states:** Adam stores momentum and variance, each in FP32: $8N$ bytes total.

**Activations:** The largest and most variable component, scaling with batch size and sequence length:

$$m_{\text{act}} = L \cdot \text{seq} \cdot \text{bs} \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot \text{seq}}{h}\right)$$

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Memory Budget: Llama 3 70B (Mixed Precision, Adam, seq=4096, bs=1)</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Component</th>
<th style="text-align:right; padding:8px;">Formula</th>
<th style="text-align:right; padding:8px;">Memory</th>
<th style="text-align:right; padding:8px;">% of Total</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Parameters (BF16)</td>
<td style="text-align:right; padding:8px;">2N</td>
<td style="text-align:right; padding:8px;">140 GB</td>
<td style="text-align:right; padding:8px;">12.5%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Gradients (BF16)</td>
<td style="text-align:right; padding:8px;">2N</td>
<td style="text-align:right; padding:8px;">140 GB</td>
<td style="text-align:right; padding:8px;">12.5%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Optimizer (FP32 master + Adam)</td>
<td style="text-align:right; padding:8px;">4N + 8N</td>
<td style="text-align:right; padding:8px;">840 GB</td>
<td style="text-align:right; padding:8px;">75.0%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ff6b6b; font-weight:bold;">Activations (selective recomp)</td>
<td style="text-align:right; padding:8px;">varies</td>
<td style="text-align:right; padding:8px;">~25 GB</td>
<td style="text-align:right; padding:8px;">~2%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold; color:#fff;">Total</td>
<td style="text-align:right; padding:8px;"></td>
<td style="text-align:right; padding:8px; font-weight:bold;">~1,120 GB</td>
<td style="text-align:right; padding:8px;">100%</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
An H100 has 80 GB. You need at minimum 14 H100s just to hold the state — and that's before increasing batch size or sequence length. Activations explode with larger batches: at bs=4, they reach ~100 GB.
</div>
</div>

The key observation: **optimizer states dominate memory for small batch sizes**, but **activations dominate at large batch sizes or long sequences**. This distinction determines which parallelism strategy helps most. ZeRO shards optimizer states/gradients/parameters but cannot touch activations. Tensor parallelism shards activations but adds communication on the critical path. The right strategy depends on which component is your bottleneck.

---

## 2. Data Parallelism: Replicate Model, Split Data

Data parallelism (DP) is the simplest scaling strategy. Replicate the entire model on $N_d$ GPUs, feed each replica a different micro-batch, run forward and backward passes independently, then synchronize gradients via **all-reduce** before the optimizer step.

The global batch size becomes:

$$\text{gbs} = \text{mbs} \times \text{grad\_acc} \times N_d$$

where $\text{mbs}$ is the micro-batch size per GPU and $\text{grad\_acc}$ is the number of gradient accumulation steps.

### Overlapping Communication with Computation

A naive implementation runs all-reduce *after* the backward pass completes — GPUs sit idle during synchronization. The critical optimization: **attach all-reduce hooks to each parameter** so gradient synchronization begins as soon as each layer's gradients are ready, overlapping with the backward pass of earlier layers. PyTorch DDP does this automatically.

A second optimization: **bucket gradients** into large contiguous buffers before communication. GPU communication primitives are far more efficient on large tensors than many small ones. PyTorch DDP groups gradients into configurable bucket sizes (default 25 MB).

A third optimization: when using gradient accumulation, **disable synchronization** on intermediate micro-batches via `model.no_sync()`, performing a single all-reduce after the final accumulation step.

### Scaling Limits

The Ultra-Scale Playbook ([[ultra-scale-playbook|paper]]) benchmarks show DP throughput degrading significantly at scale: **-6.3% at 32 GPUs, -15% at 128 GPUs, and -40.6% at 256 GPUs** for a 1B model. The root cause is that all-reduce volume scales with model size (every gradient must be communicated), and at 512+ GPUs ring latency — the time for a signal to propagate once around the ring — becomes the bottleneck, preventing full overlap of communication with computation.

**Data parallelism provides no memory savings per GPU** — each replica holds a full copy of parameters, gradients, and optimizer states. It only helps if a single forward pass already fits on one GPU. For 70B+ models, it cannot work alone.

---

## 3. ZeRO: Eliminating Memory Redundancy

The ZeRO paper ([[zero|paper]]) observed that vanilla DP stores redundant copies of optimizer states, gradients, and parameters on every GPU. The solution: **shard these across DP ranks** so each GPU stores only $1/N_d$ of each component.

ZeRO defines three progressive stages:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">ZeRO Stages: Per-GPU Memory for a Model with N Parameters (Mixed Precision, Adam)</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Stage</th>
<th style="text-align:left; padding:8px;">What's Sharded</th>
<th style="text-align:right; padding:8px;">Per-GPU Memory</th>
<th style="text-align:right; padding:8px;">70B @ 8 GPUs</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#888;">Baseline DP</td>
<td style="padding:8px;">Nothing</td>
<td style="text-align:right; padding:8px;">$2\Psi + 2\Psi + 12\Psi = 16\Psi$</td>
<td style="text-align:right; padding:8px;">1,120 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">ZeRO-1</td>
<td style="padding:8px;">Optimizer states</td>
<td style="text-align:right; padding:8px;">$2\Psi + 2\Psi + \frac{12\Psi}{N_d}$</td>
<td style="text-align:right; padding:8px;">385 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">ZeRO-2</td>
<td style="padding:8px;">Optimizer + gradients</td>
<td style="text-align:right; padding:8px;">$2\Psi + \frac{2\Psi + 12\Psi}{N_d}$</td>
<td style="text-align:right; padding:8px;">245 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">ZeRO-3 / FSDP</td>
<td style="padding:8px;">Optimizer + gradients + parameters</td>
<td style="text-align:right; padding:8px;">$\frac{2\Psi + 2\Psi + 12\Psi}{N_d}$</td>
<td style="text-align:right; padding:8px;">140 GB</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
ZeRO-3 with 8 GPUs reduces 1,120 GB to 140 GB — fitting within two H100s per model shard. DeepSeek V3 used ZeRO-1 combined with pipeline parallelism. Note: ZeRO-2 has no real overhead vs ZeRO-1 and is usually the better default.
</div>
</div>

### How ZeRO-3 / FSDP Works

In ZeRO-3 (PyTorch calls this FSDP — Fully Sharded Data Parallel), each GPU stores only $1/N_d$ of the parameters. Before each layer's forward pass, the GPU **all-gathers** the full parameters from all ranks, computes, then **discards** the non-local shards. The backward pass repeats the all-gather, computes gradients, then performs a **reduce-scatter** to distribute gradient shards.

The communication cost: $3\Psi$ total per step (two all-gathers + one reduce-scatter), compared to $2\Psi$ for vanilla DP (one all-reduce $\approx$ one all-gather + one reduce-scatter). The 50% communication overhead is the price of memory savings.

The critical optimization is **prefetching**: while computing layer $n$'s forward pass, asynchronously all-gather layer $n+1$'s parameters. When the compute-to-communication ratio is favorable (large micro-batch sizes and sequence lengths), prefetching hides most of the overhead.

### FSDP in Practice

PyTorch FSDP wraps each transformer layer as an "FSDP unit" — the granularity at which parameters are gathered and released. The key configuration choices:

- **Sharding strategy:** `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (vanilla DP)
- **Forward prefetch:** Gather next layer's params during current layer's forward
- **Backward prefetch:** `BACKWARD_PRE` (gather during backward) vs `BACKWARD_POST` (gather after)
- **Mixed precision:** Compute in BF16, communicate in BF16, maintain FP32 master weights

See [[zero-fsdp-deep-dive|excerpt]] for the communication patterns in detail.

---

## 4. Tensor Parallelism: Splitting Layers Across GPUs

Tensor parallelism (TP), introduced by Megatron-LM ([[megatron-lm|paper]]), takes a fundamentally different approach: instead of replicating layers and sharding state, **split individual weight matrices across GPUs** so each GPU computes a portion of each layer.

The mathematical foundation is the distributive property of matrix multiplication:

$$A \cdot B = A \cdot [B_1 \mid B_2] = [AB_1 \mid AB_2] \quad \text{(column-parallel)}$$

$$A \cdot B = [A_1 \mid A_2] \cdot \begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = A_1 B_1 + A_2 B_2 \quad \text{(row-parallel)}$$

### Tensor Parallelism in a Transformer Block

The Megatron-LM insight: pair **column-parallel** with **row-parallel** within each sub-layer to minimize communication. For the MLP:

1. Split the first linear (up-projection) **column-wise**: each GPU gets $W_1[:, \text{shard}]$, computes $\text{GeLU}(XW_1^{\text{shard}})$ independently
2. Split the second linear (down-projection) **row-wise**: each GPU multiplies its partial result by $W_2[\text{shard}, :]$
3. **All-reduce** the row-parallel outputs to get the correct sum

This requires only **one all-reduce per sub-layer** in the forward pass (two per transformer block: one for attention, one for MLP). No broadcast is needed because inputs are already synchronized.

For multi-head attention, column parallelism has a natural interpretation: **each GPU computes a subset of attention heads**. The Q, K, V projections are column-split, attention is computed independently per head, and the output projection is row-split. The TP degree must not exceed the number of KV heads — for Llama 3 8B with 8 KV heads, TP must be $\leq 8$.

### Communication Cost and Scaling

TP's Achilles heel: the all-reduce sits **on the critical path** of computation. Unlike DP's gradient all-reduce (which overlaps with backward), TP's all-reduce must complete before the next layer can begin. This makes TP highly sensitive to interconnect bandwidth.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">TP Scaling: Throughput vs Memory Tradeoff (3B Model, Ultra-Scale Playbook Benchmarks)</div>
<div style="display:flex; gap:24px; flex-wrap:wrap; justify-content:center;">
<div style="text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:12px; margin-bottom:8px;">Throughput Degradation</div>
<div style="display:flex; flex-direction:column; gap:4px; align-items:flex-start;">
<div style="display:flex; align-items:center; gap:8px;"><div style="background:#4ecdc4; height:16px; border-radius:3px; width:200px;"></div><span style="color:#e0e0e0; font-size:11px;">TP=2: baseline</span></div>
<div style="display:flex; align-items:center; gap:8px;"><div style="background:#4ecdc4; height:16px; border-radius:3px; width:176px;"></div><span style="color:#e0e0e0; font-size:11px;">TP=4: -12.2%</span></div>
<div style="display:flex; align-items:center; gap:8px;"><div style="background:#ffd93d; height:16px; border-radius:3px; width:156px;"></div><span style="color:#e0e0e0; font-size:11px;">TP=8: -10.8%</span></div>
<div style="display:flex; align-items:center; gap:8px;"><div style="background:#e94560; height:16px; border-radius:3px; width:114px;"></div><span style="color:#e0e0e0; font-size:11px;">TP=16: -42.7% (cross-node)</span></div>
<div style="display:flex; align-items:center; gap:8px;"><div style="background:#e94560; height:16px; border-radius:3px; width:68px;"></div><span style="color:#e0e0e0; font-size:11px;">TP=32: -65.6%</span></div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
The cliff at TP=16 is the node boundary: intra-node NVLink (900 GB/s on H100) vs inter-node InfiniBand (~400 GB/s). Rule of thumb: keep TP within a single node (TP &le; 8).
</div>
</div>

TP shards activations along the hidden dimension, reducing activation memory per GPU. But LayerNorm and dropout still require the full hidden dimension. **Sequence parallelism** (SP) — not to be confused with context parallelism — addresses this by splitting the non-TP operations along the sequence dimension. SP replaces TP's all-reduces with all-gather + reduce-scatter pairs, maintaining the same total communication volume while sharding activation memory more completely.

See the [parallelism strategy comparison](figures/parallelism-comparison.html) for an interactive visualization of how DP, TP, PP, and ZeRO divide the model.

---

## 5. Pipeline Parallelism: Splitting Layers Across Nodes

Pipeline parallelism (PP) partitions the model's layers across GPUs in depth. GPU 1 holds layers 1-10, GPU 2 holds layers 11-20, and so on. Activations flow sequentially from one "stage" to the next.

PP's communication advantage: it sends **activation tensors point-to-point** between adjacent stages — far less data than TP's all-reduce within each layer, and tolerant of lower-bandwidth inter-node connections.

PP's fundamental problem: **the pipeline bubble**. While GPU 2 computes forward on micro-batch 1, GPU 1 sits idle. The bubble fraction for naive PP:

$$r_{\text{bubble}} = \frac{(p - 1) \cdot (t_f + t_b)}{t_f + t_b} = p - 1$$

where $p$ is the number of pipeline stages. With $p = 8$, the GPU is idle **87.5%** of the time.

### Micro-batching: Shrinking the Bubble

The solution: split the global batch into $m$ micro-batches. Each micro-batch flows through the pipeline independently, filling the bubble:

$$r_{\text{bubble}} = \frac{p - 1}{m}$$

With $p = 8$ and $m = 32$: bubble is only **21.9%**. Three scheduling strategies control how micro-batches flow:

**All-Forward-All-Backward (AFAB):** Run all forward passes, then all backward passes. Simple to implement but stores activations for all $m$ micro-batches simultaneously — memory explosion.

**1F1B (One-Forward-One-Backward):** Begin backward passes as soon as the last stage completes its first forward pass. Alternates forward and backward in steady state. Activation memory capped at $p$ micro-batches (not $m$), a major improvement. Same bubble size as AFAB.

**Interleaved stages:** Assign non-contiguous layers to each GPU (e.g., GPU 1 gets layers 1-4 and 17-20). Each micro-batch passes through each GPU $v$ times, reducing the bubble by factor $v$:

$$r_{\text{bubble}} = \frac{p - 1}{v \cdot m}$$

The cost: $v\times$ more point-to-point communications. Llama 3.1 uses interleaved 1F1B with tunable depth-first vs breadth-first scheduling.

### Zero Bubble and DualPipe

DeepSeek V3 pushed further with **DualPipe**, building on the ZeroBubble observation: the backward pass decomposes into **B** (backward for inputs, needed for the next layer) and **W** (backward for weights, only needed before optimizer step). Since W can be scheduled flexibly, it fills pipeline bubbles. DualPipe runs two streams from both ends of the pipeline, interleaving them to approach **zero bubble** overhead.

See [[pipeline-schedules|excerpt]] for detailed schedule diagrams and the bubble calculus.

See the [pipeline bubble visualization](figures/pipeline-bubble.html) for an interactive walkthrough of AFAB, 1F1B, and interleaved schedules.

---

## 6. Expert Parallelism: Distributing MoE Experts

Mixture-of-Experts models (covered in [[ch-11]]) add a new parallelism dimension. Since each expert's FFN is independent, expert parallelism (EP) places each expert on a different GPU. The router determines which tokens go to which expert, and **all-to-all** communication shuffles tokens to the correct GPU.

EP is conceptually simpler than TP — no matrix splitting, just routing hidden states to the right device. But MoE training is fundamentally a **distributed systems problem** because of load balancing:

**Token imbalance:** If the router sends most tokens to a few experts, those GPUs become bottlenecks while others idle. Auxiliary load-balancing losses and capacity factors partially address this but introduce their own training dynamics.

**Node-constrained routing:** DeepSeek V3 enforces that each token is sent to experts on at most $M = 4$ nodes, limiting all-to-all communication to a subset of the cluster. This is a model architecture decision driven by distributed systems constraints.

**Communication pattern:** All-to-all is the most demanding collective — every GPU sends data to every other GPU. Unlike all-reduce (which has efficient ring implementations), all-to-all saturates bisection bandwidth. At 256 experts across 32 nodes, the communication graph becomes the training bottleneck.

EP interacts with DP: since EP only affects MoE layers, the attention blocks and non-expert modules still need data parallelism to avoid redundant computation. In practice, EP replaces a portion of the DP group — the total GPU count is divided as $N = N_{\text{DP}} \times N_{\text{EP}} \times N_{\text{TP}} \times N_{\text{PP}}$.

---

## 7. 5D Parallelism: Combining Everything

Modern training runs combine all five dimensions. The total GPU count decomposes as:

$$N_{\text{GPUs}} = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{CP}} \times N_{\text{EP}}$$

The placement strategy follows a clear hierarchy dictated by communication bandwidth requirements:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Parallelism Strategy Selection Guide</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:12px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:6px;">Strategy</th>
<th style="text-align:left; padding:6px;">Splits</th>
<th style="text-align:left; padding:6px;">Communication</th>
<th style="text-align:left; padding:6px;">Best Placement</th>
<th style="text-align:left; padding:6px;">When to Use</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:6px; color:#e94560; font-weight:bold;">TP (+SP)</td>
<td style="padding:6px;">Weights + activations along hidden dim</td>
<td style="padding:6px;">All-reduce on critical path (2 per block)</td>
<td style="padding:6px; color:#ffd93d;">Intra-node (NVLink)</td>
<td style="padding:6px;">Always, TP &le; 8</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:6px; color:#4ecdc4; font-weight:bold;">PP</td>
<td style="padding:6px;">Layers across stages</td>
<td style="padding:6px;">P2P activations between stages</td>
<td style="padding:6px; color:#4ecdc4;">Inter-node OK</td>
<td style="padding:6px;">Model too large for 1 node</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:6px; color:#ffd93d; font-weight:bold;">DP / ZeRO</td>
<td style="padding:6px;">Data (+ optionally params/grads)</td>
<td style="padding:6px;">All-reduce/reduce-scatter, overlappable</td>
<td style="padding:6px; color:#4ecdc4;">Inter-node OK</td>
<td style="padding:6px;">Scale throughput</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:6px; color:#ff6b6b; font-weight:bold;">CP</td>
<td style="padding:6px;">Activations along sequence dim</td>
<td style="padding:6px;">Ring attention for KV exchange</td>
<td style="padding:6px; color:#4ecdc4;">Inter-node OK</td>
<td style="padding:6px;">Sequences &gt; 128K</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:6px; color:#c084fc; font-weight:bold;">EP</td>
<td style="padding:6px;">Expert FFNs across GPUs</td>
<td style="padding:6px;">All-to-all token routing</td>
<td style="padding:6px; color:#ffd93d;">Prefer intra-node</td>
<td style="padding:6px;">MoE models</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Typical large-model config: TP=8 (within node) x PP=8 (across nodes) x DP=16 (remaining GPUs) x ZeRO-1 = 1024 GPUs. DeepSeek V3 used TP intra-node, PP inter-node, DP+ZeRO-1 across remaining, EP across a subset.
</div>
</div>

**Key interaction rules from the Ultra-Scale Playbook:**

1. **TP + PP are complementary:** TP handles intra-node parallelism (high bandwidth), PP handles inter-node (low bandwidth). Combining them is standard for 70B+ models.
2. **ZeRO-3 and PP solve the same problem differently:** Both distribute model state across GPUs. ZeRO-3 communicates weights; PP communicates activations. Combining them requires very large batch sizes to amortize both costs. In practice, use ZeRO-1/2 with PP instead.
3. **ZeRO-1/2 + PP is straightforward:** DeepSeek V3 used exactly this combination. ZeRO-1 shards optimizer states, PP shards layers — no interference.
4. **EP partially replaces DP:** EP groups share experts but each GPU sees different data for non-expert layers, so EP naturally combines with DP on the non-expert portions.

The Ultra-Scale Playbook's methodology for finding optimal configurations: benchmark thousands of combinations of (TP, PP, DP, mbs, grad_acc) for a given model and cluster, measuring throughput and memory. There is no closed-form solution — the interaction between parallelism dimensions, hardware topology, and model architecture is too complex for analytical optimization.

---

## Core Insights from the Literature

### Insight 1: Memory redundancy in data parallelism is the easiest win
**Paper:** Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" ([[zero|paper]])

Before ZeRO, training a 13B model required tensor or pipeline parallelism — complex, model-specific engineering. ZeRO showed that simply sharding optimizer states across DP ranks (ZeRO-1) reduces the dominant memory component by $N_d \times$ with **no communication overhead** beyond vanilla DP. ZeRO-2 adds gradient sharding for free (reduce-scatter replaces all-reduce with equal communication volume). This made large model training accessible to anyone who could run data parallelism. **Guideline:** Always use at least ZeRO-1. There is no reason not to — it's strictly better than vanilla DP. ZeRO-2 is also usually free. ZeRO-3 trades 50% more communication for full parameter sharding; use it when the model doesn't fit otherwise.

### Insight 2: Tensor parallelism works because matrix multiply is distributive
**Paper:** Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" ([[megatron-lm|paper]])

Megatron-LM's contribution was showing that tensor parallelism requires only **two all-reduce operations per transformer block** — not per parameter, not per head, but per block. The column-parallel + row-parallel pairing within each sub-layer (MLP and attention) cancels intermediate communication, making TP practical with "just a few extra lines of PyTorch." The 76% scaling efficiency on 512 GPUs (2019) proved that custom compilers were unnecessary. **Guideline:** Keep TP within a single node (TP $\leq$ 8 on standard clusters). The all-reduce is on the critical path and cannot be overlapped, so it is extremely sensitive to interconnect bandwidth. TP=8 on NVLink loses ~11% throughput; TP=16 across nodes loses ~43%.

### Insight 3: Pipeline parallelism trades bubble overhead for bandwidth tolerance
**Paper/Source:** Tazi et al., "The Ultra-Scale Playbook" ([[ultra-scale-playbook|paper]])

PP's communication is minimal (point-to-point activations between stages), making it the only parallelism strategy that scales gracefully across nodes with slow interconnects. The playbook benchmarks show PP scaling from 1 to 2 nodes loses only ~14% throughput, versus ~43% for TP in the same scenario. The cost is the pipeline bubble, but modern schedules (1F1B interleaved, ZeroBubble, DualPipe) reduce this to near-zero at the expense of implementation complexity. **Guideline:** Use PP to span nodes; use TP within nodes. The bubble fraction $\frac{p-1}{v \cdot m}$ is your knob — increase micro-batches $m$ or interleave stages $v$ to shrink it. But $m$ is bounded by target global batch size and $v$ adds communication.

### Insight 4: MoE training is a distributed systems problem, not just a model architecture choice
**Paper/Source:** Ultra-Scale Playbook ([[ultra-scale-playbook|paper]]), DeepSeek V3 Technical Report

Expert parallelism's all-to-all communication pattern is qualitatively different from the collectives used by other strategies. It requires every GPU to exchange data with every other GPU, saturating network bisection bandwidth. DeepSeek V3's architectural constraint — routing each token to at most 4 nodes — is a distributed systems decision that directly constrains the model's expressiveness. This is the clearest example of distributed training shaping model architecture, not just infrastructure. **Guideline:** When designing MoE models, co-design the routing constraints with the cluster topology. The number of experts, tokens-per-expert, and node-locality constraints are distributed systems parameters as much as model hyperparameters.

---

## Key Takeaways

1. **Four things consume GPU memory during training: parameters, gradients, optimizer states, and activations.** Optimizer states dominate for small batches; activations dominate at scale. The right parallelism strategy depends on which is your bottleneck.

2. **ZeRO/FSDP eliminates redundancy for free.** ZeRO-1 (optimizer sharding) and ZeRO-2 (+ gradient sharding) add no communication overhead over vanilla DP. ZeRO-3 adds 50% more communication but enables training arbitrarily large models via data parallelism alone.

3. **Tensor parallelism shards weights AND activations but puts communication on the critical path.** Keep TP within a single node (TP $\leq$ 8). The column-parallel + row-parallel pattern requires only two all-reduces per transformer block.

4. **Pipeline parallelism tolerates slow interconnects but wastes compute in bubbles.** The bubble fraction $\frac{p-1}{v \cdot m}$ is controlled by micro-batch count and interleaving degree. Modern schedules (DualPipe) approach zero bubble by decomposing backward into B and W phases.

5. **The 5 parallelism dimensions target different axes:** DP splits data, TP splits hidden dimension, PP splits depth, CP splits sequence, EP splits experts. They compose multiplicatively: $N = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{CP}} \times N_{\text{EP}}$.

6. **Placement follows bandwidth hierarchy:** TP on NVLink (intra-node), PP on InfiniBand (inter-node), DP/ZeRO across remaining GPUs. Violating this hierarchy — running TP across nodes, for example — causes catastrophic throughput drops (40%+).

7. **There is no analytical solution for optimal parallelism configuration.** The Ultra-Scale Playbook benchmarked 4,000+ configurations. The interaction between model architecture, hardware topology, and parallelism strategy is too complex for closed-form optimization. Benchmark systematically.

---

## References

- [[zero|Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019) (paper)]] — ZeRO-1/2/3 memory optimization
- [[megatron-lm|Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019) (paper)]] — tensor parallelism for transformers
- [[ultra-scale-playbook|Tazi et al., "The Ultra-Scale Playbook: Training LLMs on GPU Clusters" (2025) (paper)]] — comprehensive 5D parallelism benchmarks
- [[hf-nanotron|Nanotron Research, Hugging Face distributed training framework (blog)]] — reference implementation for playbook experiments
- DeepSeek V3 Technical Report (2024) — DualPipe zero-bubble pipeline parallelism, node-constrained expert routing
- [[gpipe|Huang et al., "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism" (2019) (paper)]] — original pipeline parallelism with micro-batching
- Qi et al., "Zero Bubble Pipeline Parallelism" (2024) — B/W decomposition for zero-bubble schedules
- PyTorch FSDP documentation — production ZeRO-3 implementation
