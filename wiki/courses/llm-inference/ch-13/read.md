<!-- chapter: ch-13
     phase: kernels-runtime
     title: Parallel Inference — Tensor / Pipeline / Expert
     sources: [[tensor-parallel-inference]], [[pipeline-parallel-inference]], [[expert-parallel-inference]]
     figures: figures/parallel-inference-comparison.html
-->

# Chapter 13 — Parallel Inference: Tensor / Pipeline / Expert Parallel

> **Core insight.** When one model replica no longer fits on one GPU — or one GPU is too slow to hit your TPOT target — you split *along three orthogonal axes*: tensor-parallel splits each layer's matmuls (high-bandwidth, intra-node), pipeline-parallel splits the layer stack (low-bandwidth, inter-node), expert-parallel splits MoE experts (all-to-all, token-routed). Frontier MoE inference uses **all three together**; the topology choice is what makes a 671B-parameter DeepSeek-V3 deployment economic.
>
> **Guideline.** Inside an NVLink/NVSwitch island: TP up to `tp_size = 8`. Across nodes for pure dense models: PP, not TP (no IB all-reduce on the critical path). For MoE: EP across the expert dimension + TP inside dense layers + optional PP across nodes. Never run TP=8 across InfiniBand unless you have measured the all-reduce cost is < 20% of decode latency.

---

## Why this chapter exists

The previous chapter ([[ch-12]]) closed out single-GPU serving — CUDA graphs, kernel fusion, the launch-overhead war. That entire stack assumes the model fits on one device. Once it doesn't, the latency story changes shape: every layer now has a *collective* on its critical path, and serving frameworks that ignored the network topology suddenly leave 2× speedup on the table.

This chapter is the bridge between *single-GPU* inference math ([[attention-is-all-you-need]] from ch-02, [[kv-cache-memory-formula]] from ch-03) and *production frontier-model deployment* ([[ch-20]] DeepSeek-V3 / Llama-3-405B / Qwen-3-235B-MoE serving reports). Three things you need to walk away with:

1. The exact comm pattern each parallelism induces — TP needs **all-reduce per layer**, PP needs **point-to-point activation passing**, EP needs **all-to-all per MoE layer** — and where each one lives on the bandwidth budget.
2. The per-axis overhead math: TP gives you a `2(N-1)/N · message_size` all-reduce cost per matmul; PP gives you a bubble of `(stages - 1)/microbatches`; EP gives you two all-to-alls per MoE layer.
3. Why **TP inside a node, PP across nodes, EP for MoE** is the production default and not just a preference.

All three patterns trace back to Megatron-LM ([[tensor-parallel-inference]]) and GPipe ([[pipeline-parallel-inference]]); EP traces to GShard ([[expert-parallel-inference]]). Modern serving stacks (vLLM, TensorRT-LLM, SGLang) expose them as `tensor_parallel_size`, `pipeline_parallel_size`, and `expert_parallel_size` knobs respectively — but tuning them blindly is how production deployments lose 30% throughput.

---

## 1. Tensor parallelism — split *within* each layer

TP is the Megatron-LM ([Shoeybi et al. 2019, arXiv:1909.08053](https://arxiv.org/abs/1909.08053)) trick, adapted to inference. The construction relies on a single algebraic observation: in a transformer block, the FFN and attention projections are large GEMMs that can be partitioned into independent slices, with collectives required only at the *boundaries*.

### 1.1 The MLP partition (column-parallel + row-parallel)

A transformer MLP is `y = W₂ · σ(W₁ · x)`. For `tp_size = N`, partition `W₁` **by columns** (so each rank holds `d_ff / N` output columns) and `W₂` **by rows** (so each rank holds `d_ff / N` input rows):

```
Rank i computes:   h_i = σ(W₁_i · x)          # column-parallel — no sync needed
                   y_i = W₂_i · h_i           # row-parallel partial sum
                   y   = all_reduce(y_i)      # SUM across ranks → final y
```

The activation `σ(·)` is applied *locally* on each rank because it's elementwise. Only one all-reduce is needed per MLP, not two. This was Megatron's headline contribution — naive partitioning would have required two collectives.

### 1.2 The attention partition (per-head sharding)

For multi-head attention with `H` heads, partition Q/K/V projections **column-wise** (so each rank computes `H/N` heads' worth of Q/K/V) and the output projection **row-wise**:

```
Rank i:   Q_i, K_i, V_i = W_qkv_i · x         # column-parallel, owns H/N heads
          attn_i        = softmax(Q_i K_iᵀ / √d_h) V_i      # local per-head attention
          o_i           = W_o_i · attn_i                     # row-parallel partial sum
          o             = all_reduce(o_i)                    # SUM across ranks → final o
```

Same pattern: one all-reduce per attention block. **GQA implication** ([[grouped-query-attention]] from ch-02): with `kv_heads < q_heads`, the KV cache shards stay aligned with the head partition as long as `kv_heads` is divisible by `tp_size`. Llama-3-70B has `kv_heads=8`; you can do TP={1,2,4,8} cleanly but TP=16 forces KV-head replication. Llama-3-8B has `kv_heads=8` and is typically run TP=1 or TP=2.

### 1.3 The all-reduce cost — what TP actually costs you per layer

Per transformer block: **2 all-reduces** (one MLP, one attention). Each carries a tensor of shape `[batch_tokens, d_model]`. Using bf16 (2 bytes), the per-block bytes-moved is:

```
bytes_per_block = 2 · 2 · batch_tokens · d_model
                = 4 · batch_tokens · d_model
```

Ring all-reduce moves `2(N-1)/N · message_size` bytes per rank. For a 70B model (`d_model=8192`, 80 blocks) at decode (`batch_tokens=batch`):

```
per_step_bytes = 80 · 4 · batch · 8192 · 2(N-1)/N
              ≈ 5.2 MB · batch · (N-1)/N        # for N=8 → 4.5 MB · batch
```

On NVLink 4.0 (900 GB/s peer-to-peer, ~450 GB/s effective ring bandwidth), TP=8 adds ~10 µs per decode step at batch=32 — negligible vs the ~30-50 ms full decode step. On InfiniBand (200 Gb/s ≈ 25 GB/s), the same cost is ~180 µs and accumulates fast.

This is the entire reason for **"TP within a node, PP across nodes"**: NVLink can absorb the all-reduce; IB cannot.

### 1.4 Where TP costs accumulate

- **KV cache also shards** with the head partition; each rank holds `kv_heads / tp_size` heads' worth. PagedAttention block tables ([[pagedattention]] from ch-06) are per-rank but indexed identically.
- **Vocabulary parallelism** for the LM head: vocab is split column-wise (each rank holds `vocab_size / tp_size` rows). The final all-reduce is over the logits before sampling — or you keep logits sharded and do a tree-reduced sampler.
- **Embedding parallelism**: the embedding table is similarly split; a column-parallel all-gather feeds the residual stream.

### 1.5 vLLM / TensorRT-LLM / SGLang knobs

```bash
# vLLM
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.92

# TensorRT-LLM (at engine-build time)
trtllm-build --checkpoint_dir llama-70b-fp16 \
    --output_dir engine_tp8 \
    --tp_size 8

# SGLang
python -m sglang.launch_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --tp 8
```

**Production rule of thumb** (validated against vLLM benchmarks and TRT-LLM docs):

| Model | Recommended TP on 8×H100 |
|-------|-------------------------|
| Llama-3-8B   | 1 (or 2 for low-latency single-stream) |
| Llama-3-70B  | 4 or 8 (8 for prefill-heavy, 4 for higher concurrency) |
| Qwen-2.5-72B | 4 or 8 |
| Mixtral-8x7B (45B) | 2 or 4 + EP |
| DeepSeek-V3 (671B MoE) | TP=8 inside node + EP across experts |
| Llama-3-405B (dense) | TP=8 × PP=2 across two nodes |

---

## 2. Pipeline parallelism — split *the layer stack*

PP is the GPipe ([Huang et al. 2018, arXiv:1811.06965](https://arxiv.org/abs/1811.06965)) trick: assign layers 1..L/S to GPU 1, L/S+1..2L/S to GPU 2, etc. The big win over TP: per-token cross-GPU traffic is **one activation transfer per stage boundary** (size `[batch_tokens, d_model]`), not an all-reduce per layer.

### 2.1 The basic PP loop

```
                  GPU 0           GPU 1           GPU 2           GPU 3
microbatch m₀:    layers 0..19    layers 20..39   layers 40..59   layers 60..79
microbatch m₁:                    layers 0..19    layers 20..39   layers 40..59
microbatch m₂:                                    layers 0..19    layers 20..39
microbatch m₃:                                                    layers 0..19
```

At steady state, all 4 GPUs are doing useful work simultaneously. Cross-GPU traffic: one `[batch_tokens, d_model]` send-recv per stage transition, per microbatch.

For `d_model=8192`, batch=32, bf16: each activation transfer is `32 · 8192 · 2 = 512 KB`. On IB at 25 GB/s, ~20 µs per boundary. With 3 boundaries (4 stages), that's 60 µs of cross-node activation traffic per microbatch — vs ~180 µs of all-reduce per layer if you had used TP=8 instead. PP wins decisively across nodes.

### 2.2 The bubble cost

The killer problem in PP: **pipeline bubbles**. At fill (the first few microbatches enter) and drain (the last few exit), not all stages are busy. For `S` stages and `M` microbatches:

```
bubble_fraction = (S - 1) / (S + M - 1)
```

For `S=4, M=4` → bubble = 50%. For `S=4, M=32` → bubble = 9%. The lesson: PP needs *many in-flight microbatches* to hide the bubble. This is fine for training (gradient accumulation gives you natural microbatches) but **harder for inference** — a single decode step has *one* batch, not a stream.

### 2.3 Continuous batching saves PP at inference

The fix is that serving stacks ([[continuous-batching]] from ch-04) already supply many concurrent requests. Each in-flight request becomes a microbatch from PP's perspective: at each decode step, the engine schedules `M` decode tokens across the stages, and they pipeline through.

vLLM's V1 engine ([[vllm-scheduler]] from ch-16) handles this by submitting decode tokens in PP-aware microbatch chunks; the scheduler tracks which stage each microbatch is on and refills the fill phase from the WAITING queue.

The PP-at-decode reality:

```
bubble at low QPS  →  bad (often you have fewer in-flight requests than stages)
bubble at high QPS →  good (continuous batching feeds the pipeline)
```

This is why PP at inference works mostly for high-QPS server-side deployments, not for low-latency single-stream chat.

### 2.4 Interleaved pipeline schedules

Megatron-LM ([Narayanan et al. 2021, arXiv:2104.04473](https://arxiv.org/abs/2104.04473)) introduced **interleaved 1F1B**: each stage owns *non-contiguous* layer groups (e.g. stage 0 = layers {0..9, 40..49}, stage 1 = layers {10..19, 50..59}), reducing the bubble by ~`v` (the number of virtual stages per physical stage). The cost is more activation transfers; for inference, the benefit usually doesn't justify the extra cross-node traffic.

### 2.5 PP knobs

```bash
# vLLM — typically combined with TP
vllm serve meta-llama/Llama-3-405B \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.92
```

8×H100 × 2 nodes → 16 GPUs total, TP=8 inside each node (NVLink), PP=2 across nodes (IB). This is the canonical Llama-3-405B layout.

---

## 3. Expert parallelism — split *MoE experts*

EP is the GShard ([Lepikhin et al. 2020, arXiv:2006.16668](https://arxiv.org/abs/2006.16668)) idea: a sparse MoE layer has E experts; place experts 1..E/N on rank 1, experts E/N+1..2E/N on rank 2, etc. At inference, each token's router picks top-k experts (typically k=2 in Mixtral, k=8 in DeepSeek-V3), and tokens are *routed* to the GPUs owning those experts.

### 3.1 The all-to-all communication pattern

For each MoE layer:

```
1. Local: compute router scores g_i = softmax(W_g · x_i); pick top-k experts per token.
2. Dispatch: all-to-all exchange — tokens flow to the GPUs owning their chosen experts.
3. Compute: each rank runs grouped GEMM over the batch of tokens routed to its local experts.
4. Combine: all-to-all exchange in reverse — token outputs return to their original rank.
5. Local: weighted sum of expert outputs (router weights are the combining weights).
```

Two all-to-alls per MoE layer is the dominant cost. For DeepSeek-V3 (61 layers, MoE in all but the first 3) at decode with top-8 routing, that's ~58 × 2 = 116 all-to-alls per decode step.

### 3.2 Why all-to-all is harder than all-reduce

All-reduce on `N` GPUs moves `2(N-1)/N · message_size` bytes per rank — predictable, ring-friendly.

All-to-all on `N` GPUs moves `(N-1)/N · message_size` bytes per rank — but the *messages are unbalanced*. If token routing is skewed (one expert gets 30% of tokens, another gets 5%), some GPUs receive much more data than others, the slowest link bottlenecks the collective. **Load balancing** at the router (during pretraining via aux losses; at inference via capacity factors) is the way EP doesn't fall apart.

DeepSpeed-MoE / Megatron-Core mitigate this with:
- **Capacity factor** `c`: cap how many tokens each expert receives at `c · batch_size · top_k / E`. Excess tokens are dropped or rerouted. Typical: `c = 1.25`.
- **Expert tensor parallelism (ETP)**: each expert is itself TP-sharded across a small subgroup. Reduces per-expert memory and gives a second axis for load balancing.
- **Redundant experts**: place "hot" experts on multiple GPUs.

### 3.3 The EP × TP composition

For DeepSeek-V3 (256 routed experts + 1 shared, top-8 routing, 671B params, 37B active per token):

```
World size: 64 GPUs (8 nodes × 8 H100)
Topology:   TP=8 inside each node (for dense MHA/MLA + shared expert)
            EP=64 across all GPUs (each GPU owns 4 routed experts)
            PP=1 (model fits memory-wise)
```

The router sends each token to top-8 experts → 64 GPUs participate in the all-to-all → each GPU receives ~`batch · 8 / 64 = batch/8` tokens per layer.

Why not TP=64? Because per-layer all-reduce on `tokens · d_model` traffic would saturate IB. Why not just EP=64? Because the dense MHA/MLA path still wants TP for its big QKV matmuls. The mix is forced by the architecture.

### 3.4 Mixtral as the simpler case

Mixtral 8x7B (8 experts, top-2 routing, 45B params total, ~13B active):

```
World size: 2 H100s
Topology:   TP=2 inside the node
            EP=2 across the same 2 GPUs (each owns 4 experts)
```

Mixtral 8x22B (141B params, ~39B active):

```
World size: 4 H100s
Topology:   TP=4 inside the node (NVLink)
            EP=4 (each GPU owns 2 experts)
```

vLLM exposes this as `--tensor-parallel-size 4 --enable-expert-parallel`. The all-to-all is local to the 4-GPU NVLink island; no IB traffic, no load-balancing crisis. This is why MoE *deployment* feels less scary than MoE *training*.

### 3.5 Grouped GEMM — the kernel that makes EP fast

Naive EP implementation: launch one GEMM per expert. Each GEMM is small (a few hundred tokens × FFN dims) → atrocious tensor-core utilization. The fix: **grouped GEMM** (cuBLAS `cublasLtMatmul` with grouped variant, or TRT-LLM/CUTLASS's `cutlass::GroupedGemm`):

```
inputs:   list of (M_e, K) matrices for e = 1..E_local
outputs:  list of (M_e, N) matrices
kernel:   one launch, internally pipelined across experts, single tensor-core wave
```

`E_local` typically 4-8 per GPU; M_e ranges 50-500 tokens. Grouped GEMM brings utilization from ~10% to ~60% — the difference between "MoE is slow" and "MoE serves at competitive cost per token".

---

## 4. Combined TP × PP × EP for frontier MoE

Real deployments use all three. The canonical mappings:

| Model | World size | TP | PP | EP | Notes |
|-------|-----------|----|----|----|-------|
| Llama-3-70B (dense) | 8×H100 (1 node) | 8 | 1 | — | NVLink-only |
| Llama-3-405B (dense) | 16×H100 (2 nodes) | 8 | 2 | — | TP intra-node, PP inter-node |
| Mixtral-8x7B | 2×H100 | 2 | 1 | 2 | tiny MoE, fits 1 node |
| Mixtral-8x22B | 4×H100 (1 node) | 4 | 1 | 4 | NVLink absorbs both AR + A2A |
| Qwen-3-235B-MoE | 16×H100 (2 nodes) | 8 | 2 | 16 | EP global, TP intra-node, PP inter-node |
| DeepSeek-V3 (671B MoE) | 64×H100 (8 nodes) | 8 | 1 | 64 | dense path TP=8, routed experts EP=64 |
| GPT-OSS-120B (MoE) | 8×H100 | 8 | 1 | 8 | single node, NVLink |

The pattern: **PP is the *escape hatch* for "doesn't fit in one node"**; everything else stays in the node.

### 4.1 Process-group composition

PyTorch / NCCL exposes parallel groups via `torch.distributed.new_group()`. The right composition:

```python
# World layout for Llama-3-405B on 16 GPUs, TP=8 × PP=2
# Ranks 0..7 are node 0; ranks 8..15 are node 1.
tp_groups = [list(range(0, 8)), list(range(8, 16))]      # NVLink each node
pp_groups = [[0, 8], [1, 9], [2, 10], ..., [7, 15]]      # IB across nodes

# For MoE: EP group is also typically the full world.
ep_group = list(range(world_size))
```

`tp_group` collectives ride on NVLink (cheap, frequent). `pp_group` collectives ride on IB (expensive, infrequent — once per stage boundary per microbatch). `ep_group` rides on whatever the world spans; pin tokens to expert owners with minimal hops.

### 4.2 NCCL backend choice

- **NVLink (intra-node)**: NCCL transparently uses NVSwitch fabric. P2P bandwidth ~450-900 GB/s (depends on topology).
- **InfiniBand HDR (200 Gb/s) / NDR (400 Gb/s)**: GPUDirect RDMA gives near-line-rate, but only with proper PXN/SHARP topology. Misconfigured IB → 5× slowdown.
- **Ethernet (100 Gb/s)**: NCCL works but slow; only use for PP, never TP.
- **NVLink Multi-Node Switch (Hopper Grace, NVL72)**: collapses the intra-/inter-node distinction. TP across 72 GPUs becomes feasible. DeepSeek-V3 on GB200 NVL72 can run TP=72.

### 4.3 The decision tree

```
Can model + KV cache fit on 1 GPU?
├── yes → no parallelism. TP=1 PP=1 EP=1.
├── no → fits in 1 node?
│   ├── yes (dense)   → TP=node_size (up to 8). Stop.
│   ├── yes (MoE)     → TP=node_size for dense path + EP=node_size for experts.
│   └── no (dense)    → TP=8 intra-node × PP=ceil(model_size/node_capacity).
│   └── no (MoE)      → TP=8 intra-node × EP=world_size for experts × PP if still doesn't fit.
```

---

## 5. Latency vs throughput tradeoffs across the three axes

| Axis | Per-step latency overhead | Throughput effect | Where it adds cost |
|------|---------------------------|-------------------|--------------------|
| TP=N | +1 all-reduce per layer (small if NVLink) | Higher (more compute per step) | Comm-bound at low batch |
| PP=N | +(stages-1)/microbatches bubble | Higher (more memory, more concurrent batches) | Bubble cost at low QPS |
| EP=N | +2 all-to-alls per MoE layer + load imbalance | Higher (only routed FLOPs spent) | Skewed routing |

**Latency-sensitive deployment (chat, low QPS)**: pick the *minimum* parallelism that lets the model fit. TP=2 beats TP=8 on TPOT at batch=1 because the comm cost dominates the compute savings.

**Throughput-sensitive deployment (batch jobs, high QPS)**: pick parallelism that *maximizes* concurrent capacity. TP=8 + EP=64 + continuous batching = max throughput.

There is no single right answer; you measure both for your workload.

---

## 6. Pitfalls

- **TP across IB.** TP=4 on a 4×L40S box with 100 GbE → all-reduce dominates decode latency. Stick to nodes with NVLink.
- **`kv_heads` not divisible by `tp_size`.** Llama-3-70B has 8 KV heads; TP=16 forces KV-head replication, breaking the memory savings. Stay at TP=8.
- **Pipeline bubble at low QPS.** PP=4 with one in-flight request → 75% idle. Validate that your *steady-state* concurrent batch ≥ stages × 4.
- **MoE routing collapse.** If a few experts grab >40% of traffic, EP all-to-all becomes a single-GPU bottleneck. Monitor router entropy in production; bad pretraining or insufficient capacity factor shows up as fleet-wide TPOT spikes.
- **Mixed precision on the AR.** NCCL bf16 all-reduce uses tree-reduction with rounding error that compounds in long-context decode. Use fp32 reduction for the activation AR; bf16 for the gradient AR (training-only).
- **Hidden cudaMemcpy in EP path.** Some implementations stage tokens via host memory for the all-to-all. Always verify the dispatch/combine kernel does GPU-direct send-recv. TRT-LLM's grouped-EP path is good here.
- **PP stage imbalance.** If you put 20 layers on stage 0 and 21 on stage 1, the slower stage bottlenecks the whole pipeline. Layer assignment must be exactly equal — or compensated by interleaved scheduling.
- **All-to-all token padding.** EP all-to-all expects fixed-size buffers per (sender, receiver) pair. Padding to the max-token count wastes bandwidth; dynamic-size A2A (NCCL ≥ 2.20) is faster but needs Hopper+.

---

## 7. Practitioner's cheat-sheet

```python
# vLLM offline API — explicit parallelism configuration.
from vllm import LLM, SamplingParams

# Llama-3-70B on a single 8×H100 node
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    gpu_memory_utilization=0.92,
    max_model_len=8192,
)

# Mixtral-8x7B with EP exposed
llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
)

# Llama-3-405B across 2 nodes via Ray
llm = LLM(
    model="meta-llama/Llama-3-405B-Instruct",
    tensor_parallel_size=8,
    pipeline_parallel_size=2,
    distributed_executor_backend="ray",
)
```

---

## Connections and what's next

- **[[attention-is-all-you-need]] / ch-02** — the FFN/attention matmuls TP splits.
- **[[kv-cache-memory-formula]] / ch-03** — KV-cache sharding follows the head partition; per-rank KV bytes scale as `1 / tp_size`.
- **[[continuous-batching]] / ch-04** — feeds the PP pipeline at decode.
- **[[pagedattention]] / ch-06** — block tables are per-rank but indexed identically across TP shards.
- **[[cuda-graphs-inference]] / ch-12** — TP-aware graph capture must include the all-reduce; vLLM piecewise graphs split at collective boundaries.
- **[[ch-20]] (production reports)** — DeepSeek-V3, Llama-3, Qwen-3, Mixtral, GPT-OSS serving stories ground all of this in measured numbers.

## Further reading

- [[tensor-parallel-inference]] — Megatron-LM tensor partitioning + vLLM/TRT-LLM serving knobs.
- [[pipeline-parallel-inference]] — GPipe / Megatron-LM interleaved schedules + vLLM PP.
- [[expert-parallel-inference]] — GShard + DeepSpeed-MoE + TRT-LLM expert parallelism.

## Companion visualization

**[figures/parallel-inference-comparison.html](figures/parallel-inference-comparison.html)** — interactive comparison of TP all-reduce traffic vs PP activation transfer vs EP all-to-all under varying batch size and stage count. Use it to build intuition for when each axis pays off.

## Excerpts

- [excerpts/tensor-parallel-inference.md](excerpts/tensor-parallel-inference.md) — Megatron column/row partition, all-reduce cost, TP knob in vLLM.
- [excerpts/pipeline-parallel-inference.md](excerpts/pipeline-parallel-inference.md) — bubble math, 1F1B vs interleaved, PP at decode.
- [excerpts/expert-parallel-inference.md](excerpts/expert-parallel-inference.md) — top-k routing, all-to-all dispatch/combine, grouped GEMM.
- [excerpts/moe-parallel-topology.md](excerpts/moe-parallel-topology.md) — DeepSeek-V3 / Mixtral / Qwen-3 deployment topologies.
