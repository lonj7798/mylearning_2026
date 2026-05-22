---
chapter: ch-13
course: llm-inference
phase: read
excerpt_of: "GShard (Lepikhin et al. 2020) + DeepSpeed-MoE Inference + TensorRT-LLM Expert Parallelism"
source_url: https://arxiv.org/abs/2006.16668
created_at: "2026-05-21"
---

# Excerpt: Expert Parallelism for MoE Inference

**Authors:** Dmitry Lepikhin et al. (GShard, 2020); DeepSpeed team; TensorRT-LLM team
**Year:** 2020 (GShard); 2023+ inference adaptation
**Venue:** ICLR 2021 (GShard)
**URLs:** https://arxiv.org/abs/2006.16668 ; https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html ; https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
**Raw-data source:** [[raw-data/expert-parallel-inference]]

---

## The MoE layer at inference

For input `x ∈ ℝ^d`, the router computes top-k expert assignments:

```math
g(x) = \text{softmax}(W_g \cdot x)
\quad ; \quad
\text{topk}(g, k) \to \{e_1, \ldots, e_k\}, \{w_1, \ldots, w_k\}
```

The MoE output is the weighted combination of selected expert outputs:

```math
y = \sum_{j=1}^{k} w_j \cdot E_{e_j}(x)
```

For Mixtral: `k=2`, `E=8`. For DeepSeek-V3: `k=8`, `E=256` routed + 1 shared. For Qwen-3-235B MoE: `k=8`, `E=128`.

---

## The two all-to-alls per MoE layer

With expert parallelism (EP=`N`, each rank owns `E/N` experts), a token routed to expert `e_j` must execute on the rank owning `e_j`:

```
1. Local:   each rank computes router scores for its local batch.
2. Dispatch (all-to-all):  exchange tokens — every (sender, receiver) pair sends
                           the tokens that the receiver's local experts will process.
3. Compute: grouped GEMM over per-expert mini-batches.
4. Combine (all-to-all):  reverse exchange — token outputs return to their owning rank.
5. Local:   weighted sum with router weights → produces y.
```

Per-step cost (decode, batch=`B`, top-k=`k`, EP=`N`):

```math
\text{a2a bytes per rank} = 2 \cdot \frac{N-1}{N} \cdot B \cdot k \cdot d
```

Factor 2 = dispatch + combine. Decode at `B=32`, `k=8`, `d=8192`, EP=64:

```math
2 \cdot \frac{63}{64} \cdot 32 \cdot 8 \cdot 8192 \cdot 2 \approx 8.2 \text{ MB per rank per layer}
```

Across 58 MoE layers (DeepSeek-V3 has 61 blocks with MoE in layers 4..61): ~475 MB per decode step.

---

## The load-balancing crisis

All-to-all bandwidth = `min over (sender, receiver) of capacity` — the *slowest* link bottlenecks the collective. If token routing is skewed (one expert gets 30% of tokens, another 5%), the GPU owning the hot expert is the bottleneck.

Mitigations:

- **Capacity factor `c`**: cap tokens per expert at `c · B · k / E`. Typical `c=1.25` for inference. Excess tokens are either dropped (training-style) or rerouted to the second-choice expert (inference-style).
- **Aux losses during pretraining**: penalty on router entropy enforces uniform routing. DeepSeek-V3 uses *no aux loss* but adds bias terms updated online to balance load.
- **Redundant experts**: place hot experts on multiple GPUs. Router picks the least-loaded copy.
- **Expert tensor parallelism (ETP)**: each expert is itself TP-sharded across a small subgroup → reduces per-expert memory + smooths variance.

---

## Grouped GEMM — the kernel that saves EP

Per-expert mini-batches are small (~50-500 tokens × FFN dims). Naive per-expert GEMM launches → ~10% tensor-core utilization.

**Grouped GEMM** (CUTLASS `cutlass::GroupedGemm`, cuBLAS-Lt grouped) batches `E_local` expert GEMMs into one kernel launch:

```
inputs:   list of (M_e, K) matrices for e = 1..E_local
weights:  list of (K, N) matrices for e = 1..E_local
outputs:  list of (M_e, N) matrices
kernel:   one launch, internally pipelined across experts
```

For DeepSeek-V3 (E_local=4 per GPU, M_e≈50-200 tokens, K=2048, N=2048 SwiGLU dim): ~60% utilization vs ~10% for per-expert launches. **6× throughput win**; this is the difference between MoE being competitive and being a research curiosity.

vLLM uses `fused_moe` (Triton-implemented grouped MoE kernel); TRT-LLM uses CUTLASS; SGLang uses `fused_moe` too.

---

## Composition with TP

EP and TP are orthogonal axes. For a node with 8 GPUs:

- **Mixtral-8x22B**: TP=4 × EP=4 → each GPU owns 2 experts, dense MHA is TP=4-sharded.
- **DeepSeek-V3**: TP=8 (intra-node, for dense MLA + shared expert) × EP=64 (across all 8 nodes, each GPU owns 4 routed experts).

The dense path (MHA/MLA, embedding, LM head) all-reduces on `tp_group`; the MoE path all-to-alls on `ep_group`. The two collective patterns coexist on the same physical interconnect — NCCL just multiplexes them.

```python
# Process group composition for DeepSeek-V3 on 64 GPUs (8 nodes × 8 H100)
tp_groups = [list(range(8*i, 8*(i+1))) for i in range(8)]   # 8-GPU NVLink groups
ep_group  = list(range(64))                                  # full world
```

---

## Pitfalls

- **Skewed routing kills throughput.** Monitor per-expert traffic; alert if any expert > 2× the mean.
- **All-to-all needs equal-sized buffers per pair** unless you use dynamic-size A2A (NCCL ≥ 2.20). Padding wastes bandwidth.
- **Host-staged dispatch.** Verify your EP path is GPU-direct send-recv, not staged via CPU memory.
- **CUDA graph capture is hard for EP.** Token routing varies per step → variable-shape grouped GEMM. Use piecewise capture only around the dispatch/combine boundaries.
- **Top-k arithmetic.** Bytes per AR scale with `k`. DeepSeek-V3's `k=8` is 4× the AR traffic of Mixtral's `k=2` for the same `d`.

---

## Numbers from production

| Model | Topology | Decode TPOT (ms, batch=32, H100) | Source |
|-------|----------|----------------------------------|--------|
| Llama-3-70B (TP=8) | 8×H100 NVLink | ~28 ms | vLLM benchmarks |
| Mixtral-8x22B (TP=4 EP=4) | 4×H100 NVLink | ~22 ms | TRT-LLM docs |
| DeepSeek-V3 (TP=8 EP=64) | 64×H100 IB | ~40 ms | DeepSeek tech report |
| Qwen-3-235B-MoE (TP=8 EP=16) | 16×H100 | ~35 ms | Qwen-3 tech report |

DeepSeek-V3 at 40 ms TPOT with 671B params is the most striking number — only ~37B active per token, EP keeps the compute commensurate with activated FLOPs not total params.

---

## Connections

- [[excerpts/tensor-parallel-inference]] — dense-path companion.
- [[excerpts/pipeline-parallel-inference]] — across-node escape hatch.
- [[ch-13]] — parent chapter.
