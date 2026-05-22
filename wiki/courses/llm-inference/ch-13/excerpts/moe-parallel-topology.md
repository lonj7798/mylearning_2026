---
chapter: ch-13
course: llm-inference
phase: read
excerpt_of: "Synthesis of MoE serving topologies from DeepSeek-V3, Mixtral, Qwen-3 deployment guides"
source_url: synthesis
created_at: "2026-05-21"
---

# Excerpt: MoE Parallel Topology — Deployment Patterns

**Sources synthesized:**
- DeepSeek-V3 Technical Report (2024)
- Mistral / Mixtral deployment docs (mistralai)
- Qwen-3 Technical Report (Alibaba, 2025)
- TensorRT-LLM expert-parallelism docs
- vLLM distributed serving docs
- SGLang docs on `--ep-size`

---

## The general rule

For an MoE model with:
- `d` hidden size
- `L` total layers, of which `L_moe` are MoE
- `E` routed experts, top-`k` selection
- one or more `shared` experts always active

The serving topology decomposes the world into three groups:

```
WORLD = TP_group × EP_group × PP_group     (each is a subset of ranks)
```

- **TP_group** carries dense MHA/MLA, the shared expert, the LM head, and the embedding. All-reduces ride on this group.
- **EP_group** carries the routed experts. All-to-alls ride on this group.
- **PP_group** transitions activations between layer stages. Point-to-point send-recv rides on this group.

For `S` nodes × `G` GPUs per node:

| Topology pattern | TP | EP | PP | When |
|------------------|-----|----|----|------|
| Single-node MoE | `G` | `G` (overlapped) | 1 | model fits 1 node |
| Multi-node MoE, no PP | `G` (per-node) | `S·G` (all ranks) | 1 | model fits 1 node, scale-out for throughput |
| Multi-node MoE, with PP | `G` (per-node) | `S·G` | `s` | model doesn't fit 1 node |

---

## DeepSeek-V3 (671B MoE, 37B active per token)

```
Total params:     671 B
Active per token: 37 B  (1 shared expert + 8 of 256 routed)
Hidden:           7168
Layers:           61   (MoE in layers 4..61, dense in 1..3)
MLA:              compressed K/V via low-rank projection (256 KV dim)
```

**Recommended deployment (DeepSeek tech report, Section 6.4):**

```
World:  8 nodes × 8 H100 = 64 GPUs
TP:     8   (NVLink within each node — dense MLA + shared expert)
EP:     64  (full world — routed experts, each GPU owns 4)
PP:     1   (model fits without depth-sharding)
```

Why TP=8 not TP=64? Because the dense MLA path's all-reduce would saturate IB at TP=64. By keeping TP=8 within NVLink, only the EP all-to-all crosses IB — which is what IB is good at (point-to-point, GPUDirect RDMA).

Why EP=64 not EP=8? Because 256 routed experts / EP=8 = 32 experts per GPU → memory pressure + worse load balancing. EP=64 gives 4 experts per GPU, balanced.

**Reported decode TPOT**: ~40 ms at batch=32 on H100 IB cluster — competitive with dense 70B at similar batch.

---

## Mixtral-8x7B (45B params, 13B active)

```
Total params:     45 B
Active per token: 13 B  (2 of 8 experts)
Hidden:           4096
Layers:           32
```

**Recommended deployment:**

```
World:  2 H100 = 2 GPUs (or 1 H200 for single-GPU)
TP:     2
EP:     2     (each GPU owns 4 experts)
PP:     1
```

Single-node, NVLink-only. AR and A2A both ride NVLink. No IB anywhere.

```bash
# vLLM
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel
```

---

## Mixtral-8x22B (141B params, 39B active)

```
Total params:     141 B
Active per token: 39 B  (2 of 8 experts)
Hidden:           6144
Layers:           56
```

**Recommended deployment:**

```
World:  4 H100 (1 node)
TP:     4
EP:     4    (each GPU owns 2 experts)
PP:     1
```

Still single-node. Why not TP=8 + EP=8? Because Mixtral has only 8 experts — EP=8 puts 1 expert per GPU, which both raises memory waste (each GPU loads one expert's FFN, ~6 GB) and worsens load balance.

---

## Qwen-3-235B-MoE (235B params, 22B active)

```
Total params:     235 B
Active per token: 22 B  (8 of 128 routed experts)
Hidden:           8192
Layers:           94
```

**Recommended deployment (Qwen-3 tech report):**

```
World:  2 nodes × 8 H100 = 16 GPUs
TP:     8    (NVLink intra-node)
EP:     16   (full world — 8 experts per GPU)
PP:     2    (94 layers split 47/47 across nodes)
```

Both PP and EP. PP is needed because 235B params don't fit in 1 NVLink island; EP because it's MoE.

---

## GPT-OSS-120B (MoE)

```
Total params:    ~120 B
Active per token: ~5 B   (sparse routing)
```

Fits in 1 node:

```
World:  8 H100 (1 node)
TP:     8
EP:     8
PP:     1
```

NVLink absorbs both AR and A2A. No IB traffic at all. This is the "easy" MoE case.

---

## Decision recipe

```
1. Compute total model bytes (params + per-rank KV cache headroom).
2. If bytes < single GPU memory:    TP=1 EP=1 PP=1.
3. Else if bytes < node memory:     TP=node_size; EP=node_size (if MoE); PP=1.
4. Else:                             TP=node_size; EP=world_size (if MoE);
                                     PP=ceil(bytes / node_capacity).
5. Validate kv_heads % tp_size == 0; validate routed_experts % ep_size == 0.
6. Measure TTFT / TPOT / throughput vs PP=1 baseline; tune `gpu_memory_utilization`.
```

The rule is mechanical, not aesthetic — every step is forced by either memory or interconnect bandwidth.

---

## Connections

- [[excerpts/tensor-parallel-inference]] — dense path partitioning.
- [[excerpts/expert-parallel-inference]] — EP mechanics.
- [[excerpts/pipeline-parallel-inference]] — when PP is unavoidable.
- [[ch-20]] — production model inference reports with measured numbers.
- [[ch-13]] — parent chapter.
