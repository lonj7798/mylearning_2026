---
chapter: ch-06
course: llm-inference
phase: read
excerpt_of: "Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023)"
source_url: https://arxiv.org/abs/2309.06180
created_at: "2026-05-21"
---

# Excerpt: PagedAttention — KV cache as virtual memory

**Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
**Year:** 2023 (SOSP)
**URL:** https://arxiv.org/abs/2309.06180
**Raw-data source:** [[raw-data/pagedattention]]

---

## The thesis in one sentence

> "We propose PagedAttention, an attention algorithm inspired by the classic virtual memory and paging techniques in operating systems. PagedAttention divides the KV cache into blocks, where each block contains the keys and values for a fixed number of tokens. In PagedAttention, the blocks for the KV cache are not necessarily stored in contiguous space. Therefore, we can manage the KV cache in a more flexible way as in OS's virtual memory." (§3)

The thesis is small and the consequences are large. KV cache → pages. Per-sequence layout → page table. Cross-sequence sharing → COW. Allocation/free → block pool ops. Each maps to a forty-year-old OS concept.

---

## The numbers that justify the work (paper §2)

The paper's motivating measurements on OPT-13B:

- **KV cache occupies 30 % of GPU memory** on average across the ShareGPT trace, **and grows or shrinks dynamically per request**.
- **Existing serving systems achieve only 20–38 % effective KV-memory utilization.** The rest is reservation waste + fragmentation.
- **vLLM with PagedAttention achieves >96 % utilization.** The waste shrinks to one partial block per active sequence.

Concretely (Figure 6): pre-vLLM systems had effective batch sizes of 8–15 on OPT-13B; vLLM hits 35–40 on the same hardware.

---

## The block-table mechanism (paper §3 + §4)

Logical KV layout for sequence `s`:

```
positions  0..15    16..31    32..47    48..63    ...
blocks     [ L0 ]   [ L1 ]    [ L2 ]    [ L3 ]    ...
```

Each `L_i` is a fixed-size block holding K and V for 16 contiguous token positions, for all KV heads, all layers. The physical KV pool is a global tensor:

```
K_pool: [num_blocks, num_layers, num_kv_heads, head_dim, block_size]
V_pool: [num_blocks, num_layers, num_kv_heads, head_dim, block_size]
```

The block table maps logical to physical:

```
block_table[s] = [P_47, P_3, P_91, P_12, ...]
```

To compute attention at logical position `t` for sequence `s`:
- `b = t // 16`, `o = t % 16`
- physical row: `P = block_table[s][b]`
- read K, V from `K_pool[P, :, :, :, o]`, `V_pool[P, :, :, :, o]`

The block size of 16 is the smallest that keeps kernel tiles efficient (warps of 32 threads × half-block load). The paper ablates 1, 2, 4, 8, 16, 32, 64, 128 and finds 16 minimizes the joint cost of fragmentation + indirection (Table 4).

---

## Memory accounting — the per-sequence formula

Worst-case KV bytes per sequence under paging:

```math
\text{bytes}(s) \;=\; \left\lceil \frac{T_s}{16} \right\rceil \cdot 16 \cdot 2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b
```

Where `T_s` is the actual current length of sequence `s`. The `ceil(...) × 16` rounds up to whole blocks. Maximum slack per sequence: 15 token slots. For mean sequence length 500: `15/500 = 3 %` waste — what the paper calls "internal fragmentation, bounded by one block per sequence."

Naive contiguous: bytes pre-allocated for `T_max` regardless of `T_s`. For `T_max = 8192`, `T_s = 500`, waste = `(8192-500)/8192 = 94 %` for that sequence. The waste compounds when you multiply by active sequence count.

---

## Copy-on-write for shared prefixes (paper §4.2)

Setup: a request with prompt `P` and sampling parameter `n=4` produces 4 sibling sequences sharing the prefill. The block tables look like:

```
sib_0:   [ Bp0, Bp1, Bp2, ..., Bp_k, Bd0_s0 ]   # Bd0_s0 = divergent tail block for sib 0
sib_1:   [ Bp0, Bp1, Bp2, ..., Bp_k, Bd0_s1 ]
sib_2:   [ Bp0, Bp1, Bp2, ..., Bp_k, Bd0_s2 ]
sib_3:   [ Bp0, Bp1, Bp2, ..., Bp_k, Bd0_s3 ]
```

The prompt blocks `Bp0..Bp_k` are physically allocated **once**. Per-sibling memory is just the divergent tail.

COW trigger: if the last prompt block `Bp_k` is partially full when sampling diverges, the manager:
1. Allocates a fresh physical block `Bp_k'` for the writing sibling.
2. Copies the shared portion of `Bp_k` into `Bp_k'`.
3. Updates that sibling's block table: `... Bp_k → Bp_k'`.
4. The other siblings keep pointing at the original `Bp_k`.

Net effect: parallel sampling of `n=4` on a 2048-token prompt + 200 decode tokens costs `~1.05 ×` single-sequence memory, not `4 ×`. Beam search uses the same mechanism plus pruning.

---

## The PagedAttention kernel (paper §4.4)

The kernel deviates from a standard attention kernel in two places:

1. **K/V loads go through `block_table[s, b]` indirection.** One indirect-load per block-tile.
2. **`context_lens[s]` per-sequence** — the kernel knows how many blocks are valid for each query in the batch.

The CUDA kernel signature (`csrc/attention/attention_kernels.cu`):

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,             // [batch, num_heads, head_dim]
    const scalar_t* __restrict__ q,         // [batch, num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_dim/X, block_size, X]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_dim, block_size]
    const int* __restrict__ head_mapping,   // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,   // [batch, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,   // [batch]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // optional [num_heads]
    const int q_stride, const int kv_block_stride,
    const int kv_head_stride);
```

Block-level kernel sketch:

```cpp
for (int block_idx = 0; block_idx < num_blocks_for_seq; block_idx++) {
    int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
    // load K[physical_block, head, :, :], V[physical_block, head, :, :]
    // dot(Q, K) per token in this block, softmax tile, accumulate against V
}
```

Per the paper (Table 5), this kernel hits ~98 % of the throughput of an equivalent contiguous-layout attention kernel. The indirection cost is dominated by the actual K/V load bandwidth.

---

## Throughput results (paper §6)

Headline measurements on real workloads:

| Model | Workload | vLLM throughput vs FasterTransformer | vs Orca (best) |
|---|---|---|---|
| OPT-13B | ShareGPT, A100 | **2.7×** | 1.7× |
| OPT-66B | ShareGPT, 4×A100 TP | **3.5×** | 2.0× |
| OPT-175B | ShareGPT, 8×A100 TP | **4.0×** | 1.9× |
| LLaMA-7B | Alpaca, A100 | **2.9×** | 1.7× |
| LLaMA-13B | Alpaca, A100 | **2.6×** | 1.6× |

All measurements at iso-latency SLO. The gains scale with the variance of the workload's output-length distribution — more variance → more contiguous-allocation waste → larger paging win.

---

## What the paper does *not* claim

Three honest limits:

- **Paging does not reduce per-token KV bytes.** The ch-03 formula is unchanged; paging just eliminates over-reservation.
- **Block size 16 is workload-tuned.** On very short workloads it can be too large; on very long ones it can be too small. The paper recommends 16 as a safe default and ablates the alternatives.
- **Block-table maintenance is a CPU cost on the scheduler.** For 1000s of concurrent sequences with 1000s of blocks each, the per-step serialization of block tables to GPU becomes non-trivial. vLLM V1 addresses this with delta encoding (only changed entries serialized).

---

## Connections

- [[excerpts/vllm-kv-cache-manager]] — the production-code realization of the block pool + COW.
- [[excerpts/vllm]] — overall vLLM architecture this sits inside.
- [[ch-06]] — parent synthesis.
- Forward to [[ch-07]] — prefix sharing across requests (RadixAttention) is the cross-request generalization of intra-request COW.
- Forward to [[ch-08]] — KV-cache eviction policies all assume paged allocation.
