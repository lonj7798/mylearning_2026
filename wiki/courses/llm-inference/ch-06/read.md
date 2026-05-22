<!-- chapter: ch-06
     track: kv-cache
     title: PagedAttention + Block-Based KV Cache (vLLM)
     sources: [[pagedattention]], [[vllm-kv-cache-manager]], [[vllm]], [[vllm-scheduler]]
     back: [[kv-cache-memory-formula]] (ch-03), [[continuous-batching]] (ch-04)
     forward: [[sglang-radixattention]] (ch-07), [[h2o]] (ch-08), [[vllm-scheduler]] (ch-16)
     figures: figures/pagedattention-block-table.html
-->

# Chapter 6 — PagedAttention + Block-Based KV Cache (vLLM)

> **Core insight.** A naive contiguous per-request KV cache reservation forces the serving system to pre-allocate space for *the maximum possible output length*. That wastes 30–50 % of GPU HBM on slack the request will never use. PagedAttention treats KV cache like OS virtual memory: split it into fixed-size **blocks** (16 tokens each), maintain a per-sequence **block table** that maps logical positions to physical blocks, and let the attention kernel gather K/V values *through* the table. Memory waste drops to less than one block per sequence — internal fragmentation only — and effective batch size grows 2–4× at the same HBM. Shared prefixes can point multiple block tables at the same physical block (copy-on-write on divergence), making parallel sampling and beam search nearly free.
>
> **Guideline.** Use a paged KV cache for any production serving deployment of a non-trivial model. Set `block_size = 16` (the de facto standard since the vLLM paper), set `gpu_memory_utilization = 0.85–0.95` (leaves room for activations + intermediate buffers), and let the engine size the free-block pool automatically. The block-table indirection costs ~1–3 % attention-kernel latency; the memory utilization wins back 30–80 % effective capacity.

---

## Why this chapter exists

[[kv-cache-memory-formula]] (ch-03) gave us the per-request size of the KV cache:

```math
\text{KV bytes} \;=\; 2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot b
```

For Llama-3-70B with `L=80`, `H_kv=8` (GQA), `d_head=128`, `b=2` (bf16) and `T=8192` tokens, that's `~10.5 GB` per request. On an 80 GB H100 after the 140 GB-class weights (TP=2 → 70 GB per GPU) and activation buffers (~10 GB), you have maybe `~30 GB` left for KV cache. That is space for **about 3 concurrent requests** at 8k context, or **24 at 1k**, but always at the *maximum possible* output length the scheduler is willing to admit.

The naive serving allocation makes things much worse. Suppose the scheduler reserves `max_model_len = 8192` tokens of KV per admitted request — i.e., what's needed if the request runs to its longest possible output. A request that actually generates only 200 tokens uses ~0.25 GB but locks down ~10 GB. The unused 9.75 GB is dead until the request finishes. With variable output lengths (typical for chat / code), the wasted fraction averages 30–60 %.

The [[pagedattention]] paper (Kwon et al., SOSP 2023) attacks this problem with a single insight: the KV cache is morally a virtual address space, not a physical buffer. Borrow the OS solution — pages and page tables — and apply it to GPU memory. The result is **vLLM**, the most-used open-source LLM serving engine in 2026, and a memory-management pattern every later system (SGLang, TensorRT-LLM paged KV, LMDeploy, TGI) has adopted.

This chapter walks the mechanism, the kernel changes, the copy-on-write extension, and the production knobs you actually turn.

---

## 1. The fragmentation problem in naive contiguous KV

Imagine a serving engine that allocates each request's KV cache as **one contiguous tensor** of shape `[L, 2, H_kv, T_max, d_head]`. Two kinds of waste appear:

- **Internal fragmentation.** Reserved length `T_max` minus actual length `T_used`. For chat workloads with `T_max=4096` and typical `T_used ≈ 200–800`, this is 80–95 % waste *per active request*.
- **External fragmentation.** When requests of different lengths finish and free their slabs, free GPU memory becomes a jigsaw of holes. New large requests can't be admitted even when total free memory is enough.

The combined effect is dramatic. Pre-vLLM measurements from the paper (their Figure 6 on OPT-13B) show effective batch sizes capped at ~10–15 even when the GPU has memory for ~40 fully utilized contexts. The KV-cache pre-reservation eats the difference.

Three workarounds existed pre-vLLM, all bad:

1. **Shrink `T_max`.** Cheaper per request, but truncates legitimate long generations.
2. **Re-pack the cache when a request finishes.** Expensive memory copies; serializes the scheduler.
3. **Conservative admission.** Refuse requests when free space < `T_max`. Throughput collapse under load.

PagedAttention's contribution is to make the problem disappear: with block-sized allocation there's no contiguous slab to fragment, and internal waste is bounded by one (16-token) block per sequence.

---

## 2. The virtual-memory analogy

The mechanism is a near-perfect transplant of OS paging onto GPU memory.

| OS virtual memory | PagedAttention |
|---|---|
| Process address space (virtual) | Per-sequence logical KV position (`pos = 0, 1, 2, ...`) |
| Physical RAM page | GPU KV block (16 tokens × 2 × H_kv × d_head × b bytes) |
| Page table | Block table — `block_table[seq_id]: list[physical_block_id]` |
| Page fault | Block allocation failure → preemption / wait |
| Shared library COW page | Shared prompt prefix block, COW on divergence |

The block table for sequence `s` is a list of integers indexing into the global block pool:

```
seq_s logical positions:  [   0..15  |  16..31  |  32..47  |  48..63  ]
seq_s block_table:        [    B7    |    B2    |    B19   |    B11   ]
```

When the attention kernel for sequence `s` needs `K[s, :]` at logical position 50, it computes `block_idx = 50 // 16 = 3`, `offset = 50 % 16 = 2`, looks up `physical_block = block_table[s][3] = B11`, and reads from `K_pool[B11, 2, :, :]`. Indirection costs one extra load; that's the entire price.

**Why 16 tokens per block?** The choice is empirically tuned in the paper (their Table 4):

- Smaller blocks (4, 8) waste more space on per-block metadata + extra block-table entries and add kernel overhead.
- Larger blocks (32, 64) increase internal fragmentation per sequence (the last partially full block is bigger on average).
- Block size 16 minimizes the joint cost across realistic workloads and is small enough that the per-block-table indirection stays in L1/L2 cache.

This is the same trade-off OS page-size tuning faces, and the answer is the same: power-of-two, small enough for low fragmentation, large enough to amortize bookkeeping.

---

## 3. The 96 % memory utilization claim

The paper's headline number is **96 % effective KV-memory utilization** vs ~60 % for naive contiguous allocation, on OPT-13B with the ShareGPT trace.

Where the 4 % residual waste comes from:

- **Last-block fragmentation.** Each active sequence's final block is on average half-full. With block size 16, that's 8 wasted slots per sequence. Total waste: `(8 / mean_seq_len) × num_active_seqs`. For mean seq length 500 and 40 active sequences, that's `8/500 = 1.6 %` per sequence, ~1–2 % aggregate.
- **Block-table memory.** A 4-byte block ID times (max_seq_len / 16) entries per sequence times num_sequences. For 64 sequences at 8k context, that's `64 × 512 × 4 = 128 KB` — negligible on an 80 GB GPU.
- **Free-pool reserve.** Some blocks are kept free for admission of new arrivals; the system targets ~95 % occupancy on the active pool.

The naive 60 % number is workload-dependent — it can be far worse on traces with high length variance (the worst case in the paper is ~38 % utilization with naive allocation vs 95 %+ with paging).

The downstream effect: a fixed-HBM GPU now serves 2–4× more concurrent requests. That is the entire vLLM throughput story.

---

## 4. Copy-on-write for parallel sampling and beam search

A second win from the block table abstraction: shared prefixes are physically shared.

**Parallel sampling** (n=4): a user issues "generate 4 completions for this prompt." The four sequences share the entire prompt prefix; they only diverge at the first sampled token. Naive contiguous allocation duplicates the KV cache 4 times. Paged allocation:

```
seq_0 block_table: [ B7, B2, B19, B11_seq0 ]   # all four share B7, B2, B19
seq_1 block_table: [ B7, B2, B19, B11_seq1 ]
seq_2 block_table: [ B7, B2, B19, B11_seq2 ]
seq_3 block_table: [ B7, B2, B19, B11_seq3 ]
```

The prompt blocks `[B7, B2, B19]` are physically allocated once. Each sibling gets its own *post-divergence* tail block. KV memory cost for the shared prefix: `1 ×` (not `n ×`).

**Copy-on-write trigger.** What if all four siblings still share the *last* partially-full block when sampling diverges? The vLLM block manager allocates a fresh physical block for each sibling, copies the shared portion into each, then writes the divergent token to each new block. This is exactly the OS COW pattern, and it's the only time blocks are physically copied.

**Beam search** is the same mechanism with one tree-structured twist: when a beam is pruned, its blocks are freed; when a beam is extended, its block table is extended. The per-beam state lives entirely in the block table. The KV pool sees only allocation/deallocation/COW events.

The result: parallel sampling with `n=4` on a 2k-token prompt costs ~`1.05 ×` the KV memory of a single sequence (one prompt's worth of blocks plus 4 tiny divergent tails), not `4 ×`. Beam search with 4 beams pays a similar discount until late in generation.

This sharing is the foundation of [[sglang-radixattention]] (ch-07): if siblings of one request can share blocks, why not siblings *across requests* that happen to share a system prompt?

---

## 5. The PagedAttention kernel

The mechanism imposes one requirement on the attention kernel: it must read K and V **through the block table** instead of from a contiguous stride.

The paper presents two kernel paths:

- **Prefill kernel.** Many new query tokens against the request's already-paged KV cache. The kernel loads each requested K/V block via the block-table indirection, then runs the standard tile-based attention (FlashAttention-style). The block-table lookup happens at the block-tile boundary; inside a tile, addressing is contiguous.
- **Decode kernel.** One query token (the just-sampled token) against the request's full KV history. Each block contributes 16 K/V vectors; the kernel iterates `T // 16` blocks via the table.

The kernel signature roughly looks like:

```cpp
void paged_attention_v1(
    float* out,                          // [batch, num_heads, head_dim]
    const float* q,                      // [batch, num_heads, head_dim]
    const float* k_cache,                // [num_blocks, num_kv_heads, head_dim, block_size]
    const float* v_cache,                // [num_blocks, num_kv_heads, head_dim, block_size]
    const int* block_tables,             // [batch, max_num_blocks_per_seq]
    const int* context_lens,             // [batch]
    int block_size, int num_heads, int num_kv_heads, int head_dim,
    float scale, ...
);
```

The block table is a `[batch, max_blocks_per_seq]` int32 tensor. For each query in the batch, the kernel loops `context_lens[i] / block_size` blocks; each iteration uses `block_tables[i, block_idx]` as the K/V cache row to load.

The indirection adds one indirect-load instruction per block tile. In benchmarks (paper Table 5; vLLM benchmarks v0.6) the paged kernel runs at 97–99 % of the throughput of an equivalent contiguous-layout kernel. The memory-utilization win swamps the small kernel-side cost.

Production attention backends now implement paged variants:

- **vLLM's `paged_attention` kernel** (CUDA, AMD HIP, Triton fallbacks).
- **FlashAttention 2+ paged** (Dao 2023, added in v2.4).
- **FlashInfer paged** — the kernel library many serving systems use; explicit `BatchPrefillWithPagedKVCacheWrapper` API. Covered in ch-11.
- **TensorRT-LLM paged KV** — see [[tensorrt-llm-paged-kv]].

The PagedAttention kernel is also what makes [[quest-kv]] (ch-08) feasible: query-aware sparsity loads *some* blocks per query and skips others, all via the same indirection layer.

---

## 6. The vLLM KV cache manager — the production realization

[[vllm-kv-cache-manager]] (`vllm/v1/core/kv_cache_manager.py`) is the module that implements all of the above for production. Its public surface to the scheduler is small:

```python
class KVCacheManager:
    def allocate_slots(self, request, num_tokens) -> Optional[KVCacheBlocks]: ...
    def get_computed_blocks(self, request) -> tuple[KVCacheBlocks, int]: ...
    def free(self, request) -> None: ...
    def free_block_hashes(self, request) -> None: ...
```

Five things the manager owns:

1. **Free block pool.** A `deque[int]` of unused physical block IDs. Allocation pops; free pushes back.
2. **Per-request block tables.** Stored as `request.kv_cache_blocks: list[list[int]]` (one list per KV-cache group; multi-group exists for models that need different cache specs per layer, e.g., sliding-window layers).
3. **Prefix cache.** Block-hash → physical block ID lookup. On a hit, the request inherits the existing physical block at zero cost. (Ch-07 unpacks this.)
4. **Eviction policy.** When the free pool is empty and the prefix cache holds blocks no live request references, those blocks are evicted (LRU) to refill the pool.
5. **Copy-on-write.** `allocate_slots` for a request that wants to write to a currently-shared block triggers a fresh-block copy.

Two design decisions worth flagging:

- **Block tables live in *Python objects*, not GPU memory** (with a serialized copy pushed to the GPU per step inside `SchedulerOutput`). The scheduler manipulates them cheaply; the kernel reads from the serialized form. This keeps allocation logic out of the GPU critical path.
- **`free()` is LIFO.** The most recently allocated block of a request is freed first. Combined with the LRU cache eviction, this gives the prefix cache a natural priority: shared early blocks (system prompt) stay resident longest.

---

## 7. The end-to-end story: how a request flows through paged KV

To make the abstraction concrete, here is what happens to a single request from admission to retirement:

```
Step 0  Request R arrives with prompt of length 200, max_output=512.
        Scheduler asks KVCacheManager.get_computed_blocks(R):
          - prefix-cache lookup → miss (no shared prefix).
        Scheduler calls allocate_slots(R, num_tokens=200):
          - 200 / 16 = 12.5 → 13 blocks needed.
          - free_pool.pop(13) → block_table[R] = [B7, B2, B19, ..., B41] (13 entries).
        Scheduler emits R as a prefill chunk in the SchedulerOutput.

Step k  Prefill of R completes (possibly over several chunked steps).
        R promotes to running. Each decode step:
          - allocate_slots(R, 1) → may need 0 or 1 new blocks
            (1 new block every 16 decode tokens).
          - Append sampled token's K/V to last block at offset (len % 16).

Step N  R hits stop / EOS / max_tokens. Scheduler calls free(R):
          - Blocks are returned to free_pool in LIFO order.
          - If prefix caching is on, blocks may be left in the cache
            (hash-keyed) instead of immediately freed.
```

Three operational consequences:

- **A request's worst-case KV cost is `ceil((prompt_len + max_output) / 16)` blocks**, not `T_max`. That's the entire memory-utilization win in one line.
- **KV pressure builds linearly during decode** (one new block per 16 tokens generated). Tail-latency spikes from preemption tend to happen mid-generation, not at admission.
- **Block size 16 is also the prefix-cache granularity.** A new request shares blocks with an old one only at 16-token boundaries — exact-token-prefix match. Ch-07 covers what this means for cache hit rates.

---

## 8. What PagedAttention does *not* solve

Three real limits worth flagging:

- **The KV cache is still proportional to context length.** Paging eliminates *waste*, not the underlying ch-03 formula. For 100k-token contexts you still need ~12 GB per request; paging just lets you use all of it.
- **Block-table indirection is a small but measurable kernel cost.** On extremely short contexts (<128 tokens) the indirection overhead can be ~5 %.
- **Cache-line locality is slightly worse.** Two consecutive logical positions may live in physically non-adjacent blocks. Modern attention kernels handle this fine, but kernel writers targeting paged layouts need to design around block boundaries.

The cases where paging genuinely loses are rare in serving (very-short uniform prompts is the main one, and continuous batching already handles that workload at high throughput).

---

## Connections and what's next

- **Back to [[kv-cache-memory-formula]] (ch-03)** — paging makes the per-request cost from ch-03 *achievable* by eliminating slack; the underlying bytes-per-token are unchanged.
- **Back to [[continuous-batching]] (ch-04)** — dynamic admission needed paged allocation to work in practice. Continuous batching pre-vLLM admission rates were limited by fragmentation, not compute.
- **Forward to [[sglang-radixattention]] (ch-07)** — extends block sharing from "siblings of one request" to "any request whose tokenized prefix matches an existing cache entry." Radix tree, not flat hash table.
- **Forward to [[h2o]] / [[snapkv]] / [[quest-kv]] (ch-08)** — these are KV-cache *eviction and sparsity* policies layered on top of paged allocation. They all exploit the block-table indirection to load only the blocks they want.
- **Forward to [[vllm-scheduler]] (ch-16)** — full-detail tour of the scheduler that drives the block manager.
- **Forward to [[tensorrt-llm-paged-kv]] (ch-18)** — NVIDIA's paged KV implementation; the same idea, different code path, FP8-aware blocks.

## Further reading

- [[pagedattention]] — Kwon et al. SOSP 2023. The paper.
- [[vllm-kv-cache-manager]] — production code at `vllm/v1/core/kv_cache_manager.py`.
- [[vllm]] — overall vLLM architecture and APIs.
- [[vllm-scheduler]] — how the scheduler asks for blocks and handles `None` returns.
- vLLM PagedAttention CUDA kernel source — `csrc/attention/attention_kernels.cu`.

## Companion visualization

**[figures/pagedattention-block-table.html](figures/pagedattention-block-table.html)** — interactive viewer: four sequences sharing prefix blocks, a divergence triggering COW, an eviction freeing a block back to the pool. Hover any logical position to see the block-table lookup.
