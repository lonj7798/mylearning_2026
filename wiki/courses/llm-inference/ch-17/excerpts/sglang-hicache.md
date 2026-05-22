---
chapter: ch-17
course: llm-inference
phase: read
excerpt_of: "SGLang HiCache design — hierarchical KV caching for long-context serving"
source_url: https://docs.sglang.ai/advanced_features/hicache_design.html
created_at: "2026-05-21"
---

# Excerpt: HiCache — three-tier hierarchical KV cache

**Authors:** SGLang team
**Year:** 2025–2026
**URLs:** https://docs.sglang.ai/advanced_features/hicache_design.html / https://docs.sglang.ai/advanced_features/hicache_best_practices.html
**Raw-data source:** [[raw-data/sglang-hicache]]

---

## The three tiers

```
┌──────────────────────────────────────────────────────┐
│  L1: GPU HBM                                         │
│   Capacity: 10–80 GB (after model weights)           │
│   Access: compute-speed (~µs)                        │
│   Role: hot prefix KV; decode-step working set       │
└──────────────────────────────────────────────────────┘
                       ↕  PCIe DMA (~64 GB/s)
┌──────────────────────────────────────────────────────┐
│  L2: Host RAM                                        │
│   Capacity: 100 GB – 1 TB                            │
│   Access: ~ms (PCIe transfer for one prefix)         │
│   Role: warm reusable prefixes, cross-request        │
└──────────────────────────────────────────────────────┘
                       ↕  RDMA / network (~100 Gbps)
┌──────────────────────────────────────────────────────┐
│  L3: Distributed storage                             │
│   Capacity: TB – PB                                  │
│   Access: ~10–100 ms (network + storage)             │
│   Role: cluster-shared cache; cross-node reuse;      │
│         persistence across server restarts           │
│   Backends: Mooncake, 3FS, NIXL, AIBrix KVCache      │
└──────────────────────────────────────────────────────┘
```

This is CPU-cache-hierarchy thinking applied to KV cache: each tier is bigger, slower, and shared more broadly.

---

## HiRadixTree — metadata on top of RadixAttention

Standard `RadixCache` becomes `HiRadixCache`. Each tree node now carries a *location set*:

```python
class HiRadixNode(TreeNode):
    # inherited: children, parent, key, value (GPU block IDs), lock_ref, last_access_time
    cpu_blocks: list[int] | None      # location in host memory pool (if any)
    l3_uri: str | None                # URI in distributed store (if any)
    write_back_pending: bool          # async write to L2/L3 in flight
```

A single prefix's KV can live at multiple tiers simultaneously — e.g. a recently-used prefix is in both GPU L1 *and* CPU L2, so GPU eviction is free (just drop GPU blocks; CPU copy survives and can rehydrate).

---

## The match → fetch → write-back workflow

```
1. Request arrives → HiRadixCache.match_prefix(tokens)
   - Walk tree; for matched segments, inspect each node's location set.

2. Schedule fetches:
   - L1 hit (GPU blocks present)         → no work
   - L2 hit (CPU blocks present)         → enqueue async DMA host → GPU
   - L3 hit (l3_uri set)                 → enqueue async fetch L3 → host → GPU
   - miss                                → must prefill

3. Wait for fetches to complete (overlapped with admission of other requests).

4. Run prefill on the suffix only (potentially zero if everything was cached).

5. Decode loop generates new KV; on request finish:
   - cache.insert() adds the full path to HiRadixTree
   - Async write-back: copy GPU blocks → CPU L2 (always)
   - Async write-back: copy → L3 (only if `write_through_full` or selectively
     by score / hit-rate prediction under `write_through_selective`)
```

---

## The bandwidth ceiling

A 64 GB/s PCIe link moves ~8 GB of KV per second of sustained traffic. Per [[kv-cache-memory-formula]] (ch-03), Llama-3-70B GQA-128 at 8k tokens is ~1 GB of KV. So PCIe saturates at ~8 full-prefix L2 rehydrations per second.

In practice this is fine because:
- Most matches are **partial** (only some blocks need rehydrate)
- Async fetch **overlaps** with prefill on the suffix
- L1-resident hot prefixes account for the majority of traffic

But if every request needs a full L2 rehydrate, HiCache is **bandwidth-bound** and the GPU sits idle waiting for transfers. The right metric to watch is **PCIe utilization**; if it's > 50 % of link peak, you're approaching the cliff.

---

## Configuration knobs

```bash
python -m sglang.launch_server \
    --enable-hicache \
    --hicache-ratio 2.0                          # L2 size / L1 KV size
    --hicache-write-policy write_through_selective \
    --hicache-storage-backend mooncake \
    --hicache-storage-prefetch-threshold 16384 \  # only L3-fetch prefixes > 16k tokens
    --hicache-io-backend kernel                   # 'kernel' (cuMemcpyAsync) vs 'direct' (GPUDirect)
```

| Write policy | Behavior |
|--------------|----------|
| `write_through` | Every finished request's KV is copied to L2 + L3 |
| `write_through_selective` | Only prefixes likely to be reused (heuristic on prefix depth + frequency) |
| `write_back` | Buffer writes; flush on eviction pressure |

`write_through_selective` is the production default — naive `write_through` saturates PCIe.

---

## Per-tier hit-rate metric (the only metric that matters)

```
GET /metrics
...
sglang_radix_cache_hit_rate_l1{...} 0.62
sglang_radix_cache_hit_rate_l2{...} 0.34    # over L1 misses
sglang_radix_cache_hit_rate_l3{...} 0.18    # over L1+L2 misses
sglang_hicache_pcie_utilization{...} 0.41
```

Healthy production target: L1 60–80 %, L2 20–60 % of L1 misses, L3 10–40 % of L1+L2 misses, end-to-end any-tier hit 60–95 %.

If L2 hit rate over L1 misses is < 10 %, the L2 tier is wasted overhead — disable HiCache and use bigger L1 instead.

---

## Why this matters

Long-context inference is increasingly cache management. A 100k-token document is too expensive to recompute for every question (multi-second prefill), but also too big to keep N copies of in GPU HBM. The hierarchical answer:

- Keep N=1 copy in L1 while users are actively querying.
- Spill to L2 when the user pauses; rehydrate in ~50 ms when they come back.
- Spill further to L3 for cross-session / cross-node persistence; rehydrate in ~200 ms.

For an agent / RAG / multi-QA workload, the alternative is paying full prefill each time — typically 5–50× slower than a hot HiCache hit.

---

## Connections

- [[excerpts/sglang-radixattention]] — the L1 data structure HiCache extends.
- [[excerpts/mooncake]] (ch-09) — one of the L3 backends; KV-centric distributed cache.
- [[excerpts/vllm-kv-offloading]] (ch-16) — vLLM's CPU-tier analogue; no L3 equivalent.
- [[ch-03]] — KV-cache memory formula; the per-prefix cost HiCache is amortizing.
