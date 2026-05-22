---
chapter: ch-07
course: llm-inference
phase: read
excerpt_of: "SGLang HiCache — hierarchical KV caching across GPU/CPU/distributed"
source_url: https://docs.sglang.io/docs/advanced_features/hicache_design
created_at: "2026-05-21"
---

# Excerpt: SGLang HiCache — three-tier prefix KV cache

**Authors:** SGLang team
**Year:** 2025-2026
**URL:** https://docs.sglang.io/docs/advanced_features/hicache_design
**Raw-data source:** [[raw-data/sglang-hicache]]

---

## The motivation

RadixAttention keeps prefix KV blocks in GPU HBM. That works for tens of GB of warm prefixes, but fails for two production scenarios:

- **Long documents.** A 100k-token RAG document has ~10 GB of KV at GQA-8 / 70B scale. Keeping it permanently resident on a 80 GB GPU starves active requests.
- **Fleet-shared prefixes.** A 50-GB few-shot exemplar set shared across 32 GPUs in a serving cluster is wasteful to replicate 32×.

HiCache solves both with a CPU-cache-hierarchy analogue.

---

## The three tiers

| Tier | Storage | Latency | Bandwidth | Capacity |
|---|---|---|---|---|
| L1 | GPU HBM | sub-µs | TB/s | tens of GB |
| L2 | CPU memory (DDR5) | ~10 µs | 30-50 GB/s (PCIe Gen5) | hundreds of GB |
| L3 | Distributed (Mooncake, 3FS, NIXL, AIBrix KVCache) | sub-ms via RDMA | 10-100 GB/s per node | unbounded |

Each tier has different cost characteristics. L1 is fastest but tiny. L2 is large and per-host but bound by PCIe. L3 is unbounded but bound by the network.

---

## The HiRadixTree metadata

The radix tree from RadixAttention gains location metadata per node:

```python
class HiRadixNode:
    edge_tokens: list[int]
    kv_locations: dict[int, KVLocation]   # layer -> location
    children:    dict[int, HiRadixNode]
    ref_cnt:     int

class KVLocation:
    tier: Literal["L1", "L2", "L3"]
    handle: Any                            # GPU block ID / CPU pointer / object key
```

A node's KV may be split across tiers (e.g., the most recent layers in L1, the deeper layers in L2). The scheduler walks the tree the same way as RadixAttention; the difference is what happens after a match.

---

## The cache workflow

```
On request admission with token sequence T:
  1. matched_node, matched_len = hiradix.match_prefix(T)
  2. for each layer's KV in matched_node:
       if location.tier == "L1":  ready, no transfer
       elif location.tier == "L2":  schedule async PCIe copy → L1
       elif location.tier == "L3":  schedule async RDMA fetch → L2 → L1
  3. suffix_prefill = len(T) - matched_len
  4. run suffix_prefill in mixed batch (ch-05) while transfers complete
  5. after generation, write back tail blocks to L1 (and optionally L2/L3)
```

The overlap between transfer and compute is the key: a 100k-token L2-resident prefix takes ~3 seconds to PCIe into L1 at 30 GB/s, but the suffix prefill takes some of that time anyway, so end-user TTFT is roughly `max(transfer, suffix_prefill)` — typically dominated by the transfer.

---

## Promotion/demotion policy

HiCache uses an **inclusive** cache: L1 contents are a subset of L2; L2 a subset of L3 (when L3 is enabled). Promotion (L2→L1, L3→L2) happens on demand. Demotion (L1→L2 when L1 fills, L2→L3 when L2 fills) is LRU-driven.

The capacity ratios in production are typically:

- L2 = 10–20× L1 (CPU RAM is cheap relative to HBM)
- L3 = 10–100× L2 (cluster storage is cheap relative to per-host RAM)

---

## Distributed L3 backends

HiCache treats L3 as a pluggable backend. Documented integrations:

| Backend | What it is |
|---|---|
| Mooncake | KV-cache-centric serving architecture, RDMA backbone (ch-09) |
| 3FS | Distributed file system optimized for AI workloads |
| NIXL | NVIDIA inference KV-cache transfer library |
| AIBrix KVCache | Kubernetes-native KV cache service |

The interface is small: `get(key)`, `put(key, blocks)`, `delete(key)`. The HiRadixTree decides what to fetch and store; the backend handles physical placement and replication.

---

## When HiCache pays off

The break-even is:

```
HiCache wins iff   (prefix reuse rate) × (cached prefix length) > (transfer cost from L2/L3)
```

For typical numbers (PCIe Gen5 ~30 GB/s, KV ~1-3 MB/token for 70B models):

- A 10k-token prefix takes ~300 ms to PCIe; the recompute cost (prefill) is ~1.5 seconds. → HiCache wins if reused twice.
- A 100k-token prefix takes ~3 s to PCIe; recompute is ~15 s. → HiCache wins if reused twice.

For very short prefixes (<2k tokens), the overhead of HiCache management may exceed the prefill savings. The SGLang docs recommend HiCache for workloads with shared contexts >= 4k tokens.

---

## Operational notes

From the HiCache "best practices" doc:

- **Size L2 generously.** CPU RAM is the cheap tier; 5–10× L1 is a safe default.
- **Disable L3 for single-host deployments.** The network round-trip is wasted overhead.
- **Monitor L1/L2/L3 hit rates separately.** If L1 hit rate is high (>80 %), HiCache isn't doing much; if L2/L3 hits dominate, you're getting the benefit.
- **Tokenize stably.** Same as RadixAttention — HiCache requires bit-identical tokens for matching across tiers.

---

## Connections

- [[excerpts/sglang-radixattention]] — the GPU-only foundation.
- [[excerpts/sglang-scheduler]] — the consumer of HiCache matches.
- [[ch-07]] — parent synthesis.
- Forward to [[ch-09]] — Mooncake / disaggregation as the L3 backend.
- Forward to [[ch-17]] — full SGLang internals.
