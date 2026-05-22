<!-- scope: SGLang HiCache hierarchical KV caching across GPU memory, host memory, and distributed storage
     deps: [[sglang-radixattention]], [[kv-cache-memory-formula]]
     see-also: [[vllm-kv-offloading-connector]], [[mooncake]], [[cachegen]]
-->

# SGLang HiCache — Hierarchical KV Caching
- **Core Insight:** HiCache extends SGLang's RadixAttention from GPU-only prefix reuse into a multi-tier KV hierarchy: GPU memory as L1, CPU memory as L2, and distributed storage as L3.
- **Guideline:** Use hierarchical KV caching for workloads with repeated long prefixes, multi-QA over shared documents, or agent loops where recomputing prefill dominates latency.
- **Authors:** SGLang team
- **Year:** 2025–2026
- **URL:** https://docs.sglang.io/docs/advanced_features/hicache_design
- **Relevant topics:** SGLang, HiCache, RadixAttention, hierarchical KV cache, prefix reuse, long-context serving

## Abstract
SGLang HiCache is a hierarchical KV-cache design. RadixAttention stores reusable prefix KV cache in GPU memory; HiCache extends that idea to host memory and distributed storage. Its metadata structure, HiRadixTree, tracks spans of token sequences and where their KV blocks live across local GPU, local CPU, and shared L3 storage backends such as Mooncake, 3FS, NIXL, and AIBrix KVCache.

## Key Contributions
- Turns prefix caching into a three-level cache hierarchy modeled after CPU cache design.
- Extends RadixAttention's radix tree into HiRadixTree with storage-location metadata.
- Supports local matching, prefetch, and write-back workflows.
- Integrates with distributed cache/storage backends for cluster-level KV reuse.
- Targets long-context and multi-QA workloads where many requests share document prefixes.

## Key Figures/Tables to Study
- HiCache overall architecture: L1 GPU, L2 host, L3 distributed storage.
- HiRadixTree metadata diagram: token spans mapped to KV locations.
- Workflow diagram: local match -> L3 prefetch -> GPU computation -> write-back.

## Technical Details

### Cache tiers
| Tier | Storage | Role |
|------|---------|------|
| L1 | GPU memory | hottest KV, fastest decode/prefix reuse |
| L2 | host memory | larger local cache for reusable prefixes |
| L3 | distributed storage/cache | cluster-wide cache sharing and persistence |

### Workflow
When a request arrives, HiCache first matches its token prefix in local HiRadixTree metadata. Missing spans can be prefetched from L3. After prefill computes new KV, the system may write it back to L2 or L3 so future requests can reuse it.

### Why it matters
Long-context inference increasingly looks like cache management. A 100k-token document prefix is too expensive to recompute for every question, but storing every prefix only in GPU HBM is also too expensive. HiCache is the SGLang answer to that trade-off.

## Connections
- [[sglang-radixattention]] — local GPU prefix cache foundation.
- [[vllm-kv-offloading-connector]] — vLLM's CPU-backed connector is a comparable production feature.
- [[mooncake]] — one of the distributed cache/storage backends referenced by HiCache.
- [[cachegen]] — compression/offload approach for reducing KV storage cost.
