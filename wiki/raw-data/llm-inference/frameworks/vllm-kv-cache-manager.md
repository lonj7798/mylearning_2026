<!-- scope: vLLM V1 KV cache manager, block allocation, and prefix-cache interface
     deps: [[vllm]], [[pagedattention]]
     see-also: [[vllm-scheduler]], [[sglang-radixattention]]
-->

# vLLM KV Cache Manager
- **Core Insight:** vLLM's KV cache manager hides block-pool details behind allocation, prefix lookup, caching, and eviction APIs used by the scheduler.
- **Guideline:** Treat KV blocks as the unit of serving capacity; tune block size, cache utilization, and prefix caching based on workload locality.
- **Authors:** vLLM project
- **Year:** 2024-present V1 engine
- **URL:** https://docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_manager/ and https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
- **Relevant topics:** PagedAttention, block tables, prefix caching, eviction, cache events, sliding-window cache

## Abstract
The V1 KV cache manager is the interface between scheduling policy and the physical/logical KV-cache block pool. It returns `KVCacheBlocks` to the scheduler, maps request IDs to allocated blocks, tracks computed blocks for prefix reuse, frees blocks when requests finish or are preempted, and emits cache events for observability or connectors.

## Key Contributions
- Encapsulates KV-cache block data structures so scheduler code does not manipulate block pools directly.
- Supports allocation of new slots and reuse of already-computed prefix blocks.
- Provides cache, free, evict, reset, and metrics APIs around prefix cache behavior.
- Handles multiple KV-cache groups, preparing for heterogeneous cache specs.
- Coordinates specialized managers such as full-attention and sliding-window cache managers.

## Key Figures/Tables to Study
- `KVCacheBlocks`: allocation result passed from cache manager to scheduler/model runner.
- `allocate_slots(...)`: decides which new blocks a request can receive.
- `get_computed_blocks(...)`: prefix-cache lookup path.
- `free(...)`: reverse-order freeing so tail blocks are evicted first when caching is enabled.
- `single_type_kv_cache_manager.py`: full-attention and sliding-window specialization.

## Technical Details
Public/serving entrypoint:
- Users configure cache behavior through vLLM server and engine flags, not by calling this module.
- The scheduler calls the manager during each scheduling step.

Cache approach:
- Each request's KV history is represented as block IDs per KV-cache group.
- Block tables let attention kernels address non-contiguous physical KV memory.
- Prefix caching hashes completed blocks and reuses blocks when a new request shares a token prefix.
- Freeing a request can either release blocks or leave reusable computed blocks in the prefix cache.
- Eviction removes cached blocks when the pool needs capacity for active work.

Relevant code/docs:
- KV manager source: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
- Block pool source: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py
- Single-type managers: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py
- KV utilities: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py
- API docs: https://docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_manager/

Strengths:
- Makes memory capacity a first-class scheduling resource.
- Prefix caching can greatly reduce TTFT for repeated system prompts, RAG templates, agents, and multi-turn workloads.
- Block abstraction supports paging, preemption, and future heterogeneous cache layouts.

Limitations:
- Prefix reuse depends on exact tokenized-prefix alignment and hashing granularity.
- More cache features mean more interaction effects with speculative decoding, sliding windows, and disaggregated KV.
- Misconfigured cache utilization can cause preemption or low concurrency.

## Connections
- Concrete implementation of [[pagedattention]] in [[vllm]].
- Scheduler coupling is covered in [[vllm-scheduler]].
- Compare SGLang's tree-based prefix reuse in [[sglang-radixattention]].
