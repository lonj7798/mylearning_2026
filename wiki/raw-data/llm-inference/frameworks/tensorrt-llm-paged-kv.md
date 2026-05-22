<!-- scope: TensorRT-LLM paged KV cache and in-flight batching source page
     deps: [[tensorrt-llm]]
     see-also: [[vllm-kv-cache-manager]]
-->

# TensorRT-LLM Paged KV Cache
- **Core Insight:** TensorRT-LLM combines in-flight batching with paged KV-cache blocks so request lifetimes do not require contiguous cache allocations.
- **Guideline:** Prefer paged KV mode for dynamic serving workloads unless a static/contiguous setup is simpler and sufficient.
- **Authors:** NVIDIA
- **Year:** 2023-present
- **URL:** https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html and https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-management.html
- **Relevant topics:** paged attention, in-flight batching, cache blocks, reuse, offload, prioritized eviction

## Abstract
TensorRT-LLM documents two KV-cache layouts, contiguous and paged. The paged layout divides per-layer KV memory into reusable logical blocks and lets the scheduler manage requests whose prompt and generation lengths vary. NVIDIA's KV-cache management docs also describe internal pools, block events, reuse across requests, offloading, and prioritized eviction.

## Key Contributions
- Makes paged KV cache a first-class TensorRT-LLM runtime feature.
- Couples cache layout with in-flight batching so requests can enter and exit during generation.
- Supports KV reuse across requests, improving repeated-prefix and multi-turn workloads.
- Provides event mechanisms for cache state observability and external systems.
- Includes offload and prioritized eviction tools for increasing effective reuse.

## Key Figures/Tables to Study
- Paged Attention, IFB, and Request Scheduling page: comparison of contiguous and paged KV cache.
- KV Cache Management page: hierarchy of managers, pools, blocks, and events.
- `KvCacheConfig`: user-facing configuration object.
- Runtime source around batch manager and KV cache manager in TensorRT-LLM repository.

## Technical Details
Public/serving entrypoint:
- Users configure KV behavior through TensorRT-LLM runtime/executor settings, commonly with `KvCacheConfig`.
- Paged KV affects serving through executor/Triton deployments rather than a standalone user API.

Cache approach:
- Each Transformer layer has KV storage.
- In paged mode, storage is divided into blocks/pages that can be assigned to different request positions.
- Block tables or equivalent metadata map logical sequence positions to physical blocks.
- Reuse keeps computed cache blocks available for future requests.
- Eviction and offload move or remove blocks when capacity is needed.

Relevant code/docs:
- Paged attention and IFB: https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
- KV cache management: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-management.html
- Feature source doc: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md
- KV feature source doc: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md

Strengths:
- Reduces memory fragmentation for variable-length serving workloads.
- Directly integrated with NVIDIA's optimized kernels and runtime scheduler.
- Advanced cache controls are useful for production workloads with reuse locality.

Limitations:
- Most useful on NVIDIA GPU deployments that can use TensorRT-LLM fully.
- Internal implementation details are less Python-accessible than vLLM/SGLang.
- Requires careful configuration to balance cache reuse, offload cost, and active batch capacity.

## Connections
- Equivalent design family to [[vllm-kv-cache-manager]] and [[pagedattention]].
- Scheduler side connects to [[continuous-batching]] and TensorRT-LLM in-flight batching.
- Compare with [[sglang-radixattention]] for prefix reuse policy.
