<!-- scope: SGLang RadixAttention prefix KV-cache reuse source page
     deps: [[sglang]]
     see-also: [[sglang-scheduler]], [[vllm-kv-cache-manager]]
-->

# SGLang RadixAttention
- **Core Insight:** RadixAttention stores reusable KV-cache prefixes in a radix tree so later requests can share the longest matching prefix instead of recomputing it.
- **Guideline:** Use RadixAttention-aware serving for workloads with repeated system prompts, few-shot prefixes, multi-turn state, or agent templates.
- **Authors:** SGLang project
- **Year:** 2023-present
- **URL:** https://arxiv.org/abs/2312.07104 and https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py
- **Relevant topics:** prefix caching, KV reuse, radix tree, cache-aware scheduling, HiCache

## Abstract
RadixAttention is SGLang's signature KV-cache reuse mechanism. Rather than viewing each request as isolated, it keeps completed prefix KV states in a radix tree keyed by token sequences. A new request can match the longest cached prefix, reuse those KV blocks, and only prefill the suffix that was not cached.

## Key Contributions
- Makes prefix reuse a runtime primitive rather than an application-level prompt cache.
- Uses a radix tree to share prefixes at token granularity across related requests.
- Coordinates cache matching with scheduling and memory management so reused KV blocks remain available while active.
- Supports cache eviction when memory pressure grows.
- Forms the basis for HiCache, which extends prefix caching beyond GPU memory.

## Key Figures/Tables to Study
- SGLang paper RadixAttention diagrams: tree-shaped KV reuse across program calls.
- `radix_cache.py`: node structure, match, insert, split, evict, and lock/reference logic.
- HiCache docs: how GPU-only RadixAttention generalizes to hierarchical tiers.
- Scheduler calls into radix cache: where prefix hits change prefill work.

## Technical Details
Public/serving entrypoint:
- Radix cache is part of the SGLang runtime; users normally enable/disable or tune it via server flags such as `--disable-radix-cache`.
- Requests arrive through OpenAI-compatible or native endpoints and are matched by tokenized prefix.

Cache approach:
- Token prefixes form paths in a radix tree.
- A request queries the tree for the longest prefix that is already cached.
- Matched KV blocks are locked/referenced while the request uses them.
- Only the unmatched suffix is prefetched into new KV memory.
- Eviction removes less useful cached nodes while avoiding active references.

Relevant code/docs:
- Paper: https://arxiv.org/abs/2312.07104
- Radix cache source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py
- Memory cache package: https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache
- HiCache design: https://docs.sglang.ai/advanced_features/hicache_design.html
- HiCache best practices: https://docs.sglang.ai/advanced_features/hicache_best_practices.html

Strengths:
- Particularly effective for structured programs with repeated prefixes.
- Reduces both prefill compute and time-to-first-token when cache hits are high.
- Tree representation naturally shares common prefixes across many requests.

Limitations:
- Requires token-exact prefix overlap; semantically similar prompts do not help.
- Cache hit rate depends on workload shape and eviction policy.
- Memory retained for reusable prefixes competes with active request capacity.

## Connections
- Core cache mechanism behind [[sglang]] and input signal for [[sglang-scheduler]].
- Compare against vLLM block-hash prefix caching in [[vllm-kv-cache-manager]].
- Hierarchical extension connects to host/offload KV-cache topics.
