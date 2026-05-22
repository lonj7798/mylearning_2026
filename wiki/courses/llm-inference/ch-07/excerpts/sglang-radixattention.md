---
chapter: ch-07
course: llm-inference
phase: read
excerpt_of: "SGLang RadixAttention — radix-tree prefix KV cache (Zheng et al. 2023, SGLang)"
source_url: https://arxiv.org/abs/2312.07104
created_at: "2026-05-21"
---

# Excerpt: RadixAttention — tree-keyed KV prefix cache

**Authors:** Lianmin Zheng et al. (SGLang team)
**Year:** 2023-present
**URL:** https://arxiv.org/abs/2312.07104 ; https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py
**Raw-data source:** [[raw-data/sglang-radixattention]]

---

## The thesis

> "RadixAttention is a novel technique that maintains a least-recently-used (LRU) cache of all KV cache entries inside a radix tree. It enables automatic KV cache reuse across multiple generation calls, even when they have partially overlapping prompts." (SGLang paper §3.3)

vLLM's APC matches block-hashes; RadixAttention matches token sequences in a tree. The tree representation is what makes branching workloads (agents, tree-of-thought, multi-call programs) share KV beyond the deepest single ancestor.

---

## The data structure

```python
# python/sglang/srt/mem_cache/radix_cache.py — simplified

class RadixCacheNode:
    children: dict[int, RadixCacheNode]   # first token of edge -> child
    edge_tokens: list[int]                # token span on the edge from parent
    kv_blocks:  list[int]                 # physical KV block IDs for this span
    ref_cnt:    int                       # > 0 ⇒ in use; cannot evict
    parent:     RadixCacheNode

class RadixCache:
    root: RadixCacheNode
    lru:  OrderedDict[RadixCacheNode, None]   # for eviction

    def match_prefix(self, tokens) -> tuple[list[int], int]:
        """Walk the tree, returning the longest matching prefix's KV blocks."""
        ...
    def insert(self, tokens, kv_blocks):
        """Insert a new sequence (possibly splitting an existing node)."""
        ...
    def evict(self, num_blocks_to_free) -> list[int]:
        """LRU-evict unreferenced nodes; return freed block IDs."""
        ...
```

Three operations:

- `match_prefix(tokens)` — walks from root, follows the child whose first edge token equals `tokens[i]`, continues until a mismatch. Returns the matched KV blocks and the matched length.
- `insert(tokens, kv_blocks)` — appends a new child to the deepest matched node, splitting an existing node if the match ends mid-edge.
- `evict(num_blocks_to_free)` — walks the LRU list for nodes with `ref_cnt == 0`, frees their blocks, removes them from the tree.

---

## The mid-edge split (the operation that gives token-level granularity)

vLLM's APC matches at 16-token block boundaries. RadixAttention matches at arbitrary token positions because edges can be split:

```text
Before:
    root
    └── edge "[hello world how are you today]"  (6 tokens, blocks B0..B0)
                kv_blocks = [B0]
                ref_cnt = 0

A new request arrives with tokens "hello world how is your day".
  Match walks the edge until token index 3 ("how"), then diverges
  ("are" ≠ "is"). Split:

After:
    root
    └── edge "[hello world how]"   (3 tokens, blocks B0:0..3)
            ├── edge "[are you today]"  (3 tokens, blocks B0:3..6)
            │      (old continuation)
            └── edge "[is your day]"    (3 tokens, blocks B_new)
                   (new request's continuation)
```

Physical KV blocks are *not* moved; only the metadata changes. The first half of B0 now belongs to the parent node (shared); the second half belongs only to the original sequence's continuation. The new request shares 3 tokens that APC's 16-token granularity would have rounded down to 0.

---

## Reference counting and eviction safety

Each node has `ref_cnt`. When a request is admitted with a matched prefix, every node along the matched chain has its `ref_cnt += 1`. When the request completes, the chain is `ref_cnt -= 1`. Only nodes with `ref_cnt == 0` are evictable.

This is what makes RadixAttention safe under concurrent admission/eviction:

- An admitted request's matched blocks cannot be evicted out from under it.
- An evicted node is guaranteed to have no live consumer.
- The LRU ordering only affects priority among the safe-to-evict set.

The same machinery exists in vLLM's APC — `KVCacheBlock.ref_cnt` — but in the radix variant it applies to whole spans (variable length), not fixed-size blocks.

---

## Scheduler integration

The SGLang scheduler ([[excerpts/sglang-scheduler]]) calls `match_prefix` during admission and uses the matched length to set `request.cached_tokens`:

```python
def schedule_new_request(req):
    matched_kv, matched_len = radix.match_prefix(req.input_ids)
    req.cached_tokens = matched_len
    req.prefill_to_do = req.input_len - matched_len
    radix.lock(matched_kv)                      # ref_cnt += 1 for matched nodes
    # remaining prefill is now matched_len shorter
```

Two consequences:

1. **TTFT scales with suffix length, not full prompt length.** A 1000-token prompt with a 900-token cached prefix takes the time of a 100-token prefill.
2. **Cache-aware admission.** The scheduler can sort the waiting queue by `cached_tokens / input_len` ratio, admitting high-hit requests first to free token budget for others sooner. This is the second-order win on top of the cache itself.

---

## Hit rates from the paper (§6)

The SGLang paper reports hit rates on several program-style workloads:

| Workload | RadixAttention hit rate | Speedup vs vLLM-without-cache |
|---|---|---|
| Few-shot in-context learning (LLaMA-7B) | 91 % | 5.1× throughput |
| Multi-turn dialogue (LLaMA-7B) | 71 % | 2.5× throughput |
| ReAct agent loop (LLaMA-7B) | 89 % | 3.7× throughput |
| Tree-of-thought (LLaMA-7B) | 95 % | 6.4× throughput |
| JSON decoding pipeline | 88 % | 4.3× throughput |

The largest gains are on workloads where the *tree-shaped* prefix matching matters — tree-of-thought (where every branch shares the root) and agent loops (where every iteration shares the persistent scratchpad). vLLM APC catches the first two; the tree structure catches the rest.

---

## What RadixAttention does *not* solve

- **Tokenization changes invalidate everything.** A new chat template or special-token revision purges the tree.
- **Cache memory still competes with active request memory.** A workload with no prefix sharing makes the radix tree useless (and adds a small bookkeeping overhead).
- **Mid-span splits are O(L) walks.** Acceptable at typical prompt lengths, but at 100k tokens this needs care.
- **Single-GPU only by default.** For cross-GPU or cross-node sharing, you need [[sglang-hicache]] (next excerpt) or vLLM's KVConnector.

---

## Connections

- [[excerpts/sglang]] — overall SGLang architecture.
- [[excerpts/sglang-scheduler]] — scheduler integration with the cache.
- [[excerpts/sglang-hicache]] — multi-tier extension.
- [[excerpts/vllm-kv-cache-manager]] (ch-06) — vLLM APC, the flat-hash alternative.
- [[ch-07]] — parent synthesis.
- Forward to [[ch-17]] — SGLang full internals tour.
