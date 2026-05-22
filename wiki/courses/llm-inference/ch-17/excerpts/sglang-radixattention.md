---
chapter: ch-17
course: llm-inference
phase: read
excerpt_of: "Efficiently Programming Large Language Models using SGLang — RadixAttention section"
source_url: https://arxiv.org/abs/2312.07104
created_at: "2026-05-21"
---

# Excerpt: RadixAttention — the data structure

**Authors:** Lianmin Zheng et al.
**Year:** 2023–present
**URLs:** https://arxiv.org/abs/2312.07104 (paper) / https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py (source)
**Raw-data source:** [[raw-data/sglang-radixattention]]

---

## The tree

A **radix tree** (compressed prefix trie) keyed by token sequences. Each edge owns one or more KV-cache blocks (typically 16 tokens per block, matching the paged-attention block size).

```
root
 └── ["sys-prompt tokens (200)", "doc tokens (5000)"]  ← 5.2k tokens cached on one edge
      ├── ["question 1 tokens (40)"]
      │     └── ["answer 1 tokens (256)"]
      ├── ["question 2 tokens (35)"]
      │     └── ["answer 2 tokens (192)"]
      └── ["question 3 tokens (60)"]
            └── ["answer 3 tokens (218)"]
```

A request asking "question 4" against the same document walks `root → sys+doc edge → branches`, sees no match for question 4, and only needs to prefill `40` new tokens — saving 5.2k tokens of prefill compute.

---

## Node structure (simplified from `radix_cache.py`)

```python
class TreeNode:
    children: dict[token_chunk, "TreeNode"]   # edge labels
    parent: "TreeNode"
    key: list[int]                            # token IDs on this edge
    value: list[int]                          # KV-block indices for those tokens
    lock_ref: int                             # ref count from live requests
    last_access_time: float                   # for LRU eviction

class RadixCache:
    root_node: TreeNode
    evictable_size: int                       # bytes of unpinned KV
    protected_size: int                       # bytes of pinned KV (lock_ref > 0)

    def match_prefix(self, key: list[int]) -> tuple[list[int], TreeNode]:
        # Walk from root, greedy longest match.
        # Returns: (KV-block indices for matched prefix, deepest matched node)
        ...

    def insert(self, key: list[int], value: list[int]) -> int:
        # Add a new path; split existing edges on partial match.
        # Returns: number of tokens that were already cached (the prefix-hit length)
        ...

    def evict(self, num_tokens: int) -> None:
        # LRU-evict leaf nodes with lock_ref == 0 until num_tokens freed.
        ...

    def inc_lock_ref(self, node: TreeNode) -> None: ...
    def dec_lock_ref(self, node: TreeNode) -> None: ...
```

---

## The match → admit → release lifecycle

```
1. Request R arrives with token_ids = [t_0, ..., t_n].

2. Scheduler calls cache.match_prefix(token_ids).
   → returns (matched_kv_blocks, node)
   → matched_kv_blocks is a list of physical block IDs the request can reuse for free.
   → node is the deepest tree node along the match path.

3. Scheduler calls cache.inc_lock_ref(node).
   → Pins node and all its ancestors; they cannot be evicted until R finishes.

4. Allocate new KV blocks for the unmatched suffix (n - len(matched)) tokens.
   Run prefill on suffix only.

5. Decode loop: generate tokens one at a time, appending to R's allocated KV blocks.

6. On finish, scheduler calls cache.insert(R's full token sequence, R's KV blocks).
   → A new path is added to the tree (or the existing edge extended).
   → Future requests can now match R's full output as a cached prefix.

7. cache.dec_lock_ref(node).  R's prefix path is now eligible for LRU eviction
   when memory pressure rises.
```

---

## Eviction policy

LRU among leaf nodes with `lock_ref == 0`. Internal nodes are never directly evicted — but when a leaf is freed, the parent is checked, and if it now has zero children and zero ref count, it too becomes evictable.

The protected invariant: **active requests always have their full prefix path resident**. Eviction only touches "cold" prefixes — paths last used by a request that has finished and not yet been revisited.

---

## What it costs

| Cost | Magnitude |
|------|-----------|
| Tree walk per `match_prefix` | O(n) in prompt length, but very cache-friendly traversal (~µs for 8k tokens) |
| Insert + split per finished request | O(k) where k is the divergence position; usually negligible vs forward pass |
| Memory overhead | ~64 bytes / tree node + edge label storage (~negligible vs KV blocks themselves) |
| Eviction sweep | O(log N) with a heap of `(last_access_time, node)` pairs |

The whole machinery adds <1 % serving overhead and pays itself back many times over when hit rate > 30 %.

---

## Hit-rate numbers in practice

From the SGLang paper Section 4 + production reports:

| Workload | Hit rate | Throughput vs no-cache |
|----------|---------:|-----------------------:|
| Multi-turn chat, 2k system prompt | 60–80 % | 1.3–1.5× |
| Few-shot (8 shots, 64 queries) | 80–95 % | 2–3× |
| Tree-of-thought (16-way fork) | 90–95 % | 3–5× |
| Multi-QA over 100k document | 95–99 % | 4–6× |
| Arbitrary user prompts | 0–20 % | ~1× (parity) |

---

## Connections

- [[excerpts/sglang-architecture]] — where this data structure sits in the stack.
- [[excerpts/sglang-scheduler]] — how `match_prefix` results drive admission.
- [[excerpts/vllm-kv-cache-manager]] (ch-16) — the alternative: block-hash prefix cache.
- [[ch-07]] — algorithm-level treatment of prefix caching including this structure.
