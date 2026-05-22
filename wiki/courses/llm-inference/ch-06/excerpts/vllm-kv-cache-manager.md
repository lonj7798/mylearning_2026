---
chapter: ch-06
course: llm-inference
phase: read
excerpt_of: "vLLM V1 KV cache manager — vllm/v1/core/kv_cache_manager.py"
source_url: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
created_at: "2026-05-21"
---

# Excerpt: vLLM V1 KV cache manager — paged allocation in production code

**Authors:** vLLM project
**Year:** 2024-present (V1 engine)
**URLs:** https://docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_manager/ ; https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
**Raw-data source:** [[raw-data/vllm-kv-cache-manager]]

---

## Module scope

The manager is the single owner of the GPU KV-block pool. It exposes a small, contract-stable API to the scheduler:

```python
class KVCacheManager:
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
    ) -> Optional[KVCacheBlocks]: ...

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[KVCacheBlocks, int]: ...   # (blocks, num_cached_tokens)

    def free(self, request: Request) -> None: ...
    def free_block_hashes(self, request: Request) -> None: ...
    def reset_prefix_cache(self) -> None: ...
```

The scheduler never manipulates the block pool directly. Every allocation, COW, and eviction happens through these four methods.

---

## Data structures

```python
# Simplified — see vllm/v1/core/block_pool.py and kv_cache_manager.py
class BlockPool:
    block_size: int                       # default 16
    num_gpu_blocks: int                   # = (free HBM after weights+activations) / per-block bytes
    free_block_queue: deque[int]          # FIFO of free physical block IDs
    cached_block_hash_to_block: dict[BlockHash, KVCacheBlock]  # prefix cache
    null_block: KVCacheBlock              # reserved for "this slot is dummy"

class KVCacheBlock:
    block_id: int
    ref_cnt: int                          # > 1 ⇒ shared (prefix cache or COW source)
    block_hash: Optional[BlockHash]       # set when cacheable
```

The free queue is the active pool; the hash dict is the prefix cache. Total physical blocks `num_gpu_blocks` is computed once at engine start from `gpu_memory_utilization` and the per-block byte cost.

---

## `allocate_slots` — the load-bearing method

```python
def allocate_slots(self, request, num_new_tokens):
    # 1. How many new blocks do we need?
    total_blocks_needed = ceil(
        (request.num_computed_tokens + num_new_tokens) / self.block_size
    )
    new_blocks_needed = total_blocks_needed - len(request.kv_cache_blocks[0])
    if new_blocks_needed <= 0:
        return KVCacheBlocks([])                      # fits in current last block

    # 2. Is there room?
    if new_blocks_needed > len(self.block_pool.free_block_queue):
        return None                                   # caller will preempt or defer

    # 3. Pull blocks from the free pool.
    new_blocks = [self.block_pool.free_block_queue.popleft()
                  for _ in range(new_blocks_needed)]
    request.kv_cache_blocks[0].extend(new_blocks)
    return KVCacheBlocks(new_blocks)
```

Three subtleties:

- **Return value of `None` is the failure protocol.** The scheduler reacts by preempting a lower-priority request via `free(victim)` and retrying.
- **Allocation is block-granular.** A request extending by 1 token usually gets 0 new blocks (it fits in the current partial block) — but every 16 tokens triggers a 1-block allocation.
- **Block IDs are integers**, never pointers. The actual K/V data lives in the global pool tensors `K_pool`, `V_pool`. The block ID is just an index.

---

## `get_computed_blocks` — the prefix-cache integration point

```python
def get_computed_blocks(self, request):
    """Look up blocks already cached for this request's prompt prefix."""
    if not self.enable_caching:
        return KVCacheBlocks([]), 0

    block_hashes = hash_request_tokens(self.block_size, request)
    cached = []
    for h in block_hashes:
        if h in self.block_pool.cached_block_hash_to_block:
            blk = self.block_pool.cached_block_hash_to_block[h]
            blk.ref_cnt += 1                          # pin against eviction
            cached.append(blk)
        else:
            break                                     # prefixes are contiguous
    return KVCacheBlocks(cached), len(cached) * self.block_size
```

Two design choices to flag:

- **Hashes are token-exact.** A block hash incorporates the tokens in that block *and the parent block's hash* — so block 5's hash depends on the actual content of blocks 0-4. This is the chain-hash design that makes prefix matching equivalent to "is the entire prefix bit-identical?"
- **Prefix breaks on first miss.** The scan stops at the first uncached block hash, because a chained hash means downstream blocks can't be valid without their predecessor.

The integer `num_cached_tokens` returned here is what the scheduler stores as `request.num_computed_tokens` — and what makes "chunked prefill" with prefix-cache hit silently skip the cached work entirely.

---

## `free` — LIFO order matters

```python
def free(self, request):
    blocks = request.kv_cache_blocks[0]
    # Reverse so the tail (most recently allocated) is enqueued first.
    for blk in reversed(blocks):
        blk.ref_cnt -= 1
        if blk.ref_cnt == 0:
            if blk.block_hash is not None:
                # Cached: leave in the hash dict; will be evicted only under pressure.
                self.block_pool.add_to_free_queue(blk)  # but stays cacheable
            else:
                self.block_pool.free_block_queue.append(blk.block_id)
```

Why LIFO: the tail of a sequence is the *most divergent* part (the actually-sampled tokens); the head is the most-shareable part (the system prompt). Freeing tail-first means the shared head sits in the cache longest, which is what you want for hit rate.

---

## Eviction — implicit through the cache

The manager does not run an eviction *thread*; eviction happens lazily when `allocate_slots` finds the free pool empty. The block pool then walks `cached_block_hash_to_block` in LRU order and evicts cached-but-unreferenced blocks back to the free pool until the allocation succeeds (or until no evictable cached blocks remain, in which case `allocate_slots` returns `None` and the scheduler must preempt a live request).

```python
def _try_evict(self, n_blocks_needed):
    while len(self.free_block_queue) < n_blocks_needed:
        blk = self.lru_cache_head()
        if blk is None or blk.ref_cnt > 0:
            return False                              # no evictable cache entries
        del self.cached_block_hash_to_block[blk.block_hash]
        blk.block_hash = None
        self.free_block_queue.append(blk.block_id)
    return True
```

The result: free-pool exhaustion → prefix-cache eviction (free, since the data is recomputable) → preemption (expensive, loses partial KV). The two-tier escalation is what keeps tail latency bounded under memory pressure.

---

## Production knobs

| Flag | Effect |
|---|---|
| `--block-size 16` | Block granularity. Almost never changed. |
| `--gpu-memory-utilization 0.9` | Sets `num_gpu_blocks` indirectly (larger ⇒ more KV, less activation headroom). |
| `--enable-prefix-caching` (default True since V1) | Turns on the hash dict; off ⇒ blocks never re-enter cache after free. |
| `--num-gpu-blocks-override N` | Manual override for the block pool size; useful for benchmarking. |
| `--swap-space 4` | Reserve N GiB of CPU memory for swap-out preemption (vs recompute). |

The first two are the only ones most operators ever touch.

---

## What this code does *not* do

The manager is intentionally narrow. It does not:

- Run the attention kernel. (Workers do.)
- Decide *which* requests to admit or preempt. (Scheduler does.)
- Manage the radix-tree variant of prefix caching. (SGLang's path, ch-07.)
- Cross-GPU coordination of paged blocks. (`KVConnector` does for disaggregated serving, ch-09.)

This isolation is what lets vLLM swap in features like sliding-window cache, encoder cache, and speculative-decoding `K+1`-slot allocation by writing tiny adapters around the same four-method contract.

---

## Connections

- [[excerpts/pagedattention]] — the algorithm this code implements.
- [[excerpts/vllm-scheduler]] (ch-05) — the caller. The scheduler is the only client.
- [[ch-06]] — parent synthesis.
- Forward to [[ch-07]] — `get_computed_blocks` is the entry point for vLLM's APC prefix caching, contrasted with SGLang's RadixAttention.
- Forward to [[ch-16]] — full vLLM internals tour.
