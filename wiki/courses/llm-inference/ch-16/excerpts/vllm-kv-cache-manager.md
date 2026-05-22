---
chapter: ch-16
course: llm-inference
phase: read
excerpt_of: "vLLM V1 KV Cache Manager: vllm/v1/core/kv_cache_manager.py + block_pool.py"
source_url: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
created_at: "2026-05-21"
---

# Excerpt: vLLM V1 KV Cache Manager

**Source files:**
- `vllm/v1/core/kv_cache_manager.py` (top-level manager)
- `vllm/v1/core/block_pool.py` (free-list mechanics)
- `vllm/v1/core/single_type_kv_cache_manager.py` (per-group specialization)
- `vllm/v1/core/kv_cache_utils.py` (hash + utility functions)

**Raw-data source:** [[raw-data/vllm-kv-cache-manager]]

---

## The interface (KVCacheManager)

```python
class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len, enable_caching, ...):
        self.block_size = kv_cache_config.block_size  # default 16 tokens
        self.num_blocks = kv_cache_config.num_blocks
        self.block_pool = BlockPool(self.num_blocks)
        self.enable_caching = enable_caching   # APC on/off
        self.cached_block_hash_to_id: dict[BlockHash, int] = {}  # APC index

    def allocate_slots(self, request, num_new_tokens, computed_blocks=None) -> KVCacheBlocks:
        """Allocate enough blocks for num_new_tokens; reuse computed_blocks if APC hit."""

    def has_capacity_for(self, request, num_new_tokens) -> bool:
        """Pre-check: is there enough free capacity?"""

    def get_computed_blocks(self, request) -> list[int]:
        """APC lookup — return prefix of blocks already in cache."""

    def free(self, request):
        """Release a request's blocks back to the pool (or to APC cache if enabled)."""

    def reset_prefix_cache(self):
        """Wipe APC entirely. Useful for benchmarks."""
```

This is the surface the scheduler sees. Everything below is implementation detail.

---

## BlockPool — the free-list

```python
class BlockPool:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_block_queue: deque[int] = deque(range(num_blocks))
        self.ref_counts: dict[int, int] = {bid: 0 for bid in range(num_blocks)}
        self.cached_blocks: dict[int, BlockHash] = {}  # block_id → its content hash

    def allocate(self, n: int) -> list[int]:
        if len(self.free_block_queue) < n:
            # Try to evict LRU cached blocks
            self._evict_cached(n - len(self.free_block_queue))
        return [self.free_block_queue.popleft() for _ in range(n)]

    def free(self, block_ids: list[int]):
        for bid in block_ids:
            self.ref_counts[bid] -= 1
            if self.ref_counts[bid] == 0:
                if bid in self.cached_blocks:
                    # Keep in APC cache, append to LRU end (recently used)
                    self.free_block_queue.append(bid)
                else:
                    self.free_block_queue.append(bid)
```

Reference counting handles copy-on-write for parallel sampling (`n=4` samples on one prompt → 4 refs to each prompt block; on divergence, one ref drops, others keep the original).

LRU eviction is implicit in queue order: blocks at the head are oldest (will be evicted first). Recently-used blocks (just freed) go to the tail.

---

## Block tables — per-request KV addressing

```python
@dataclass
class KVCacheBlocks:
    blocks: list[int]   # block IDs for this request
    new_block_ids: list[int]   # subset that's newly allocated this step
    computed_block_ids: list[int]  # subset that came from APC

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def append_token_ids(self, token_ids):
        # Trigger hash computation when a block fills up
        pass
```

The attention kernel receives `request.block_table` (the `blocks` list) and uses it as an indirection table:

```
For token at logical position p:
  block_idx = p // block_size
  in_block_offset = p % block_size
  physical_block = block_table[block_idx]
  kv = kv_cache_pool[physical_block, in_block_offset]
```

This is the [[pagedattention]] (ch-06) indirection. The block table is `O(num_tokens / block_size)` integers per request — small, fits in registers for kernel addressing.

---

## APC — the hash chain

When `enable_caching=True`, each full block (16 tokens) gets a content hash that includes the previous block's hash:

```python
def hash_block(prev_hash: BlockHash, token_ids: list[int], extra_info: dict) -> BlockHash:
    """
    extra_info includes: model, tokenizer hash, LoRA adapter ID, etc.
    This ensures cached blocks from one model aren't matched against another's.
    """
    return hash((prev_hash, tuple(token_ids), tuple(sorted(extra_info.items()))))
```

The hash chain means: if block N has hash H_N, then H_N depends on the entire token sequence from position 0 to position (N+1)·block_size − 1. So matching H_N means the *entire* prefix matches — guaranteed correctness on prefix reuse.

### Cache lookup

```python
def get_computed_blocks(self, request) -> list[int]:
    if not self.enable_caching:
        return []
    computed = []
    prev_hash = NULL_HASH
    for chunk in chunks_of(request.prompt_token_ids, self.block_size):
        if len(chunk) < self.block_size:
            break  # partial block — don't cache
        h = hash_block(prev_hash, chunk, extra_info=request.extra_hash_info)
        if h in self.cached_block_hash_to_id:
            block_id = self.cached_block_hash_to_id[h]
            computed.append(block_id)
            self.block_pool.ref_counts[block_id] += 1  # bump ref
            prev_hash = h
        else:
            break  # first miss → stop
    return computed
```

Linear walk from prefix start; first miss terminates. So if the system prompt is 200 tokens (12 full blocks), and a request has that exact system prompt as prefix, we get 12 free blocks — 192 tokens of "free" prefill.

### Cache write-back

When a block fills (16 tokens get appended), it's hashed and registered in `cached_block_hash_to_id`. So the cache populates *automatically* as blocks fill during generation.

---

## Eviction

LRU within the free queue. When `allocate(n)` is called and there aren't enough free blocks, the cache manager evicts the oldest cached blocks first:

```python
def _evict_cached(self, num_needed):
    candidates = [bid for bid in self.free_block_queue
                  if self.ref_counts[bid] == 0 and bid in self.cached_blocks]
    for bid in candidates[:num_needed]:
        h = self.cached_blocks[bid]
        del self.cached_block_hash_to_id[h]
        del self.cached_blocks[bid]
        # bid stays in free_block_queue; will be reallocated next time
```

Important: in-use blocks (ref_count > 0) are never evicted — that would break running requests. Eviction only touches *free + cached* blocks.

---

## Multiple KV cache groups

Some models (Mistral 7B with sliding-window attention, Gemma with hybrid attention) have heterogeneous KV requirements. V1 supports this via `KVCacheGroup`:

```python
class KVCacheGroup:
    name: str  # e.g. "full_attention" or "sliding_window"
    block_size: int
    num_blocks: int
    block_pool: BlockPool
```

The top-level `KVCacheManager` dispatches per-group requests to per-group `SingleTypeKVCacheManager` instances. This was added in late 2024 to support Gemma-3 and Mistral 7B's mixed-attention modes.

---

## Pitfalls

- **APC hash collisions**. Hashing uses Python's `hash()` which is keyed (PYTHONHASHSEED). Across worker processes (multiproc executor), hash seeds match because they fork from a common parent. But if you spawn workers via `spawn` rather than `fork`, hashes mismatch → APC entries are per-process. Use `--worker-multiproc-method fork`.
- **Block size choice**. Default 16. Larger blocks (32, 64) → less per-block overhead but more wasted bytes per request (each request has avg ~block_size/2 wasted tokens). Smaller blocks (8) → more overhead but better packing. Don't change unless you've profiled.
- **`enable_caching=True` adds ~5% overhead per step** due to hash computation. The TTFT savings on prefix hits dominate this, but for workloads with no shared prefixes, APC is slight loss.
- **Free vs evict in metrics**. `vllm_cached_blocks_total` is the APC pool size; `vllm_free_blocks_total` is the free queue. APC hits show up as `vllm_prefix_cache_hits_total` going up.

---

## Connections

- [[excerpts/vllm-scheduler]] — calls `allocate_slots()` and `has_capacity_for()` every step.
- [[pagedattention]] — the attention kernel that consumes the block tables.
- [[excerpts/vllm-production-knobs]] — `--enable-prefix-caching` and `--block-size` tuning.
- [[ch-16]] — parent chapter.
