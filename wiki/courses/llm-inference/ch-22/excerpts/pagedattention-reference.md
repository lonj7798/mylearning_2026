---
chapter: ch-22
course: llm-inference
phase: read
excerpt_of: "PagedAttention reference implementation skeleton (for Option 1)"
source_url: https://arxiv.org/abs/2309.06180
created_at: "2026-05-21"
---

# Excerpt: PagedAttention reference implementation skeleton

**Source:** [[pagedattention]] (Kwon et al. 2023, arXiv 2309.06180)
**Raw-data:** [[raw-data/pagedattention]]

---

## What this excerpt is

A starting skeleton for the **Option 1** (PagedAttention) capstone. Not a complete implementation — it's the scaffolding you complete during Step 2 (pure-PyTorch reference) and Step 3 (model hook) of the reproduction workflow.

The skeleton covers the **memory metric** path (the cleanly reproducible 96 % utilisation headline). The throughput path requires a Triton kernel and is the optional next step.

---

## Block-pool data structure

```python
import torch
from dataclasses import dataclass, field

BLOCK_SIZE = 16  # tokens per block; per [[pagedattention]] §4, this is the canonical value

@dataclass
class PagedKVPool:
    """Global pool of fixed-size KV blocks. Per [[pagedattention]] Fig. 3."""
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_blocks: int
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    def __post_init__(self):
        # Per-layer K and V pools: (n_blocks, BLOCK_SIZE, n_kv_heads, head_dim)
        self.K = torch.zeros(self.n_layers, self.n_blocks, BLOCK_SIZE,
                              self.n_kv_heads, self.head_dim,
                              dtype=self.dtype, device=self.device)
        self.V = torch.zeros_like(self.K)
        # Free list
        self.free_blocks: list[int] = list(range(self.n_blocks))
        # Allocation: seq_id -> list of physical block ids
        self.block_tables: dict[int, list[int]] = {}
        # Track per-seq filled token count (for utilisation accounting)
        self.filled_tokens: dict[int, int] = {}

    def allocate_for(self, seq_id: int, n_tokens: int) -> list[int]:
        n_blocks_needed = (n_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        if len(self.free_blocks) < n_blocks_needed:
            raise MemoryError(f"OOB: need {n_blocks_needed} blocks, have {len(self.free_blocks)}")
        ids = [self.free_blocks.pop() for _ in range(n_blocks_needed)]
        self.block_tables[seq_id] = ids
        self.filled_tokens[seq_id] = n_tokens
        return ids

    def append_token(self, seq_id: int, layer: int, k_vec, v_vec):
        """k_vec, v_vec: shape (n_kv_heads, head_dim). Add one token."""
        n_filled = self.filled_tokens[seq_id]
        block_idx_local = n_filled // BLOCK_SIZE
        slot = n_filled % BLOCK_SIZE
        # Allocate new block if at boundary
        if slot == 0 and block_idx_local >= len(self.block_tables[seq_id]):
            new_block = self.free_blocks.pop()
            self.block_tables[seq_id].append(new_block)
        physical = self.block_tables[seq_id][block_idx_local]
        self.K[layer, physical, slot] = k_vec
        self.V[layer, physical, slot] = v_vec
        if layer == self.n_layers - 1:
            # Bump filled count once per token (after last layer's write)
            self.filled_tokens[seq_id] = n_filled + 1

    def gather_kv(self, seq_id: int, layer: int):
        """Return contiguous K, V for attention computation. Slow reference."""
        blocks = self.block_tables[seq_id]
        n_filled = self.filled_tokens[seq_id]
        K_chunks = [self.K[layer, b] for b in blocks]
        V_chunks = [self.V[layer, b] for b in blocks]
        K = torch.cat(K_chunks, dim=0)[:n_filled]   # (n_tokens, n_kv_heads, head_dim)
        V = torch.cat(V_chunks, dim=0)[:n_filled]
        return K, V

    def free(self, seq_id: int):
        for b in self.block_tables.pop(seq_id):
            self.free_blocks.append(b)
        self.filled_tokens.pop(seq_id)

    def utilisation(self) -> float:
        """Fraction of allocated KV slots that hold real tokens (per [[pagedattention]] §6)."""
        if not self.block_tables:
            return 0.0
        allocated_slots = sum(len(bs) * BLOCK_SIZE for bs in self.block_tables.values())
        filled_slots = sum(self.filled_tokens.values())
        return filled_slots / allocated_slots
```

The `utilisation()` method is the function whose output is your headline number.

---

## Verify the bounded-waste property

The paper's claim: internal fragmentation is bounded by `BLOCK_SIZE - 1` per sequence (at most one partially-filled last block). Test it:

```python
pool = PagedKVPool(n_layers=32, n_kv_heads=8, head_dim=128, n_blocks=2000)

# Simulate 100 sequences of varying lengths
import random
random.seed(42)
seq_lens = [random.randint(50, 2000) for _ in range(100)]

for seq_id, L in enumerate(seq_lens):
    pool.allocate_for(seq_id, L)

util = pool.utilisation()
print(f"Utilisation: {util:.4f}")
print(f"Total filled tokens: {sum(pool.filled_tokens.values())}")
print(f"Allocated slots: {sum(len(b)*BLOCK_SIZE for b in pool.block_tables.values())}")

# Bound check
total_filled = sum(seq_lens)
max_waste = len(seq_lens) * (BLOCK_SIZE - 1)
expected_min_util = total_filled / (total_filled + max_waste)
assert util >= expected_min_util - 1e-6, "Waste bound violated!"
```

For typical mixed-length workloads, this should print utilisation around 0.96–0.98 — matching the paper's headline.

---

## Hook into a HuggingFace Llama-3-8B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

class PagedDynamicCache(DynamicCache):
    """HF Cache wrapper that stores into a PagedKVPool. Subclasses 4.36+ Cache API."""
    def __init__(self, pool: PagedKVPool, seq_id: int):
        super().__init__()
        self.pool = pool
        self.seq_id = seq_id

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states / value_states: (batch=1, n_kv_heads, n_new_tokens, head_dim)
        # Append each new token to the pool
        kt = key_states.squeeze(0).transpose(0, 1)   # (n_new_tokens, n_kv_heads, head_dim)
        vt = value_states.squeeze(0).transpose(0, 1)
        if self.seq_id not in self.pool.block_tables:
            self.pool.allocate_for(self.seq_id, kt.shape[0])
        for i in range(kt.shape[0]):
            self.pool.append_token(self.seq_id, layer_idx, kt[i], vt[i])
        # Return everything cached so far (for attention to attend over)
        K, V = self.pool.gather_kv(self.seq_id, layer_idx)
        return K.transpose(0, 1).unsqueeze(0), V.transpose(0, 1).unsqueeze(0)

    def get_seq_length(self, layer_idx=0):
        return self.pool.filled_tokens.get(self.seq_id, 0)

# Smoke test: identical greedy generation
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
)
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Explain attention in one sentence."
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

pool = PagedKVPool(n_layers=32, n_kv_heads=8, head_dim=128, n_blocks=2000)
paged_cache = PagedDynamicCache(pool, seq_id=0)
out_paged = model.generate(ids, max_new_tokens=64, do_sample=False,
                            past_key_values=paged_cache)

out_baseline = model.generate(ids, max_new_tokens=64, do_sample=False)

assert torch.equal(out_paged, out_baseline), "Paged generation diverged from baseline!"
print("LOSSLESS:", tok.decode(out_paged[0]))
```

If `torch.equal` fails, the most likely cause is the layer-write ordering in `append_token` — verify that `filled_tokens[seq_id]` is bumped exactly once per token (after the last layer's write).

---

## Memory utilisation under realistic workload

The real measurement: replay ShareGPT through this paged cache and report utilisation over time.

```python
import json
sharegpt = json.load(open("sharegpt.json"))[:1000]

pool = PagedKVPool(n_layers=32, n_kv_heads=8, head_dim=128, n_blocks=4000)
utilisations = []
for i, conv in enumerate(sharegpt):
    prompt = conv["conversations"][0]["value"]
    n_tokens = len(tok(prompt).input_ids) + 100  # +100 simulated output
    if len(pool.free_blocks) < (n_tokens // BLOCK_SIZE + 1):
        # Free oldest seq to make room (simple LRU)
        oldest = min(pool.block_tables.keys())
        pool.free(oldest)
    pool.allocate_for(i, n_tokens)
    pool.filled_tokens[i] = n_tokens
    if i % 50 == 0:
        utilisations.append(pool.utilisation())

print(f"Mean utilisation: {sum(utilisations) / len(utilisations):.4f}")
# Target: ~0.96 per [[pagedattention]] §6
```

---

## What this reference does NOT do

- **It does not run fast.** The `torch.cat` in `gather_kv` is O(blocks) and rebuilds the contiguous K, V every step. A real implementation uses a paged attention CUDA kernel that reads K, V through block-table indirection directly. Your memo can cite that limit explicitly.
- **It does not handle copy-on-write.** For parallel sampling / beam search, sibling sequences should share blocks until divergence. The reference allocates fresh blocks per sequence. Adding CoW is ~50 lines (reference-count blocks; on write, if refcount>1, allocate a new block and copy).
- **It does not handle preemption.** A real serving system evicts blocks under memory pressure. The reference just OOMs.

For the capstone memo, document each missing piece — they're the "what the paper didn't tell me" entries.

---

## Connections

- [[ch-06]] — the chapter where PagedAttention is introduced; re-read before starting.
- [[vllm-kv-cache-manager]] — the production-grade implementation; use as *verification*, not as a starting point.
- [[excerpts/debugging-tree]] — when `torch.equal` fails or utilisation looks wrong, the diagnostic tree.
