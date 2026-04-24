# Excerpt: KV Cache Implementation — From Scratch to Production

**Source:** [[raschka-kv-cache|Raschka, "Understanding and Coding the KV Cache in LLMs from Scratch" (2025)]]

---

## Why the KV Cache Exists

During autoregressive generation, each new token must attend to all previous tokens. Without caching:

- Generating token 2: compute K/V for token 1, then attend
- Generating token 3: **recompute** K/V for tokens 1 and 2, then attend
- Generating token $n$: recompute K/V for all $n-1$ previous tokens

Total K/V computations across $n$ tokens: $1 + 2 + \ldots + (n-1) = O(n^2)$.

With caching, each step computes K/V only for the new token ($O(1)$) and retrieves cached K/V for previous tokens. Total: $O(n)$.

## The Naive Implementation

```python
def forward(self, x, use_cache=False):
    keys_new = self.W_key(x)
    values_new = self.W_value(x)
    queries = self.W_query(x)

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
```

**Critical implementation detail:** The cache must be reset between generation calls. Without reset, "queries of a new prompt [attend] to stale keys left over from the previous sequence" — a context-leaking bug.

## Position Tracking

The cache introduces a position tracking requirement. Without cache, position IDs are simply `[0, 1, ..., seq_len-1]`. With cache, new tokens must receive position IDs relative to their actual position in the full sequence:

```python
if use_cache:
    pos_ids = torch.arange(
        self.current_pos, self.current_pos + seq_len,
        device=in_idx.device, dtype=torch.long
    )
    self.current_pos += seq_len
```

Without this, new tokens would be treated as starting from position 0, breaking RoPE and other position-dependent mechanisms.

## Generation Loop with Cache

The key optimization: after the initial prefill, **only feed the single new token** at each step:

```python
def generate_text_simple_cached(model, idx, max_new_tokens):
    model.reset_kv_cache()
    with torch.no_grad():
        logits = model(idx[:, -ctx_len:], use_cache=True)  # Prefill

    for _ in range(max_new_tokens):
        next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_idx], dim=1)
        logits = model(next_idx, use_cache=True)  # Only new token!
    return idx
```

## Performance: 5.3x Speedup

On Mac Mini M4 (CPU), 124M parameter model, 200 tokens:

| Configuration | Time | Speedup |
|---|---|---|
| Without cache | ~17.5s | 1x |
| With cache (naive) | ~3.3s | 5.3x |
| Pre-allocated cache | ~2.8s | 6.25x |
| Pre-allocated + compiled | ~2.4s | 7.3x |

## The `torch.cat` vs Pre-allocation Tradeoff

**`torch.cat` (naive):** Each step allocates a new tensor, copies all cached data, appends the new token. $O(n)$ copy per step, $O(n^2)$ total copies across $n$ steps. Flexible — only uses memory proportional to actual sequence length.

**Pre-allocation:** Reserve a buffer for the maximum sequence length upfront:

```python
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim))
# Write into slices:
cache_k[:, :, current_pos, :] = new_key
```

$O(1)$ per step (slice assignment), but wastes memory when actual sequence is shorter than maximum. For Llama 3 with 128K context, pre-allocating the full cache per request wastes enormous memory for short requests.

**This is exactly the problem PagedAttention solves** — block-based allocation combines the memory efficiency of dynamic allocation with the performance of pre-allocation.

## GPU vs CPU Caching Behavior

CPUs benefit most from KV caching (5-10x improvement). GPUs show smaller relative gains for small models because:

1. GPU parallel compute already amortizes some redundant computation
2. Device transfer overhead (CPU→GPU) can dominate for small models
3. The cache benefit scales with model size and sequence length

For large models (7B+) on GPUs, the KV cache is essential — without it, generation would be impractically slow.

---

**Key takeaway:** The KV cache is a textbook space-time tradeoff. The $O(n)$ memory cost buys $O(n^2) \to O(n)$ computation savings. But the *management* of that memory — allocation strategy, position tracking, cache reset — introduces engineering complexity that scales with production serving requirements.
