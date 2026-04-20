<!-- scope: KV cache implementation and optimization
     deps: [[ch-03]]
     see-also: [[raschka-attention-variants]], [[alammar-illustrated-gpt2]]
-->

# Understanding and Coding the KV Cache in LLMs from Scratch

- **Core Insight:** KV cache turns O(n^2) generation into O(n) per token but introduces memory management challenges.
- **Guideline:** Architecture choices (GQA, MLA, SWA) determine KV cache size — design attention with serving in mind.

- **Author:** Sebastian Raschka, PhD
- **URL:** https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- **Relevant chapters:** Inference optimization, attention mechanism, autoregressive generation

## Summary
A hands-on guide to implementing KV cache in LLMs from scratch, explaining what the cache does, why it matters, how to implement it in PyTorch, and optimization strategies including pre-allocation and sliding windows. Includes benchmarks showing 5.3x speedup on a 124M parameter model.

## Key Content

### The Problem

During autoregressive text generation, LLMs process tokens sequentially. Without caching:
- Generating token 2: recomputes K/V for token 1
- Generating token 3: recomputes K/V for tokens 1 and 2
- This creates O(n^2) computational complexity

### The Solution

Store previously computed key and value vectors and retrieve them, reducing per-step complexity to O(n).

### Step-by-Step: Without vs With Cache

**Without cache** (text: "Time flies fast"):
1. Process "Time" -> output "flies"
2. Reprocess "Time flies" -> output "fast"
3. Reprocess "Time flies fast" -> output next token

**With cache:**
1. Process "Time", cache its K/V -> output "flies"
2. Retrieve cached K/V for "Time", only compute new K/V for "flies" -> output "fast"
3. Retrieve cached K/V, only compute new K/V for "fast" -> output next token

### Implementation

**Registering cache buffers:**
```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

**Forward pass with caching:**
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

**Cache must be reset between separate generation calls** to prevent "queries of a new prompt attending to stale keys left over from the previous sequence."

**Position tracking:** Track token position to ensure new queries align correctly:
```python
if use_cache:
    pos_ids = torch.arange(
        self.current_pos, self.current_pos + seq_len,
        device=in_idx.device, dtype=torch.long
    )
    self.current_pos += seq_len
```

**Generation with cache** — only feed the new token:
```python
def generate_text_simple_cached(model, idx, max_new_tokens):
    model.reset_kv_cache()
    with torch.no_grad():
        logits = model(idx[:, -ctx_len:], use_cache=True)
    
    for _ in range(max_new_tokens):
        next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_idx], dim=1)
        logits = model(next_idx, use_cache=True)  # Only new token!
    return idx
```

### Performance Results

On Mac Mini M4 chip (CPU), 124M parameter model, 200 tokens:

| Configuration | Time |
|---|---|
| Without cache | ~17.5 seconds |
| With cache | ~3.3 seconds |
| **Speedup** | **~5.3x** |

### Optimization Strategies

**Problem with naive implementation:** Repeatedly using `torch.cat` causes memory fragmentation and reallocation overhead.

**Optimization 1: Pre-allocate memory**
```python
max_seq_len = 1024
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim))
```
Write new values into tensor slices rather than creating new tensors.

**Optimization 2: Sliding window truncation**
```python
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
```
Trade-off: Model loses context beyond window size.

**Optimized performance:**

| Implementation | Time (Mac Mini M4 CPU) |
|---|---|
| Basic cache | ~3.3s |
| Pre-allocated | ~2.8s |
| Pre-allocated + compiled | ~2.4s |

### Practical Considerations

For large models with massive context windows (Qwen3: 41k, Llama 3: 131k), pre-allocating all positions consumes ~8GB extra memory. The `torch.cat` approach may be more practical despite slight performance trade-offs.

**GPU vs CPU:** CPUs benefit most from KV cache (5-10x improvements). GPUs show diminished gains for small models due to device transfer overhead.

## Notable Insights
- KV cache is a textbook space-time tradeoff: O(n) extra memory for O(n) computation savings per step. The cumulative savings during autoregressive generation are dramatic.
- Position tracking is a subtle but critical implementation detail — without it, new tokens get treated as if they start from position 0.
- The `torch.cat` vs pre-allocation tradeoff is practical engineering: pre-allocation is faster but wastes memory for short sequences; concatenation is flexible but fragments memory.
- GPU gains from KV cache are diminished compared to CPU gains — this is because GPU's parallel compute already amortizes some of the redundancy, while CPU bottlenecks are more severely affected.
- Cache must be reset between generations — a common bug in naive implementations leads to "context leaking" between separate prompts.
