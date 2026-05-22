<!-- scope: Practical KV-cache memory formula for decoder-only inference
     deps: [[attention-is-all-you-need]], [[multi-query-attention]], [[grouped-query-attention]]
     see-also: [[prefill-vs-decode]], [[batching-for-inference]]
-->

# KV Cache Memory Formula
- **Core Insight:** KV-cache memory scales linearly with batch size, sequence length, layers, KV heads, head dimension, and bytes per element.
- **Guideline:** Estimate KV cache before serving: `2 * layers * batch * seq_len * n_kv_heads * head_dim * bytes`, then add allocator/block overhead.
- **Authors:** Derived from Transformer attention; serving implications emphasized by Shazeer (MQA), Ainslie et al. (GQA), and Kwon et al. (PagedAttention/vLLM)
- **Year:** 2017-2023
- **URL:** https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/1911.02150 ; https://arxiv.org/abs/2305.13245 ; https://arxiv.org/abs/2309.06180
- **Relevant topics:** KV cache, memory planning, MQA, GQA, batch size, context length

## Abstract
Autoregressive Transformer inference caches each layer's key and value tensors for prior tokens. This avoids recomputing the prompt at every decode step, but the cache often becomes the limiting memory object in serving. The formula is simple and should be memorized.

## Key Contributions
- Turns abstract attention internals into a concrete GPU memory budget.
- Explains why MQA/GQA matter: they reduce `n_kv_heads`.
- Explains why long context and continuous batching compete for the same memory pool.
- Separates KV memory from parameter memory and activation scratch memory.
- Provides the basis for PagedAttention-style block allocation.

## Key Figures/Tables to Study
- **Attention equations in Transformer/MQA/GQA papers:** Identify what is cached.
- **PagedAttention Figure 1:** KV cache can consume a large fraction of GPU memory.
- **Model configs:** `num_hidden_layers`, `num_key_value_heads`, `hidden_size`, `num_attention_heads`.
- **Serving memory dashboards:** Compare parameter bytes vs live KV bytes.

## Technical Details
For decoder-only inference:

```text
KV bytes = 2 * L * B * S * H_kv * D_head * bytes_per_elem
```

Where:
- `2` = key and value tensors
- `L` = number of layers
- `B` = active sequences or batch size
- `S` = cached tokens per sequence
- `H_kv` = number of key/value heads
- `D_head` = head dimension
- `bytes_per_elem` = 2 for fp16/bf16, 1 for int8/fp8-style cache formats

Example: `L=32`, `B=16`, `S=4096`, `H_kv=8`, `D_head=128`, `bf16=2 bytes` gives about `8.6 GB` of raw KV cache. Add allocator fragmentation, block tables, padding, CUDA graph reserves, and temporary attention workspace.

## Connections
- [[multi-query-attention]]: sets `H_kv=1`, greatly shrinking cache.
- [[grouped-query-attention]]: uses intermediate `H_kv` to trade quality for cache efficiency.
- [[prefill-vs-decode]]: prefill writes many KV entries at once; decode appends one per sequence per step.
- [[batching-for-inference]]: maximum batch is frequently a KV-cache memory problem.
