<!-- scope: practitioner explanation of KV cache mechanics, memory growth, and serving implications
     deps: [[attention-is-all-you-need]], [[kv-cache-memory-formula]]
     see-also: [[pagedattention]], [[vllm-kv-cache-manager]], [[sglang-radixattention]]
-->

# KV Cache Explained
- **Core Insight:** KV cache makes autoregressive decoding fast by storing each layer's past key/value tensors, but it turns context length and concurrency into a live GPU-memory problem.
- **Guideline:** Teach KV cache before serving systems: most inference design choices, including PagedAttention, prefix caching, MQA/GQA, offload, and eviction, are responses to KV memory growth.
- **Authors:** Practitioner synthesis from Transformer attention, MQA/GQA, vLLM, SGLang, and Hugging Face inference documentation
- **Year:** 2017–2026
- **URL:** https://huggingface.co/docs/transformers/main/en/cache_explanation ; https://arxiv.org/abs/2309.06180 ; https://docs.vllm.ai/
- **Relevant topics:** KV cache, autoregressive decoding, memory planning, prefill, decode, prefix caching, paging

## Abstract
During autoregressive inference, a decoder-only Transformer repeatedly predicts the next token. Without caching, every decode step would recompute keys and values for the full prefix at every layer. KV cache stores those tensors once and appends only the new token's K/V at each step. This changes decode compute from repeated full-prefix recomputation to one-token incremental attention, but the cache grows with active tokens, layers, KV heads, head dimension, and dtype.

## Key Contributions
- Gives the intuitive explanation for why decode can reuse prior K/V states.
- Connects prefill and decode: prefill creates a large initial cache, decode appends one token per active sequence.
- Explains why long contexts and many concurrent requests compete for the same memory pool.
- Motivates MQA/GQA as architectural reductions in KV head count.
- Motivates PagedAttention and prefix caching as serving-system memory management.

## Key Figures/Tables to Study
- Hugging Face cache explanation diagrams: past K/V tensors reused at each step.
- vLLM PagedAttention figures: block tables and non-contiguous KV allocation.
- SGLang RadixAttention diagrams: prefix-tree reuse for shared prompts.
- KV memory formula card: `2 * layers * batch * seq_len * kv_heads * head_dim * bytes`.

## Technical Details

### Prefill
The model processes the whole prompt in parallel and writes K/V tensors for every prompt token at every layer. This stage is usually compute-heavy and has a large one-time attention workload.

### Decode
For each new token, the model computes only the new query/key/value, appends K/V to cache, and attends the new query over cached keys and values. Decode is often memory-bandwidth-bound because it repeatedly reads weights and growing K/V cache while doing relatively little compute per token.

### Serving consequences
- Larger context means larger per-request cache.
- Higher concurrency means more active caches.
- MQA/GQA reduce cache by reducing KV heads.
- Prefix caching avoids recomputing shared prompts.
- Paging/offload/eviction are needed when GPU memory is the bottleneck.

## Connections
- [[kv-cache-memory-formula]] — exact memory budget.
- [[pagedattention]] — block-based KV allocation.
- [[vllm-kv-cache-manager]] — vLLM implementation of block allocation and prefix caching.
- [[sglang-radixattention]] / [[sglang-hicache]] — prefix tree and hierarchical cache reuse.
