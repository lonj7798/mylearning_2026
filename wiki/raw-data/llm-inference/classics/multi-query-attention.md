<!-- scope: Multi-query attention for lower decode-time KV bandwidth
     deps: [[attention-is-all-you-need]], [[kv-cache-memory-formula]]
     see-also: [[grouped-query-attention]], [[batching-for-inference]]
-->

# Fast Transformer Decoding: One Write-Head is All You Need
- **Core Insight:** Share one set of key/value heads across many query heads to drastically reduce KV-cache size and memory bandwidth during autoregressive decoding.
- **Guideline:** Use MQA when decode throughput and memory bandwidth dominate, but verify quality because full KV-head sharing can be too aggressive for some models.
- **Authors:** Noam Shazeer
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1911.02150
- **Relevant topics:** multi-query attention, KV cache, memory bandwidth, fast decoding

## Abstract
Multi-query attention (MQA) keeps multiple query heads but uses a single shared key head and value head. During incremental decoding, this makes the cached keys and values much smaller than standard multi-head attention. The paper targets the memory-bandwidth bottleneck of Transformer decoding.

## Key Contributions
- Introduced MQA as "one write-head" attention for faster autoregressive decode.
- Separated the number of query heads from the number of key/value heads.
- Reduced memory traffic for reading cached keys and values at every decode step.
- Preserved much of multi-head attention's expressiveness through many query projections.
- Motivated later grouped-query attention as a quality/speed compromise.

## Key Figures/Tables to Study
- **Architecture diagrams/equations:** Compare MHA vs MQA projections.
- **Incremental decoding analysis:** Focus on cached `K,V` reads.
- **Speed results:** Look for wall-clock improvements at decode time.
- **Quality comparisons:** Identify tasks where MQA tracks MHA and where it degrades.

## Technical Details
Standard MHA has `H_q = H_kv` heads. MQA uses:

```text
H_q = many query heads
H_kv = 1 key/value head
```

The KV-cache memory becomes proportional to `H_kv`, not `H_q`. For a model with 32 query heads, MQA can reduce KV tensors by about 32x compared with full MHA, ignoring implementation overheads.

Decode speed improves because every new token attends over all previous cached keys and values in every layer. Reading fewer KV bytes is often more important than reducing arithmetic on modern GPUs for small-batch decode.

## Connections
- [[grouped-query-attention]]: generalizes MQA with more than one KV head to recover quality.
- [[kv-cache-memory-formula]]: MQA changes the formula by setting `n_kv_heads = 1`.
- [[batching-for-inference]]: smaller KV cache permits larger batches and higher throughput.
- [[prefill-vs-decode]]: MQA mainly helps decode; prefill still computes full prompt attention.
