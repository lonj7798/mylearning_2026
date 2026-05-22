<!-- scope: CacheGen paper on compressing and streaming reusable KV cache
     deps: [[pagedattention]]
     see-also: [[mooncake]], [[h2o]], [[snapkv]]
-->

# CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving
- **Core Insight:** Reusing long-context KV cache is only useful if fetching the cache is faster than recomputing it; compress and stream KV tensors to reduce transfer delay.
- **Guideline:** For shared long contexts, optimize KV-cache transport format and bandwidth adaptation, not just GPU compute.
- **Authors:** Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.07240
- **Relevant topics:** KV-cache reuse, compression, streaming, long context, bandwidth adaptation

## Abstract
CacheGen targets applications that reuse a long context across inputs. Recomputing the context prefill is expensive, but fetching a full KV cache over the network can also dominate latency. CacheGen encodes KV cache into compact bitstreams using tensor distribution properties, then adapts compression levels to network bandwidth and quality requirements.

## Key Contributions
- Defines KV-cache loading as a bottleneck for context reuse.
- Designs a custom KV tensor encoder with low decode overhead.
- Adapts compression level by cache segment and bandwidth condition.
- Trades off fetching compressed cache versus recomputing parts on the fly.
- Reports multi-x reductions in KV size and context-loading delay with little quality impact.

## Key Figures/Tables to Study
- KV-cache size and network-delay motivation figures.
- Encoder/decoder pipeline figure.
- Bandwidth-adaptive compression policy.
- Quality and latency comparisons versus recomputation or uncompressed transfer.

## Technical Details
The system assumes a reusable context whose KV cache can be stored and loaded for later requests. Because KV size scales with layers, KV heads, head dimension, token count, and dtype, a large shared context can be too large to move naively.

CacheGen compresses the cache and streams it so generation can begin after enough state is available. It can choose stronger compression for less sensitive segments or under lower bandwidth, and can recompute some KV when fetching would be slower or lower quality.

## Connections
- Complements [[pagedattention]]: paging manages GPU residency, CacheGen manages external transfer format.
- Related to [[mooncake]]'s disaggregated KV-cache storage.
- Different from eviction methods such as [[h2o]] and [[snapkv]], which reduce the active cache.

## Notes
CacheGen is most relevant when a shared context is reused often enough to justify storing KV state.
