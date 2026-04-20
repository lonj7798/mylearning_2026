<!-- scope: MQA — single KV head per layer for fast decoding
     deps: [[attention-is-all-you-need]]
     see-also: [[gqa]], [[paged-attention]]
-->

# Fast Transformer Decoding: One Write-Head is All You Need
- **Core Insight:** A single shared KV head per layer cuts KV cache by H-fold with only minor quality loss, because decoding is memory-bandwidth-bound.
- **Guideline:** If inference latency dominates your cost and you can retrain, MQA gives the maximum KV-cache reduction; otherwise prefer GQA for a gentler tradeoff.
- **Authors:** Noam Shazeer
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1911.02150
- **Relevant chapters:** attention mechanisms, inference optimization, KV cache, memory bandwidth

## Abstract
Multi-head attention layers, as used in the Transformer neural sequence model, are a powerful alternative to RNNs for moving information across and between sequences. While training these layers is generally fast and simple, due to parallelizability across the length of the sequence, incremental inference (where such paralleization is impossible) is often slow, due to the memory-bandwidth cost of repeatedly loading the large "keys" and "values" tensors. We propose a variant called multi-query attention, where the keys and values are shared across all of the different attention "heads", greatly reducing the size of these tensors and hence the memory bandwidth requirements of incremental decoding. We verify experimentally that the resulting models can indeed be much faster to decode, and incur only minor quality degradation from the baseline.

## Key Contributions
- Identifies that autoregressive Transformer decoding is memory-bandwidth-bound, not compute-bound, due to repeatedly loading large KV tensors
- Proposes multi-query attention (MQA): all query heads share a single set of keys and values, reducing KV tensor size by a factor of H (number of heads)
- Demonstrates that MQA achieves significant decoding speedup with only minor quality degradation
- Establishes the foundational insight that led to GQA and modern KV cache optimization techniques
- One of the earliest papers to analyze Transformer inference through the lens of memory bandwidth rather than FLOPs

## Architecture Details
- **Standard multi-head attention:** For H heads, the projections are Q = xW_Q, K = xW_K, V = xW_V, where W_K and W_V each have shape (d_model, H * d_k). This means H separate key and value vectors per position
- **Multi-query attention:** W_K and W_V are reduced to shape (d_model, d_k) — a single key and value head. All H query heads attend to the same keys and values. Q projection remains unchanged with H separate heads
- **Memory bandwidth analysis:** During incremental decoding, each new token requires loading the entire KV cache. With MQA, the KV cache is H times smaller, so loading it requires H times less memory bandwidth. Since decoding is bandwidth-bound, this translates nearly directly to H-fold speedup
- **Parameter reduction:** MQA reduces total model parameters slightly (the K and V projection matrices are smaller), but the main benefit is the reduced KV cache size during inference
- **Training impact:** Training speed is largely unaffected because training parallelizes across sequence length (all positions computed simultaneously), making the KV cache size irrelevant
- **Incremental decoding bottleneck:** At each decoding step, the model generates one token. The compute (one matrix multiply per layer) is tiny, but loading all cached K and V tensors from HBM is expensive. MQA directly targets this bottleneck

## Tradeoffs Discussed
- MQA incurs "only minor quality degradation from the baseline" — the paper acknowledges that sharing K/V across heads reduces the model's representational capacity
- The quality loss is task-dependent: some tasks are more sensitive to reduced KV diversity than others
- MQA provides the most benefit during inference/decoding; it offers no speedup during training
- The extreme compression (all heads share one KV) may be more aggressive than necessary, which later motivated GQA as a middle ground
- Models must be trained from scratch with MQA; it cannot be trivially applied to existing MHA checkpoints (this limitation was later addressed by the GQA paper's uptraining recipe)
