<!-- scope: Flash-Decoding technique for efficient attention during one-token-at-a-time LLM decoding
     deps: flashattention-2
     see-also: flashinfer, speculative-decoding, cuda-graphs-inference
-->

# Flash-Decoding for Long-Context Inference
- **Core Insight:** Decode attention has query length 1, so FlashAttention's training/prefill parallelization underutilizes GPUs unless the key/value length is split across more work.
- **Guideline:** Use decode-specific attention kernels for long-context generation; prefill-optimized FlashAttention kernels are not automatically optimal for token-by-token decode.
- **Authors:** Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov
- **Year:** 2023
- **URL:** https://pytorch.org/blog/flash-decoding/
- **Relevant topics:** decode attention, KV cache, long context, split-K attention, low batch inference, token latency

## Abstract
This is a source page for an official PyTorch blog and associated FlashAttention implementation rather than a standalone paper. Flash-Decoding targets the decode phase of autoregressive inference, where each step has a single query token attending over a long KV cache. It parallelizes over the KV sequence length, computes partial attention outputs and log-sum-exp statistics, and combines them to recover the exact output.

## Key Contributions
- Identifies that decode attention is poorly parallelized when batch size and query length are small.
- Splits keys and values across blocks so more GPU SMs participate for one-token queries.
- Separately rescales and combines partial softmax results to preserve exact attention.
- Reports up to 8x faster generation for very long sequences in the PyTorch blog.
- Clarifies the prefill/decode distinction for attention kernel design.

## Key Figures/Tables to Study
- Motivation section: explains why q-length 1 can use less than 1% of an A100 for batch size 1.
- Multi-head attention for decoding section: anchors the formula `softmax(q K^T) V`.
- Performance charts: show speedup growing with context length.
- FlashAttention repository KV-cache functions: inspect `flash_attn_with_kvcache` paths for implementation context.

## Technical Details
During decoding, the model appends a new K/V vector to the cache and computes attention for the current query against all previous keys. Flash-Decoding partitions the KV cache into chunks. Each chunk computes a local attention result and the local log-sum-exp normalization data. A reduction then combines chunk outputs using the same softmax-rescaling identity used in FlashAttention.

The method helps most when context length is large and batch size is too small to fill the GPU. It is less important when large serving batches already provide enough parallel work.

## Connections
- [[flashattention]] and [[flashattention-2]] optimize prefill/training-style attention.
- [[flashinfer]] generalizes decode/prefill attention kernels for serving engines and KV-cache formats.
- [[speculative-decoding]] reduces the number of target decode steps; Flash-Decoding reduces the cost of each decode attention step.
- [[cuda-graphs-inference]] can reduce per-step launch overhead around decode kernels.
- [[flashattention-3]] is a separate hardware-generation improvement, mostly for Hopper prefill/training-style kernels.
