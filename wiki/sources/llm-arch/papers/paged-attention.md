<!-- scope: PagedAttention / vLLM — virtual memory for KV cache
     deps: [[mqa]], [[gqa]]
     see-also: [[flash-attention]], [[speculative-decoding]]
-->

# Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Core Insight:** Applying OS-style virtual memory paging to KV cache eliminates fragmentation waste and enables 2-4x higher LLM serving throughput.
- **Guideline:** Use vLLM (or a PagedAttention-based engine) as the default serving backend; tune block size for your sequence length distribution.
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.06180
- **Relevant chapters:** inference serving, memory management, KV cache, systems optimization

## Abstract
High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4x with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM's source code is publicly available at https://github.com/vllm-project/vllm

## Key Contributions
- Identifies KV cache memory fragmentation and duplication as the primary bottleneck limiting LLM serving throughput
- Introduces PagedAttention: an attention algorithm that manages KV cache in fixed-size, non-contiguous memory blocks (pages), inspired by OS virtual memory
- Achieves near-zero KV cache memory waste, enabling larger batch sizes and 2-4x throughput improvement over state-of-the-art systems
- Enables flexible KV cache sharing within a request (e.g., beam search candidates sharing prefix cache) and across requests (e.g., system prompts), further reducing memory
- Builds vLLM, a production-grade serving system that has become the most widely used open-source LLM inference engine

## Architecture Details
- **The KV cache problem:** Each request's KV cache grows token-by-token during generation and varies in final size. Pre-allocating max-length memory wastes space; dynamic allocation causes fragmentation. Existing systems waste 60-80% of KV cache memory
- **Paging analogy:** Like OS virtual memory, PagedAttention separates logical KV cache (contiguous per-request) from physical memory (non-contiguous blocks/pages). A page table maps logical to physical blocks
- **Block structure:** KV cache is divided into fixed-size blocks (e.g., 16 tokens per block). Each block stores the K and V vectors for a contiguous chunk of tokens in one layer. Blocks are allocated on demand as tokens are generated
- **PagedAttention kernel:** A custom attention kernel that reads K/V from non-contiguous memory locations according to the block table, computes attention, and writes the output. This replaces the standard attention kernel that assumes contiguous KV storage
- **Copy-on-write sharing:** Multiple sequences can share the same physical KV blocks (e.g., beam search candidates sharing a common prefix). When a shared block needs modification, it is copied — exactly like OS copy-on-write
- **Cross-request sharing:** Requests with the same system prompt can share KV cache for the prompt tokens, avoiding redundant computation and storage
- **Memory management:** A block allocator tracks free and used blocks. Blocks are reclaimed when a request finishes. This eliminates fragmentation because all blocks are the same size
- **Scheduling:** vLLM uses a first-come-first-served scheduler with preemption. If GPU memory is full, lower-priority requests can have their KV cache swapped to CPU memory or recomputed later
- **Throughput gains:** 2-4x over FasterTransformer and Orca; improvements increase with longer sequences (more KV cache to manage) and complex decoding (beam search, parallel sampling)

## Tradeoffs Discussed
- The non-contiguous block layout requires a custom attention kernel that is slightly less efficient than standard contiguous-memory attention, due to the indirection through the block table
- Block size is a tuning parameter: smaller blocks reduce internal fragmentation but increase block table overhead and kernel complexity; larger blocks waste more memory on the last partially-filled block
- Copy-on-write adds complexity to memory management and requires reference counting; incorrect reference handling could lead to memory leaks or corruption
- The system is optimized for throughput (maximizing batch size) rather than single-request latency; individual requests may experience slightly higher latency due to sharing GPU resources with a larger batch
- PagedAttention's benefits are most pronounced with longer sequences and larger models; for short sequences with small KV caches, the overhead of block management may not be justified
