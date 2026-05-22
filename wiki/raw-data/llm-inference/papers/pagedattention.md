<!-- scope: vLLM paper introducing PagedAttention for KV-cache paging and sharing
     deps: [[orca]]
     see-also: [[continuous-batching]], [[cachegen]], [[h2o]]
-->

# Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Core Insight:** Treat the KV cache like virtual memory: split each sequence's cache into fixed-size logical blocks and map them to non-contiguous physical GPU blocks.
- **Guideline:** Use paged KV allocation when serving variable-length requests, because contiguous per-request reservations waste memory and reduce batch size.
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.06180
- **Relevant topics:** KV cache, PagedAttention, vLLM, continuous batching, memory fragmentation, prefix sharing

## Abstract
The paper argues that LLM serving throughput is often limited by KV-cache memory, not only raw compute. Because request lengths grow and shrink dynamically, conventional allocation produces fragmentation and duplicate cache storage. PagedAttention stores KV tensors in fixed-size blocks with a block table that translates logical token positions to physical cache blocks. vLLM builds on this mechanism to reduce waste, support larger batches, and share cache blocks across prompts, parallel sampling, and beam search.

## Key Contributions
- Introduces PagedAttention, an attention algorithm that reads KV values through block-table indirection.
- Shows that KV cache can occupy a large fraction of GPU memory and that fragmentation directly limits batching.
- Separates logical sequence layout from physical cache placement, avoiding large contiguous reservations.
- Adds copy-on-write sharing for prompts reused by multiple continuations.
- Implements vLLM, a serving engine combining PagedAttention with request scheduling.
- Reports 2-4x throughput gains over FasterTransformer and Orca at similar latency.

## Key Figures/Tables to Study
- Figure 1: memory breakdown and throughput collapse from KV-cache pressure.
- Figure 2: analogy between OS paging and PagedAttention block mapping.
- Figure 3: block table and physical KV blocks for multiple sequences.
- Evaluation tables: throughput under different model sizes, sequence lengths, and decoding modes.

## Technical Details
KV cache memory grows roughly with `2 * layers * kv_heads * head_dim * tokens * dtype_bytes * batch`, where the factor 2 is for keys and values. PagedAttention groups tokens into blocks and maintains a per-sequence block table, so the attention kernel gathers K/V blocks by logical block id rather than assuming contiguous memory.

The final block of a sequence can be partially full, but earlier blocks are densely packed, so waste is bounded by one block per active sequence. Shared prefixes point multiple logical block tables to the same physical blocks. When a shared block must be modified, vLLM uses copy-on-write instead of duplicating the whole prefix up front.

Scheduling is tied to memory accounting: a request can be admitted when enough free KV blocks exist for its prefill/decode progress. If memory pressure rises, blocks can be evicted or swapped depending on the serving policy. PagedAttention is therefore not just a kernel optimization; it changes admission, batching, and prefix reuse.

## Connections
- Builds on [[orca]]'s iteration-level scheduling but attacks the KV-memory bottleneck.
- Enables [[continuous-batching]] by making dynamic batch membership cheap in memory.
- Complements cache compression/offload work such as [[cachegen]], [[h2o]], and [[snapkv]].
