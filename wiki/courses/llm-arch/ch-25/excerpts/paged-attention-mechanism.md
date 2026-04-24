# Excerpt: PagedAttention — OS Virtual Memory Applied to KV Cache

**Source:** [[paged-attention|Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)]]

---

## The Problem: 60-80% Memory Waste

Existing LLM serving systems manage KV cache memory in one of two ways, both wasteful:

1. **Pre-allocation:** Reserve contiguous memory for the maximum possible sequence length per request. If a request generates 200 tokens but the max is 4096, 95% of reserved memory is wasted (internal fragmentation).

2. **Dynamic contiguous allocation:** Allocate contiguous memory on demand as tokens are generated. As requests arrive and complete at different times, freed memory fragments into gaps too small for new allocations (external fragmentation).

The paper measured that FasterTransformer and Orca wasted 60-80% of KV cache memory through these mechanisms.

## The Solution: Paging

PagedAttention borrows the core idea of OS virtual memory:

- **Physical memory** is divided into fixed-size **blocks** (pages). Each block stores KV vectors for a fixed number of tokens (typically 16).
- **Logical memory** appears contiguous to each request. A **block table** (page table) maps logical block indices to physical block locations.
- Blocks are allocated **on demand** from a free-block pool — no pre-allocation, no contiguous requirement.
- All blocks are the same size — freed blocks are **immediately reusable** with zero external fragmentation.

### Block Table Example

```
Request A block table:
  Logical 0 → Physical 7
  Logical 1 → Physical 3
  Logical 2 → Physical 12

Request B block table:
  Logical 0 → Physical 1
  Logical 1 → Physical 9
```

Requests A and B see contiguous KV caches. In physical GPU memory, their blocks are interleaved. Neither request knows or cares about the other's layout.

## Custom Attention Kernel

Standard attention kernels assume contiguous KV storage — a single pointer plus stride covers all keys. PagedAttention requires a custom kernel that:

1. Reads the block table for the current request
2. Gathers key/value vectors from non-contiguous physical blocks
3. Computes attention scores and weighted values as usual
4. Writes output

The indirection through the block table adds slight per-operation overhead. But the throughput gain from fitting 2-4x more concurrent requests in GPU memory vastly outweighs this cost.

## Copy-on-Write for Beam Search

For beam search, multiple candidate sequences share a common prefix. Without sharing, each beam copies the entire prefix KV cache. PagedAttention uses **copy-on-write**:

- Beams share physical blocks for the common prefix (reference-counted)
- When a beam diverges and needs to write different values, the affected block is copied to a new physical location
- Only the divergent block is duplicated — shared prefix blocks remain shared

This eliminates prefix duplication for beam search with $k$ beams: instead of $k$ copies of the prefix cache, there is 1 shared copy plus copy-on-write overhead at divergence points.

## Results

| Metric | vs. FasterTransformer | vs. Orca |
|--------|----------------------|----------|
| Throughput (OPT-13B) | 2-4x | 2-4x |
| KV cache waste | <4% (vs. 60-80%) | <4% (vs. 60-80%) |
| Improvement at longer sequences | Greater | Greater |

The improvement increases with longer sequences (more KV cache to manage), larger models (bigger per-token KV footprint), and complex decoding (beam search benefits from copy-on-write).

## The vLLM System

vLLM is the production serving system built on PagedAttention. It combines:
- PagedAttention for memory management
- Continuous batching for request scheduling
- Preemption (swapping KV cache to CPU or recomputing) when GPU memory is full
- Prefix caching for shared system prompts

As of 2025, vLLM is the most widely used open-source LLM serving engine, with PagedAttention adopted by essentially every major serving framework.

---

**Key takeaway:** The KV cache management problem is a systems problem, not a model architecture problem. PagedAttention's contribution is recognizing that decades of OS research on memory management applies directly — and the payoff is 2-4x serving throughput improvement with zero model changes.
