# Excerpt: Architectural Tradeoffs in Hybrid SSM-Attention Models

<!-- source: [[jamba|report]], [[mamba|paper]] -->

## The Complexity Tax

Jamba's triple hybrid is not free. Every architectural benefit comes with an implementation cost that must be weighed against the alternative of simply using a well-optimized Transformer.

## Training Complexity

### Three Compute Primitives in One Forward Pass

A Jamba forward pass executes three fundamentally different computations:

1. **Mamba selective scan** — a parallel scan (prefix sum) over the input-dependent SSM recurrence. Requires a custom CUDA kernel that keeps the state in SRAM (the same IO-awareness principle as Flash Attention). Cannot be expressed as a standard matrix multiply.

2. **Transformer attention** — standard GQA with Flash Attention. Well-supported by existing libraries but must coexist with the SSM computation in the same model forward pass.

3. **MoE routing** — top-2 expert selection from 16 experts, with all-to-all communication in distributed settings. The routing occurs on both Mamba and attention layers' MLP components.

Each primitive has different parallelization requirements:
- Mamba's parallel scan parallelizes across the batch and channel dimensions but is sequential along the time dimension (within each scan segment)
- Attention parallelizes across batch, heads, and (in Flash Attention 2) the sequence dimension
- MoE requires expert parallelism — distributing experts across GPUs and routing tokens between them

Training infrastructure must handle all three simultaneously. AI21's in-house framework uses FSDP, tensor parallelism, sequence parallelism, *and* expert parallelism. This is strictly more complex than training a pure Transformer (which needs only the first three) or a pure MoE Transformer (which does not need SSM scan support).

### Hyperparameter Explosion

The hybrid introduces architectural hyperparameters that do not exist in pure models:

- **Attention-to-Mamba ratio** (1:7 in Jamba, but other ratios are viable)
- **Attention layer placement** (which positions within each block get attention)
- **MoE frequency** (every 2nd layer in Jamba, but could be every 3rd, 4th, etc.)
- **Which layer types get MoE** (both Mamba and attention in Jamba, but could be selective)
- **SSM state dimension** (N=16 in standard Mamba, but tunable)
- **Mamba vs. Mamba-2 choice** (different SSM variants have different efficiency/expressivity profiles)

Each combination affects quality, throughput, and memory differently. The ablation cost for searching this space is higher than for a pure Transformer, where the main architectural choices are layer count, width, head count, and GQA group count.

## Serving Complexity

### Inference Engine Requirements

At the time of Jamba's release (early 2024), no mainstream inference engine natively supported hybrid SSM-attention models. Serving Jamba required:

1. **Custom Mamba kernel integration.** The selective scan kernel from the Mamba repository must be compiled and integrated into the serving framework. This kernel has specific CUDA version requirements and is not part of standard libraries like cuBLAS or cuDNN.

2. **Dual state management.** The serving engine must manage two types of per-request state simultaneously:
   - **KV cache** for the 4 attention layers: grows with context length, requires memory allocation and management (similar to vLLM's PagedAttention for Transformers)
   - **SSM hidden state** for the 28 Mamba layers: fixed-size per layer, reset per sequence (or carried across turns in multi-turn dialogue)

3. **Heterogeneous compute scheduling.** Different layers have different compute profiles:
   - Mamba layers: memory-bound during generation (like Transformer FFN layers), small state to load
   - Attention layers: memory-bound during generation (like Transformer attention), larger KV cache to load
   - MoE layers: routing overhead + expert loading, with different memory patterns than dense layers

4. **Batching across sequence lengths.** In a pure Transformer, all requests in a batch share the same compute graph. In Jamba, the prefill phase (processing the full prompt) uses the Mamba parallel scan, while the decoding phase (generating token by token) uses the Mamba recurrence. These have different compute kernels, complicating continuous batching.

### The Tooling Gap

The practical consequence: as of mid-2024, Jamba was harder to deploy efficiently than Mixtral or LLaMA despite being architecturally designed for efficient deployment. The theoretical advantage (4 GB vs. 32 GB KV cache) was partially offset by suboptimal inference engine support.

This mirrors the pattern observed in [[ch-07]] for MLA: architectural innovation outpaces inference tooling. GQA dominates not because it is the best attention variant, but because it is the best-supported one. Similarly, pure Transformers dominate serving not because they are the most efficient architecture, but because inference engines are built for them.

## The Fundamental Tradeoff

The hybrid approach trades:

**Gains:**
- 32x KV cache reduction at 256K context
- 3x throughput at equivalent quality
- Single-GPU deployment for long-context workloads
- Qualitatively different memory scaling (fewer caching layers, not just smaller caches)

**Costs:**
- Three different compute kernels in one forward pass
- Larger architectural hyperparameter search space
- Immature inference engine support
- Debugging complexity (failure modes span SSM dynamics, attention patterns, and routing decisions)
- Talent pool: fewer engineers have experience with SSM architectures than with Transformers

## When the Tradeoff Favors the Hybrid

The hybrid wins when:
1. **Context length is the binding constraint.** At 256K context, no pure Transformer fits on a single GPU. The hybrid is the only option short of multi-GPU serving.
2. **Hardware is fixed and limited.** If you have one A100 and need long-context capability, the hybrid's memory efficiency is decisive.
3. **Throughput matters more than latency.** The 3x throughput gain comes from fitting larger batches, which helps throughput but does not reduce per-request latency.

The hybrid loses when:
1. **Context is short (< 32K).** At short context, KV cache is not the bottleneck, and the tooling disadvantage outweighs the memory savings.
2. **Multi-GPU serving is available.** With 4+ GPUs, a pure Transformer can serve 256K context via tensor parallelism, and inference engines are highly optimized for this.
3. **Maximum quality is the only goal.** The MMLU and HumanEval gaps, while small, are real. For applications where every benchmark point matters, a pure Transformer trained with more compute may be preferred.
