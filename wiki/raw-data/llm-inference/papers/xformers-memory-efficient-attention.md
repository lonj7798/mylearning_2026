<!-- scope: xFormers memory-efficient attention operator family and its paper lineage
     deps: flashattention
     see-also: flashattention-2, flashinfer
-->

# xFormers Memory-Efficient Attention
- **Core Insight:** Memory-efficient attention should be exposed as a dispatchable operator family so models can use the best available backend without rewriting attention code.
- **Guideline:** Use `xformers.ops.memory_efficient_attention` or framework integrations when you need exact attention without materializing the full attention matrix and can accept backend-dependent constraints.
- **Authors:** facebookresearch/xformers; Markus N. Rabe and Charles Staats for the core `O(n)` memory paper lineage
- **Year:** 2021-2022
- **URL:** https://github.com/facebookresearch/xformers ; https://arxiv.org/abs/2112.05682
- **Relevant topics:** memory-efficient attention, PyTorch operators, backend dispatch, attention bias, block-diagonal masks, inference attention

## Abstract
This page summarizes an official code/docs artifact plus the related paper "Self-attention Does Not Need O(n^2) Memory." xFormers provides optimized transformer building blocks, with memory-efficient attention as a flagship operator. The operator computes exact attention without explicitly allocating the full attention matrix and dispatches among multiple backends such as CUTLASS, FlashAttention-family kernels, and specialized inference masks.

## Key Contributions
- Exposes a single PyTorch-level `memory_efficient_attention` API over multiple fast attention implementations.
- Supports attention bias objects such as causal and block-diagonal masks for packed variable-length batches.
- Enables diffusion and transformer applications to reduce memory use with minimal model-code changes.
- Integrates FlashAttention backends as they become available.
- Provides diagnostic errors that reveal dtype, device, head-dimension, and build constraints.

## Key Figures/Tables to Study
- xFormers GitHub README: installation, feature list, and exact-attention claim.
- `xformers.ops.fmha` docs/source: operator signatures and backend selection.
- `attn_bias` classes: study block-diagonal and causal-with-offset masks for packed batches and KV-cache cases.
- Rabe/Staats Algorithm 2: linear-memory attention recurrence lineage.

## Technical Details
The xFormers operator accepts query, key, value tensors plus optional attention bias and dropout. Rather than forcing a specific kernel, it dispatches to an implementation compatible with the tensor layout, dtype, GPU capability, and requested mask.

For LLM inference, the important pieces are packed variable-length support and mask objects that avoid padding overhead. The library is not a full serving engine: it provides attention operators that frameworks can call inside their own batching, KV-cache, and scheduling systems.

## Connections
- [[flashattention]] is the most influential exact IO-aware backend design.
- [[flashattention-2]] and [[flashattention-3]] represent backend improvements that can be exposed through libraries.
- [[flashinfer]] takes a more serving-specific approach to KV-cache formats and graph-compatible scheduling.
- [[flashdecoding]] covers decode-specific attention where xFormers-style dispatch alone is not the whole serving story.
- [[cuda-graphs-inference]] is orthogonal to xFormers operators and targets repeated launch overhead.
