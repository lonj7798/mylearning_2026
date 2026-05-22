<!-- scope: Quadratic attention cost and efficient Transformer context
     deps: [[attention-is-all-you-need]]
     see-also: [[kv-cache-memory-formula]], [[prefill-vs-decode]], [[batching-for-inference]]
-->

# Efficient Transformers: A Survey
- **Core Insight:** Full self-attention scales quadratically with sequence length, making long-context inference expensive even when model weights fit in memory.
- **Guideline:** Separate attention compute complexity from KV-cache memory; prefill pays the quadratic attention matrix cost, while decode pays repeated linear reads over cached context.
- **Authors:** Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2009.06732
- **Relevant topics:** attention complexity, efficient Transformers, long context, sparse attention

## Abstract
Efficient Transformers surveys architectural approaches that reduce the memory or compute cost of self-attention. For inference fundamentals, its main value is the taxonomy around why vanilla attention is expensive: token-to-token interactions form an `n x n` attention matrix per head.

## Key Contributions
- Summarized efficient-attention families: sparse, low-rank, kernelized, memory, recurrence, and compression methods.
- Made quadratic sequence scaling the central bottleneck for long-context Transformers.
- Compared efficiency methods by complexity, mechanism, and empirical behavior.
- Clarified that not all efficient attention methods preserve full-attention quality.
- Provides vocabulary for later serving optimizations that target different bottlenecks.

## Key Figures/Tables to Study
- **Survey taxonomy tables:** Map each efficient Transformer to its mechanism.
- **Complexity comparisons:** Standard self-attention vs approximate/sparse alternatives.
- **Long-sequence discussion:** Why `n^2` becomes the dominant cost.
- **Tradeoff sections:** Efficiency gains can affect quality or hardware efficiency.

## Technical Details
For sequence length `n`, hidden size `d`, and heads `h`, full attention forms attention logits shaped roughly:

```text
[batch, heads, n, n]
```

This is the source of `O(n^2)` score memory and attention compute during full-context processing. The MLP and projection layers are usually `O(n d^2)`, so the bottleneck depends on sequence length, hidden size, implementation, and hardware.

During autoregressive decode with a KV cache, the new token attends to `n` cached keys, so the attention score shape per step is closer to `[batch, heads, 1, n]`. That avoids recomputing the whole `n x n` prompt matrix but still grows linearly with context length per generated token.

## Connections
- [[prefill-vs-decode]]: prefill is where the full prompt attention matrix appears.
- [[kv-cache-memory-formula]]: decode speed relies on storing KV tensors instead of recomputing them.
- [[rope]] and [[alibi]]: position methods alter scores/representations but not full-attention scaling.
- [[batching-for-inference]]: long contexts reduce batch capacity because attention and KV memory both grow.
