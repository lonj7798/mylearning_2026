<!-- scope: Transformer attention foundation for modern LLM inference
     deps: []
     see-also: [[attention-complexity]], [[kv-cache-memory-formula]], [[multi-query-attention]], [[rope]], [[alibi]]
-->

# Attention Is All You Need
- **Core Insight:** Replace recurrence and convolution with self-attention so every token can directly condition on every other token and training can parallelize across sequence positions.
- **Guideline:** Treat scaled dot-product attention, multi-head projections, causal masking, and residual/normalization blocks as the baseline mental model for all LLM inference optimizations.
- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- **Year:** 2017
- **URL:** https://arxiv.org/abs/1706.03762
- **Relevant topics:** transformer architecture, self-attention, multi-head attention, positional encoding, decoder-only inference

## Abstract
The Transformer removes recurrent sequence processing and builds encoder-decoder models entirely from attention and feed-forward layers. Self-attention lets each position compute a weighted mixture over all positions, while multi-head attention allows different learned subspaces to attend to different relations. For inference, the decoder-side masked self-attention block is the ancestor of GPT-style autoregressive generation.

## Key Contributions
- Introduced scaled dot-product attention: `softmax(QK^T / sqrt(d_k))V`.
- Introduced multi-head attention, splitting projections into multiple heads and concatenating their outputs.
- Removed recurrence, enabling much higher training parallelism than RNN sequence models.
- Used causal masking in the decoder so positions cannot attend to future tokens.
- Added sinusoidal positional encodings to inject order without recurrence.
- Established the residual + normalization + position-wise MLP block pattern used throughout modern LLMs.

## Key Figures/Tables to Study
- **Figure 1:** Encoder-decoder Transformer block layout; map the decoder block to autoregressive LLM inference.
- **Section 3.2:** Scaled dot-product and multi-head attention equations.
- **Table 1:** Complexity comparison: self-attention is `O(n^2 d)` but path length is `O(1)`.
- **Section 3.5:** Sinusoidal positional encoding, later replaced by RoPE/ALiBi in many LLMs.

## Technical Details
For one attention head:

```text
Q = X W_Q
K = X W_K
V = X W_V
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
```

In autoregressive decoding, the model runs one new query token at a time against all cached prior keys and values. Without caching, every decode step would recompute the full prompt history. With a KV cache, each layer appends only the new token's `K,V` tensors and reuses previous tensors.

The original Transformer used encoder-decoder attention for translation. GPT-style LLMs keep the masked decoder stack and train next-token prediction. This turns inference into repeated application of the same decoder block until an end condition or maximum token budget is reached.

## Connections
- [[attention-complexity]]: full self-attention has quadratic sequence-time/memory behavior during prefill.
- [[kv-cache-memory-formula]]: cached `K,V` tensors make decode fast but consume memory proportional to layers, KV heads, head dimension, batch, and context length.
- [[multi-query-attention]] and [[grouped-query-attention]]: reduce KV-cache bandwidth and memory while keeping many query heads.
- [[rope]] and [[alibi]]: modern positional schemes inserted into or around attention scores.
