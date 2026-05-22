<!-- scope: ALiBi positional bias for train-short/test-long length extrapolation
     deps: [[attention-is-all-you-need]]
     see-also: [[rope]], [[attention-complexity]]
-->

# Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
- **Core Insight:** Add a head-specific linear penalty to attention scores based on token distance, encouraging recency and improving extrapolation beyond training length.
- **Guideline:** ALiBi is a positional-bias method, not a memory optimization; it can help length extrapolation but does not remove quadratic attention or KV-cache growth.
- **Authors:** Ofir Press, Noah A. Smith, Mike Lewis
- **Year:** 2021
- **URL:** https://arxiv.org/abs/2108.12409
- **Relevant topics:** ALiBi, positional bias, context extrapolation, long context

## Abstract
ALiBi removes explicit positional embeddings and instead adds a fixed linear bias to attention scores. The bias penalizes attending to distant past tokens, with different slopes per attention head. The paper shows that models trained on shorter sequences can extrapolate to longer sequences more effectively than with several position embedding baselines.

## Key Contributions
- Proposed adding linear distance penalties directly to attention logits.
- Avoided learned position embeddings and their fixed maximum-position table.
- Demonstrated train-short/test-long extrapolation.
- Reported faster and lower-memory training than training sinusoidal baselines at longer sequence length.
- Emphasized positional method choice as a driver of inference-time context behavior.

## Key Figures/Tables to Study
- **Bias equation:** Attention logits receive `-m_h * distance`.
- **Length extrapolation results:** Train on 1024, evaluate on longer lengths.
- **Slope schedule:** Different heads receive different recency biases.
- **Comparisons against sinusoidal/RoPE/T5-style biases:** Useful for position-method tradeoffs.

## Technical Details
For head `h`, ALiBi modifies causal attention logits:

```text
score(i, j) = q_i dot k_j / sqrt(d) - m_h * (i - j)
```

where `j <= i` and `m_h` is a fixed positive slope. The penalty grows with distance into the past. Since it is added to the attention matrix, no position embedding vector is added to token embeddings.

At inference, ALiBi requires correct distance computation for cached tokens. It does not add KV-cache tensors, but the attention kernel must incorporate the bias for each query/key position pair.

## Connections
- [[rope]]: RoPE encodes position through rotations; ALiBi encodes it through score bias.
- [[attention-complexity]]: ALiBi keeps full attention complexity unless combined with sparse/windowed attention.
- [[prefill-vs-decode]]: during decode, the bias for the new query spans all cached positions.
- [[kv-cache-memory-formula]]: ALiBi does not alter KV-cache memory.
