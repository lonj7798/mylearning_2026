<!-- scope: Rotary position embeddings for relative position behavior in attention
     deps: [[attention-is-all-you-need]]
     see-also: [[alibi]], [[attention-complexity]]
-->

# RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Core Insight:** Encode positions by rotating query and key vectors so dot products depend naturally on relative positions.
- **Guideline:** For LLM inference, remember that RoPE is applied to `Q,K` before attention; context extension tricks usually modify RoPE scaling, not the KV-cache formula.
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu
- **Year:** 2021
- **URL:** https://arxiv.org/abs/2104.09864
- **Relevant topics:** RoPE, positional encoding, relative position, long context

## Abstract
RoFormer introduces Rotary Position Embedding (RoPE), which injects token position into attention by applying position-dependent rotations to query and key vectors. Unlike absolute position embeddings added to token embeddings, RoPE changes the attention dot product directly and supports relative-position behavior.

## Key Contributions
- Proposed rotary transformations for query/key vectors in self-attention.
- Showed that attention scores can depend on relative token offsets through rotation algebra.
- Avoided learned absolute position tables in the core method.
- Became the default positional method for many GPT-style open LLMs.
- Enabled later practical context-extension methods through frequency scaling.

## Key Figures/Tables to Study
- **RoPE equations:** Complex-number or block-rotation form.
- **Relative position derivation:** Why `q_m^T k_n` depends on `m-n`.
- **Model comparison tables:** RoFormer vs baseline Transformer.
- **Implementation detail:** RoPE applies to each head's `Q,K`, not to `V`.

## Technical Details
RoPE pairs dimensions and rotates them by an angle determined by token position and frequency:

```text
q'_pos = rotate(q, pos)
k'_pos = rotate(k, pos)
score(pos_i, pos_j) = q'_i dot k'_j
```

Because both `Q` and `K` are rotated, the resulting dot product contains relative offset information. During decoding, cached keys are stored after RoPE has been applied for their positions; the new query receives the rotation for the current position.

RoPE does not reduce attention complexity or KV-cache size. It changes the position signal. Long-context variants usually interpolate or scale RoPE frequencies so models trained on shorter contexts behave better at longer positions.

## Connections
- [[alibi]]: alternative that adds linear biases to attention scores instead of rotating vectors.
- [[attention-is-all-you-need]]: replaces or augments the original sinusoidal position encoding idea.
- [[kv-cache-memory-formula]]: RoPE changes neither number of cached tensors nor their shape.
- [[prefill-vs-decode]]: correct absolute position indexing matters when continuing from a cache.
