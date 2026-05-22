---
chapter: ch-02
course: llm-inference
phase: read
excerpt_of: "Train Short, Test Long: Attention with Linear Biases (Press et al. 2021)"
source_url: https://arxiv.org/abs/2108.12409
created_at: "2026-05-21"
---

# Excerpt: ALiBi — linear distance bias on attention scores

**Authors:** Ofir Press, Noah A. Smith, Mike Lewis
**Year:** 2021
**Venue:** ICLR 2022
**URL:** https://arxiv.org/abs/2108.12409
**Raw-data source:** [[raw-data/alibi]]

---

## The mechanism in one equation

Instead of rotating queries and keys (RoPE) or adding position embeddings (sinusoidal/learned), ALiBi adds a fixed linear bias to attention logits:

```math
\mathrm{score}(i, j) = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot (i - j), \quad j \leq i
```

For head `h`, the slope `m_h` is a fixed (non-learned) positive constant. The bias grows linearly with the distance from query `i` to key `j` — distant past keys are penalized.

No position embedding is added to token embeddings. No rotation is applied to `Q, K`. Only the score matrix gets a per-head additive bias.

---

## The slope schedule

The original paper proposed geometrically-spaced slopes:

```
m_h = 2^(-8 · h / H)        for h = 1, ..., H
```

For `H = 8`: `m_h ∈ {2⁻¹, 2⁻², 2⁻³, ..., 2⁻⁸}`. The first head's bias is steep (recency-biased), the last head's is gentle (allows long-range). Different heads thus specialize on different time-scales of context — by hardware-imposed inductive bias rather than learning.

---

## Why ALiBi extrapolates

The killer feature: a model trained on 1024-token contexts can generate at 16k-token contexts with much smaller perplexity degradation than RoPE or sinusoidal baselines.

Mechanism: the bias `−m_h · (i − j)` is well-defined for *any* distance, including distances unseen at training. The model learns to combine the bias and the dot-product score; at test time, longer distances simply produce more negative biases, smoothly downweighting distant content. There's no out-of-distribution behavior because the bias function is linear and unbounded.

Compare RoPE: distances unseen at training correspond to RoPE angles never seen — out-of-distribution by construction. NTK/YaRN fix this, but ALiBi doesn't need a fix.

---

## Compute cost and cache impact

```python
def alibi_attention(q, k, v, slopes, positions):
    # positions[i]: absolute position of query i
    # positions[j]: absolute position of key j (cached)
    distance = positions[:, None] - positions[None, :]      # [L, L]
    bias = -slopes[:, None, None] * distance                # [H, L, L]
    scores = q @ k.transpose(-2, -1) / sqrt(d_head)
    scores = scores + bias                                  # broadcast over batch
    scores = scores.masked_fill(distance < 0, -float("inf"))
    return softmax(scores) @ v
```

- **Compute overhead**: one extra add per `(query, key, head)` pair. Negligible vs `QKᵀ` matmul.
- **Cache impact**: zero. `K, V` cache shape unchanged. The bias is applied per-step from absolute positions.
- **Kernel fusion**: FlashAttention supports ALiBi as an option (`alibi_slopes` kernel parameter).

---

## ALiBi vs RoPE: head-to-head

| | RoPE | ALiBi |
|---|---|---|
| Modifies | `Q, K` (rotation) | Attention scores (additive bias) |
| Cache impact | Rotated `K` stored | None |
| In-distribution quality | Stronger | Slightly weaker |
| Extrapolation beyond train length | Needs YaRN/NTK | Native |
| Compute overhead | `O(d)` rotation per token | `O(L)` bias per query |
| Heads specialize | Implicitly via training | Explicitly via slope schedule |
| Adopted by | Llama, Qwen, Mistral, DeepSeek, Phi, Gemma | BLOOM, MPT, Falcon-1B/7B (older) |

**Why RoPE won.** In-distribution quality matters more than extrapolation for most production deployments. And YaRN/NTK closed the extrapolation gap convincingly. As of 2026, ALiBi is mostly a legacy choice — but it remains a useful baseline for "what should extrapolation look like" arguments.

---

## When ALiBi still matters

1. **Research baselines**: any paper claiming long-context capability should compare against ALiBi as the "free extrapolation" baseline.
2. **Edge inference of older models**: BLOOM, MPT-7B, Falcon-7B are still deployed; kernels and serving stacks need to support ALiBi.
3. **Hybrid schemes**: some recent architectures combine ALiBi-like recency bias with RoPE for the best of both (rarely shipped at scale).

---

## Common pitfalls

- **Forgetting per-head slopes in the kernel**. FlashAttention requires an `alibi_slopes` tensor of shape `[H]`. Pass zeros to disable; a partial schedule breaks attention.
- **Reusing RoPE positional indexing for ALiBi**. RoPE rotates by absolute position; ALiBi uses pairwise distance. Conflating them produces silent wrong scores.
- **Assuming ALiBi removes the `O(L²)` attention cost**. It doesn't — it changes the scores, not the computation. You still pay full attention complexity.

---

## Connections

- [[excerpts/rope]] — the alternative; rotation-based and now dominant.
- [[excerpts/attention-complexity]] — ALiBi adds `O(L)` per query, negligible vs `O(L·d)` attention.
- [[raw-data/flashattention-2]] — kernel supports both ALiBi and no-positional-bias paths.
- [[ch-20]] — model survey; check `position_embedding_type` in each model's config.
