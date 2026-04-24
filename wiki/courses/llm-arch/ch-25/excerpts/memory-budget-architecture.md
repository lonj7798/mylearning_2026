# Excerpt: Memory Budget — How Architecture Determines Serving Capacity

**Sources:** [[llama-3|report]], [[deepseek-v2|report]], [[mistral-7b|report]]

---

## The Central Equation

For a GPU with $M_{\text{GPU}}$ bytes of memory:

$$\text{Max concurrent requests} = \frac{M_{\text{GPU}} - M_{\text{weights}} - M_{\text{overhead}}}{M_{\text{KV per request}}}$$

where:

$$M_{\text{KV per request}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times S_{\text{avg}} \times \text{bytes\_per\_param}$$

The numerator is fixed by the GPU and model size. The denominator — the KV cache per request — is determined entirely by the **attention variant** chosen at training time. This means that serving capacity is an architectural decision, locked in months before deployment.

## Worked Example: Llama 3 8B on A100 80GB

Llama 3 8B configuration:
- 32 layers, 8 KV heads (GQA), $d_{\text{head}} = 128$, FP16

Step-by-step:

1. **Model weights:** $8 \times 10^9 \times 2 = 16$ GB
2. **Available for KV cache:** $80 - 16 - 2$ (activations) $= 62$ GB
3. **KV per token per layer:** $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes
4. **KV per token (all 32 layers):** $4{,}096 \times 32 = 131{,}072$ bytes $= 128$ KB
5. **KV per request at 2K avg seq:** $128 \times 2{,}048 = 262{,}144$ KB $= 256$ MB
6. **Max concurrent requests:** $62{,}000 / 256 \approx 242$

## The GQA Multiplier

Same model with full MHA (32 KV heads instead of 8):

- KV per token per layer: $2 \times 32 \times 128 \times 2 = 16{,}384$ bytes
- KV per request at 2K: $16{,}384 \times 32 \times 2{,}048 = 1$ GB
- Max concurrent: $62{,}000 / 1{,}024 \approx 60$

**GQA with 8 heads gives 4x the serving capacity** compared to MHA. This is the reason Llama 3 uses 8 KV heads at every model size — including the 405B model, which has 128 query heads but only 8 KV heads (a 16:1 ratio).

## DeepSeek-V2's MLA Advantage

MLA caches a 576-dimensional latent ($d_c = 512$ + RoPE component $d_h^R = 64$) per token per layer, independent of head count:

- **GQA-8 equivalent:** $2 \times 8 \times 128 = 2{,}048$ dims/token/layer
- **MLA:** $576$ dims/token/layer

Ratio: $2{,}048 / 576 \approx 3.6\times$ advantage over GQA-8.

The paper reports 5.76x generation throughput increase over the MHA-based DeepSeek 67B predecessor, driven by:
- 93.3% KV cache reduction enabling larger batch sizes
- PagedAttention-compatible block-based serving
- Efficient custom inference kernels for the up-projection step

## Mistral 7B: Bounded Budget via SWA

Mistral 7B adds a second dimension of memory control: **sliding window attention** caps the number of cached positions at $W = 4096$, regardless of actual sequence length.

$$M_{\text{KV per request}} = 2 \times 8 \times 128 \times 32 \times \min(S_{\text{avg}}, 4096) \times 2$$

At any sequence length above 4096, the KV cache is constant:
- $2 \times 8 \times 128 \times 32 \times 4096 \times 2 = 536$ MB

The rolling buffer implementation:
```python
cache_index = t % W  # Position t overwrites position t - W
```

This makes the memory budget **predictable** — capacity planning does not depend on generation length distribution.

## The Multiplicative Composition

GQA and SWA reduce different dimensions:
- **GQA** reduces the per-token cache width (fewer KV heads)
- **SWA** bounds the total number of cached tokens
- Together: width reduction $\times$ length bound = multiplicative savings

Mistral 7B: GQA (4x reduction per token) + SWA (bounded at 4096 positions) means:
- At 32K context: $4\times$ from GQA $\times$ $8\times$ from SWA = effective $32\times$ reduction vs hypothetical MHA at 32K

## Design Implications

| Architectural Choice | Serving Impact |
|---------------------|----------------|
| MHA → GQA-8 | 4-8x more concurrent requests |
| GQA-8 → MLA | Additional 3.6x capacity |
| Add SWA ($W = 4096$) | Fixed memory regardless of context length |
| Reduce $d_{\text{head}}$ | Linear reduction but affects quality |
| Fewer layers | Linear reduction but affects quality |

The first three are "free" in the sense that they preserve model quality. The last two degrade it.

---

**Key takeaway:** The serving capacity equation should be computed *before* the attention configuration is finalized. Every model that will be deployed at scale is implicitly making a serving cost decision when it chooses its attention variant. Llama 3's 8 KV heads, DeepSeek-V2's MLA, and Mistral 7B's SWA are all design decisions where serving economics overrode pure modeling considerations.
