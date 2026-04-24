# Excerpt: Attention Interleaving Design Rationale

<!-- source: [[gemma-3|report]], [[mistral-7b|report]], [[raschka-attention-variants|blog]] -->

## The Core Tradeoff

Global attention at every layer is redundant. Each layer's global attention re-reads the full KV cache, but consecutive layers produce highly correlated attention patterns. The information deposited into the residual stream by one global layer persists for the next several layers, making repeated global reads wasteful.

Gemma 3 exploits this redundancy with a 5:1 local-to-global ratio:

```
Layers 0-4:   LOCAL  (sliding window, W=1024)
Layer  5:     GLOBAL (full context, up to 128K)
Layers 6-10:  LOCAL
Layer  11:    GLOBAL
...
```

## KV Cache Arithmetic

For a model with $L$ layers, GQA with $G$ KV groups, head dimension $d_k$, sequence length $S$, and window $W$:

**All-global baseline:**
$$\text{Cache}_\text{total} = L \times 2 \times G \times d_k \times S$$

**5:1 interleaved:**
$$\text{Cache}_\text{total} = \left\lfloor \frac{L}{6} \right\rfloor \times 2 G d_k S + \left(L - \left\lfloor \frac{L}{6} \right\rfloor\right) \times 2 G d_k W$$

For $S = 128K$, $W = 1024$:
- Each global layer caches $125\times$ more than a local layer
- With 5:1 ratio, total cache drops to ~15-17% of all-global

## Why Not 3:1 or 7:1?

The Gemma 3 report states that ablations showed "minimal perplexity degradation" at 5:1. The reasoning:

- **1:1** (alternating): Still expensive. Half the layers cache full context.
- **3:1**: More global layers than needed. Diminishing returns from additional global refresh.
- **5:1**: Sweet spot. One global refresh every 6 layers provides sufficient long-range signal.
- **7:1 or higher**: Risk of information loss. The residual stream signal from a global layer may degrade across too many local layers, especially for tasks requiring precise long-range retrieval.

The 5:1 ratio also maps cleanly to typical layer counts: 18, 30, 36, 42, 48 layers all divide into groups of 6 with minimal remainder.

## Dual RoPE Frequency: Why It Matters

Using the same RoPE base for local and global layers would waste representational capacity:

- **10K base at global scale (128K context):** Position angles for distant tokens become nearly indistinguishable. The rotation "wraps around" too quickly, aliasing distant positions to nearby ones.
- **1M base at local scale (1K window):** Position angles for nearby tokens are nearly identical. The rotation is too slow to provide meaningful positional discrimination within the window.

Gemma 3's solution assigns each layer type its own optimal frequency:

| Layer Type | RoPE Base | Effective Range | Resolution |
|-----------|-----------|----------------|------------|
| Local | 10,000 | ~1K tokens | High (nearby) |
| Global | 1,000,000 | ~128K tokens | Lower but sufficient |

This is analogous to using different lens focal lengths for different distances: a telephoto lens (10K base) for nearby detail, a wide-angle lens (1M base) for the full panorama.

## Comparison: Mistral vs Gemma 3 vs OLMo 3

| Feature | Mistral 7B | Gemma 3 | OLMo 3 |
|---------|-----------|---------|--------|
| Approach | Pure SWA | 5:1 local/global | Similar hybrid |
| Window | 4,096 | 1,024 | varies |
| Global layers | 0 | Every 6th | Periodic |
| RoPE | Single freq | Dual freq | Single freq |
| Max context | 8K (trained) | 128K | 32K |

Mistral proved SWA was viable. Gemma 3 showed that combining aggressive SWA with periodic global refresh enables long context. The dual-frequency RoPE is the mechanism that makes this work at 128K.

## Information Propagation

Consider a token at position 100,000 in a 128K-context input:

1. **Layer 5 (first global layer):** Attends to position 100,000 directly. Writes its representation into the residual stream.
2. **Layers 6-10 (local):** Cannot attend to position 100,000 directly. But read the residual stream, which now contains information deposited by Layer 5.
3. **Layer 11 (second global layer):** Attends to position 100,000 again, but now with enriched representations from 5 layers of local processing.

Each global layer acts as a "checkpoint" that ensures long-range information remains accessible. The local layers in between refine and compose this information without the cost of re-reading the entire context.
