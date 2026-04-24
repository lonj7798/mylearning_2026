# Excerpt: Qwen 3.5 — The Gated DeltaNet Hybrid Architecture

<!-- source: [[qwen-3-5|report]] -->

## Why Replace Attention at All?

Standard self-attention computes pairwise interactions between all tokens in the sequence. For a sequence of length $L$, this costs $O(L^2)$ in both compute and memory (the KV cache stores all past keys and values). At 128K tokens, the KV cache for a 64-head, 128-dimension model is already substantial. At 262K or 1M tokens, it becomes the dominant bottleneck — not the FLOPs for the matrix multiplications, but the memory bandwidth to read and write the growing cache.

The insight behind Qwen 3.5's design: **most layers do not need full pairwise attention.** Many layers perform "soft" contextual mixing — updating token representations based on broad context rather than precise token-pair interactions. A linear-attention mechanism that maintains a fixed-size state can handle this efficiently. The few layers that do need precise pairwise interactions (entity coreference, long-range syntactic binding, exact copying) retain full quadratic attention.

## Gated DeltaNet: The Mechanism

Gated DeltaNet (GDN) is a linear-attention variant based on the delta rule from associative memory theory. The core idea: instead of storing all key-value pairs in a growing cache, maintain a fixed-size recurrent state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ and update it with a correction rule.

### The Delta Rule Update

For each token at position $t$, the model computes queries $q_t$, keys $k_t$, values $v_t$, and two gating scalars $\alpha_t$, $\beta_t$ via linear projections followed by lightweight convolutions and normalization. The state update is:

$$S_t = \alpha_t \cdot S_{t-1} + \beta_t \cdot k_t (v_t - S_{t-1}^T k_t)^T$$

The critical term is $(v_t - S_{t-1}^T k_t)$ — the **prediction error**. $S_{t-1}^T k_t$ is what the current state *predicts* the value should be for key $k_t$. The difference between the actual value $v_t$ and this prediction is the error signal. The update corrects the state by this error, scaled by the gate $\beta_t$ and projected by the key $k_t$.

This is fundamentally different from standard linear attention (which simply accumulates $k_t v_t^T$ into the state). Naive accumulation leads to interference — old key-value associations degrade as new ones are added, because the state cannot distinguish between entries. The delta rule addresses this by *correcting* existing associations rather than blindly adding new ones.

### Why the Delta Rule Gives Better Retrieval

Consider a concrete example. Suppose the model has seen "The capital of France is Paris" earlier in the sequence. In standard linear attention, the state accumulates the key-value pair (key="France", value="Paris") along with thousands of other pairs. When later queried with key="France", the retrieved value is the sum of "Paris" plus interference from all other accumulated associations. At long sequence lengths, this interference can drown the signal.

In DeltaNet, when the (key="France", value="Paris") pair arrives, the update checks: what does the current state predict for "France"? If the state already has some noisy association, the delta rule corrects it — pushing the state toward the correct "Paris" association and implicitly reducing the interference. The result: cleaner retrieval at long contexts.

### The Gating Mechanism

The two gates serve distinct purposes:

- **$\alpha_t$ (decay gate):** Controls how much of the previous state to retain. Values near 1 preserve history; values near 0 allow rapid forgetting. This gives the model token-level control over its effective context window — some tokens (like paragraph boundaries) may reset the state, while others preserve it.

- **$\beta_t$ (write gate):** Controls how strongly to write the correction into the state. This allows the model to selectively update — a filler token ("the", "is") may write weakly, while a content-bearing token ("Paris", "42.7") may write strongly.

The combination of decay and write gating gives GDN more expressive control than either standard linear attention (no gating) or simple gated RNNs (single gate).

## The 3:1 Hybrid Ratio

Qwen 3.5 arranges layers in a strict 3:1 pattern: three GDN layers followed by one full Gated Attention layer.

```
Layer 1:  Gated DeltaNet → MoE FFN
Layer 2:  Gated DeltaNet → MoE FFN
Layer 3:  Gated DeltaNet → MoE FFN
Layer 4:  Gated Attention → MoE FFN    ← full quadratic attention
Layer 5:  Gated DeltaNet → MoE FFN
Layer 6:  Gated DeltaNet → MoE FFN
...
```

For the 397B flagship with 60 layers, this gives 45 GDN layers and 15 full-attention layers. For the 27B dense model with 64 layers, this gives 48 GDN layers and 16 full-attention layers.

### Why Not 100% GDN?

Pure linear attention, even with the delta-rule correction, fundamentally cannot represent all the attention patterns that full quadratic attention can. Specifically:

1. **Exact copy operations.** Tasks that require the model to copy a specific token from earlier in the sequence (e.g., reproducing a variable name, citing a number) benefit from the sharp attention distributions that full attention produces. GDN's fixed-size state smooths these distributions.

2. **Long-range syntactic agreement.** Subject-verb agreement across hundreds of intervening tokens requires attending to a specific distant token. Full attention can attend to any position; GDN's recurrent state compresses distant information.

3. **Multi-hop reasoning.** Queries like "What is the capital of the country where X was born?" require attending to multiple specific facts. Full attention can compose these lookups; GDN must encode the necessary information in its fixed-size state.

By retaining full attention every 4th layer, the model ensures these capabilities are preserved. The GDN layers handle the "easy" contextual mixing that constitutes the majority of computation, while the attention layers provide the "hard" precise interactions where needed.

### Why Not 50/50?

The choice of 75% GDN (rather than 50/50 or 90/10) reflects a throughput-quality Pareto analysis. The Qwen team's finding: going beyond 75% GDN degrades quality noticeably on retrieval-heavy benchmarks; going below 75% leaves significant throughput on the table. The 3:1 ratio hits the sweet spot — enough attention for quality, enough GDN for efficiency.

## Memory and Throughput Implications

The practical payoff of the hybrid design:

| Component | Full Attention Model | Hybrid 3:1 Model |
|-----------|---------------------|-------------------|
| KV cache layers | 60 (all) | 15 (25%) |
| KV cache memory (262K context) | ~100% baseline | ~25% baseline |
| GDN state memory | 0 | Fixed per layer (independent of L) |
| Total memory scaling | $O(L)$ per layer | $O(L)$ for 25% + $O(1)$ for 75% |
| Reported throughput gain | 1x | ~19x at long contexts |

The 19x throughput improvement is not purely from reduced memory. It also reflects reduced memory bandwidth pressure — reading a smaller KV cache per decoding step means the GPU spends less time waiting for memory transfers and more time computing. At long contexts where decoding is memory-bandwidth-bound, this is transformative.

## Gated Attention: Not Standard GQA

The full-attention layers in Qwen 3.5 are not identical to Qwen 3's GQA layers. They use **Gated Attention** — a variant that applies similar gating mechanisms to the attention output, providing the model with a learned interpolation between the attention result and the residual. The Gated Attention layers also use a different head configuration: for the 397B flagship, GA layers use 64 query heads but details on KV heads differ from the GDN layers. The GA layers use a larger head dimension (256 vs. 128 for GDN), providing higher per-head capacity for the layers where full attention matters most.

## Implications for Future Architectures

Qwen 3.5's hybrid DeltaNet design represents a concrete data point in the long-running search for sub-quadratic attention. Unlike prior linear-attention proposals that remained research curiosities (Linear Transformer, RetNet, RWKV, Mamba), the GDN hybrid is:

1. **Deployed at scale** — 397B parameters, not a toy model
2. **Open-weight** — reproducible, auditable, measurable
3. **Competitive with full attention** — not a quality sacrifice for efficiency
4. **Hybridized** — acknowledging that linear attention alone is insufficient

The 3:1 ratio is not necessarily optimal for all model sizes or task distributions. But it establishes a production-validated starting point that future architectures can refine.
