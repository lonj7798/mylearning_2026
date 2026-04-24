<!-- scope: GPT-1 vs GPT-2 vs GPT-3 architecture differences, parameter-level comparison, parent: [[ch-04]] -->

# GPT-1 / GPT-2 / GPT-3: Architecture Diff Table

The GPT lineage is often summarized as "same architecture, just bigger." That summary is mostly correct but obscures important refinements that were prerequisite to stable training at scale. This excerpt catalogues every architectural difference across the three generations, distinguishing changes that were essential for scaling from those that were incidental.

---

## Full Comparison Table

| Component | GPT-1 ([[gpt-1\|paper]]) | GPT-2 ([[gpt-2\|paper]]) | GPT-3 ([[gpt-3\|paper]]) |
|-----------|------|------|------|
| **Parameters** | 117M | 1.5B (13x) | 175B (117x GPT-2) |
| **Layers** | 12 | 48 | 96 |
| **Model dim ($d_\text{model}$)** | 768 | 1600 | 12288 |
| **Attention heads** | 12 | 25 | 96 |
| **Head dim ($d_k$)** | 64 | 64 | 128 |
| **FFN dim** | 3072 (4x) | 6400 (4x) | 49152 (4x) |
| **Context window** | 512 | 1024 | 2048 |
| **Vocab size** | ~40K (BPE) | 50,257 (byte-BPE) | 50,257 (byte-BPE) |
| **LayerNorm** | Post-norm | Pre-norm | Pre-norm |
| **Residual init** | Standard | $1/\sqrt{N}$ | $1/\sqrt{N}$ |
| **Positional enc.** | Learned | Learned | Learned |
| **Activation** | GELU | GELU | GELU |
| **Attention pattern** | Dense | Dense | Alternating dense + sparse |
| **Batch size** | 64 (fixed) | 512 (fixed) | 32K to 3.2M tokens (ramped) |
| **Training data** | BooksCorpus (800M words) | WebText (40GB, 8M pages) | 300B tokens (filtered blend) |
| **Embedding tying** | No | Yes | Yes |
| **Training paradigm** | Pre-train + fine-tune | Zero-shot | Few-shot / in-context |

---

## Essential-for-Scaling Changes

### Pre-Norm LayerNorm (GPT-2)

GPT-1 used the original Transformer's **post-norm** placement: LayerNorm is applied after the residual addition.

```
# Post-norm (GPT-1)
x = x + Attention(x)
x = LayerNorm(x)
```

GPT-2 moved to **pre-norm**: LayerNorm at the input of each sub-block, with the residual connection bypassing the normalization.

```
# Pre-norm (GPT-2, GPT-3)
x = x + Attention(LayerNorm(x))
```

This matters because the residual pathway is now unobstructed -- the signal flows directly through addition without passing through normalization. In deep networks (48+ layers), post-norm creates gradient instability because the residual stream is repeatedly squeezed through normalization. Pre-norm keeps the residual magnitude stable regardless of depth.

### Residual Weight Scaling (GPT-2)

Residual layer weights are initialized with a scale factor of $1/\sqrt{N}$ where $N$ is the number of residual layers. At initialization, the contribution of each residual block to the output is proportional to $1/\sqrt{N}$, preventing the signal from exploding as it passes through $N$ blocks. Without this scaling, the variance of the residual stream grows linearly with depth, causing training instability in deep networks.

### Batch Size Warmup (GPT-3)

GPT-3 gradually ramped batch size from 32K tokens to 3.2M tokens during training. The motivation: early in training, the loss landscape is noisy and high-variance. Small batches provide more frequent parameter updates (more gradient steps per token), which is more efficient for navigating the noisy early landscape. As training progresses and the loss landscape smooths, larger batches become more compute-efficient -- each gradient estimate is more reliable, so fewer steps are needed per token.

---

## Incidental Changes

### Byte-Level BPE (GPT-2)

GPT-1's BPE operated on Unicode characters and required an unknown token for unseen characters. GPT-2's byte-level BPE operates on raw bytes (256 base tokens), eliminating the UNK token entirely. This was a tokenization improvement ([[ch-05]]) rather than an architecture change, but it enabled open-vocabulary handling that proved essential for generalization.

### Sparse Attention (GPT-3)

GPT-3 uses alternating dense and locally-banded sparse attention patterns, similar to the Sparse Transformer. The paper does not emphasize this change, and it is unclear how much it contributed to GPT-3's capabilities versus scaling alone. Later models (LLaMA, Mistral) reverted to fully dense attention, suggesting sparse attention was not the critical factor.

### Embedding-Output Weight Tying (GPT-2)

GPT-2 reuses the input embedding matrix (transposed) as the output projection for generating logits. This is both a parameter efficiency trick (saves $V \times d_\text{model}$ parameters) and a structural insight: the model's "understanding" of a token as input is tied to its "prediction" of that token as output ([[alammar-illustrated-gpt2|blog]]).

---

## The Pattern

The progression reveals a clear pattern: **architecture changes that mattered were stability improvements, not capacity improvements.** Pre-norm, residual scaling, and batch size warmup all address training stability at depth. The capacity came from raw scaling -- more layers, wider layers, more data. The architecture was already sufficient at GPT-1; the challenge was making it trainable at 100x the scale.

This is why the decoder-only Transformer has proven so durable. The base design from 2017 needed only minor stability patches to scale from 117M to 175B and beyond. No fundamental architectural change was required.

---

## References

- [[gpt-1|Radford et al. "Improving Language Understanding by Generative Pre-Training" (2018) (paper)]]
- [[gpt-2|Radford et al. "Language Models are Unsupervised Multitask Learners" (2019) (paper)]]
- [[gpt-3|Brown et al. "Language Models are Few-Shot Learners" (2020) (paper)]]
- [[alammar-illustrated-gpt2|Alammar, "The Illustrated GPT-2" (blog)]]
