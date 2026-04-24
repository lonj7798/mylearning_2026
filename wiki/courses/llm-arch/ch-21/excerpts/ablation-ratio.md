# Excerpt: Attention-to-Mamba Ratio Ablation Analysis

<!-- source: [[jamba|report]], [[mamba|paper]] -->

## The Ablation Setup

The Jamba technical report tested multiple attention-to-Mamba ratios at controlled scale (ablation models up to 250B training tokens) to determine the optimal interleaving pattern. The key question: **how many attention layers are necessary to recover in-context learning capability from a predominantly-SSM model?**

The ratios tested span the full spectrum:

| Configuration | Attention Layers | Mamba Layers | Attention Fraction |
|---------------|------------------|--------------|--------------------|
| Pure Mamba | 0 | 32 | 0% |
| 1:15 | 2 | 30 | 6.25% |
| 1:7 | 4 | 28 | 12.5% |
| 1:3 | 8 | 24 | 25% |
| 1:1 | 16 | 16 | 50% |
| Pure Transformer | 32 | 0 | 100% |

## Key Findings

### Finding 1: Quality Saturates Quickly with Attention

Standard language modeling perplexity shows minimal variation across configurations — all ratios from 1:7 to pure Transformer achieve comparable perplexity on held-out text. The pure Mamba configuration also achieves competitive perplexity, confirming findings from [[mamba|paper]] that SSMs match Transformers on standard LM quality.

The divergence appears on **in-context learning tasks**: few-shot classification, format adherence (following specific output templates), and needle-in-a-haystack retrieval.

- **Pure Mamba (0 attention):** Fails to reliably follow output format specifications. On few-shot tasks requiring the model to mimic the format of provided examples, pure Mamba degrades relative to Transformers. The fixed-size state does not reliably preserve format patterns from the in-context examples.

- **1:7 ratio (4 attention layers):** Recovers essentially all of the in-context learning capability. The model develops functional induction heads in the attention layers — attention patterns that match the current token against earlier context and copy what followed. Four layers is sufficient for this circuit to form.

- **1:3 to pure Transformer:** Marginal quality improvement on in-context tasks, not worth the KV cache cost. Each additional attention layer adds linearly to the KV cache without proportional quality gain.

### Finding 2: Induction Heads Emerge with Minimal Attention

The emergence of induction heads — the attention pattern most directly associated with in-context learning — with only 4 attention layers is architecturally significant.

In a pure Transformer, induction heads typically form through a two-layer circuit: one attention head copies previous tokens' keys, and another uses those keys to look up what followed. This circuit requires at least two attention layers in sequence (or within computational reach through the residual stream).

In Jamba, the 4 attention layers are separated by blocks of 7 Mamba layers. The induction-head circuit must form across attention layers that are 8 layers apart in the full stack. This works because:

1. **Mamba layers preserve and enrich positional information** in the residual stream between attention layers. The recurrent state carries implicit position encoding, so the attention layers receive inputs that already contain positional signal.

2. **The attention layers operate on pre-processed features**, not raw token embeddings. By the time input reaches the first attention layer (layer 8), it has been processed by 7 Mamba layers. The representations are richer and more structured than raw embeddings, making the attention computation more effective per layer.

3. **Four attention layers (one per block) provide four "retrieval checkpoints"** distributed evenly through the network depth. Information that cannot be retrieved at the first checkpoint may be retrievable at the second, third, or fourth — after further Mamba processing has refined the representations.

### Finding 3: The Memory-Quality Pareto Frontier

The ablation reveals a sharp Pareto frontier: the 1:7 ratio sits near the optimal point where further attention layers provide negligible quality gain relative to their memory cost.

Quantitatively, moving from 1:7 to 1:3 doubles the KV cache (from 4 to 8 attention layers) for less than 1% quality improvement on in-context tasks. Moving from 1:7 to 1:1 quadruples the KV cache for similarly marginal gains. The 1:7 ratio captures approximately 95% of a pure Transformer's in-context learning capability at 12.5% of its KV cache cost.

## Implications for Hybrid Architecture Design

The ablation results suggest a design principle: **in hybrid SSM-attention architectures, the number of attention layers should be determined by the minimum needed for in-context learning, not by conventional expectations about attention prevalence.**

The conventional assumption — that attention should dominate because it is the "smarter" primitive — is wrong for this design space. Most of a language model's computation is feature transformation (what the SSM and MLP layers do), not precise retrieval (what attention does). Allocating 87.5% of layers to the efficient primitive and 12.5% to the precise-but-expensive primitive is a rational division of labor.

## The KV Cache Arithmetic at Each Ratio

To make the memory implications concrete, consider a model with Jamba's dimensions (GQA with 8 KV heads, $d_k = 128$) at 256K context in FP16:

| Ratio | Attn Layers | KV Cache (256K, FP16) | Relative to Pure Transformer |
|-------|-------------|----------------------|------------------------------|
| Pure Transformer | 32 | 32 GB | 1.0x |
| 1:1 | 16 | 16 GB | 0.5x |
| 1:3 | 8 | 8 GB | 0.25x |
| **1:7 (Jamba)** | **4** | **4 GB** | **0.125x** |
| 1:15 | 2 | 2 GB | 0.0625x |
| Pure Mamba | 0 | 0 GB | 0x |

The cache scales linearly with attention layer count. Each step down the ratio ladder cuts cache linearly, but the quality curve is non-linear — most quality is recovered at the 1:7 point. This asymmetry between linear cache cost and diminishing-returns quality gain is what creates the sharp Pareto frontier.

At 256K context, the difference between 32 GB (pure Transformer with GQA) and 4 GB (Jamba) is the difference between needing multi-GPU serving and fitting comfortably on a single 80GB GPU with room for batching.

## Connection to the SSM State Bottleneck

The ablation results connect directly to the theoretical analysis in [[mamba|paper]] regarding SSM state capacity. Mamba's hidden state has dimension $D \times N$ per layer, where $D$ is the expanded model dimension and $N$ is the state dimension (typically 16). This is a fixed-size representation regardless of sequence length.

The state bottleneck means that as context grows, the SSM must compress more information into the same fixed-size state. At 256K tokens, each Mamba layer's state represents a 16-dimensional compressed summary of 256K tokens of context. This compression is lossy — the SSM learns which information to propagate and which to discard via the selective mechanism ($B_t$, $C_t$, $\Delta_t$ as functions of input). But it cannot guarantee that any specific token can be recovered.

The attention layers resolve this by providing lossless access to the full context (within their window). The 1:7 ratio works because the SSM layers handle the majority of the computation — gradual feature extraction, long-range dependency propagation, contextual enrichment — while the attention layers handle the minority case of precise retrieval that the SSM state cannot guarantee.

This division of labor is analogous to the CPU cache hierarchy: most memory accesses hit the fast, small L1/L2 cache (SSM state), and the slower, larger main memory (attention KV cache) is accessed only when the cache misses. The hybrid architecture is effectively implementing a "memory hierarchy for sequence information."

## Caveats

- The ablations were conducted at moderate scale (up to 250B training tokens). The optimal ratio might shift at much larger training budgets, especially if larger training sets demand more precise few-shot patterns.
- The evaluation focused on English text tasks. Multilingual or code-heavy workloads might require different ratios if those domains demand more precise retrieval or exact pattern matching.
- The 1:7 ratio reflects Jamba's specific layer width and state dimensions. Different SSM state sizes ($N$) or attention head configurations might shift the optimal ratio.
- Attention layer *placement* within each block was not extensively ablated in the public report. Whether the attention layer should be first, last, or in the middle of each block may also affect quality, though the residual stream likely mitigates placement sensitivity.
