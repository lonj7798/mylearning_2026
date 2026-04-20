<!-- scope: attention mechanism families (MHA, GQA, MLA, SWA, hybrid)
     deps: [[ch-03]]
     see-also: [[raschka-kv-cache]], [[flash-attention-explained]]
-->

# A Visual Guide to Attention Variants in Modern LLMs

- **Core Insight:** Attention is a design space with 7+ families, each optimizing different tradeoffs.
- **Guideline:** Pick attention variant based on your deployment constraint (memory, latency, quality).

- **Author:** Sebastian Raschka, PhD
- **URL:** https://magazine.sebastianraschka.com/p/visual-attention-variants
- **Relevant chapters:** Attention mechanisms, KV cache, efficient inference, hybrid architectures

## Summary
A visual survey of seven attention mechanism families used in modern LLMs (as of March 2026), drawn from the LLM Architecture Gallery with 45+ documented entries. Covers Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), Sliding Window Attention (SWA), DeepSeek Sparse Attention (DSA), Gated Attention, and Hybrid architectures combining linear and quadratic attention.

## Key Content

### 1. Multi-Head Attention (MHA)

Standard transformer mechanism. Multiple self-attention heads operate in parallel with different learned projections.

**Mathematical foundation:**
- Q, K, V projections from input X using weight matrices W_q, W_k, W_v
- Raw relevance scores: QK^T
- Softmax normalizes to attention matrix A
- Output: Z = A x V

Different heads specialize in different token relationships — short dependencies, semantic links, syntactic structure.

**Example architectures:** GPT-2, OLMo 2 7B, OLMo 3 7B

### 2. Grouped-Query Attention (GQA)

Multiple query heads share the same key-value projections, reducing KV cache substantially.

**Why it became standard:**
- Significantly cheaper KV caching than MHA
- Modest modeling degradation
- Simpler implementation than MLA
- Spectrum between multi-query attention (1 group) and full MHA

Memory savings become increasingly pronounced as context windows grow. Despite advances like MLA, GQA remains popular for robustness and easier training.

**Examples:** Llama 3 8B, Qwen3 4B, Gemma 3 27B, Mixtral 8x7B

### 3. Multi-Head Latent Attention (MLA)

Rather than reducing how many K/V heads are stored (GQA), MLA compresses what gets cached into latent representations. Stores compressed latent representation; reconstructs usable state during inference.

**Performance data from DeepSeek-V2 ablations:**
- GQA: Performance degradation vs MHA at scale
- MLA: Remained competitive with MHA; could slightly outperform when properly tuned

Works best at ~100B+ parameters; smaller models often benefit more from GQA.

**Examples:** DeepSeek V3, Kimi K2, GLM-5, Mistral Large 3

### 4. Sliding Window Attention (SWA)

Each token attends only to a fixed window of recent tokens — "local attention." Many architectures use hybrid: local layers + occasional global attention layers.

**Gemma 3 configuration:**
- 5:1 local-to-global ratio
- 1,024-token window (down from 4,096)
- Ablation showed minimal modeling performance impact

Often paired with GQA: SWA limits context, GQA reduces state per token.

**Examples:** Gemma 3, OLMo 3, Arcee Trinity, Step 3.5 Flash

### 5. DeepSeek Sparse Attention (DSA)

Unlike SWA's fixed-window locality, DSA uses learned sparse patterns to select which prior tokens deserve attention.

**Mechanism:**
1. Lightning Indexer: Computes relevance scores using MLA's compressed representations
2. Token Selector: Retains only top-k high-scoring positions

Key advantage: Model learns which past tokens matter rather than hard-coding locality rules.

**Examples:** DeepSeek V3.2, GLM-5

### 6. Gated Attention

Modified full-attention block for stability within hybrid stacks:
1. Output gate scaling attention result before residual addition
2. Zero-centered QK-Norm replacing standard RMSNorm
3. Partial RoPE

Appears as periodic full-attention breaks in hybrid stacks (e.g., 3:1 with Gated DeltaNet in Qwen3-Next/3.5).

### 7. Hybrid Attention Architectures

Replace most full-attention layers with cheaper alternatives (linear attention, state-space models) while retaining selective full-attention for content retrieval.

**Gated DeltaNet (Qwen3-Next/3.5):**
- 3:1 pattern: Three DeltaNet blocks per Gated Attention block
- Delta-rule fast-weight memory updates with learned gates controlling information flow
- Substantially flatter memory growth curves than full attention

**Kimi Linear:** Channel-wise gating instead of scalar gate per head for finer memory control.

**Ling 2.5 (Lightning Attention):** Simpler recurrent linear-attention variant paired with MLA. "Substantially faster than Kimi K2 at 32k tokens."

**Nemotron (Mamba-Transformer Hybrid):** Interleaves Mamba-2 blocks with sparse MoE. Pushes farthest from transformer baseline.

### KV Cache Savings Progression

MHA (baseline) -> GQA (significant reduction) -> MLA (additional compression via latent storage) -> SWA (flattened memory curve) -> Hybrids (most aggressive savings)

### Modeling Quality Hierarchy (at equivalent efficiency)

MHA (reference) -> GQA (minor degradation) -> MLA (competitive or superior at 100B+) -> Hybrids (acceptable degradation for extreme efficiency)

## Notable Insights
- GQA persists despite theoretically superior alternatives because robustness and implementation simplicity matter in production. Several 2025 releases deliberately maintained classic GQA.
- MLA's sweet spot is 100B+ parameters — at smaller scales, the implementation complexity isn't justified over GQA.
- Qwen3.5's promotion of the hybrid architecture from experimental to flagship signals that hybrid attention is transitioning from novelty to mainstream.
- The MiniMax-M2 retreat from linear attention is a cautionary tale: linear attention degraded multi-turn and reasoning performance enough to justify returning to quadratic attention.
- Inference stack maturity matters: locally-run models with GQA often achieve better tok/sec throughput than theoretically superior architectures due to better tooling support.
