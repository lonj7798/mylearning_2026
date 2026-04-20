<!-- scope: modern LLM architecture comparison
     deps: [[ch-03]]
     see-also: [[raschka-attention-variants]], [[hf-mixture-of-experts]]
-->

# The Big LLM Architecture Comparison

- **Core Insight:** Modern LLMs converge on similar components (RoPE, GQA/MLA, SwiGLU, RMSNorm) with variation in expert routing.
- **Guideline:** Compare architectures by their departures from the shared baseline, not by their common components.

- **Author:** Sebastian Raschka, PhD
- **URL:** https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- **Relevant chapters:** All architecture chapters; especially attention variants, normalization, MoE, positional encoding

## Summary
A comprehensive comparison of 15+ major LLM architectures from 2025-2026, covering DeepSeek V3/R1, OLMo 2/3, Gemma 3, Llama 4, Qwen3, SmolLM3, Kimi K2, gpt-oss, Grok 2.5, GLM-4.5, Qwen3-Next, MiniMax-M2, and more. Traces how fundamental transformer structures remain similar despite seven years of development, with key refinements in positional encodings (absolute -> RoPE), attention (MHA -> GQA), and activations (GELU -> SwiGLU).

## Key Content

### Core Architectural Evolution (GPT-2 to 2025+)

Despite seven years of development, fundamental transformer structures remain similar. Key refinements:
- **Positional Embeddings:** Absolute -> Rotational (RoPE)
- **Attention Mechanisms:** Multi-Head Attention -> Grouped-Query Attention (GQA)
- **Activation Functions:** GELU -> SwiGLU

### Major Model Architectures

**DeepSeek V3/R1 (671B total, 37B active)**
- Multi-Head Latent Attention (MLA): Compresses K/V tensors into lower-dimensional space before caching — reduces memory while slightly improving performance over MHA
- MoE: 256 experts, only 9 active per token
- Shared Expert: Always-active expert in every MoE module
- MLA outperformed GQA in ablation studies from DeepSeek-V2

**OLMo 2 (7B-32B)**
- Post-Norm: RMSNorm placed after attention/FFN (contrasts with GPT's pre-norm)
- QK-Norm: Additional RMSNorm on queries and keys before RoPE
- Retains standard MHA (not GQA)
- Post-norm + QK-norm improved training stability

**Gemma 3 (27B)**
- Sliding Window Attention: 5:1 ratio of sliding to global attention
- Hybrid normalization: RMSNorm both before and after attention/FFN
- 1024-token sliding window (reduced from Gemma 2's 4096)
- Ablation: minimal perplexity degradation despite substantial memory savings

**Llama 4 (400B)**
- GQA attention
- MoE: 2 active experts (8,192 hidden size each)
- Alternating MoE and dense modules every other layer

**Qwen3 (0.6B-235B-A22B)**
- Dense variants: Deeper architectures with more transformer blocks, fewer attention heads
- MoE variants: No shared experts, 8 active experts with 2,048 hidden dimensions

**SmolLM3 (3B)**
- No Positional Embeddings (NoPE): Omits RoPE entirely in every 4th layer; relies on causal masking
- NoPE benefits: Better length generalization

**Kimi K2 (1 trillion parameters)**
- Adopts DeepSeek V3 design scaled larger
- First production use of Muon optimizer (instead of AdamW)

**gpt-oss (20B, 120B)**
- Wider architecture: embedding dim 2880 vs 2048
- Fewer layers: 24 blocks (vs Qwen3's 48) — width over depth
- Attention Sinks: Learned per-head bias logits appended to attention scores
- Sliding Window Attention in alternating layers

**Qwen3-Next (80B-A3B) — Hybrid Architecture**
- Gated DeltaNet + Gated Attention: 3:1 ratio replacing full attention with linear attention
- Multi-Token Prediction (MTP): Predicts k future tokens simultaneously; enables speculative decoding
- Extended context: 262k tokens
- DeltaNet: Fast-weight delta rule update with convolution-based kernels

**MiniMax-M2 (230B)**
- Abandoned linear attention from M1 predecessor; returned to full attention for better accuracy
- "Linear attention degraded multi-turn and reasoning performance"
- Only 4.37% active parameters (10B active)
- Partial RoPE: Rotations only on first rotary_dim channels

### Attention Mechanism Evolution

| Mechanism | Description | Examples |
|-----------|-------------|----------|
| MHA | Original standard, parameter-intensive | GPT-2, OLMo 2 |
| GQA | Multiple query heads share K/V projections | Llama 3/4, Gemma 3, Qwen3 |
| MLA | Compress K/V to lower dimensions before caching | DeepSeek V3, Kimi K2 |
| SWA | Local context windows, O(n*w) complexity | Gemma 3, OLMo 3 |
| Linear variants | DeltaNet, Gated DeltaNet — O(n) | Qwen3-Next, Kimi Linear |

### Normalization Strategies

- **Pre-Norm (GPT standard):** RMSNorm before attention/FFN
- **Post-Norm (OLMo 2):** RMSNorm after modules, within residual paths
- **Hybrid (Gemma 3):** Both pre-norm and post-norm
- **QK-Norm:** Additional per-layer normalization for query/key stability

### Expert Configuration Patterns

- **Shared Experts:** Always-active modules (DeepSeek V3, GLM-4.5, Grok 2.5)
- **Many Small Experts:** Recent preference (256 in DeepSeek V3, 2048 in Qwen3-Next)
- **Few Large Experts:** Older approach (8 in Grok 2.5)
- **Activation:** Typically 1-9 active experts per token from pools of 32-2048

### Parameter Counts

| Model | Total | Active | Architecture |
|-------|-------|--------|-------------|
| Kimi K2 | 1T | 37B | MoE |
| DeepSeek V3 | 671B | 37B | MoE |
| Llama 4 | 400B | 17B | MoE |
| GLM-4.5 | 355B | ~32B | MoE |
| Qwen3-235B | 235B | 22B | MoE |
| MiniMax-M2 | 230B | 10B | MoE |
| Gemma 3 | 27B | 27B | Dense |
| Olmo 3 | 32B | 32B | Dense |

## Notable Insights
- Modern LLM progress emphasizes efficiency over raw parameter scaling. Successful 2025 models combine carefully tuned normalization, sparse expert routing, hybrid attention, and training innovations.
- MLA demonstrated superior performance vs GQA at scale — but GQA persists due to simplicity and robustness. The choice depends on engineering maturity and scale.
- MiniMax-M2's retreat from linear attention is cautionary: linear attention degraded multi-turn and reasoning performance, suggesting the quadratic attention cost may be a necessary price for certain capabilities.
- The NoPE approach (omitting positional encoding entirely) in SmolLM3 is a provocative finding that challenges a fundamental assumption of transformer design.
- The trend toward hybrid architectures (mixing linear and quadratic attention) suggests the future is not one-size-fits-all but rather heterogeneous attention within a single model.
