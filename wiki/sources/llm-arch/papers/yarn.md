<!-- scope: YaRN — temperature scaling of RoPE for context extension
     deps: [[rope]]
     see-also: [[alibi]], [[flash-attention]]
-->

# YaRN: Efficient Context Window Extension of Large Language Models
- **Core Insight:** Temperature-scaling of RoPE frequencies extends context window well beyond training length with minimal fine-tuning.
- **Guideline:** To extend a RoPE model's context, apply NTK-aware interpolation + attention temperature scaling and fine-tune for ~400 steps on long data.
- **Authors:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.00071
- **Relevant chapters:** positional encoding, context extension, RoPE, long-context modeling

## Abstract
Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset.

## Key Contributions
- Proposes YaRN (Yet another RoPE extensioN), a compute-efficient method for extending the context window of RoPE-based language models
- Requires 10x fewer tokens and 2.5x fewer training steps than prior context extension methods (e.g., position interpolation, Code Llama's approach)
- Demonstrates effective extrapolation beyond the fine-tuning context length, not just interpolation to the target length
- Surpasses prior state-of-the-art methods for context window extension on perplexity and downstream benchmarks
- Combines multiple techniques (NTK-aware interpolation, attention temperature scaling, dynamic scaling) into a unified approach

## Architecture Details
- **The problem:** RoPE encodes position using frequencies theta_i = base^(-2i/d). Models trained at context length L fail at length L' > L because the high-frequency RoPE dimensions experience position values outside their training distribution
- **Position Interpolation (PI):** The simplest approach scales all positions by L/L', so position m becomes m * L/L'. This keeps all positions within the training range but compresses fine-grained positional distinctions
- **NTK-aware interpolation:** Instead of uniformly scaling all frequencies, YaRN modifies the RoPE base frequency: base' = base * (L'/L)^(d/(d-2)). This effectively interpolates low frequencies (which need it) while preserving high frequencies (which encode local position and need less adjustment)
- **Frequency-dependent interpolation:** YaRN further refines this by applying different scaling factors to different frequency bands. High-frequency dimensions (local position info) are left mostly unscaled. Low-frequency dimensions (global position info) are interpolated more aggressively. The interpolation factor for dimension i is a function of the wavelength relative to the original context length
- **Attention temperature scaling:** YaRN applies a learned or fixed temperature scaling factor t to the attention logits: softmax(q^T k / (sqrt(d) * t)). This compensates for the increased entropy in attention distributions that occurs when the context is extended (more positions to attend to)
- **Dynamic scaling:** For applications where the actual sequence length varies, YaRN can dynamically adjust the scaling factor based on the current sequence length rather than a fixed target length. This avoids degradation on short sequences while still enabling long sequences
- **Fine-tuning recipe:** Start from a pretrained RoPE model, apply YaRN scaling, and fine-tune on a small amount of long-context data (400 steps on 64K-length sequences is sufficient for LLaMA). The 10x token efficiency comes from the better initialization provided by NTK-aware interpolation
- **Extrapolation capability:** YaRN models can generalize to context lengths beyond their fine-tuning length. For example, a model fine-tuned at 64K can maintain quality at 128K, unlike PI which degrades sharply beyond the fine-tuning length

## Tradeoffs Discussed
- YaRN still requires some fine-tuning (not zero-shot context extension), though the amount is small (400 steps). Truly training-free extension remains an open problem
- The frequency-dependent interpolation introduces additional hyperparameters (the ramp function boundaries) that need to be set per model architecture
- Attention temperature scaling is an approximation; the optimal temperature may vary by layer and head, but YaRN uses a single global factor for simplicity
- Dynamic scaling adds a small computational overhead to compute the scaling factor per sequence, and the quality is slightly lower than static scaling when the target length is known in advance
- YaRN is specific to RoPE-based models; it does not apply to models using ALiBi, learned positional embeddings, or other position encoding methods
- Very extreme extension ratios (e.g., 4K -> 1M) may still degrade quality; YaRN has been validated up to moderate extension ratios (e.g., 4K -> 128K)
