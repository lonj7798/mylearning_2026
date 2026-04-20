<!-- scope: RoPE — relative position via rotation in complex space
     deps: [[attention-is-all-you-need]]
     see-also: [[alibi]], [[yarn]]
-->

# RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Core Insight:** Encoding relative position via rotation in complex space enables length extrapolation while keeping absolute position information.
- **Guideline:** Use RoPE as the default positional encoding; its rotary formulation is both efficient and pairs well with context-extension methods.
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **Year:** 2021
- **URL:** https://arxiv.org/abs/2104.09864
- **Relevant chapters:** positional encoding, attention mechanisms, long-context modeling

## Abstract
Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface.

## Key Contributions
- Proposes Rotary Position Embedding (RoPE), which encodes absolute position via a rotation matrix while naturally capturing relative position information in the attention dot product
- Achieves decaying inter-token dependency as relative distance increases, providing a built-in locality inductive bias
- Enables flexible sequence length handling without retraining or interpolation
- Extends relative position encoding to linear self-attention variants, which prior methods could not do
- Provides theoretical analysis showing why RoPE's rotary formulation preserves relative position information through the inner product

## Architecture Details
- **Core formulation:** For a query/key vector at position m, RoPE applies a rotation: f(x, m) = R(m)x, where R(m) is a block-diagonal rotation matrix. Each 2D block rotates by angle m * theta_i, where theta_i = 10000^(-2i/d) for dimension pair i
- **Relative position via inner product:** The dot product q_m^T k_n depends only on the relative position (m - n) because R(m)^T R(n) = R(n - m). This elegantly unifies absolute and relative position encoding
- **Rotation matrix structure:** R_theta,m = diag(R(m*theta_1), R(m*theta_2), ..., R(m*theta_{d/2})), where each R(angle) is a standard 2D rotation matrix [[cos, -sin], [sin, cos]]
- **Long-range decay:** The inner product between rotated vectors naturally decays with increasing relative distance, providing a soft locality bias without explicit windowing
- **Implementation:** Equivalent to element-wise multiplication of the query/key with sinusoidal functions of position, making it computationally cheap (no additional parameters)
- **Compatibility:** Works with any transformer variant; adopted widely in LLaMA, PaLM, GPT-NeoX, and most modern open-weight LLMs

## Tradeoffs Discussed
- RoPE encodes position only in the query-key dot product, not in the value projection, meaning value representations are position-agnostic
- While RoPE supports arbitrary sequence lengths in principle, models trained with RoPE at a fixed context length do not automatically extrapolate well to much longer sequences (motivating later work like YaRN and NTK-aware interpolation)
- The sinusoidal frequency schedule (base 10000) is a hyperparameter; suboptimal choices can hurt performance at certain context lengths
- Adds minimal computational overhead compared to learned positional embeddings, but requires careful implementation for numerical stability at very long sequences
