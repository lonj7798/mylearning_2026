<!-- scope: GLU variants — gated activations in Transformer FFNs
     deps: [[attention-is-all-you-need]]
     see-also: [[rmsnorm]], [[gpt-1]]
-->

# GLU Variants Improve Transformer
- **Core Insight:** Gated linear units consistently outperform standard ReLU/GELU activations in Transformer FFN sublayers.
- **Guideline:** Use SwiGLU as the default FFN activation; set d_ff to 8/3 * d_model (rounded to a convenient multiple) to keep parameter count matched.
- **Authors:** Noam Shazeer
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2002.05202
- **Relevant chapters:** feed-forward networks, activation functions, transformer architecture

## Abstract
Gated Linear Units (arXiv:1612.08083) consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feed-forward sublayers of the Transformer (arXiv:1706.03762) sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations.

## Key Contributions
- Proposes replacing the standard Transformer FFN activation (ReLU or GELU) with gated linear unit (GLU) variants
- Tests multiple GLU variants: GEGLU, SwiGLU, ReGLU, and others, showing consistent quality improvements on language modeling
- SwiGLU (using the Swish/SiLU activation) emerges as particularly effective and has been adopted by most modern LLMs (LLaMA, PaLM, Mistral, etc.)
- Demonstrates that the gating mechanism is more important than the specific choice of activation function
- Provides a simple, drop-in replacement for the FFN sublayer that improves quality with minimal architectural change

## Architecture Details
- **Standard Transformer FFN:** FFN(x) = W_2 * activation(W_1 * x + b_1) + b_2, with two projections of shapes (d_model, d_ff) and (d_ff, d_model)
- **GLU formulation:** GLU(x) = (W_1 * x) ⊙ sigma(W_2 * x), where ⊙ is element-wise multiplication and sigma is the sigmoid function. This uses two parallel linear projections instead of one
- **GLU variants replace sigma with other activations:**
  - **ReGLU:** GLU with ReLU: (W_1 * x) ⊙ ReLU(V * x)
  - **GEGLU:** GLU with GELU: (W_1 * x) ⊙ GELU(V * x)
  - **SwiGLU:** GLU with Swish/SiLU: (W_1 * x) ⊙ Swish(V * x), where Swish(x) = x * sigmoid(beta * x)
- **Three-matrix FFN:** The gated variants require three weight matrices (W_1, V, W_2) instead of two (W_1, W_2), because the gating branch needs its own projection. To keep parameter count comparable, d_ff is typically reduced from 4 * d_model to (8/3) * d_model (often rounded to a multiple of 256)
- **Parameter budget adjustment:** With the standard FFN, d_ff = 4d. With GLU variants having 3 matrices, setting d_ff = (8/3)d keeps total parameters roughly equal: 3 * d * (8/3)d = 8d^2 vs. 2 * d * 4d = 8d^2
- **SwiGLU in practice:** LLaMA uses SwiGLU with d_ff = (8/3) * d_model (rounded). PaLM also uses SwiGLU. This has become the default FFN variant in modern LLMs
- **Evaluation:** Tested on machine translation and language modeling tasks, showing perplexity improvements across the board

## Tradeoffs Discussed
- GLU variants add a third weight matrix to the FFN, increasing the number of matrix multiplications from 2 to 3 per FFN sublayer (though d_ff is reduced to compensate on total parameters)
- The paper does not provide a clear theoretical explanation for why gating helps — the improvements are empirically observed
- The gating mechanism adds a multiplicative interaction that may complicate gradient flow compared to simple activation functions, though this does not cause training instability in practice
- No single GLU variant dominates across all tasks; SwiGLU and GEGLU are close, with SwiGLU slightly preferred in most modern implementations
- The reduced d_ff (to match parameter count) means each individual FFN "neuron" has less capacity, but the gating mechanism more than compensates
