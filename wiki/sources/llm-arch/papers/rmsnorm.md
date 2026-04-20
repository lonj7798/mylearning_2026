<!-- scope: RMSNorm — simpler normalization by dropping mean-centering
     deps: [[layer-norm]]
     see-also: [[pre-norm-vs-post-norm]], [[glu-variants]]
-->

# Root Mean Square Layer Normalization
- **Core Insight:** Dropping mean-centering from LayerNorm works just as well and is faster; only the re-scaling component is essential.
- **Guideline:** Use RMSNorm instead of LayerNorm in all new Transformer models; it is simpler, faster, and the default in modern LLMs.
- **Authors:** Biao Zhang, Rico Sennrich
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1910.07467
- **Relevant chapters:** normalization techniques, training stability, computational efficiency

## Abstract
Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and significantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%~64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.

## Key Contributions
- Hypothesizes and validates that the mean-centering component of LayerNorm is dispensable — only the scaling (variance normalization) component is essential
- Proposes RMSNorm: a simpler normalization that uses root mean square instead of mean and variance, eliminating the mean computation
- Introduces partial RMSNorm (pRMSNorm) that estimates RMS from only p% of inputs for further efficiency
- Demonstrates 7-64% wall-clock speedup over LayerNorm with comparable model quality across tasks
- RMSNorm has become the default normalization in most modern LLMs (LLaMA, GPT-NeoX, Mistral, PaLM, etc.)

## Architecture Details
- **LayerNorm formula:** LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta, requiring computation of both mean and variance across the hidden dimension
- **RMSNorm formula:** RMSNorm(x) = gamma * x / RMS(x), where RMS(x) = sqrt((1/n) * sum(x_i^2)). No mean subtraction, no beta bias term
- **Computational savings:** RMSNorm eliminates the mean computation (one reduction over the hidden dimension) and the mean subtraction (one element-wise operation). On GPUs, this reduces kernel launch overhead and memory traffic
- **Re-scaling invariance:** RMSNorm provides re-scaling invariance: if the input is scaled by a constant c, the output is unchanged (RMSNorm(cx) = RMSNorm(x) * sign(c)). This property stabilizes training by making the network invariant to input scale
- **Implicit learning rate adaptation:** The re-scaling invariance means that the effective learning rate adapts based on the norm of the weights, similar to weight normalization
- **Partial RMSNorm (pRMSNorm):** Estimates RMS from a random or fixed subset of p% of the hidden dimensions. This further reduces computation while maintaining the re-scaling property. In practice, p=25% often suffices
- **No learnable bias:** Unlike LayerNorm which has both gamma (scale) and beta (shift) parameters, RMSNorm uses only gamma. This simplifies the layer and slightly reduces parameters
- **Pre-norm placement:** In modern LLMs, RMSNorm is typically applied before attention and FFN sublayers (pre-norm architecture), which further improves training stability

## Tradeoffs Discussed
- Removing the mean-centering component sacrifices re-centering invariance — the network is no longer invariant to constant shifts in the input. The paper argues empirically this does not hurt performance
- RMSNorm provides smaller speedups when the hidden dimension is small (e.g., 7% for small models) because the mean computation is a smaller fraction of total work. Speedups are larger for RNNs and models with larger hidden dimensions
- pRMSNorm introduces a slight approximation that could theoretically affect model quality, though experiments show the effect is negligible
- The simplicity of RMSNorm means fewer opportunities for the normalization layer to correct distribution issues (no centering), which could matter for specific architectures or tasks not tested in the paper
