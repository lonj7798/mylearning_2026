<!-- scope: original Transformer architecture — self-attention replaces recurrence
     deps: [[ch-01]]
     see-also: [[bahdanau-attention]], [[flash-attention]], [[rope]]
-->

# Attention Is All You Need
- **Core Insight:** Self-attention alone (no recurrence, no convolution) is sufficient for sequence modeling.
- **Guideline:** Start every architecture from the Transformer blueprint; deviate only when you have evidence attention is the bottleneck.
- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- **Year:** 2017
- **URL:** https://arxiv.org/abs/1706.03762
- **Relevant chapters:** Transformer architecture, self-attention mechanism, positional encoding, encoder-decoder design

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## Key Contributions
- Introduced the Transformer architecture, eliminating recurrence and convolutions entirely in favor of pure attention mechanisms
- Proposed multi-head self-attention, allowing the model to jointly attend to information from different representation subspaces at different positions
- Introduced scaled dot-product attention with the scaling factor 1/sqrt(d_k) to counteract vanishing gradients in large dimensions
- Demonstrated that attention-only models achieve state-of-the-art on machine translation while training significantly faster than recurrent/convolutional alternatives
- Introduced sinusoidal positional encodings to inject sequence order information without learned parameters

## Key Figures/Tables to Study
- **Figure 1** (The Transformer architecture): The full encoder-decoder diagram -- the single most referenced architecture figure in modern deep learning. Study the data flow, residual connections, and layer normalization placement.
- **Figure 2** (Scaled Dot-Product Attention and Multi-Head Attention): Shows the core computation. Understand Q, K, V projections and how multiple heads are concatenated.
- **Table 2** (Variations on the Transformer): Ablation study showing the effect of varying number of heads, key/value dimensions, model size, dropout, and attention type. Critical for understanding design tradeoffs.
- **Table 3** (English-to-German and English-to-French BLEU results): Final translation results showing the Transformer outperforming all prior models at a fraction of the training cost.

## Architecture Details
- **Model dimension (d_model):** 512
- **Feed-forward inner dimension (d_ff):** 2048
- **Number of attention heads:** 8
- **Head dimension (d_k = d_v):** 64 (d_model / h)
- **Number of encoder layers:** 6
- **Number of decoder layers:** 6
- **Dropout:** 0.1 (applied to sub-layer outputs, attention weights, and embeddings)
- **Optimizer:** Adam with beta_1=0.9, beta_2=0.98, epsilon=1e-9
- **Learning rate schedule:** Warmup + inverse square root decay (warmup_steps=4000)
- **Label smoothing:** 0.1
- **Training hardware:** 8 NVIDIA P100 GPUs
- **Base model training time:** ~12 hours (100K steps)
- **Big model training time:** 3.5 days (300K steps)
- **Positional encoding:** Sinusoidal (sin/cos functions of different frequencies)
- **Attention function:** Scaled dot-product: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- **Residual connections:** Around each sub-layer, followed by layer normalization
- **Vocabulary:** Byte-pair encoding with ~37K tokens (shared source-target)
