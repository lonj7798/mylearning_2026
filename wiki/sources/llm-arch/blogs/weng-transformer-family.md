<!-- scope: transformer variant taxonomy
     deps: [[ch-03]]
     see-also: [[raschka-attention-variants]], [[eleutherai-rope]]
-->

# The Transformer Family Version 2.0

- **Core Insight:** Transformer variants form a taxonomy: position encoding x attention pattern x computation allocation.
- **Guideline:** Classify new architectures by which of these three dimensions they innovate on.

- **Author:** Lilian Weng
- **URL:** https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
- **Relevant chapters:** Attention mechanisms, efficient transformers, positional encoding, context extension, sparse attention

## Summary
An encyclopedic survey (45-minute read) of transformer architectural variants since the original 2017 paper. Covers positional encoding schemes (sinusoidal, learned, relative, RoPE), context extension techniques (Transformer-XL, Compressive Transformer, kNN-LM), sparse attention patterns (Longformer, Big Bird, Reformer), low-rank attention (Linformer, Performers), adaptive computation, and transformers for RL. This is Version 2.0, doubling the length of the original 2020 post.

## Key Content

### Attention Fundamentals

**Scaled dot-product attention:** "The output is a weighted sum of the value vectors, where the weight assigned to each value slot is determined by the dot-product of the query with the corresponding key."

**attn(Q, K, V) = softmax(QK^T / sqrt(d_k)) V**

**Multi-head self-attention** splits inputs into smaller chunks and computes attention across subspaces in parallel. "Independent attention outputs are simply concatenated and linearly transformed into expected dimensions."

### Positional Encoding Variants

**Sinusoidal:** Uses sinusoid functions of different wavelengths, ranging from 2pi to 10000*2pi, assigning different sine/cosine pairs to different dimensions.

**Learned:** Assigns each position a learned column vector. Potentially varies per layer.

**Relative Position (Shaw et al. 2018):** Incorporates relative positional information between keys and values. Maximum relative position clipped to value k, yielding 2k+1 unique edge labels.

**Rotary Position Embedding (RoPE):** Encodes absolute position using rotation matrices multiplied with key/value matrices at each attention layer. Rotations performed on d/2 subspaces with angles proportional to position indices. "Essentially equivalent to sinusoidal positional encoding but formulated as a rotation matrix."

**ALiBi (Attention with Linear Biases):** Adds constant bias terms to scores proportional to distances: softmax(q_i K^T + alpha_i * [0, -1, -2, ..., -(i-1)]). Head-specific weights form geometric sequences rather than being learned.

### Context Extension Techniques

**Transformer-XL:** Reuses hidden states between segments by concatenating previous segment hidden states with current. "Queries only consume hidden states at the current step" while keys and values use extended states.

**Compressive Transformer:** Extends Transformer-XL with both regular memory (FIFO queue) and compressed memory slots. Uses compression functions (max pooling, convolution, dilated convolution) to compress old activations.

**kNN-LM:** Enhances language models by retrieving nearest neighbors from external datastores. SPALM combines short-term memory (Transformer-XL style) with long-term key-value stores using gating mechanisms.

**Memorizing Transformer:** Maintains FIFO caches of past key-value pairs in a specialized kNN-augmented attention layer with learnable per-head gating.

### Distance-Enhanced Attention

**DA-Transformer:** Multiplies attention scores by learnable distance-dependent biases using a sigmoid-like function with per-head parameters.

**ALiBi:** Adds constant bias terms proportional to token distances. Simpler than DA-Transformer — no learned parameters, just geometric head-specific weights.

### Adaptive Modeling

**Adaptive Attention Span:** Different attention heads have different optimal spans. A soft mask function m_z(x) = clip((1/R)(R+z-x), 0, 1) controls effective span. Parameter z learned separately per head with L1 regularization.

**Depth-Adaptive Transformer / CALM:** Attach output classifiers to intermediate layers, allowing tokens to exit early. CALM uses the Learn-then-Test framework for threshold calibration.

### Sparse Attention Patterns

**Fixed Local Context:** Restricts each token to attending only to nearby positions. Reduces complexity from O(L^2) to linear in L.

**Sparse Transformer (Strided):** Factorizes attention through sparse matrix decomposition:
- Strided attention with stride sqrt(n)
- Fixed attention where small token sets summarize locations

**Longformer:** Combines local attention windows with global tokens and dilated sliding windows.

**Big Bird:** Uses local attention, global tokens, and random token attention.

**Reformer (LSH Attention):** Uses Locality-Sensitive Hashing to bucket queries and keys by similarity. Queries only attend to matching hash buckets. Complexity: O(L log L). Also introduces reversible residual layers: y1 = x1 + F(x2); y2 = x2 + G(y1), enabling activation recovery without storage.

**Routing Transformer:** Online k-means clustering instead of static hashing. Complexity: O(L^1.5).

### Low-Rank Attention

**Linformer:** Projects key and value matrices to lower-rank subspaces using learned projection matrices, reducing dimensions from L x d to k x d.

**Performers:** Use random feature maps to approximate softmax. Through orthogonal features and positive random feature maps, achieves linear complexity attention approximation.

### Recurrent Mechanisms

**Universal Transformer:** Combines self-attention with recurrence by sharing parameters across layers. Hidden states evolve through iterative refinement. Can use Adaptive Computation Time for dynamic iteration counts per token.

### Transformers for RL

**Gated Transformer-XL (GTrXL):** Two modifications for stability:
1. Layer normalization only on input streams (not shortcuts)
2. GRU-style gating replacing residual connections: g(x,y) = (1-z)*x + z*h_hat

**Decision Transformer:** Formulates RL as conditional sequence modeling. Feeds (return-to-go, state, action) triplets to Transformer, outputting optimal actions for specified returns.

## Notable Insights
- Sparse attention reduces complexity but may harm expressiveness — there is no free lunch in attention efficiency.
- ALiBi's simplicity (just adding distance-proportional biases, no learned parameters) is notable given it achieves competitive extrapolation to longer sequences.
- Reformer's reversible residual layers are elegant: by storing only the final activations, the intermediate activations for backprop can be recovered mathematically, halving memory for activations.
- The Decision Transformer reframing of RL as sequence modeling is conceptually powerful: instead of learning value functions, you learn to predict actions conditioned on desired outcomes.
- Context extension remains an active area — the fundamental tension between memory cost and information retention is not fully resolved by any single approach.
