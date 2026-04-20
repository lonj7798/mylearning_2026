<!-- scope: ALiBi — linear attention bias for length extrapolation
     deps: [[attention-is-all-you-need]]
     see-also: [[rope]], [[yarn]]
-->

# Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
- **Core Insight:** A simple linear distance penalty on attention scores is simpler than learned position embeddings and extrapolates better to unseen lengths.
- **Guideline:** If you need training-free length extrapolation and can tolerate a strong recency bias, ALiBi is the zero-parameter option; otherwise prefer RoPE for flexibility.
- **Authors:** Ofir Press, Noah A. Smith, Mike Lewis
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2108.12409
- **Relevant chapters:** positional encoding, length extrapolation, attention mechanisms, long-context modeling

## Abstract
Since the introduction of the transformer model by Vaswani et al. (2017), a fundamental question has yet to be answered: how does a model achieve extrapolation at inference time for sequences that are longer than it saw during training? We first show that extrapolation can be enabled by simply changing the position representation method, though we find that current methods do not allow for efficient extrapolation. We therefore introduce a simpler and more efficient position method, Attention with Linear Biases (ALiBi). ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance. We show that this method trains a 1.3 billion parameter model on input sequences of length 1024 that extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048 but training 11% faster and using 11% less memory. ALiBi's inductive bias towards recency also leads it to outperform multiple strong position methods on the WikiText-103 benchmark.

## Key Contributions
- Demonstrates that length extrapolation in Transformers is primarily a function of the position encoding method, not the model architecture
- Introduces ALiBi, which replaces positional embeddings entirely with a simple linear bias added to attention scores
- Achieves effective extrapolation to 2x training length (1024 -> 2048) with no quality loss, while using 11% less memory and training 11% faster
- Shows that the recency bias inherent in ALiBi improves language modeling quality on benchmarks like WikiText-103
- Provides a much simpler positional encoding scheme with zero learned parameters for position

## Architecture Details
- **Core mechanism:** Instead of adding positional embeddings to token embeddings, ALiBi adds a static bias to the query-key dot product: attention(q_i, k_j) = q_i^T k_j - m * |i - j|, where m is a head-specific slope and |i - j| is the distance between positions
- **No positional embeddings:** ALiBi completely removes positional embeddings from the input. Position information enters only through the attention bias, meaning the token embeddings are purely semantic
- **Head-specific slopes:** Each attention head h has a different slope m_h, set as a geometric sequence: m_h = 2^(-8h/H) for h = 1, ..., H, where H is the total number of heads. This gives slopes like 1/2, 1/4, 1/8, ... for 8 heads
- **Recency inductive bias:** Heads with larger slopes (steeper penalty) focus on nearby tokens; heads with smaller slopes attend more broadly. This creates a multi-scale attention pattern from local to global
- **Linear penalty:** The penalty is strictly linear in distance, meaning the bias grows without bound for distant tokens. This naturally implements a soft windowing effect that becomes stronger for more distant positions
- **No learned parameters:** The slopes are fixed (not learned), making ALiBi a zero-parameter position encoding method. This reduces model size and eliminates a source of overfitting
- **Extrapolation mechanism:** Because the bias is a simple linear function of distance, it generalizes naturally to distances longer than seen in training — there are no learned embeddings that become undefined at unseen positions
- **Implementation:** ALiBi can be implemented as a simple matrix addition to the attention logits before softmax, with negligible computational cost

## Tradeoffs Discussed
- ALiBi's linear distance penalty is a strong inductive bias toward recency, which helps language modeling but may hurt tasks requiring long-range uniform attention (e.g., retrieval, global reasoning)
- Extrapolation is demonstrated up to 2x training length; extrapolation to much longer sequences (e.g., 10x) may still degrade, and the paper does not test extreme extrapolation ratios
- The fixed geometric slope schedule is a design choice that may not be optimal for all architectures or tasks; there is no mechanism to learn or adapt the slopes
- ALiBi has been largely superseded by RoPE in modern LLMs, partly because RoPE's rotary formulation provides more flexibility and pairs better with context extension methods like YaRN
- The strict distance penalty may discard useful positional information that absolute or rotary encodings can capture (e.g., absolute position within a document)
