<!-- scope: Mamba-2 / SSD — SSMs and linear attention are mathematical duals
     deps: [[mamba]], [[attention-is-all-you-need]]
     see-also: [[flash-attention-2]]
-->

# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- **Core Insight:** SSMs and linear attention are mathematically dual via structured semiseparable matrices; this unified view enables a 2-8x faster Mamba-2 architecture.
- **Guideline:** Use the SSD framework to pick the faster computation path (recurrence vs. quadratic form) depending on sequence length and hardware.
- **Authors:** Tri Dao, Albert Gu
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.21060
- **Relevant chapters:** state space models, attention mechanisms, theoretical foundations, efficient architectures

## Abstract
While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling.

## Key Contributions
- Proves a deep theoretical connection between Transformers (attention) and state space models, showing they are duals of each other through structured semiseparable matrix decompositions
- Develops the State Space Duality (SSD) framework that unifies SSMs and attention variants under a common mathematical structure
- Designs Mamba-2, a refined architecture that is 2-8x faster than Mamba-1 while maintaining competitive language modeling quality
- Shows that the attention matrix in certain Transformer variants can be expressed as a semiseparable matrix, which is exactly the structure computed by SSMs
- Enables algorithmic flexibility: the same computation can be executed via either the SSM recurrence or the attention-like matrix form, choosing whichever is faster for the given hardware and sequence length

## Architecture Details
- **Semiseparable matrices:** A matrix M is semiseparable if every submatrix contained entirely in the lower-triangular part has rank at most N (the state dimension). Causal attention produces such matrices when the head dimension is small. SSM recurrences also produce semiseparable output matrices
- **State Space Duality:** The key insight is that the SSM recurrence y_t = C_t h_t, h_t = A_t h_{t-1} + B_t x_t computes a matrix-vector product y = Mx where M is a structured semiseparable matrix. The same M can also be computed via a quadratic (attention-like) form
- **Dual computation paths:** For short sequences, the quadratic "attention" form is faster (leverages matrix multiply hardware). For long sequences, the linear "recurrence" form is faster. Mamba-2 can switch between them
- **Refined selective SSM:** Mamba-2's core layer uses a scalar-times-identity structure for A (i.e., A_t = a_t * I), which constrains the state transition but enables much more efficient computation. This is the key simplification that yields the 2-8x speedup
- **Multi-head SSM:** Analogous to multi-head attention, Mamba-2 uses multiple SSM "heads" with different state dimensions, increasing expressivity
- **Chunk-wise computation:** The sequence is divided into chunks. Within each chunk, the quadratic form is used. Across chunks, the recurrent form propagates state. This hybrid approach optimizes for GPU hardware
- **Architecture:** Similar to Mamba-1's block structure (linear projection, convolution, SSM, gating, projection) but with the refined SSM layer and additional flexibility from the SSD framework
- **Competitive quality:** On language modeling benchmarks, Mamba-2 matches or slightly exceeds Mamba-1 quality while being significantly faster

## Tradeoffs Discussed
- The scalar-identity constraint on A (A_t = a_t * I) restricts expressivity compared to Mamba-1's diagonal A; this is the price paid for the 2-8x speedup
- Mamba-2 remains "competitive with Transformers" rather than clearly surpassing them, suggesting that the efficiency gains do not translate to quality advantages at current scales
- The theoretical framework assumes causal (lower-triangular) structure; bidirectional or non-causal settings require additional work
- The chunk-wise algorithm introduces a chunk size hyperparameter that affects the compute-memory tradeoff; suboptimal chunk sizes reduce efficiency
- While the SSD framework provides theoretical elegance, the practical implementation still requires custom CUDA kernels, similar to FlashAttention and Mamba-1
