# Excerpt: Structured State Space Duality (SSD)

Source: [[mamba-2|paper]] — Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024)

---

## The Central Claim

The title alone — "Transformers are SSMs" — is a provocation. But the paper backs it up with a rigorous algebraic proof, not a hand-wavy analogy.

> We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices.

This is not a loose analogy. The paper proves an exact algebraic equivalence: selective SSMs and causal linear attention both produce outputs that can be expressed as $y = Mx$ where $M$ is a lower-triangular semiseparable matrix. The "structured" in SSD refers to the fact that this matrix has additional structure beyond raw semiseparability — it decomposes in specific ways that enable efficient algorithms.

## Semiseparable Matrices

A matrix $M$ is $N$-semiseparable if every submatrix contained entirely in the lower-triangular part has rank at most $N$. This is exactly the structure produced by an SSM with state dimension $N$:

$$M_{ts} = C_t \left(\prod_{r=s+1}^{t} \bar{A}_r\right) \bar{B}_s$$

The rank constraint comes from the state bottleneck: all information from position $s$ to position $t$ passes through the $N$-dimensional state, so the rank of any lower-triangular submatrix is at most $N$.

Similarly, causal linear attention produces:

$$M_{ts} = q_t^\top k_s \qquad \text{(rank at most } d_k \text{)}$$

When the head dimension $d_k$ is small, this is also semiseparable.

## Two Algorithms, One Matrix

> Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is a refinement of Mamba's selective SSM that is 2-8X faster.

The duality provides algorithmic flexibility:

**Recurrent path (SSM view):** Compute $y$ by running the recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$, then $y_t = C_t h_t$. Cost: $O(LN)$. Best for long sequences where $L \gg N$.

**Quadratic path (attention view):** Compute the full matrix $M$, then $y = Mx$. Cost: $O(L^2)$. Best for short sequences where the matrix multiply leverages tensor core hardware.

**Chunk-wise hybrid:** Divide the sequence into chunks of size $C$. Within each chunk, use the quadratic path (a small $C \times C$ matmul). Across chunks, propagate state via the recurrent path. This is Mamba-2's default training algorithm.

The chunk-wise approach is deeply analogous to Flash Attention's tiling. Flash Attention tiles Q, K, V into SRAM-sized blocks to avoid materializing the full $N \times N$ matrix in HBM. Mamba-2's chunking tiles the *computation itself* — using the quadratic form where tensor cores are most efficient (within-chunk) and the recurrent form where linear scaling matters (across-chunk). Both exploit the same insight: decompose a large computation into hardware-friendly pieces.

## The Scalar-Identity Simplification

> Mamba-2's core layer uses a scalar-times-identity structure for A (i.e., A_t = a_t * I), which constrains the state transition but enables much more efficient computation.

In Mamba-1, $A$ is a diagonal matrix with $N$ independent values. In Mamba-2, $A_t = a_t \cdot I$ — a single scalar per timestep. This means every state dimension decays at the same rate, which is less expressive. The compensation: **multi-head SSM** (analogous to multi-head attention), where each head has its own scalar $a_t$.

The scalar constraint is what enables the chunked computation to be expressed as a standard matrix multiply, directly using tensor core hardware. Without it, each state dimension would require separate handling, preventing the use of optimized GEMM routines.

Result: **2-8x speedup** over Mamba-1 with equivalent or slightly better language modeling quality.

## What the Duality Means

> Enables algorithmic flexibility: the same computation can be executed via either the SSM recurrence or the attention-like matrix form, choosing whichever is faster for the given hardware and sequence length.

The practical implications:

1. **No more SSM vs attention debate.** They compute the same thing. The question is which algorithm is faster for your hardware and sequence length.
2. **Transfer of improvements.** Any optimization to SSM scan algorithms also improves the "attention" path, and vice versa. Flash Attention-style tiling applies to SSM computation; SSM-style chunking applies to attention.
3. **Unified architecture search.** The state dimension $N$ and head dimension $d_k$ are analogous hyperparameters controlling the rank of the same semiseparable matrix.

## Remaining Limitations

> Mamba-2 remains "competitive with Transformers" rather than clearly surpassing them, suggesting that the efficiency gains do not translate to quality advantages at current scales.

The duality is with *linear* attention, not softmax attention. The exponential nonlinearity in softmax produces sharper, more selective attention patterns that semiseparable matrices cannot represent. This is why pure SSMs still lag behind softmax Transformers on tasks requiring precise retrieval — and why hybrid architectures like Jamba ([[jamba|report]]) add a few softmax attention layers.

---

## Key Equation

Both SSM and linear attention compute:
$$y_t = \sum_{s=0}^{t} M_{ts} \, x_s$$

where $M$ is a lower-triangular, $N$-semiseparable matrix. The SSM computes this via recurrence in $O(LN)$; linear attention computes it via matrix product in $O(L^2)$.

The choice between these two paths is purely algorithmic, not mathematical — both produce the identical output vector $y$. This is the core of the duality: the SSM recurrence and the attention matrix multiply are two decompositions of the same structured linear map.

## Implications for Future Architectures

The SSD framework suggests that the future of efficient sequence modeling is not "SSM or attention" but rather a unified design space parameterized by the rank $N$ of the semiseparable output matrix. Low $N$ means aggressive compression (efficient but lossy); high $N$ means richer representation (expressive but costly). Softmax attention sits outside this space entirely — its nonlinearity produces full-rank matrices, which is why it excels at precise retrieval but costs $O(L^2)$.
