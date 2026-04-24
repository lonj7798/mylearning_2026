# Excerpt: The SSD Framework — Why SSMs and Attention Are Dual

<!-- source: [[mamba-2|paper]], Dao & Gu (2024) -->

## The Semiseparable Matrix Connection

The central theorem of the SSD paper can be stated informally as: *the input-output map of a selective SSM is a structured semiseparable matrix, and this is exactly the same class of matrices produced by linear attention with causal masking.*

### Formal Statement

Consider a discrete-time selective SSM with scalar state transitions:

$$h_t = \alpha_t \cdot h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t$$

where $h_t \in \mathbb{R}^N$, $x_t \in \mathbb{R}$, $B_t \in \mathbb{R}^N$, $C_t \in \mathbb{R}^N$, and $\alpha_t \in \mathbb{R}$ is a scalar.

Unrolling the recurrence, the output at time $i$ can be written as:

$$y_i = \sum_{j=0}^{i} C_i^\top \left(\prod_{k=j+1}^{i} \alpha_k\right) B_j \, x_j$$

This defines a matrix $M$ where:

$$M_{ij} = \begin{cases} C_i^\top \left(\prod_{k=j+1}^{i} \alpha_k\right) B_j & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}$$

The key observation: for any submatrix of the lower-triangular part, the rank is at most $N$ (the state dimension), because each entry factors through the $N$-dimensional vectors $C_i$ and $B_j$. This is exactly the definition of a semiseparable matrix of order $N$.

### The Attention Side

Now consider linear attention (attention without softmax) with causal masking:

$$y_i = \sum_{j \leq i} (q_i^\top k_j) \, v_j$$

where $q_i, k_j \in \mathbb{R}^d$. The corresponding matrix has entries $M_{ij} = q_i^\top k_j$ for $i \geq j$, which is also semiseparable of order $d$.

**The duality:** Setting $N = d$ (SSM state dimension equals attention head dimension), the two computations produce the same class of matrices. The SSM's $C_i$ corresponds to attention's $q_i$, the SSM's $B_j$ corresponds to attention's $k_j$, and the SSM's cumulative decay $\prod \alpha_k$ corresponds to the causal mask.

### What the Duality Enables

The practical payoff is algorithmic flexibility. The same mathematical computation can be performed via:

1. **Sequential recurrence** (SSM form): $O(TN)$ time, $O(N)$ memory. Best for autoregressive generation.
2. **Dense matrix multiply** (attention form): $O(T^2N)$ time, $O(T^2)$ memory. Best for short sequences where matmul hardware is fast.
3. **Chunk-wise hybrid**: divide sequence into chunks of size $C$. Within chunks, use the quadratic form. Across chunks, use the recurrence. Total: $O(TCN)$ time, $O(C^2 + N)$ memory.

The chunk-wise algorithm is the practical workhorse of Mamba-2: it uses tensor cores within each chunk (maximizing hardware utilization) while maintaining overall linear scaling via inter-chunk recurrence.

### What the Duality Does *Not* Cover

The equivalence is with **linear** attention. Standard softmax attention produces a dense, full-rank matrix that is not semiseparable for any finite $N$. The softmax nonlinearity provides:

- **Sparsification**: concentrating probability mass on a few positions
- **Competitive normalization**: increasing attention to one position decreases others
- **Dynamic range**: exponential weighting enables very precise retrieval

These properties are precisely what SSMs lack, explaining their persistent weaknesses in in-context learning and exact retrieval tasks.

### The Scalar A Constraint

Mamba-2's restriction to $A_t = \alpha_t \cdot I$ (scalar-times-identity) is more restrictive than Mamba-1's diagonal $A = \text{diag}(a_1, \ldots, a_N)$. With diagonal $A$, each state dimension has its own decay rate, allowing the model to maintain some state components for many timesteps while rapidly forgetting others. With scalar $A$, all dimensions decay at the same rate.

The tradeoff: scalar $A$ makes the semiseparable structure *more structured*, enabling decomposition into efficient matmul operations. The speedup (2-8x) comes from this additional structure. Empirically, Mamba-2 compensates for the expressivity loss by using larger state dimensions $N$ (affordable because of the faster computation) and multiple heads.

### Worked Example: 4-Token Sequence

Consider a sequence of $T = 4$ tokens with state dimension $N = 2$. The SSM produces the lower-triangular matrix:

$$M = \begin{pmatrix} C_1^\top B_1 & 0 & 0 & 0 \\ C_2^\top (\alpha_2) B_1 & C_2^\top B_2 & 0 & 0 \\ C_3^\top (\alpha_3 \alpha_2) B_1 & C_3^\top (\alpha_3) B_2 & C_3^\top B_3 & 0 \\ C_4^\top (\alpha_4 \alpha_3 \alpha_2) B_1 & C_4^\top (\alpha_4 \alpha_3) B_2 & C_4^\top (\alpha_4) B_3 & C_4^\top B_4 \end{pmatrix}$$

This can be decomposed as the element-wise product $(L \odot CB^\top)$ where $CB^\top$ is the rank-2 outer product matrix and $L$ is the causal decay mask with entries $L_{ij} = \prod_{k=j+1}^{i} \alpha_k$. Both components are computable via matmul. With $C = 2$ (chunk size = sequence length), the entire computation maps to a single $4 \times 4$ masked matmul — exactly what a GPU tensor core is optimized for.

---

## Key Equations Summary

| Quantity | SSM View | Attention View |
|----------|----------|----------------|
| Query vector | $C_i$ | $q_i$ |
| Key vector | $B_j$ | $k_j$ |
| Causal mask | $\prod_{k=j+1}^{i} \alpha_k$ | $\mathbb{1}[i \geq j]$ |
| State dimension | $N$ | $d$ (head dim) |
| Matrix entry | $C_i^\top (\prod \alpha_k) B_j$ | $q_i^\top k_j$ |
| Matrix rank | $\leq N$ | $\leq d$ |
| Recurrence cost | $O(TN)$ | Not native |
| Matmul cost | Not native (Mamba-1) | $O(T^2 d)$ |
| SSD cost | $O(TCN)$ | $O(TCN)$ |
