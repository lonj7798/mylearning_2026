<!-- scope: RoPE mathematical derivation and implementation
     deps: [[ch-03]]
     see-also: [[hf-positional-encoding-design]], [[weng-transformer-family]]
-->

# Rotary Embeddings: A Relative Revolution

- **Core Insight:** RoPE derives from first principles: represent position as rotation in complex space.
- **Guideline:** Understanding the mathematical derivation reveals why RoPE extrapolates — it's relative by construction.

- **Author:** Stella Biderman, Sid Black, Charles Foster, Leo Gao, Eric Hallahan, Horace He, Ben Wang, Phil Wang (EleutherAI)
- **URL:** https://blog.eleuther.ai/rotary-embeddings/
- **Relevant chapters:** Positional encoding, attention mechanism, transformer components

## Summary
EleutherAI's deep dive into Rotary Position Embeddings (RoPE), the positional encoding method used in most modern LLMs. Covers the mathematical derivation from first principles, the geometric intuition behind encoding position as rotation in complex space, implementation in both PyTorch and JAX, experimental results showing consistent improvements over alternatives, and extension to multiple dimensions.

## Key Content

### The Problem

Existing positional encoding methods have constraints:
- Learned absolute encodings lack generalization and may not transfer across contexts
- Many relative position methods require full NxN attention matrices, incompatible with efficient attention
- A unified approach working across both standard and efficient attention was needed

### Core Intuition

The geometric insight: the dot product between vectors equals ||q|| ||k|| cos(theta_qk), depending on magnitudes and angular separation. RoPE leverages this by representing embeddings as complex numbers with position-dependent rotations applied multiplicatively.

**Key property:** When both query and key undergo identical positional rotations, their relative angle — and thus dot product — remains unchanged. This preserves relative positional information while encoding absolute position.

### Illustrative Example

For a single element dimension:
```
RoPE(x, m) = x * e^{m*i*epsilon}

<RoPE(q_j, m), RoPE(k_j, n)> = q_j * k_j * e^{(m-n)*i*epsilon}
```

The result depends solely on the relative position difference (m-n).

### Rigorous Derivation

**Setup:**
- Work in complex space C^{d/2} by pairing consecutive embedding dimensions
- Each pair (x_i, x_{i+1}) becomes x_i + i*x_{i+1}
- Define f(x, l) as the position-encoding function
- Goal: ensure <f(q,m), f(k,n)> = g(q, k, m-n) (depends only on relative position)

**Exponential decomposition in polar form:**
f(q,m) = R_f(q,m) * e^{i*Theta_f(q,m)}

Two constraints from the inner product:
1. **Magnitude:** R_f(q,m) * R_f(k,n) = R_g(q,k,m-n)
2. **Phase:** Theta_f(q,m) - Theta_f(k,n) = Theta_g(q,k,m-n)

**Critical steps:**
- Setting m=n with f(x,0)=x shows R_f is independent of position: R_f(x,y) = x
- Phase decomposes as Theta_f(x,y) = Theta(x) + phi(y), where phi is arithmetic: phi(m) = m*theta

**Final formula:**
f(q,m) = q * e^{i*(Theta(q) + m*theta)} = sum_{j=1}^{d/2} q_j * e^{i*m*theta_j} * e_j

### Matrix Form (Implementation)

Block-diagonal rotation matrices applied to query/key vectors:

M_j = [[cos(m*theta_j), -sin(m*theta_j)],
       [sin(m*theta_j),  cos(m*theta_j)]]

Each 2x2 block is a standard rotation matrix, applied independently to each pair of embedding dimensions.

### Efficient Implementation

**PyTorch (GPT-NeoX):**
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin),
           (k * cos) + (rotate_half(k) * sin)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return cat((-x2, x1), dim=-1)
```

### Computational Overhead

- Naive: 4-5x cost of additive positional embeddings
- With Torchscript fusion: 2-2.5x cost
- Overall transformer impact with fusion: 1-3% overhead (negligible)

### Extension to Multiple Dimensions

RoPE naturally extends to multidimensional data by applying independent 1D rotations to dimension pairs:

<f(q,m,i), f(k,n,j)> = g1(q_{:d/2}, k_{:d/2}, m-n) + g2(q_{d/2:}, k_{d/2:}, i-j)

Enables simultaneous relative position encoding across multiple axes (e.g., timing and pitch in music).

### Comparison with Sinusoidal Embeddings

Two key differences:
1. **Coordinate mixing:** Sinusoidal treats each coordinate independently; RoPE pairs and mixes coordinates through complex multiplication
2. **Operation type:** Sinusoidal adds terms additively; RoPE uses multiplication

### Experimental Results

**125M parameter models (GPT-NeoX on OpenWebText2):**

| Method | Loss | Perplexity |
|--------|------|-----------|
| Learned Absolute | 2.809 | 16.59 |
| T5 RPE | 2.801 | 16.46 |
| **RoPE** | **2.759** | **15.78** |

RoPE showed ~25% faster convergence over learned absolute.

**1.4B parameter models (Mesh Transformer JAX on The Pile):**

| Method | Loss | Perplexity |
|--------|------|-----------|
| Learned Absolute | 2.240 | 9.393 |
| T5 RPE | 2.223 | 9.234 |
| **RoPE** | **2.173** | **8.784** |

~30% improvement over learned absolute, 10-20% over T5 RPE.

**Efficient attention (Performer on Enwik8):** RoPE substantially improved validation loss and convergence.

## Notable Insights
- RoPE emerged from rigorous first-principles mathematical reasoning, not empirical tinkering — the derivation constrains the solution space until rotation matrices are the only answer.
- The multiplicative application (rotating Q and K) vs additive application (adding to embeddings) is the key difference from sinusoidal encoding. Multiplication preserves vector norms while encoding position through angles.
- RoPE is "essentially equivalent to sinusoidal positional encoding but formulated as a rotation matrix" — the mathematical relationship is deep, but the multiplicative formulation performs better in practice.
- The 1-3% computational overhead after fusion makes RoPE essentially free in practice, which explains its universal adoption in modern LLMs (Llama, Mistral, Qwen, etc.).
- Results generalized across codebases (PyTorch, JAX), scales, and datasets — a rare property noted as "few and far between" in transformer improvements.
