# Excerpt: Mamba-2 vs Mamba-1 — What Changed and Why

<!-- source: [[mamba-2|paper]], [[mamba|paper]] -->

## Architecture Differences

### State Transition Matrix

| Property | Mamba-1 | Mamba-2 |
|----------|---------|---------|
| State transition A | Diagonal: $\text{diag}(a_1, \ldots, a_N)$ | Scalar: $\alpha_t \cdot I_N$ |
| Per-dimension decay | Each state dim has own rate | All dims decay together |
| Expressivity | Higher — independent decay rates | Lower — uniform decay |
| Matmul compatibility | Low — parallel scan only | High — structured semiseparable matmul |
| Typical state dim N | 16 | 64-128 |

The scalar constraint is the single most important architectural change. It sacrifices per-dimension decay control but unlocks the entire SSD computation framework.

**Why diagonal A was hard to accelerate:** With $A = \text{diag}(a_1, \ldots, a_N)$, the state transition between positions $j$ and $i$ is:

$$\prod_{k=j+1}^{i} A_k = \text{diag}\left(\prod_{k=j+1}^{i} a_{k,1}, \ldots, \prod_{k=j+1}^{i} a_{k,N}\right)$$

This is an $N$-dimensional object that varies for every $(i,j)$ pair, preventing simple factorization of the semiseparable matrix into matmul-friendly form.

**Why scalar A enables matmul:** With $A_t = \alpha_t \cdot I$:

$$\prod_{k=j+1}^{i} A_k = \left(\prod_{k=j+1}^{i} \alpha_k\right) \cdot I_N$$

The product collapses to a single scalar times identity. This scalar can be precomputed as a cumulative product over the sequence, and the full matrix $M$ factors as:

$$M_{ij} = C_i^\top B_j \cdot \underbrace{\prod_{k=j+1}^{i} \alpha_k}_{\text{scalar decay mask } L_{ij}}$$

The $C_i^\top B_j$ term is a standard outer product (computed via matmul: $CB^\top$), and $L_{ij}$ is a lower-triangular mask. The full computation becomes: $Y = (L \odot CB^\top) X$ — a masked matmul, which tensor cores handle efficiently.

### Multi-Head Structure

Mamba-1 processed the full hidden dimension through a single SSM. Mamba-2 splits into $H$ independent heads:

| Aspect | Mamba-1 | Mamba-2 |
|--------|---------|---------|
| Heads | 1 (implicit) | H (configurable, e.g., 64) |
| Per-head input dim | $d_\text{model}$ | $d_\text{model} / H$ |
| Per-head state | $N$ | $N$ |
| Total state | $N$ | $H \times N$ |
| Analogy | Single massive SSM | Multi-head attention equivalent |

Multiple heads increase the total state capacity ($H \times N$ vs $N$) and allow different heads to specialize in tracking different types of information — analogous to how different attention heads learn different patterns.

### Block Structure Comparison

**Mamba-1 block:**
1. Linear expand (1x to ~2x width)
2. 1D depthwise convolution (kernel size 4)
3. SiLU activation
4. Selective SSM (diagonal A, parallel scan)
5. Gating (element-wise multiply with projected bypass)
6. Linear project down

**Mamba-2 block:**
1. Linear expand
2. 1D convolution
3. SSD layer (scalar A, multi-head, chunk-wise matmul)
4. **Normalization** (new — stabilizes training)
5. Gating
6. Linear project down

Key differences:
- The SSD layer replaces the selective SSM
- Normalization is added between the SSD output and gating
- The multi-head structure adds a new dimension of configurability
- Both architectures omit separate MLP blocks — the gated structure serves double duty

## Speed Comparison

The speedup varies with configuration:

### Scaling with State Dimension N

At sequence length 2K:
- N=16: Mamba-2 is ~2-3x faster than Mamba-1
- N=64: Mamba-2 is ~4-6x faster
- N=128: Mamba-2 is ~6-8x faster

The speedup increases with N because Mamba-1's parallel scan cost grows as $O(TN^2)$ (for diagonal A, the scan involves $N \times N$ operations at each step), while Mamba-2's chunk-wise matmul cost grows as $O(TCN)$ — linear in $N$. The ratio of costs thus grows linearly with $N$.

### Scaling with Sequence Length T

At state dimension N=64:
- T=1K: ~3-4x speedup
- T=4K: ~5-6x speedup
- T=8K: ~6-8x speedup

Longer sequences benefit more because the overhead of chunk boundary processing (state propagation between chunks) is amortized over more within-chunk matmul computation.

### Why Mamba-1 Cannot Be Easily Accelerated

Mamba-1's parallel scan is inherently a *prefix sum* operation — each output depends on a cumulative reduction of all preceding inputs. While parallel scans can be computed in $O(\log T)$ parallel steps, each step involves $N \times N$ state operations and irregular memory access patterns. GPU tensor cores are designed for dense, regular matrix multiplications, not prefix scans with small irregular matrices.

Mamba-2's insight is that constraining the architecture (scalar A instead of diagonal A) converts the computation into something tensor cores *are* good at. This is a recurring theme in ML systems: matching the algorithm to the hardware often matters more than optimizing the algorithm in isolation.

## Quality Comparison

On language modeling benchmarks, Mamba-2 matches or slightly exceeds Mamba-1 quality at equivalent parameter counts. This is somewhat surprising given the expressivity reduction from diagonal to scalar A. Two factors compensate:

1. **Larger affordable state dimension.** Mamba-2 can practically use N=64 or N=128 where Mamba-1 was limited to N=16. The total state capacity per head ($N$ values) is much larger, partially compensating for the loss of per-dimension decay control.

2. **Multi-head diversity.** Multiple heads with independent B, C projections provide representational diversity that a single large SSM cannot match, similar to how multi-head attention outperforms single-head attention even at the same total dimension.

The net effect: Mamba-2 trades *type* of expressivity (per-dimension decay rates) for *amount* of expressivity (larger state, more heads), while gaining 2-8x speed. The exchange is favorable at every scale tested in the paper.
