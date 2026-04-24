<!-- scope: SwiGLU parameter cost derivation, parent: [[ch-08]] -->

# SwiGLU Parameter Cost Derivation

Switching from a standard two-matrix FFN to a gated three-matrix FFN (SwiGLU) changes the parameter budget. This excerpt derives the iso-parameter hidden dimension, explains why real models exceed it, and traces the compute implications through a full Transformer block.

---

## The Standard FFN: Two Matrices, 8d^2

The original Transformer FFN ([[attention-is-all-you-need|paper]]):

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

where:
- $W_1 \in \mathbb{R}^{d_{ff} \times d_{model}}$ (up-projection)
- $W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$ (down-projection)

With the conventional ratio $d_{ff} = 4 \times d_{model}$:

$$\text{Params}_{\text{standard}} = d_{model} \times d_{ff} + d_{ff} \times d_{model} = 2 \times d \times 4d = 8d^2$$

(Ignoring biases, which modern models drop.)

**FLOPs per token:** Each matrix multiply costs $2 \times \text{rows} \times \text{cols}$ FLOPs (multiply-accumulate). So:

$$\text{FLOPs}_{\text{standard}} = 2 \times d \times 4d + 2 \times 4d \times d = 16d^2$$

---

## The Gated FFN: Three Matrices

SwiGLU ([[glu-variants|paper]]) replaces the single up-projection + activation with two parallel projections and an element-wise product:

$$\text{FFN}_{\text{SwiGLU}}(x) = \left(\text{Swish}(W_1 x) \odot V x\right) W_2$$

Three weight matrices:
- $W_1 \in \mathbb{R}^{d_{ff}' \times d}$ (gate branch)
- $V \in \mathbb{R}^{d_{ff}' \times d}$ (value branch)
- $W_2 \in \mathbb{R}^{d \times d_{ff}'}$ (down-projection)

$$\text{Params}_{\text{SwiGLU}} = 3 \times d \times d_{ff}'$$

---

## The Iso-Parameter Derivation

Setting SwiGLU parameters equal to standard FFN parameters:

$$3 \times d \times d_{ff}' = 8d^2$$

$$d_{ff}' = \frac{8}{3} d \approx 2.667d$$

This is the **iso-parameter** hidden dimension: using $d_{ff}' = \frac{8}{3}d$ with SwiGLU gives the same total parameter count as $d_{ff} = 4d$ with the standard FFN.

### FLOPs at Iso-Parameter

$$\text{FLOPs}_{\text{SwiGLU}} = \underbrace{2 \times d \times \frac{8}{3}d}_{W_1 x} + \underbrace{2 \times d \times \frac{8}{3}d}_{Vx} + \underbrace{2 \times \frac{8}{3}d \times d}_{W_2(\cdot)} = 3 \times 2 \times d \times \frac{8}{3}d = 16d^2$$

Same FLOPs as the standard FFN at iso-parameter. The element-wise Swish activation and the Hadamard product $\odot$ are negligible ($O(d_{ff}')$ operations vs $O(d \times d_{ff}')$ for the matmuls).

**So at iso-parameter, SwiGLU costs the same in params and FLOPs but empirically performs better.** This is why Shazeer's result was so impactful: it was essentially free quality improvement.

---

## Why Real Models Exceed 8/3

The theoretical $\frac{8}{3} \approx 2.667$ is a floor. In practice:

| Model | $d_{model}$ | $d_{ff}$ | Ratio | Total FFN Params | vs 8d^2 |
|-------|-------------|---------|-------|-----------------|---------|
| Llama 3 8B ([[llama-3|report]]) | 4,096 | 14,336 | 3.50x | $3 \times 4096 \times 14336 = 176.2\text{M}$ | 1.31x |
| Llama 3 70B | 8,192 | 28,672 | 3.50x | $3 \times 8192 \times 28672 = 704.6\text{M}$ | 1.31x |
| Llama 3 405B | 16,384 | 53,248 | 3.25x | $3 \times 16384 \times 53248 = 2.617\text{B}$ | 1.22x |
| Mistral 7B ([[mistral-7b|report]]) | 4,096 | 14,336 | 3.50x | $176.2\text{M}$ | 1.31x |

All these models spend 22-31% **more** FFN parameters than iso-parameter would require. Why?

1. **The FFN quality returns are worth the extra parameters.** The FFN-as-memory hypothesis suggests wider FFNs store more knowledge per layer. Labs have empirically determined that the marginal parameter in the FFN has higher returns than spending it elsewhere.

2. **Hardware alignment.** The hidden dimension must be a multiple of 128 (or 256) for efficient tensor core utilization. Rounding up from $\frac{8}{3} \times 4096 = 10{,}923$ to $14{,}336$ happens to land at $3.5\times$.

3. **The $\frac{8}{3}$ baseline was calibrated to the standard 4x ratio.** But nothing says the 4x ratio was optimal to begin with. It was an early design choice from Vaswani et al. (2017) that stuck. The SwiGLU transition was an opportunity to re-optimize the FFN allocation.

---

## FFN's Share of Total Model Parameters

Consider a single Transformer block with SwiGLU FFN and GQA attention ($G$ KV groups):

**Attention parameters:**
$$P_{\text{attn}} = d \times (H \times d_k) + d \times (G \times d_k) \times 2 + (H \times d_k) \times d$$

For Llama 3 8B ($d = 4096$, $H = 32$, $G = 8$, $d_k = 128$):
$$P_{\text{attn}} = 4096 \times 4096 + 4096 \times 1024 \times 2 + 4096 \times 4096 = 42.0\text{M}$$

($W_Q$: 16.8M, $W_K$: 4.2M, $W_V$: 4.2M, $W_O$: 16.8M)

**FFN parameters (SwiGLU, 3.5x):**
$$P_{\text{FFN}} = 3 \times 4096 \times 14336 = 176.2\text{M}$$

**FFN's share:** $176.2 / (176.2 + 42.0) = 80.7\%$

With GQA reducing attention parameters (from the MHA baseline of $4 \times d^2 = 67.1$M to 42.0M), the FFN's relative share has actually **increased** from the original ~67% to ~81%. The modern Transformer block is overwhelmingly FFN by parameter count.

---

## Compute Cost Per Transformer Block

For a single token through one Llama 3 8B block:

| Component | FLOPs | Share |
|-----------|-------|-------|
| $W_Q$ projection | $2 \times 4096 \times 4096 = 33.6\text{M}$ | 8.0% |
| $W_K$ projection | $2 \times 4096 \times 1024 = 8.4\text{M}$ | 2.0% |
| $W_V$ projection | $2 \times 4096 \times 1024 = 8.4\text{M}$ | 2.0% |
| Attention ($QK^T + PV$) | $4 \times N \times d_k \times H$ | varies with $N$ |
| $W_O$ projection | $2 \times 4096 \times 4096 = 33.6\text{M}$ | 8.0% |
| $W_1 x$ (gate) | $2 \times 4096 \times 14336 = 117.4\text{M}$ | 28.0% |
| $Vx$ (value) | $2 \times 4096 \times 14336 = 117.4\text{M}$ | 28.0% |
| $W_2(\cdot)$ (down) | $2 \times 14336 \times 4096 = 117.4\text{M}$ | 28.0% |
| **Total (excl. attention)** | **436.2M** | |

The three FFN matmuls together account for 84% of non-attention FLOPs. This is why SwiGLU's 50% increase in FFN matmuls (from 2 to 3) was a real cost that needed to be justified by quality gains.

---

## The SwiGLU Matmul Fusion Opportunity

In practice, $W_1 x$ and $Vx$ can be computed as a single fused matmul by concatenating $W_1$ and $V$ into a $(2 \times d_{ff}') \times d$ matrix:

$$[W_1; V] \cdot x = \begin{bmatrix} W_1 x \\ Vx \end{bmatrix}$$

The result is split and the Swish + Hadamard product applied. This reduces kernel launch overhead and improves GPU utilization. Most modern inference frameworks (vLLM, TensorRT-LLM) implement this fusion.

---

## References

- [[glu-variants|Shazeer, "GLU Variants Improve Transformer" (2020) (paper)]]
- [[attention-is-all-you-need|Vaswani et al., "Attention Is All You Need" (2017) (paper)]]
- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]]
- [[mistral-7b|Mistral AI, "Mistral 7B" (2023) (report)]]
