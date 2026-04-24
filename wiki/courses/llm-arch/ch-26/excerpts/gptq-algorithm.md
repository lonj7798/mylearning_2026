# Excerpt: GPTQ — Layer-Wise Optimal Weight Quantization

<!-- source: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2023) -->

## The Problem GPTQ Solves

Given a pre-trained weight matrix $W \in \mathbb{R}^{m \times n}$ and a set of calibration inputs $X$, find quantized weights $\hat{W}$ that minimize the layer-wise reconstruction error:

$$\arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

Naive round-to-nearest (RTN) quantizes each weight independently — it ignores the fact that errors in one weight can be partially compensated by adjusting other weights. GPTQ uses second-order information (the Hessian of the reconstruction error) to find a jointly better set of quantized weights.

## Algorithm Walkthrough

### Step 1: Compute the Hessian

The Hessian of the layer-wise reconstruction error with respect to the weights is:

$$H = 2 X X^\top \in \mathbb{R}^{n \times n}$$

This matrix captures the sensitivity of the output to each weight: $H_{ii}$ measures how much error a perturbation of weight $i$ introduces, and $H_{ij}$ measures the cross-sensitivity between weights $i$ and $j$. The Hessian is computed once per layer from the calibration set and reused for all rows of $W$.

### Step 2: Process Columns Left-to-Right

For each column $i = 0, 1, \ldots, n-1$:

1. **Quantize:** $\hat{w}_i = \text{quant}(w_i)$ using the target format (e.g., INT4 with group scale)

2. **Compute error:** $\delta_i = w_i - \hat{w}_i$ (the rounding error for column $i$)

3. **Compensate remaining columns:** For every unquantized column $j > i$:
   $$w_j \leftarrow w_j - \frac{\delta_i \cdot H_{ij}}{H_{ii}}$$

The compensation step (3) is the core insight. The Hessian tells us: "if column $i$ was rounded up by $\delta_i$, then reducing column $j$ by $\delta_i \cdot H_{ij}/H_{ii}$ minimizes the total output error." This is a closed-form solution derived from Optimal Brain Surgeon theory.

### Step 3: Block Processing for Speed

Processing truly column-by-column is $O(n^3)$ per layer — too slow for billion-parameter models. GPTQ processes columns in blocks of size $B$ (typically $B = 128$):

1. Within each block, apply the column-by-column compensation as described above
2. After the block is complete, apply a single "lazy batch" update to all remaining columns using the accumulated error across the block
3. Move to the next block

This reduces the complexity to $O(n^2 B)$ per block update, making quantization of 175B-parameter models feasible in ~4 GPU-hours on a single A100.

## Why the Column Order Matters

OBQ (the predecessor) chose the quantization order based on which weight would introduce the least error — a greedy strategy requiring $O(n)$ Hessian inversions. GPTQ's key practical insight: **the order barely matters at LLM scale.** With thousands of columns, the error-compensation mechanism is powerful enough that any fixed order (e.g., left-to-right) produces results within 1% of the optimal greedy order, while being dramatically faster.

## Quantitative Results (from the paper)

On OPT-175B quantized to INT4 (4-bit weights, group size 128):

| Method | WikiText-2 Perplexity | C4 Perplexity | Time |
|--------|----------------------|---------------|------|
| FP16 (baseline) | 8.34 | 10.13 | — |
| RTN (round-to-nearest) | 14.89 | 18.22 | minutes |
| GPTQ | 8.68 | 10.56 | ~4 hours |

GPTQ loses only 0.34 perplexity points on WikiText-2 — a 2x improvement over RTN's 6.55-point degradation. The error compensation from the Hessian is responsible for nearly all of this gain.

## Connection to AWQ

AWQ achieves similar results through a different mechanism: instead of compensating errors after quantization (GPTQ), AWQ protects important weights before quantization by scaling them up. Both methods use activation information to identify which weights matter most — GPTQ through the Hessian ($H = 2XX^\top$), AWQ through activation magnitudes ($\text{mean}(|X_j|)$). In practice, AWQ is faster (no Hessian computation) and generalizes slightly better across tasks, while GPTQ sometimes achieves lower perplexity on the calibration distribution.

## The Hessian Inverse: Why It Works

The compensation formula $w_j \leftarrow w_j - \delta_i \cdot H_{ij}/H_{ii}$ deserves closer attention. This is a single step of the Optimal Brain Surgeon (OBS) update. The intuition:

- $H_{ii}$ measures how sensitive the layer output is to weight $i$. A large $H_{ii}$ means even a small quantization error in weight $i$ causes a large output error.
- $H_{ij}$ measures the correlation between weights $i$ and $j$ in their effect on the output. When $H_{ij}$ is large, adjusting weight $j$ can effectively compensate for errors in weight $i$.
- The ratio $H_{ij}/H_{ii}$ is the optimal compensation coefficient — how much to adjust weight $j$ per unit error in weight $i$.

This is mathematically equivalent to projecting the quantization error onto the remaining weight dimensions using the Hessian as a metric tensor. The Hessian defines the "geometry" of the loss landscape, and GPTQ follows the minimum-error path through this geometry as it quantizes each weight.

## Implementation Notes

- **Calibration set size:** 128 examples is typical; GPTQ is not very sensitive to the specific examples chosen, but the domain should roughly match the target use case
- **Group size:** $g = 128$ is standard. Each group of 128 consecutive weights within a column shares one FP16 scale factor, adding 0.125 bits/weight overhead (so "INT4" is really 4.125 bits/weight)
- **Asymmetric vs symmetric quantization:** Asymmetric (with zero-point) typically gives 0.1-0.2 perplexity points improvement over symmetric
- **Order of layers:** Layers are quantized sequentially in model order; the output of each quantized layer becomes the input (calibration data) for the next layer
- **Damping:** A small constant ($\lambda \approx 0.01 \cdot \text{mean}(\text{diag}(H))$) is added to the Hessian diagonal before inversion. This prevents numerical instability when $H_{ii}$ is near zero (weights that the calibration data barely activates). Without damping, these weights get massive compensations that destabilize the model.
