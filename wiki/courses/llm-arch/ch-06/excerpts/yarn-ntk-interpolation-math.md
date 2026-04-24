<!-- scope: YaRN and NTK-aware interpolation mathematics, frequency-dependent scaling, parent: [[ch-06]] -->

# YaRN and NTK-Aware Interpolation: The Math

Context extension for RoPE-based models is fundamentally a frequency-domain problem. This excerpt derives the mathematics of position interpolation, NTK-aware scaling, and YaRN's attention temperature correction, explaining why each component is necessary and how they compose.

---

## The Extrapolation Problem

RoPE uses frequencies $\theta_j = \text{base}^{-2j/d}$ with $\text{base} = 10000$. At position $m$, dimension pair $j$ is rotated by angle $m \cdot \theta_j$.

A model trained at maximum context length $L$ has seen rotation angles in:

$$\mathcal{A}_j = \{m \cdot \theta_j : m \in [0, L)\}$$

At inference with $m > L$, the rotation angle $m \cdot \theta_j$ exceeds the maximum training-time angle $L \cdot \theta_j$. The model encounters angles it has never been optimized for.

**Key insight**: Not all dimensions are equally vulnerable. The angle range depends on the frequency:
- **High-frequency dimensions** (large $\theta_j$, small $j$): The angle $m \cdot \theta_j$ wraps around $2\pi$ many times even within the training length $L$. Extending to $L' > L$ does not introduce fundamentally new angles -- the model has seen all possible angles already.
- **Low-frequency dimensions** (small $\theta_j$, large $j$): The angle $m \cdot \theta_j$ covers less than one full period within $L$. Extending to $L'$ introduces angles the model has never seen.

---

## Position Interpolation (PI): The Baseline

The simplest fix: linearly scale all positions so they fit within the training range.

$$m' = m \cdot \frac{L}{L'}$$

Now the maximum angle at any dimension is $\frac{L}{L'} \cdot L' \cdot \theta_j = L \cdot \theta_j$, which is within the training distribution.

**The problem with PI**: It compresses ALL frequencies uniformly. High-frequency dimensions that encode fine-grained local position distinctions (e.g., "is this token 1 or 2 positions away?") are unnecessarily compressed. Nearby tokens become harder to distinguish, degrading local attention patterns.

Formally, the positional resolution at dimension $j$ is:

$$\Delta\alpha_j = \theta_j \cdot \frac{L}{L'}$$

For high-frequency dimensions, this resolution was already adequate at the original scale. Compressing it further is wasteful.

---

## NTK-Aware Interpolation: Frequency-Dependent Scaling

Instead of scaling positions, modify the base frequency to change the frequency distribution:

$$\text{base}' = \text{base} \cdot s^{d/(d-2)}$$

where $s = L'/L$ is the extension ratio.

This modifies each frequency as:

$$\theta_j' = (\text{base}')^{-2j/d} = \text{base}^{-2j/d} \cdot s^{-2j/(d-2)}$$

**The effect is frequency-dependent:**
- When $j$ is small (high frequency): $s^{-2j/(d-2)} \approx 1$. High frequencies are barely changed.
- When $j$ is large (low frequency): $s^{-2j/(d-2)} \ll 1$. Low frequencies are scaled down significantly.

This is exactly the desired behavior: preserve high-frequency local resolution while interpolating low-frequency global position.

**Why "NTK"?** The name references Neural Tangent Kernel theory. The connection is that the RoPE frequency distribution resembles the spectral decomposition of an NTK, and modifying the base frequency is analogous to adjusting the kernel's frequency content while preserving its overall structure.

---

## YaRN's Frequency Partitioning

YaRN refines NTK-aware interpolation by explicitly partitioning frequencies into three bands based on their wavelength $\lambda_j = 2\pi / \theta_j$ relative to the original context length $L$:

### Band 1: High-frequency (wavelength $\ll L$)

Dimensions where $\lambda_j \ll L$. These cycle many times within the training context. Leave them untouched:

$$\theta_j' = \theta_j$$

### Band 2: Medium-frequency (wavelength $\sim L$)

Dimensions where $\lambda_j \sim L$. These need partial interpolation. YaRN applies a smooth ramp function:

$$\theta_j' = \theta_j \cdot (1 - \gamma_j) + \theta_j / s \cdot \gamma_j$$

where $\gamma_j$ is the ramp factor that transitions smoothly from 0 (no scaling) to 1 (full interpolation) based on the ratio $\lambda_j / L$.

### Band 3: Low-frequency (wavelength $\gg L$)

Dimensions where $\lambda_j \gg L$. These have not completed even one cycle during training. Fully interpolate:

$$\theta_j' = \theta_j / s$$

---

## Attention Temperature Scaling

Extending the context from $L$ to $L'$ increases the number of positions the attention mechanism distributes probability over. The softmax denominator grows:

$$\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{L'} e^{s_j}}$$

With $L' > L$, the denominator has more terms, reducing each attention weight. The attention distribution becomes more uniform (higher entropy), diluting the model's ability to focus on specific positions.

YaRN compensates with a temperature factor $t$:

$$\text{attention}(q, K) = \text{softmax}\left(\frac{qK^T}{\sqrt{d_k} \cdot t}\right)$$

The temperature $t > 1$ sharpens the distribution, counteracting the dilution. The optimal $t$ scales approximately as:

$$t \approx \frac{\log L'}{\log L}$$

This is a heuristic derived from the observation that attention entropy grows logarithmically with context length under a uniform prior.

---

## The Complete YaRN Recipe

Starting from a pretrained RoPE model at context length $L$, extending to $L'$:

1. **Compute frequency-dependent scaling** using the NTK-aware base modification or explicit band partitioning
2. **Apply attention temperature** $t$ to all attention computations
3. **Fine-tune for ~400 steps** on long-context data at the target length $L'$

The fine-tuning is necessary because:
- The model's learned attention patterns were calibrated for the original frequency distribution
- The temperature scaling is an approximation; the model needs to adjust its internal calibration
- Long-range attention patterns (which frequencies to use for distant dependencies) need recalibration

But only 400 steps suffice because:
- The NTK-aware initialization is already close to the optimal frequency distribution
- The model's local attention patterns (high-frequency dimensions) are preserved exactly
- Only the global attention patterns (low-frequency dimensions) need adjustment

---

## Comparison of Extension Methods

| Method | Frequency treatment | Temp. scaling | Fine-tuning | Max validated ratio |
|--------|-------------------|---------------|-------------|-------------------|
| Position Interpolation | Uniform | No | ~1000 steps | ~8x |
| NTK-Aware (static) | Frequency-dependent | No | ~600 steps | ~16x |
| YaRN | Frequency-dependent + banded | Yes | ~400 steps | ~32x |
| iRoPE ([[ch-06]]) | Remove PE from some layers | Inference-time | Trained from scratch | ~40x |

---

## References

- [[yarn|Peng et al. "YaRN: Efficient Context Window Extension" (2023) (paper)]]
- [[rope|Su et al. "RoFormer" (2021) (paper)]]
- Chen et al. "Extending Context Window of Large Language Models via Position Interpolation" (2023)
- bloc97, "NTK-Aware Scaled RoPE" (2023)
