# Excerpt: YaRN's Frequency-Dependent Scaling

<!-- source: [[yarn|paper]], [[eleutherai-rope|blog]] -->

## The Core Problem: Not All Frequencies Fail Equally

RoPE assigns each dimension pair a frequency $\theta_i = \text{base}^{-2i/d}$. The corresponding wavelength is $\lambda_i = 2\pi / \theta_i$. The relationship between wavelength and training context length $L$ determines whether a dimension extrapolates:

- **Wavelength $\lambda_i \ll L$**: The dimension has completed many full rotations during training. At extended length $L'$, it simply continues rotating through angles it has already seen. **No failure.**
- **Wavelength $\lambda_i \gg L$**: The dimension has not completed a single rotation during training. At $L'$, it encounters genuinely novel rotation angles. **Extrapolation failure.**
- **Wavelength $\lambda_i \approx L$**: Boundary case. Partial extrapolation.

## Position Interpolation: The Uniform Mistake

PI scales all positions by $L/L'$, which divides all frequencies by $s = L'/L$:

$$\theta_i^{\text{PI}} = \theta_i / s$$

This eliminates extrapolation failure (all angles stay within training range) but **uniformly compresses** all dimensions. High-frequency dimensions that were already fine get compressed unnecessarily, reducing the model's ability to distinguish nearby token positions.

For a 4x extension: positions 1, 2, 3, 4 now occupy the angle slots of 0.25, 0.5, 0.75, 1.0. The model must discriminate 4x finer angular differences — degrading local positional resolution.

## NTK-Aware: Better, but One-Parameter

NTK-aware scaling modifies the base:

$$\text{base}' = \text{base} \cdot s^{d/(d-2)}$$

This has a non-uniform effect: low-frequency dimensions (large $i$) are scaled more than high-frequency ones (small $i$). But it is still a single-parameter adjustment — the shape of the scaling curve is fixed by the formula.

## YaRN's Three-Part Solution

### 1. Frequency-Dependent Ramp

YaRN explicitly partitions dimensions by wavelength:

```
For each dimension i:
  wavelength = 2 * pi / theta_i
  
  if wavelength < L:       # high-freq: already extrapolates
    gamma = 0              # leave unscaled
  elif wavelength > 2*L:   # low-freq: fully OOD
    gamma = 1              # apply full interpolation (like PI)
  else:                    # boundary
    gamma = (wavelength - L) / L  # smooth ramp
    
  theta_i_new = theta_i * (1 - gamma) + (theta_i / s) * gamma
```

The ramp function smoothly transitions from "leave alone" (high frequency) to "fully interpolate" (low frequency). This preserves local positional resolution while fixing the out-of-distribution low-frequency dimensions.

### 2. Attention Temperature Scaling

Longer context means more positions compete in softmax, increasing entropy (more uniform distribution). YaRN applies temperature:

$$\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k} \cdot \sqrt{t}}\right)$$

where $t \approx 0.1 \ln(s) + 1$. At 4x extension: $t \approx 1.14$. At 32x: $t \approx 1.35$. The logarithmic relationship means even large extensions need only modest temperature adjustment.

### 3. Minimal Fine-Tuning

Because NTK-aware + ramp provides a much better initialization than PI alone:
- **PI requires**: ~1000+ training steps on long-context data
- **YaRN requires**: ~400 steps (10x fewer tokens, 2.5x fewer steps)

The better initialization means YaRN needs to learn much less during fine-tuning — the frequency adjustments already place the model in a good part of the loss landscape.

## Critical Result: Extrapolation Beyond Fine-Tuning Length

YaRN-extended models can generalize *beyond* their fine-tuning context:
- Fine-tuned at 64K, maintain quality at 128K
- PI models degrade sharply beyond their fine-tuning length

This happens because YaRN's ramp preserves high-frequency dimensions in their natural extrapolation regime. PI compresses these dimensions, removing their extrapolation capability.

## The Wavelength Boundary

The critical insight from both the EleutherAI analysis and the YaRN paper: the boundary between "extrapolates" and "fails" is approximately $\lambda_i = L$. This is the wavelength equal to the training context length. Dimensions with shorter wavelengths are safe. Dimensions with longer wavelengths need intervention.

For a model with base=10000 and d=128, trained at L=4096:
- Dimension 0: $\lambda_0 = 2\pi \approx 6.3$ (safe: $\lambda \ll L$)
- Dimension 32: $\lambda_{32} \approx 628$ (safe: $\lambda < L$)  
- Dimension 56: $\lambda_{56} \approx 39,800$ (fails: $\lambda \gg L$)
- Dimension 63: $\lambda_{63} \approx 62,800$ (fails: $\lambda \gg L$)
