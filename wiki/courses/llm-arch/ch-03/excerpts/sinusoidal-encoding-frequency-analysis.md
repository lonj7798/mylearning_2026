<!-- scope: sinusoidal positional encoding frequency analysis, parent: [[ch-03]] -->

# Sinusoidal Positional Encoding: Frequency Analysis

This excerpt dissects the sinusoidal positional encoding from "Attention Is All You Need" in mathematical detail: why the specific frequency spectrum was chosen, how relative position is encoded as a linear transformation, and why this scheme was ultimately replaced by RoPE.

---

## 1. The Encoding Formula

For a token at position $pos$ and embedding dimension index $i$:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $i \in \{0, 1, \ldots, d_{\text{model}}/2 - 1\}$.

Each pair of dimensions $(2i, 2i+1)$ encodes position using a sinusoid at a specific frequency $\omega_i$:

$$\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}$$

---

## 2. The Frequency Spectrum

The frequencies form a geometric progression from high to low:

| Dimension pair $(2i, 2i+1)$ | Frequency $\omega_i$ | Wavelength $\lambda_i = 2\pi / \omega_i$ |
|---|---|---|
| 0, 1 | 1.0 | $2\pi \approx 6.28$ positions |
| 64, 65 | $10000^{-0.25} \approx 0.1$ | $\approx 63$ positions |
| 128, 129 | $10000^{-0.5} = 0.01$ | $\approx 628$ positions |
| 256, 257 | $10000^{-0.75} \approx 0.001$ | $\approx 6{,}283$ positions |
| 510, 511 | $10000^{-1} = 0.0001$ | $\approx 62{,}832$ positions |

*(Assuming $d_{\text{model}} = 512$)*

**Interpretation:** The lowest dimensions oscillate rapidly -- they can distinguish positions 1 apart but "wrap around" after ~6 positions. The highest dimensions oscillate extremely slowly -- they distinguish positions thousands apart but cannot resolve fine differences. Together, the full spectrum creates a unique "fingerprint" for every position, analogous to how binary numbers use both high-order and low-order bits to represent any integer.

---

## 3. Why 10,000?

The base constant 10,000 determines the ratio between the fastest and slowest frequencies:

$$\frac{\omega_{\max}}{\omega_{\min}} = \frac{1}{10000^{-1}} = 10{,}000$$

This gives a 4-order-of-magnitude spread. The choice is empirical -- the paper does not derive it from first principles -- but the rationale is:

- **Too small a base** (e.g., 100): The slowest frequencies would still have relatively short wavelengths, making it hard to distinguish positions far apart. For sequences of length 500+, multiple positions would have nearly identical encodings.
- **Too large a base** (e.g., $10^6$): The slowest frequencies would have wavelengths of millions of positions -- wasted capacity for distinguishing positions that never appear in training data of length 512.
- **10,000**: The slowest wavelength (~63K positions) comfortably exceeds the maximum training length (512), ensuring unique encodings for all training positions while leaving headroom for extrapolation.

---

## 4. Relative Position as Linear Transformation

The paper's key theoretical claim: "for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$."

**Proof:** Consider dimension pair $(2i, 2i+1)$. Using the angle addition formulas:

$$\sin(\omega_i(pos + k)) = \sin(\omega_i \cdot pos)\cos(\omega_i k) + \cos(\omega_i \cdot pos)\sin(\omega_i k)$$

$$\cos(\omega_i(pos + k)) = \cos(\omega_i \cdot pos)\cos(\omega_i k) - \sin(\omega_i \cdot pos)\sin(\omega_i k)$$

In matrix form:

$$\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}$$

This is a **rotation matrix** $R_k^{(i)}$ that depends only on $k$ and $i$, not on $pos$. For the full $d_{\text{model}}$-dimensional encoding, the transformation from $PE_{pos}$ to $PE_{pos+k}$ is a block-diagonal rotation:

$$PE_{pos+k} = \text{diag}(R_k^{(0)}, R_k^{(1)}, \ldots, R_k^{(d_{\text{model}}/2-1)}) \cdot PE_{pos}$$

**Why this matters:** The attention score between positions $pos$ and $pos + k$ depends on $q_{pos} \cdot k_{pos+k}$. If the model learns query/key projections that are sensitive to these rotation matrices, it can effectively attend based on relative position $k$ without seeing $k$ explicitly. The linear transformation makes this learnable by standard linear layers.

---

## 5. Dot Product Decay with Distance

A useful property: the dot product between positional encodings at positions $pos$ and $pos + k$ decreases (on average) as $|k|$ increases. This gives the model a built-in distance bias.

$$PE_{pos} \cdot PE_{pos+k} = \sum_{i=0}^{d/2-1} \cos(\omega_i k)$$

Each frequency contributes a $\cos(\omega_i k)$ term. For small $k$, most terms are close to 1 (the cosines have not yet oscillated significantly). For large $k$, the high-frequency terms oscillate rapidly and cancel out, while only the low-frequency terms contribute coherently. The result is a smooth decay:

```
k=0:   dot product = d/2  (maximum, all cosines = 1)
k=1:   dot product ≈ d/2 - small correction
k=10:  dot product < d/2  (high-freq terms decorrelate)
k=100: dot product << d/2 (most terms decorrelate)
```

This means nearby positions are more "similar" in the encoding space -- a useful inductive bias for language, where local context is usually more relevant than distant context.

---

## 6. Limitations of Sinusoidal Encoding

### Additive pollution of the residual stream

The encoding is added to the token embedding:

$$x_0 = \text{Embed}(token) + PE_{pos}$$

This means position and content information are mixed from the very first layer. Every downstream computation operates on this mixture. The model must learn to disentangle position from content -- a burden that grows with depth because the position signal propagates additively through the residual stream.

### Absolute position is wasteful

The encoding assigns a unique vector to each absolute position (0, 1, 2, ...). But most linguistic structure depends on **relative** position. "The word 3 positions before the verb" is linguistically meaningful; "the word at position 47" is not. The model must learn to convert absolute encodings into relative position information through the attention weights -- a learnable but indirect path.

### Extrapolation fails in practice

Despite the theoretical argument that sinusoidal encodings should extrapolate to unseen lengths, trained models degrade significantly beyond their training context length. The problem is not in the encoding itself but in the attention weights: the model learns to expect certain position-dependent patterns in the dot products $q_{pos} \cdot k_{pos'}$, and positions beyond the training range produce dot products outside the learned distribution.

---

## 7. The Path to RoPE

Rotary Positional Embedding (RoPE, Su et al. 2021, discussed in [[ch-09]]) addressed all three limitations:

| Issue | Sinusoidal | RoPE |
|---|---|---|
| Position injection | Additive to embedding | Applied to Q/K only, not residual stream |
| Position type | Absolute | Relative (encoded in attention scores) |
| Extrapolation | Poor in practice | Better with NTK-aware scaling |

RoPE applies the same rotation matrix to query and key vectors before the dot product:

$$q_{pos}' = R_{pos} \cdot q_{pos}, \quad k_{pos'}' = R_{pos'} \cdot k_{pos'}$$

The attention score then depends on the rotation difference:

$$q_{pos}' \cdot k_{pos'}' = q_{pos}^\top R_{pos-pos'} k_{pos'}$$

This directly encodes relative position $pos - pos'$ in the attention score, without touching the residual stream. The value vectors (and thus the information that flows through the stream) are position-free.

The mathematical machinery of RoPE -- applying rotations in 2D subspaces at geometrically spaced frequencies -- is directly descended from the sinusoidal encoding's frequency structure. The key innovation was moving the rotation from the embedding to the attention computation.

*Source: [[attention-is-all-you-need|paper]], Su et al. (2021)*
