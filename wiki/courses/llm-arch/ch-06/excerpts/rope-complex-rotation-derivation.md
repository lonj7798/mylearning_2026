<!-- scope: RoPE derivation from first principles in complex space, uniqueness proof, parent: [[ch-06]] -->

# RoPE: The Complex Rotation Derivation

RoPE is not one design among many -- it is the **unique solution** to the constraint "inner product depends only on relative position." This excerpt walks through the full derivation from first principles, following the EleutherAI blog ([[eleutherai-rope|blog]]) and the original RoFormer paper ([[rope|paper]]), with every step made explicit.

---

## The Problem Statement

We want a function $f(\mathbf{x}, m)$ that encodes position $m$ into vector $\mathbf{x}$ such that the inner product of two encoded vectors depends only on their content and relative position:

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

Three constraints:
1. **Relative position dependence**: The inner product depends on $m - n$, not on $m$ and $n$ individually
2. **Identity at origin**: $f(\mathbf{x}, 0) = \mathbf{x}$ (no encoding at position 0)
3. **Magnitude preservation**: Position encoding should not change the vector's norm

---

## Step 1: Lift to Complex Space

Pair consecutive real dimensions: $(x_1, x_2) \to z = x_1 + ix_2$. A $d$-dimensional real vector becomes a $d/2$-dimensional complex vector. We work one complex dimension at a time; the final solution applies independently to each pair.

For a single complex dimension, the encoding function maps a complex number $q \in \mathbb{C}$ and a position $m \in \mathbb{Z}$ to a complex number $f(q, m) \in \mathbb{C}$.

---

## Step 2: Polar Decomposition

Write $f(q, m)$ in polar form:

$$f(q, m) = R_f(q, m) \cdot e^{i\Theta_f(q, m)}$$

where $R_f$ is the magnitude and $\Theta_f$ is the phase (angle).

The inner product of two complex numbers $f(q, m)$ and $f(k, n)$ is:

$$\langle f(q, m), f(k, n) \rangle = \text{Re}\left[f(q, m) \cdot \overline{f(k, n)}\right]$$

$$= R_f(q, m) \cdot R_f(k, n) \cdot \cos\left(\Theta_f(q, m) - \Theta_f(k, n)\right)$$

For this to equal $g(q, k, m - n)$, we need both the magnitude product and the phase difference to depend only on $m - n$ (not on $m$ and $n$ individually).

---

## Step 3: Magnitude Is Position-Independent

Set $m = n$ in the inner product constraint:

$$\langle f(q, m), f(k, m) \rangle = g(q, k, 0)$$

This must hold for all $m$. Combined with the identity constraint $f(x, 0) = x$:

$$\langle f(q, 0), f(k, 0) \rangle = \langle q, k \rangle = g(q, k, 0)$$

So $\langle f(q, m), f(k, m) \rangle = \langle q, k \rangle$ for all $m$. This forces:

$$R_f(q, m) = |q| \quad \text{for all } m$$

**The magnitude is position-independent.** Position encoding can only change the angle of the vector, never its length. This is the "rotation preserves magnitude" property.

---

## Step 4: Phase Decomposes Additively

The phase constraint requires:

$$\Theta_f(q, m) - \Theta_f(k, n) = \Theta_g(q, k, m - n)$$

Set $k = q$ and $n = 0$:

$$\Theta_f(q, m) - \Theta_f(q, 0) = \Theta_g(q, q, m)$$

The left side depends on $m$ but not on $n$. Set $q = k$ and $m = 0$:

$$\Theta_f(q, 0) - \Theta_f(q, n) = \Theta_g(q, q, -n)$$

Now consider the general case. The phase must decompose as:

$$\Theta_f(q, m) = \Theta(q) + \phi(m)$$

where $\Theta(q)$ is a content-dependent base phase and $\phi(m)$ is a position-dependent offset. Substituting:

$$\Theta(q) + \phi(m) - \Theta(k) - \phi(n) = \Theta(q) - \Theta(k) + \phi(m) - \phi(n)$$

For this to depend only on $m - n$, we need $\phi(m) - \phi(n) = h(m - n)$ for some function $h$. The only continuous solution is $\phi(m) = m\theta$ for some constant $\theta$:

$$\phi(m) - \phi(n) = m\theta - n\theta = (m - n)\theta$$

---

## Step 5: The Unique Solution

Combining Steps 3 and 4:

$$f(q, m) = |q| \cdot e^{i(\Theta(q) + m\theta)} = q \cdot e^{im\theta}$$

where the last equality uses $q = |q| \cdot e^{i\Theta(q)}$.

**For the full $d/2$ complex dimensions**, each pair $j$ gets its own frequency $\theta_j$:

$$f(\mathbf{q}, m) = \sum_{j=1}^{d/2} q_j \cdot e^{im\theta_j} \cdot \mathbf{e}_j$$

The frequency schedule $\theta_j = 10000^{-2j/d}$ is the same geometric progression used in sinusoidal encoding. This is not a coincidence -- both are derived from the same underlying requirement for multi-scale position representation.

---

## Step 6: Verify the Relative Position Property

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = \sum_{j=1}^{d/2} \text{Re}\left[q_j e^{im\theta_j} \cdot \overline{k_j e^{in\theta_j}}\right]$$

$$= \sum_{j=1}^{d/2} \text{Re}\left[q_j \bar{k}_j \cdot e^{i(m-n)\theta_j}\right]$$

$$= \sum_{j=1}^{d/2} |q_j||k_j| \cos\left(\angle q_j - \angle k_j + (m-n)\theta_j\right)$$

This depends on $\mathbf{q}$, $\mathbf{k}$, and $(m-n)$ only. The constraint is satisfied.

---

## Uniqueness

The derivation admits exactly one family of solutions: position-proportional rotation with content-dependent base phase. You cannot satisfy the three constraints (relative dependence, identity at origin, magnitude preservation) with any other mechanism.

This is why RoPE has proven so hard to improve upon. The only way to do "better" is to relax the constraints -- which is exactly what iRoPE does by removing positional encoding from some layers entirely ([[ch-06]], Section 7).

---

## From Complex to Real: The Block-Diagonal Matrix

In real coordinates, $f(q, m) = q \cdot e^{im\theta_j}$ for the $j$-th complex dimension becomes a 2x2 rotation matrix applied to the real pair $(q_{2j-1}, q_{2j})$:

$$\begin{pmatrix} q_{2j-1}' \\ q_{2j}' \end{pmatrix} = \begin{pmatrix} \cos m\theta_j & -\sin m\theta_j \\ \sin m\theta_j & \cos m\theta_j \end{pmatrix} \begin{pmatrix} q_{2j-1} \\ q_{2j} \end{pmatrix}$$

The full rotation is block-diagonal: $d/2$ independent 2x2 rotation blocks, each rotating at its own frequency.

---

## The Connection to Sinusoidal Encoding

The sinusoidal encoding from 2017 used the same frequencies $\theta_j = 10000^{-2j/d}$ and contained rotation matrices as a mathematical consequence of the angle addition theorem ([[hf-positional-encoding-design|blog]]). The difference:
- **Sinusoidal**: Rotation applied **additively** to the embedding (pollutes semantics)
- **RoPE**: Rotation applied **multiplicatively** to Q and K (preserves semantics, enters where position is used)

The math was identical. The architectural insight -- where to apply it -- took four years.

---

## References

- [[rope|Su et al. "RoFormer" (2021) (paper)]]
- [[eleutherai-rope|EleutherAI "Rotary Embeddings: A Relative Revolution" (blog)]]
- [[hf-positional-encoding-design|Fleetwood "You Could Have Designed State of the Art Positional Encoding" (HF blog)]]
