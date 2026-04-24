<!-- scope: residual stream view — deep dive on skip connections and information flow, parent: [[ch-03]] -->

# The Residual Stream: A Deep Dive

This excerpt develops the "residual stream" interpretation of Transformers in detail: why skip connections are not auxiliary but define the model's information backbone, how they enable compositional features across layers, and what this means for interpretability and architecture design.

---

## 1. The Degradation Problem and Its Solution

He et al. (2015, [[resnet|paper]]) observed a paradox in deep networks: adding more layers to a plain (no-skip-connection) network made performance *worse*, even on the training set. This was not overfitting -- it was an optimization failure. Deeper networks were harder to optimize, not less expressive.

**The mathematical argument:** Consider a shallow network that achieves some accuracy. A deeper network should be *at least* as good: the extra layers could learn the identity function, passing input through unchanged. But in practice, learning the identity through a stack of nonlinear layers is hard -- the optimization landscape makes it unlikely that SGD will find identity-like solutions.

**The residual fix:** Instead of learning $H(x)$ directly, learn the residual $F(x) = H(x) - x$:

$$y = x + F(x)$$

Now if the optimal transformation is close to identity ($H(x) \approx x$), the network only needs $F(x) \approx 0$ -- much easier for SGD to find. The skip connection provides a "default path" that passes information through unchanged, and each layer only needs to learn useful corrections.

---

## 2. From ResNet to the Transformer's Residual Stream

The Transformer applies residual connections to every sub-layer. For a single Transformer layer with attention (Attn) and feed-forward network (FFN):

**Post-LN (original Transformer):**
$$h' = \text{LayerNorm}(x + \text{Attn}(x))$$
$$h'' = \text{LayerNorm}(h' + \text{FFN}(h'))$$

**Pre-LN (modern Transformers):**
$$h' = x + \text{Attn}(\text{LayerNorm}(x))$$
$$h'' = h' + \text{FFN}(\text{LayerNorm}(h'))$$

In Pre-LN, the residual stream is cleaner to analyze. The full forward pass through $L$ layers is:

$$x_L = x_0 + \sum_{\ell=1}^{L} \left[\text{Attn}_\ell(\text{LN}(x_{\ell-1})) + \text{FFN}_\ell(\text{LN}(x_{\ell-1} + \text{Attn}_\ell(\text{LN}(x_{\ell-1}))))\right]$$

Or, more compactly: the output is the input embedding plus the sum of all sub-layer contributions:

$$x_L = x_0 + \sum_{\ell=1}^{L} \Delta_{\text{attn}}^{(\ell)} + \sum_{\ell=1}^{L} \Delta_{\text{ffn}}^{(\ell)}$$

**The stream metaphor:** Think of $x_\ell \in \mathbb{R}^{d_{\text{model}}}$ at each token position as a "river" flowing from input embedding to output head. Each attention and FFN sub-layer is a tributary that reads from the river and writes an additive update back to it. The river carries all information; sub-layers are specialized readers/writers.

---

## 3. Why Additive Updates Enable Composition

The additive structure has a profound consequence: **information written by early layers is not overwritten by later layers**. It persists in the stream and can be read by any downstream layer.

### Example: Induction Heads

Elhage et al. (2021) discovered a two-layer compositional pattern called an **induction head**. It requires two attention heads working together across layers:

**Layer $\ell$: "Previous token" head.** At position $t$, this head attends to position $t-1$ and copies information about token $x_{t-1}$ into the residual stream at position $t$. After this head, position $t$'s stream contains information about "what token preceded me."

**Layer $\ell + k$: "Pattern matching" head.** This head's query looks for "positions where the preceding token matched $x_{t-1}$." It finds an earlier position $s$ where $x_{s-1} = x_{t-1}$, then copies the token at $s$ (i.e., $x_s$) to predict $x_t$.

The critical point: this composition only works because the "previous token" head's output persists in the residual stream and is still available $k$ layers later. In a pipeline architecture without skip connections, layer $\ell$'s output would be processed (and potentially destroyed) by layers $\ell+1$ through $\ell+k-1$ before the pattern-matching head could read it.

---

## 4. Gradient Flow Through the Stream

The residual stream provides a **gradient superhighway**. During backpropagation, the gradient from the loss at the output flows back through two paths at each layer:

$$\frac{\partial \mathcal{L}}{\partial x_\ell} = \frac{\partial \mathcal{L}}{\partial x_{\ell+1}} \cdot \left(\mathbf{I} + \frac{\partial F_{\ell+1}(x_\ell)}{\partial x_\ell}\right)$$

The identity matrix $\mathbf{I}$ from the skip connection ensures that gradients pass through unchanged. Even if $\partial F / \partial x$ is small (vanishing sublayer gradient), the identity path preserves the gradient magnitude.

**Quantitative impact:** In a 96-layer Transformer without skip connections, gradients would need to pass through 96 matrix multiplications. Even with careful initialization, the gradient magnitude would shrink exponentially. With skip connections, the gradient at layer 1 has a direct additive path from layer 96 -- the identity contribution does not shrink.

This is why Transformers can be trained with 100+ layers while plain feedforward networks of the same depth would be untrainable.

---

## 5. The Residual Stream as a Communication Bus

The stream interpretation reveals a clear functional separation:

| Component | Role | Analogy |
|---|---|---|
| Residual stream | Communication bus | Shared memory / data bus |
| Attention heads | Inter-position communication | Message passing between processors |
| FFN layers | Per-position computation | Local compute on each processor |
| Layer normalization | Signal conditioning | Voltage regulation |
| Output head (unembedding) | Final readout | Output port |

**Attention writes cross-position information** to the stream: "token at position 5 is relevant to the prediction at position 12." The FFN reads the accumulated information at each position and computes nonlinear transformations: "given all the context gathered by attention, what factual knowledge applies here?"

Geva et al. (2021) showed that FFN layers function as **key-value memories**: each row of $W_1$ (the up-projection) acts as a pattern detector ("key"), and the corresponding column of $W_2$ (the down-projection) is the information written to the stream when that pattern activates ("value"). ReLU (or SiLU/GELU in modern models) provides sparsity -- only a fraction of patterns activate for any given input.

---

## 6. Implications for Architecture Design

### Layer ordering is flexible
Because each sub-layer reads from and writes to a shared stream, the order of layers is less constrained than in a pipeline. Some architectures (Sandwich Transformers, hybrid attention-SSM models like Jamba) interleave different sub-layer types without degradation. The stream accommodates any ordering as long as the necessary information is written before it is read.

### Width (d_model) determines stream bandwidth
A wider residual stream can carry more simultaneous signals without interference. This is one reason modern LLMs have grown $d_{\text{model}}$ from 512 (original Transformer) to 8192 (Llama 2 70B): the stream needs bandwidth proportional to the model's knowledge and capability.

### Depth enables composition, not just capacity
Adding layers does not just add parameters (you could add parameters by widening). Each new layer adds a read-write cycle on the stream, enabling more complex compositions. A 2-layer model can compose two attention patterns; a 32-layer model can compose 32. The number of layers determines the compositional depth of the features the model can represent.

### Model editing targets specific stream positions
Techniques like ROME and MEMIT edit specific factual associations by modifying FFN weights in specific layers. This works precisely because the residual stream architecture makes it possible to localize where specific information is written. Without the clean additive structure, factual associations would be distributed non-additively across layers, making surgical editing impossible.

*Source: [[resnet|paper]], Elhage et al. (2021), Geva et al. (2021), [[attention-is-all-you-need|paper]]*
