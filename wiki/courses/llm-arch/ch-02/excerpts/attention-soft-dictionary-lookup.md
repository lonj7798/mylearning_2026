<!-- scope: attention as soft dictionary lookup — full derivation, parent: [[ch-02]] -->

# Attention as Soft Dictionary Lookup: A Derivation

This excerpt builds the formal connection between attention and dictionary lookup, starting from basic retrieval and arriving at the scaled dot-product formulation. The goal is to make precise what the analogy captures and where it breaks down.

---

## 1. Hard Dictionary Lookup

A dictionary stores key-value pairs $\{(k_j, v_j)\}_{j=1}^{n}$. Given a query $q$, hard lookup returns:

$$\text{Lookup}(q) = v_{j^*} \quad \text{where} \quad j^* = \arg\max_j \, \text{sim}(q, k_j)$$

This is a discrete operation: you get exactly one value back. It is non-differentiable (the argmax has zero gradient almost everywhere), so it cannot be trained with backpropagation.

---

## 2. Soft Dictionary Lookup

Replace the hard selection with a weighted combination:

$$\text{SoftLookup}(q) = \sum_{j=1}^{n} w_j \cdot v_j$$

where the weights $w_j$ are a function of query-key similarity:

$$w_j = \frac{\exp(\text{sim}(q, k_j))}{\sum_{m=1}^{n} \exp(\text{sim}(q, k_m))}$$

The softmax normalizes the weights to form a probability distribution. If the similarities are sharply peaked (one key matches much better than the rest), the weights approximate a one-hot vector and the output approximates hard lookup. If the similarities are diffuse, the output blends multiple values.

**This is exactly attention** with $\text{sim}(q, k) = q \cdot k / \sqrt{d_k}$.

---

## 3. From Soft Lookup to Scaled Dot-Product Attention

Let's derive the standard attention formula step by step.

**Step 1: Choose a similarity function.** The dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ is the simplest bilinear similarity. It measures the alignment between two vectors in $\mathbb{R}^{d_k}$.

**Step 2: Scale for variance control.** If $q$ and $k$ have entries drawn from $\mathcal{N}(0, 1)$, then:

$$\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$$

The standard deviation grows as $\sqrt{d_k}$. For $d_k = 64$: $\sigma = 8$. Dot products of magnitude $\pm 16$ or more are common, which pushes softmax into its saturated regime where gradients vanish.

Dividing by $\sqrt{d_k}$ normalizes the variance to 1:

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

**Step 3: Apply softmax row-wise.** For a batch of queries $Q \in \mathbb{R}^{n \times d_k}$ and keys $K \in \mathbb{R}^{n \times d_k}$:

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$$

Each row of $A$ is a probability distribution over positions. $A_{ij}$ is the weight that query $i$ assigns to key $j$.

**Step 4: Weighted value retrieval.** Multiply the attention weights by the values:

$$\text{Output} = A \cdot V \in \mathbb{R}^{n \times d_v}$$

Row $i$ of the output is the weighted sum $\sum_j A_{ij} v_j$ -- the soft lookup result for query $i$.

**Full formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

---

## 4. The Projection Step: Why Q, K, V Are Not the Raw Input

In self-attention, queries, keys, and values are all derived from the same input $X$ via learned linear projections:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

This is essential. Without projections, the dot product $X X^\top$ would be a fixed function of the input -- the model could not learn *what* to attend to. The learned projections $W_Q, W_K, W_V$ define three different "views" of each token:

- **$W_Q$**: "What am I looking for?" (the question this token asks)
- **$W_K$**: "What do I advertise?" (the label this token presents to queries)
- **$W_V$**: "What information do I provide?" (the content this token contributes)

A token's query and key can encode entirely different aspects of its meaning. A verb might query for "a noun in nominative case" (syntactic) while its key advertises "an action in past tense" (semantic). The projections allow the model to decouple these roles.

---

## 5. Geometric Interpretation

In the projected space, attention is computing a **kernel density estimate** in $\mathbb{R}^{d_k}$. Each key $k_j$ is a point in this space, and the query $q_i$ induces a softmax kernel centered at $q_i$:

$$w_j = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_m \exp(q_i \cdot k_m / \sqrt{d_k})}$$

This is a von Mises-Fisher kernel (for unit-norm vectors) or a Gaussian kernel (for general vectors) over the angle between $q_i$ and $k_j$. Keys that are close to the query in angular distance receive high weight.

**Implication:** Attention is performing a kind of nearest-neighbor interpolation in a learned embedding space, weighted by angular proximity. This is fundamentally different from the fixed-window local operations of convolutions -- attention can route information from any position, with the routing determined dynamically by content.

---

## 6. Where the Analogy Breaks Down

The "soft dictionary" analogy has three important limitations:

**1. Keys and values share the same source.** In a real dictionary, the key "cat" maps to a definition (the value) that is a completely different kind of object. In self-attention, both $k_j = x_j W_K$ and $v_j = x_j W_V$ are projections of the same input token $x_j$. The "key" and "value" are not independently chosen -- they are entangled through their shared origin.

**2. The output is a blend, not a retrieval.** Dictionary lookup returns a single entry. Attention returns a convex combination of all values. This means the output can represent information that no single value contains -- it can synthesize. For example, if two tokens at positions 3 and 7 each contribute 50% of the attention weight, the output is a vector halfway between $v_3$ and $v_7$, which may encode a novel composite meaning.

**3. The "database" is the same as the "query set."** In self-attention, every token is simultaneously a query, a key, and a value. There is no separation between the "searcher" and the "searched." This circularity is what gives self-attention its power (it can discover relationships within a single sequence) but makes it harder to reason about than a traditional query-database system.

**A better analogy:** Attention is **content-based addressing into a soft associative memory**, where the addresses are computed dynamically by comparing learned query representations against learned key representations, and the retrieved content is a weighted interpolation of learned value representations.

*Source: [[attention-is-all-you-need|paper]], [[bahdanau-attention|paper]], [[alammar-illustrated-transformer|blog]]*
