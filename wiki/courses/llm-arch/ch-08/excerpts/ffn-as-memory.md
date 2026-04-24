<!-- scope: FFN-as-memory hypothesis evidence, parent: [[ch-08]] -->

# The FFN-as-Memory Hypothesis: Evidence and Implications

The claim that Transformer feed-forward networks function as key-value memories is one of the most productive interpretive frameworks in mechanistic interpretability. This excerpt collects the primary evidence, examines the strongest counterarguments, and traces the implications for architecture design.

---

## The Core Claim

Geva et al. (2021) proposed that the FFN sublayer:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x)$$

operates as a key-value memory where:
- **Keys** = rows of $W_1$ ($d_{ff}$ patterns, each $d_{model}$-dimensional)
- **Values** = columns of $W_2$ ($d_{ff}$ vectors, each $d_{model}$-dimensional)
- **Retrieval** = $W_1 x$ computes similarity between input $x$ and all keys
- **Activation** = $\sigma(\cdot)$ selects which keys match (thresholding)
- **Output** = $W_2 \cdot \sigma(W_1 x)$ sums the selected values

Under this interpretation, the FFN is not a generic nonlinear function approximator -- it is a **retrieval system** that stores and recalls patterns from its weights.

---

## Evidence For

### 1. Interpretable Key Patterns

Geva et al. analyzed individual rows of $W_1$ across GPT-2 and found that many correspond to semantically coherent patterns:
- Specific token sequences ("New York")
- Syntactic patterns (subject-verb agreement)
- Semantic categories (dates, locations, numbers)

When the input activates a particular $W_1$ row (high dot product), the corresponding $W_2$ column contributes to the output. The model literally looks up stored patterns by matching input against key vectors.

### 2. Factual Knowledge Localization (ROME)

Meng et al. (2022) demonstrated the strongest evidence: specific factual associations are localized to specific FFN layers. Their experiment:

1. Identify which layer is most responsible for the association "Eiffel Tower is in [Paris]"
2. Perform a rank-one edit to $W_2$ at that layer, changing the stored "value" to a different city
3. The model now consistently outputs the new city for queries about the Eiffel Tower's location

This **Rank-One Model Editing (ROME)** technique works because the FFN stores factual knowledge in specific key-value pairs. If the FFN were a generic function, targeted rank-one edits would not have such clean, predictable effects.

Key findings from ROME:
- Factual associations are concentrated in **middle layers** (layers 15-25 of 48 in GPT-J)
- Early layers handle syntactic patterns; late layers handle task-specific output formatting
- The causal tracing technique identifies a "decisive" layer where the fact is recalled

### 3. Sparse Activation Patterns

In ReLU-based FFNs, only a small fraction of neurons activate for any given input. Empirical measurements show:
- GPT-2: ~3-10% of FFN neurons active per token
- Mixture-of-Experts models make this sparsity structural (only 8/256 experts activate in DeepSeek-V3)

This sparsity is consistent with the memory interpretation: most keys don't match the current input, so most values aren't retrieved. A generic MLP would be expected to activate a larger, less structured subset of neurons.

### 4. Knowledge Neurons

Dai et al. (2022) extended the analysis to identify "knowledge neurons" -- individual FFN neurons that activate specifically for certain factual knowledge. Suppressing these neurons degrades the model's ability to recall the associated facts, while suppressing random neurons has minimal effect.

### 5. Scaling Behavior

Under the memory hypothesis, model capacity for factual knowledge should scale with $d_{ff} \times L$ (total number of key-value slots across all layers). Empirically, larger models (wider FFNs, more layers) do store more facts, and factual recall accuracy scales predictably with total FFN capacity.

---

## Evidence Against (or Complicating)

### 1. Distributed Representations

Not all knowledge is cleanly localized to individual neurons or layers. Many facts require distributed computation across multiple layers and attention heads. The memory metaphor may be too clean -- the actual mechanism is messier, with partial contributions from many neurons.

### 2. Superposition

Elhage et al. (2022, Anthropic) showed that models encode more features than they have dimensions, using nearly-orthogonal directions. This means a single $W_1$ row might encode *multiple* overlapping patterns in superposition, complicating the "one row = one key" interpretation.

### 3. SwiGLU Complicates the Interpretation

The gated FFN:

$$\text{FFN}_{\text{SwiGLU}}(x) = (Swish(W_1 x) \odot Vx) \cdot W_2$$

introduces a multiplicative interaction between two projections. The "key" is no longer simply a row of $W_1$ -- it's a combination of patterns from both $W_1$ and $V$. The gating mechanism makes the retrieval more selective but harder to interpret as a simple lookup table.

### 4. Attention Also Stores Knowledge

Recent work (e.g., Gould et al., 2023) shows that attention heads also participate in factual recall, particularly through "induction heads" and "factual recall heads." The FFN is not the sole repository of knowledge; it's the primary one, but attention contributes.

---

## Architectural Implications

### Width = Memory Capacity

If FFNs are memories, then $d_{ff}$ directly controls how many key-value pairs each layer stores. This reframes the width-vs-depth tradeoff:

- **More width** ($d_{ff} \uparrow$) = more storage slots per layer = better factual recall
- **More depth** ($L \uparrow$) = more composition steps = better reasoning

This is empirically supported: models that underperform on knowledge-intensive tasks (trivia, factual QA) tend to have narrow FFNs relative to their depth.

### MoE as Explicit Memory Banking

Mixture-of-Experts ([[ch-14]]) makes the memory interpretation structural:

$$\text{MoE-FFN}(x) = \sum_{e \in \text{selected}} g_e(x) \cdot \text{FFN}_e(x)$$

Each expert is a memory bank. The router selects which banks to query. This is the logical endpoint of the FFN-as-memory hypothesis: if most keys don't match, don't compute them.

DeepSeek-V3 ([[deepseek-v3|report]]) uses 256 routed experts per layer, each with its own $W_1, V, W_2$ matrices. The total number of key-value slots is $256 \times d_{expert}$, but each token only queries $8 \times d_{expert}$ slots. This gives massive memory capacity at modest compute cost.

### Model Editing

The FFN-as-memory view directly enables model editing techniques:
- **ROME** (rank-one edits to $W_2$): Change individual facts
- **MEMIT** (mass editing): Edit many facts simultaneously across layers
- **Knowledge erasure**: Zero out specific neurons to remove knowledge

These techniques work *because* knowledge is localized in FFN weights. They would not work if knowledge were fully distributed across the entire network.

### Knowledge Distillation

If the FFN stores knowledge, then distilling a large model into a smaller one is fundamentally limited by the smaller model's FFN capacity ($d_{ff} \times L$). A 7B model cannot store as many facts as a 70B model, regardless of how good the distillation technique is. This explains why small distilled models often fail on long-tail factual knowledge while performing well on common knowledge.

---

## The Residual Stream Connection

The FFN-as-memory view connects to the residual stream interpretation ([[ch-09]]). Each FFN reads from the residual stream (the current context representation), retrieves relevant stored information, and writes a delta back to the stream. The attention mechanism routes information *between* positions; the FFN retrieves stored information *at* each position. Together, they implement a retrieve-then-reason pipeline.

---

## References

- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021)
- Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, 2022)
- Dai et al., "Knowledge Neurons in Pretrained Transformers" (2022)
- Elhage et al., "Toy Models of Superposition" (Anthropic, 2022)
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]]
- [[glu-variants|Shazeer, "GLU Variants Improve Transformer" (2020) (paper)]]
