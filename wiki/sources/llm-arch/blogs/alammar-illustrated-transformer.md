<!-- scope: visual transformer architecture walkthrough
     deps: [[ch-02]]
     see-also: [[alammar-illustrated-gpt2]], [[weng-transformer-family]]
-->

# The Illustrated Transformer

- **Core Insight:** Visual decomposition of attention into 6 concrete steps makes the mechanism intuitive.
- **Guideline:** Use visual step-by-step decomposition when learning or teaching attention.

- **Author:** Jay Alammar
- **URL:** https://jalammar.github.io/illustrated-transformer/
- **Relevant chapters:** Transformer architecture, attention mechanisms, encoder-decoder, positional encoding

## Summary
A visual, step-by-step walkthrough of the Transformer architecture from "Attention Is All You Need." Alammar breaks down self-attention, multi-head attention, positional encoding, the encoder-decoder structure, and the training process with clear explanations suitable for building deep intuition about how transformers work.

## Key Content

### High-Level Architecture

The Transformer is a sequence-to-sequence model with two main components:
- **Encoding component:** Stack of encoders (typically 6)
- **Decoding component:** Stack of decoders (matching encoder quantity)

**Encoder structure** (each identical encoder contains two sub-layers):
1. Self-attention layer (allows encoders to examine other input words)
2. Feed-forward neural network (applied independently per position)

**Decoder structure** includes both encoder components plus an additional encoder-decoder attention layer positioned between them, helping decoders focus on relevant input portions.

### Tensor and Vector Flow

Words transform into 512-dimensional vectors via embedding algorithms. Embedding only occurs at the bottom encoder; subsequent encoders receive outputs from lower encoders.

**Parallel processing insight:** "The word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel."

### Self-Attention Mechanism (6 Steps)

Self-attention allows the model to examine other input positions when encoding specific words. Example: understanding that "it" in "The animal didn't cross the street because it was too tired" refers to "animal," not "street."

**Step 1: Create Query, Key, Value Vectors**
Each input vector multiplies by three trained weight matrices (W_Q, W_K, W_V), producing:
- Query vector (Q) — 64-dimensional
- Key vector (K) — 64-dimensional
- Value vector (V) — 64-dimensional

The 64-dim size (vs 512-dim embeddings) is an architectural choice supporting multi-head attention.

**Step 2: Calculate Attention Scores**
Score = dot product of Query vector with Key vector of each position.
For word position 1: score_1 = q_1 . k_1, score_2 = q_1 . k_2, etc.

**Step 3: Normalize Scores**
Divide all scores by 8 (square root of key vector dimension: sqrt(64) = 8). Produces more stable gradients.

**Step 4: Apply Softmax**
All scores become positive and sum to 1.0. The resulting distribution indicates focus allocation across input positions.

**Step 5: Weight Value Vectors**
Multiply each value vector by its corresponding softmax score. Preserves attended words while suppressing irrelevant ones.

**Step 6: Sum Weighted Values**
Sum all weighted value vectors to produce the self-attention output for that position.

### Matrix Formula

All six steps compress into:

**Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V**

### Multi-Head Attention

The paper enhanced self-attention through "multi-headed" attention:

1. **Expanded focus:** Multiple attention heads allow simultaneous focus on different positions
2. **Multiple representation subspaces:** Eight separate Q/K/V weight matrix sets enable projections into eight distinct representational spaces

**Process:**
- Eight complete self-attention calculations occur in parallel using eight different W_Q/W_K/W_V matrix sets
- Each produces independent Z matrices (Z_0 through Z_7)
- Concatenate all eight matrices, multiply by weight matrix W_O:

**MultiHead(Q, K, V) = Concat(head_0, ..., head_7) W_O**

### Positional Encoding

The architecture lacks inherent positional information. Solution: add positional encoding vectors to each input embedding using sine and cosine functions:
- One half generated using sine functions
- One half generated using cosine functions
- Results concatenated to form complete encoding vectors

"It gives the advantage of being able to scale to unseen lengths of sequences."

### Residual Connections and Layer Normalization

Each encoder sub-layer incorporates:
- Residual connections wrapping the sub-layer
- Layer normalization following the sub-layer

These stabilize training and improve gradient flow through deep stacks.

### Decoder Operations

**Phase 1 — Encoding:** The encoder stack processes the input completely. The top encoder outputs generate attention key (K) and value (V) matrices that feed into each decoder's encoder-decoder attention layer.

**Phase 2 — Decoding:** Iterative, producing one output element per step:
1. Previous output embedded with positional encoding
2. Bottom decoder processes encoded input
3. Each decoder layer bubbles results upward
4. Repeat until end-of-sequence symbol

**Masked self-attention in decoders:** "The self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to -inf) before the softmax step."

**Encoder-decoder attention:** Queries from decoder, keys and values from encoder stack outputs.

### Output Generation

1. **Linear layer:** Fully connected network projects decoder output to logits vector matching vocabulary size
2. **Softmax layer:** Converts logits to probabilities; highest probability word becomes output

### Training Process

- Model compares output probability distribution against target using cross-entropy or KL divergence
- Backpropagation adjusts all weights to move predictions closer to targets

**Decoding strategies:**
- **Greedy decoding:** Select highest-probability word at each step
- **Beam search:** Maintain top-k candidates simultaneously; trade computational cost for quality

## Notable Insights
- The feed-forward layers enable parallel execution across all positions, unlike sequential RNNs — this is the key efficiency advantage.
- Multiple attention heads provide interpretability: different heads learn distinct linguistic phenomena (syntactic structure, semantic links, short-range dependencies).
- Positional encoding compensates for the architecture's position-agnostic operations — without it, the model treats input as a set, not a sequence.
- The 64-dimensional Q/K/V vectors (vs 512 model dimension) exist specifically to make multi-head attention computationally feasible: 8 heads x 64 = 512.
