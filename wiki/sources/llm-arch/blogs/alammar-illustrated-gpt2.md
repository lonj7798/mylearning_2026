<!-- scope: decoder-only transformer architecture (GPT-2)
     deps: [[ch-02]]
     see-also: [[alammar-illustrated-transformer]], [[raschka-kv-cache]]
-->

# The Illustrated GPT-2 (Visualizing Transformer Language Models)

- **Core Insight:** Decoder-only architecture is the transformer with causal masking, KV cache, and no encoder.
- **Guideline:** Understand GPT-2 as the minimal viable LLM architecture.

- **Author:** Jay Alammar
- **URL:** https://jalammar.github.io/illustrated-gpt2/
- **Relevant chapters:** Decoder-only architecture, language modeling, autoregressive generation, GPT architecture

## Summary
A visual guide to GPT-2's decoder-only transformer architecture, explaining how it differs from the original encoder-decoder transformer and BERT. Covers the self-attention mechanism in detail, masked attention, the autoregressive generation process, and how the architecture extends beyond language modeling to translation, summarization, and music generation.

## Key Content

### GPT-2 and Language Modeling

A language model predicts the next word based on previous context. GPT-2 was trained on WebText, a 40GB internet dataset. Model sizes vary primarily in depth:
- Small: 12 blocks (~117M parameters)
- Medium: 24 blocks
- Large: 36 blocks

### Key Difference from BERT

GPT-2 uses **auto-regression** — generating one token at a time, then feeding output back as input. BERT processes bidirectional context simultaneously. GPT-2 sacrifices bidirectional understanding for autoregressive generation capability.

### Transformer Block Evolution

- **Encoder Block (Original):** Full self-attention without masking (up to 512 tokens)
- **Decoder Block (Original):** Masked self-attention preventing future token attention, plus encoder-decoder attention
- **Decoder-Only Block (GPT-2):** Eliminates encoder entirely; each block contains masked self-attention and a feed-forward neural network

### Input Processing

1. **Token Embedding:** Words converted to 768-dimensional embedding vectors (GPT-2 small) via learned embedding matrix
2. **Positional Encoding:** Separate learned matrix provides position information for each of the 1024 possible token positions, added to embeddings before processing

### Self-Attention Mechanics

**Step 1: Create Q, K, V Vectors**
- **Query (Q):** Current word's search pattern
- **Key (K):** Labels for all words in the segment
- **Value (V):** Actual semantic representations to be combined

**Step 2: Score**
Query dot-producted against all keys, normalized via softmax.

**Step 3: Sum**
Values multiplied by their scores and summed — creates weighted blend of context representations.

**Masked Self-Attention:** Before softmax, mask values set future positions to -1 billion (in GPT-2), preventing access to uncongenerated tokens.

### Multi-Head Attention

GPT-2 small conducts 12 parallel "heads," each operating on different vector subsets. Results concatenated and projected through a final weight matrix.

### Feed-Forward Network

After self-attention:
1. First layer: Expands to 4x model dimension (768 -> 3,072 for small)
2. Second layer: Projects back to model dimension (3,072 -> 768)

### Output Generation

The final block's output vector is multiplied by the embedding matrix (transposed), producing logits for all 50,000 vocabulary words. Softmax converts to probabilities.

**Sampling strategies:**
- **top-k=1:** Greedy — selects highest-probability word
- **top-k=40:** Samples from 40 most likely words (balanced creativity)
- Full distribution sampling: Considers all words weighted by probability

### Autoregressive Generation Process

1. Start token processed, producing top word prediction
2. Predicted word appended to input sequence
3. Entire sequence processes again (reusing cached key/value pairs)
4. New prediction generated from only the newest position
5. Repeat until end-of-sequence or 1024 token limit

**Efficiency via caching:** Only the latest token requires full computation; previous layers' key/value vectors are cached.

### Model Parameters (GPT-2 Small ~117-124M)

- 1 embedding matrix (word embeddings)
- 1 positional encoding matrix
- 12 transformer blocks, each containing:
  - Q/K/V projection matrix
  - Multi-head attention output projection matrix
  - Two feed-forward network weight matrices

### Beyond Language Modeling

**Machine Translation:** Decoder-only transformers can translate without explicit encoder structure. The decoder learns to parse source language and generate target language through masked attention.

**Text Summarization:** Trained on Wikipedia articles (main body -> summary). Transfer learning with GPT-2 achieves strong results in low-data scenarios, sometimes exceeding encoder-decoder baselines.

**Music Generation:** The Music Transformer applies identical decoder architecture to musical performance data. Musical notes represented as vectors with pitch, velocity, and timing. Self-attention learns recurring melodic patterns for long-range musical coherence.

## Notable Insights
- GPT-2's success derived from scale (117M+ params, 40GB data), not architectural novelty — the decoder-only design was already known.
- Masked attention is fundamentally about causality: preventing the model from seeing the future maintains the autoregressive property needed for generation.
- The embedding matrix is reused (transposed) as the output projection, tying input and output representations — a parameter-efficient design choice.
- KV caching during autoregressive generation avoids recomputing attention for all previous tokens, making inference practical.
- The decoder-only architecture is surprisingly versatile: the same structure handles language, translation, summarization, and music.
