# Excerpt: The "Lost in the Middle" Problem

<!-- source: Liu et al. (2023), architectural analysis -->

## The Phenomenon

When relevant information is placed at different positions within a long context, model performance follows a **U-shaped curve**:

- **Beginning of context (positions 0-10%):** High accuracy. The model reliably retrieves and uses information here.
- **End of context (positions 90-100%):** High accuracy. Recent tokens receive strong attention.
- **Middle of context (positions 30-70%):** Significantly degraded accuracy. Information here is systematically underweighted.

This is not a subtle effect. On needle-in-a-haystack tasks with 128K context, placing the target fact at the midpoint can reduce retrieval accuracy by **20-40%** compared to placing it at the start or end.

## Three Compounding Causes

### 1. Primacy Bias (Attention Sinks)

The earliest tokens in the sequence accumulate a structural advantage through the residual stream. In a multi-layer transformer, the first token's representation is updated by attention from every subsequent position across every layer. This creates "attention sinks" — positions that attract disproportionate attention weight regardless of their semantic content.

The mechanism:
- At layer 1, token 0 is just another token
- At layer 2, token 0's representation includes aggregated information from layer 1's attention over all other tokens
- By layer L, token 0 has accumulated L layers of cross-position information
- This makes its key vector a good match for many queries, attracting even more attention

Some models explicitly leverage this by adding a dedicated "sink" token (e.g., `<bos>`). Mistral's rolling buffer cache always retains the first few positions for this reason.

### 2. Recency Bias

Multiple mechanisms favor recent tokens:

- **RoPE's distance decay:** The inner product between RoPE-rotated vectors naturally decreases with relative distance, providing a soft bias toward nearby tokens.
- **Causal masking:** During training, the model generates each token conditioned primarily on its recent predecessors. The training distribution therefore overrepresents short-range dependencies.
- **Sliding window attention:** In hybrid architectures, the majority of layers (e.g., 5 out of 6 in Gemma 3) only see the most recent W tokens. Information from early positions must survive propagation through the residual stream.

### 3. Middle Neglect

Tokens in the middle receive neither advantage:
- They are too far from position 0 to benefit from the attention sink effect
- They are too far from the generation position to benefit from recency bias
- In multi-layer processing, their signal must compete with both primacy and recency-biased positions at every layer

The compound effect is multiplicative across layers: a token that receives slightly less attention at each layer has its information exponentially attenuated by the time it reaches the output layer.

## Quantitative Evidence

From Liu et al.'s experiments with 16K-128K contexts:

| Position of target info | Accuracy (multi-document QA) |
|------------------------|------------------------------|
| First document | 82% |
| Middle documents | 56% |
| Last document | 79% |

The ~26 percentage point gap between beginning/end and middle is consistent across models and tasks. It is worse for:
- Longer contexts (more positions = more competition)
- Smaller models (less capacity for distributed representations)
- Models trained without explicit long-context data

## Mitigations

### Training-Side

**Uniform position sampling:** Construct training examples where the relevant information appears at uniformly random positions within the context. Standard language modeling training naturally overrepresents information at the "beginning" of documents (due to how data is chunked) and the "end" (due to causal prediction). Explicitly training with mid-context relevance reduces the U-shaped bias.

### Architecture-Side

**Position-free layers (iRoPE):** NoPE layers attend based purely on content similarity. They have no positional bias — a semantically relevant token at position 50K receives the same attention as one at position 1K. Interleaving NoPE layers throughout the model provides content-addressed retrieval that is immune to the position-dependent biases.

**Attention temperature scaling:** Sharpening the attention distribution counteracts the entropy dilution from long sequences. When attention is more focused, it is more likely to attend to the semantically relevant token regardless of position.

### Inference-Side

**Document ordering heuristic:** Place the most important information at the beginning or end of the prompt. This is a pragmatic workaround, not a solution, but it can improve retrieval accuracy by 20+ percentage points at zero compute cost.

**Retrieval-augmented generation:** Instead of placing all information in a single long context, use RAG to retrieve and place the most relevant chunks near the end of the prompt (just before the query). This converts a long-context problem into a short-context problem where recency bias works in your favor.

## Implications for Architecture Design

The lost-in-the-middle problem reveals that **nominal context length does not equal effective context length**. A model with a 128K context window that exhibits strong U-shaped bias has an effective uniform-quality context length much shorter than 128K.

The models that best mitigate this problem are those with:
1. Training data that distributes relevant information uniformly across positions
2. Architectural features (NoPE layers, attention temperature) that reduce positional bias
3. Sufficient model capacity to maintain distributed representations across many positions

This is why context-length benchmarks are moving beyond "does the model process N tokens without crashing" toward "does the model equally utilize information at all positions" — the needle-in-a-haystack test at varied depths.
