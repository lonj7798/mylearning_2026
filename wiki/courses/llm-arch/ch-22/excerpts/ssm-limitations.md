# Excerpt: Where Pure SSMs Fall Short — The Information-Theoretic Argument

<!-- source: [[mamba|paper]], [[mamba-2|paper]] -->

## The Fundamental Capacity Constraint

A Transformer's "memory" during inference is its KV cache, which stores explicit key and value vectors for every position in the context. For a sequence of length $T$ with head dimension $d$ and $G$ KV groups, this cache holds $2GTd$ values — growing linearly with $T$.

An SSM's memory is its recurrent state $h \in \mathbb{R}^{H \times N}$, which is a fixed-size tensor regardless of $T$. For Mamba-2 with $H = 64$ heads and $N = 64$ state dimension, this is $64 \times 64 = 4096$ values. Period. Whether the model has seen 100 tokens or 1 million tokens, it has 4096 values to encode everything it needs from the past.

**The information-theoretic bound:** A state of $H \times N$ floating-point values can store at most $H \times N \times 16$ bits of information (for FP16). At $T = 100K$ tokens, the Transformer's KV cache stores roughly $2 \times 8 \times 128 \times 100000 \times 16 \approx 3.3 \times 10^9$ bits. Mamba-2's state stores $4096 \times 16 = 65536$ bits — roughly 50,000x less.

The SSM must therefore perform *lossy compression*. The selectivity mechanism (input-dependent $B$, $C$, $\alpha$) determines what to keep and what to discard, but it cannot escape the fundamental limit: a 4096-dimensional state cannot losslessly represent 100K arbitrary tokens.

## Task-Specific Failure Modes

### In-Context Learning (ICL)

**The task:** Given $k$ (input, output) examples in the prompt, infer the mapping and apply it to a new input.

**Why Transformers succeed:** Attention allows the model to compare the current query against each cached example, forming explicit cross-position similarity scores. The model can retrieve structurally similar examples and copy their output patterns. This is a *comparison* operation that operates on explicit representations of past positions.

**Why SSMs struggle:** The SSM must encode all $k$ examples into its fixed state. For ICL to work, the state must encode not just the individual examples but the *relational structure* between them — which inputs map to which outputs, and what pattern generates that mapping. As $k$ grows, the compression becomes increasingly lossy.

Empirical findings:
- For $k \leq 5$ examples with simple mappings, SSMs perform comparably to Transformers
- For $k \geq 20$ or complex mappings (requiring multi-step reasoning about the pattern), SSMs degrade significantly
- The degradation is *graceful*, not catastrophic — SSMs do worse, not randomly

### Needle-in-a-Haystack Retrieval

**The task:** Find a specific piece of information (the "needle") embedded somewhere in a long context (the "haystack").

**Why Transformers succeed:** The query attends directly to the position containing the needle. The attention mechanism's sharpening (via softmax) allows it to assign near-zero weight to irrelevant positions and near-one weight to the target position. This is $O(T)$ lookup with precise retrieval.

**Why SSMs struggle:** The needle information must survive in the state through all subsequent tokens. Each subsequent token updates the state, potentially overwriting the needle representation. Whether the needle survives depends on:

1. **How distinctive the needle was** — a highly unusual token sequence is easier for the selectivity mechanism to flag for preservation
2. **How much interference followed** — more subsequent tokens mean more opportunities for state overwrite
3. **Where the needle was** — needles early in the context must survive more state updates

Retrieval accuracy degrades with:
- Longer total context (more intervening tokens)
- More needles to find simultaneously (state must encode all of them)
- Less distinctive needles (harder for selectivity to prioritize)

### Sequence Copying

**The task:** Reproduce an input sequence verbatim.

**Why Transformers succeed:** Trivially — attend from output position $i$ to input position $i$ and copy the value. Each position is independently retrievable from the KV cache.

**Why SSMs fail for long sequences:** Copying $T$ tokens requires storing $T$ values in a state of dimension $H \times N$. For $T > H \times N$, lossless encoding is impossible (pigeonhole principle). Even for $T \leq H \times N$, the state must learn an encoding that maps each token to a unique state configuration and can later decode it — a non-trivial representational problem.

In practice, SSMs can copy sequences up to moderate lengths by learning efficient encodings, but fail on sequences longer than roughly $2N$ to $4N$ tokens. This has practical implications for:
- Code refactoring (copy existing code with modifications)
- Long-form summarization (preserving exact quotes)
- Translation (preserving named entities character-by-character)

## The Softmax Gap

The SSD framework proves SSMs are equivalent to *linear* attention. The gap between linear attention and softmax attention is well-characterized:

### Sharpening

Softmax attention weights: $w_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$

For scores $s = [10, 1, 1, 1]$, softmax produces approximately $[0.9999, 0.0000, 0.0000, 0.0000]$ — near-binary attention to one position. Linear attention (without softmax) would produce proportional weights $[10/13, 1/13, 1/13, 1/13]$ — much more diffuse.

The ability to sharpen attention to a single position is what makes exact retrieval possible. SSMs, being equivalent to linear attention, cannot achieve this level of precision.

### Competitive Normalization

Softmax's denominator creates an implicit competition: increasing $s_j$ for one position necessarily decreases the attention weight on all others (because the denominator grows). This winner-take-all dynamic is absent in linear attention and SSMs, where each position's contribution is independent.

### Dynamic Range

The exponential in softmax creates enormous dynamic range. A score difference of 10 between two positions creates a weight ratio of $e^{10} \approx 22000$. Linear operations cannot achieve this selectivity without extremely large weight magnitudes.

## Mitigation Strategies

1. **Increase state dimension $N$:** More state = more capacity, but also more computation. Mamba-2's efficiency helps here — it can afford $N = 64$ or $N = 128$ where Mamba-1 was limited to $N = 16$.

2. **Hybrid architectures:** Jamba ([[ch-21]]) interleaves SSM layers with attention layers. SSM layers handle long-range propagation efficiently; attention layers handle precise retrieval. This captures most of both advantages at intermediate cost.

3. **Task selection:** Deploy pure SSMs on tasks where approximate recall suffices — code generation, long-document summarization, streaming applications — and use Transformers or hybrids for tasks requiring exact retrieval.

4. **Retrieval augmentation:** Pair the SSM with an external retrieval system (RAG) that handles the precise-lookup cases the SSM's state cannot support.

The honest assessment: pure SSMs trade retrieval precision for computational efficiency. For many practical applications, this trade is favorable. For others, it is not. The hybrid approach ([[ch-21]]) may be the pragmatic middle ground.
