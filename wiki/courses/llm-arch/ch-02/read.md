# Chapter 2: The Attention Mechanism

<!-- scope: from RNNs to attention, scaled dot-product QKV, multi-head attention, O(n^2) cost analysis
     deps: [[ch-01]]
     see-also: [[ch-03]] (the original Transformer), [[ch-07]] (attention variants: MQA, GQA, MLA, Flash Attention)
-->

## Overview

Chapter 1 established that language modeling is probability assignment over token sequences, and that autoregressive models optimize cross-entropy loss via teacher forcing. But we left a critical question open: *how does the model decide which previous tokens matter when predicting the next one?*

The attention mechanism is the answer. It replaced the fixed-length bottleneck of encoder-decoder RNNs with a dynamic, content-based addressing scheme that lets the model look at every input position and decide, per query, how much each one matters. This single idea — introduced by Bahdanau et al. (2014) ([[bahdanau-attention|paper]]) for machine translation, then generalized by Vaswani et al. (2017) ([[attention-is-all-you-need|paper]]) into self-attention — is the computational primitive that makes Transformers work. Every architecture you study in this course, from LLaMA to DeepSeek-V3 to Mamba hybrids, is either built on attention, designed to replace it, or both.

This chapter covers the mechanism itself: the mathematical formulation, the design decisions behind it (why scale? why multiple heads? why this particular form of softmax weighting?), and the quadratic cost that constrains every deployment decision. Understanding attention deeply here will pay off repeatedly — in [[ch-07]] when we study the variants designed to make it cheaper, in [[ch-16]] when we tackle long-context scaling, and in [[ch-25]] when KV-cache management becomes the serving bottleneck.

---

## 1. The Information Bottleneck That Motivated Attention

### The Seq2Seq Baseline

Sutskever et al. (2014) ([[seq2seq|paper]]) established the encoder-decoder paradigm for sequence-to-sequence tasks. An encoder LSTM reads an input sequence $(x_1, \ldots, x_T)$ and compresses it into a single fixed-length hidden state $h_T$. A decoder LSTM then generates the output sequence conditioned on this vector.

The architecture achieved a BLEU score of 34.8 on WMT'14 English-to-French — competitive with phrase-based statistical MT systems (33.3 BLEU). Two surprising findings emerged:

1. **Depth matters.** Four-layer LSTMs significantly outperformed shallow ones. Deep encoder-decoder stacks were essential for capturing the hierarchical structure of language.
2. **Reversing source sentences helps.** Feeding the source sentence backwards improved performance markedly. Why? It created shorter-range dependencies between corresponding source and target tokens at the start of generation, where getting the trajectory right matters most.

That second finding is a red flag. If you need to hack the input ordering to help the model, the architecture has a fundamental information-flow problem.

### The Bottleneck

The problem is architectural: the entire source sentence must be compressed into a single vector $h_T \in \mathbb{R}^d$ before decoding begins. For a 4096-dimensional hidden state encoding a 50-word sentence, that's roughly 80 dimensions per source word — and those dimensions must simultaneously encode word identity, position, syntactic role, and semantic relationships. For longer sentences, the compression becomes lossy.

Empirically, Cho et al. (2014) showed that encoder-decoder performance degrades sharply on sentences longer than ~20 tokens. The fixed-length vector is a bandwidth bottleneck: the decoder has no way to go back and re-examine specific parts of the input.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:sans-serif;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:20px; font-weight:bold;">The Fixed-Length Bottleneck</div>
<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; justify-content:center;">
<div style="display:flex; flex-direction:column; gap:6px; align-items:center;">
<div style="color:#888; font-size:11px;">Source tokens</div>
<div style="display:flex; gap:4px;">
<div style="background:#0f3460; padding:8px 12px; border-radius:6px; color:#e94560; font-size:12px;">x₁</div>
<div style="background:#0f3460; padding:8px 12px; border-radius:6px; color:#e94560; font-size:12px;">x₂</div>
<div style="background:#0f3460; padding:8px 12px; border-radius:6px; color:#e94560; font-size:12px;">x₃</div>
<div style="background:#0f3460; padding:8px 12px; border-radius:6px; color:#e94560; font-size:12px;">...</div>
<div style="background:#0f3460; padding:8px 12px; border-radius:6px; color:#e94560; font-size:12px;">x_T</div>
</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#e94560; font-size:20px;">→</div>
<div style="color:#666; font-size:10px;">compress</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="background:#e94560; padding:14px 24px; border-radius:8px; color:#fff; font-weight:bold; font-size:13px;">h_T</div>
<div style="color:#e94560; font-size:10px; font-weight:bold;">bottleneck</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#e94560; font-size:20px;">→</div>
<div style="color:#666; font-size:10px;">decode</div>
</div>
<div style="display:flex; flex-direction:column; gap:6px; align-items:center;">
<div style="color:#888; font-size:11px;">Target tokens</div>
<div style="display:flex; gap:4px;">
<div style="background:#16213e; padding:8px 12px; border-radius:6px; color:#a8d8ea; font-size:12px;">y₁</div>
<div style="background:#16213e; padding:8px 12px; border-radius:6px; color:#a8d8ea; font-size:12px;">y₂</div>
<div style="background:#16213e; padding:8px 12px; border-radius:6px; color:#a8d8ea; font-size:12px;">y₃</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
All information about the source must pass through a single d-dimensional vector. For long sentences, this is lossy.
</div>
</div>

---

## 2. Bahdanau Attention: Learning to Align

Bahdanau, Cho, and Bengio (2014) ([[bahdanau-attention|paper]]) proposed a direct fix: instead of forcing the decoder to work from a single vector, let it **look back at all encoder hidden states** at every decoding step, and learn which ones to focus on.

At each decoder step $t$, the model computes an **alignment score** $e_{tj}$ between the current decoder state $s_{t-1}$ and each encoder hidden state $h_j$:

$$e_{tj} = a(s_{t-1}, h_j)$$

where $a$ is a learned alignment function (a small feedforward network). These scores are normalized via softmax to produce **attention weights**:

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T} \exp(e_{tk})}$$

The **context vector** is a weighted sum of encoder hidden states:

$$c_t = \sum_{j=1}^{T} \alpha_{tj} h_j$$

This context vector $c_t$ changes at every decoding step — the decoder dynamically focuses on different source positions as it generates each target word. The attention weights $\alpha_{tj}$ are interpretable: they form a soft alignment matrix between source and target positions.

### What Bahdanau Got Right

The key insight was framing attention as **soft alignment** rather than hard selection. The model doesn't pick one source position — it takes a weighted combination of all positions. This is differentiable (no need for REINFORCE or other discrete optimization tricks), stable to train, and naturally handles one-to-many and many-to-one alignments that are common in translation.

Critically, the attention weights $\alpha_{tj}$ learned to match known linguistic alignment patterns without any alignment supervision. The model discovered word correspondences purely from the translation objective. This was the first strong evidence that attention functions as a general-purpose information routing mechanism, not just a translation trick.

---

## 3. Scaled Dot-Product Attention

Vaswani et al. (2017) ([[attention-is-all-you-need|paper]]) generalized Bahdanau's additive attention into the form used by every modern Transformer. The formulation introduces three explicit roles — **Query**, **Key**, and **Value** — and replaces the learned alignment network with a simple dot product.

Given input representations $X \in \mathbb{R}^{n \times d_{model}}$, we project them into three spaces:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_K \in \mathbb{R}^{d_{model} \times d_k}$, $W_V \in \mathbb{R}^{d_{model} \times d_v}$.

The attention computation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

In the Transformer base model: $d_{model} = 512$, $d_k = d_v = 64$, with 8 attention heads.

### The Q-K-V Analogy and Its Limits

The standard analogy: attention is a **soft dictionary lookup**. The query asks a question, the keys are the labels on stored entries, and the values are the stored content. The dot product $QK^T$ measures how well each key matches the query, softmax normalizes the match scores into a probability distribution, and the output is a weighted sum of values.

This analogy is useful for initial intuition but breaks down in important ways:

1. **Keys and values come from the same token.** In a real dictionary, the key ("cat") and the value (the definition of cat) are different things. In self-attention, $K$ and $V$ are both linear projections of the same input — they share the same source information, just projected into different subspaces.

2. **Queries also come from the same sequence.** In self-attention, the "question" and the "database" are the same set of tokens. Every token is simultaneously a query (asking "what should I attend to?") and a key-value pair (answering other tokens' queries).

3. **The output is a weighted blend, not a retrieval.** Dictionary lookup returns a discrete entry. Attention returns a convex combination of all values — it can synthesize information from multiple positions simultaneously.

A more precise analogy: attention is **content-based addressing into a soft memory**, where the address is computed by comparing a query against stored keys, and the output is an interpolation of all stored values weighted by address similarity.

### Why Scale by $\sqrt{d_k}$?

This is a frequently asked interview question, and the real answer has more nuance than "to prevent large dot products."

For queries and keys with entries drawn independently from $\mathcal{N}(0, 1)$, the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. As $d_k$ grows, the dot products grow in magnitude, pushing the softmax into regions where its gradients are extremely small (the tails of the softmax saturate).

Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping the softmax in its sensitive (non-saturated) region regardless of dimension:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

**Why this matters in practice:** In the Transformer base model, $d_k = 64$, so $\sqrt{d_k} = 8$. Without scaling, the dot products have standard deviation 8 — large enough to push softmax outputs toward one-hot vectors, effectively making attention a hard (non-differentiable) selection instead of a soft blend. The model would lose the ability to attend to multiple positions simultaneously, and gradients would vanish for all but the highest-scoring key.

Vaswani et al. tested this directly: additive attention (Bahdanau's formulation) outperformed unscaled dot-product attention for large $d_k$, but **scaled** dot-product attention matched additive attention while being faster (matrix multiplication is highly optimized on GPUs). The scaling factor is what makes the simpler formulation viable.

---

## 4. Attention as Weighted Value Retrieval: A Computational View

Let's trace through the computation for a concrete example to build geometric intuition.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:sans-serif;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:20px; font-weight:bold;">Scaled Dot-Product Attention — Step by Step</div>

<div style="display:flex; flex-direction:column; gap:16px;">

<div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
<div style="background:#16213e; padding:12px 16px; border-radius:8px; min-width:120px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Step 1</div>
<div style="color:#aaa; font-size:11px; margin-top:4px;">Linear Projections</div>
</div>
<div style="color:#e0e0e0; font-size:12px; font-family:monospace; flex:1;">
X (n x d_model) --W_Q--> Q (n x d_k)<br/>
X (n x d_model) --W_K--> K (n x d_k)<br/>
X (n x d_model) --W_V--> V (n x d_v)
</div>
</div>

<div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
<div style="background:#16213e; padding:12px 16px; border-radius:8px; min-width:120px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Step 2</div>
<div style="color:#aaa; font-size:11px; margin-top:4px;">Attention Scores</div>
</div>
<div style="color:#e0e0e0; font-size:12px; font-family:monospace; flex:1;">
S = QK^T / sqrt(d_k)  -->  (n x n) matrix<br/>
S[i,j] = similarity of query_i with key_j
</div>
</div>

<div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
<div style="background:#16213e; padding:12px 16px; border-radius:8px; min-width:120px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Step 3</div>
<div style="color:#aaa; font-size:11px; margin-top:4px;">Softmax (per row)</div>
</div>
<div style="color:#e0e0e0; font-size:12px; font-family:monospace; flex:1;">
A = softmax(S, dim=-1)  -->  (n x n) matrix<br/>
Each row sums to 1.0; A[i,j] = how much position i attends to j
</div>
</div>

<div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
<div style="background:#16213e; padding:12px 16px; border-radius:8px; min-width:120px; text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:13px;">Step 4</div>
<div style="color:#aaa; font-size:11px; margin-top:4px;">Weighted Sum</div>
</div>
<div style="color:#e0e0e0; font-size:12px; font-family:monospace; flex:1;">
Output = A V  -->  (n x d_v) matrix<br/>
output_i = sum_j A[i,j] * value_j
</div>
</div>

</div>

<div style="color:#888; font-size:11px; margin-top:20px; text-align:center; border-top:1px solid #333; padding-top:12px;">
For n=1024, d_k=64: QK^T produces a 1024x1024 score matrix (1M entries). This is the O(n^2) cost.
</div>
</div>

### What the Softmax Does (and Doesn't Do)

Softmax converts raw scores into a probability distribution over positions. But the distribution is typically **far from uniform** — it's sharply peaked. In trained Transformers, most attention heads produce near-sparse distributions where 5-10 positions receive the bulk of the weight.

This means attention is operationally closer to a **soft top-k selection** than a uniform blending. The softmax isn't spreading information evenly; it's concentrating it. This property is important for understanding why attention works — it enables selective information routing — and why it's expensive — you must compute all $n^2$ scores even though most of them contribute negligibly after softmax.

---

## 5. Computational Cost: $O(n^2 d)$ Time and $O(n^2)$ Memory

### Time Complexity

The dominant computation is the matrix multiplication $QK^T$:

- $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$
- $QK^T \in \mathbb{R}^{n \times n}$: costs $O(n^2 d_k)$ FLOPs
- $\text{softmax}(S) V$ where $V \in \mathbb{R}^{n \times d_v}$: costs $O(n^2 d_v)$ FLOPs

Total per attention layer: $O(n^2 d)$ where $d = d_k \approx d_v$.

For comparison, the feed-forward network (FFN) in each Transformer layer costs $O(n \cdot d_{model} \cdot d_{ff})$. In the standard Transformer, $d_{ff} = 4 \cdot d_{model}$, so FFN costs $O(n \cdot d_{model}^2)$ — linear in sequence length.

**The crossover point:** Attention dominates the compute budget when $n > d_{model}$. For the base Transformer ($d_{model} = 512$), attention becomes the bottleneck at sequences longer than 512 tokens. For modern models ($d_{model} = 4096$), the crossover is at 4096 tokens. This is why the quadratic cost was "manageable" in 2017 (sequences rarely exceeded 512 tokens) but became critical as context windows grew to 128K+ tokens.

### Memory Complexity

The attention score matrix $S \in \mathbb{R}^{n \times n}$ must be materialized (or its rows must be computed on-the-fly). In standard (non-Flash) attention:

| Sequence Length $n$ | Score Matrix Size (FP16) | Per Head |
|---|---|---|
| 512 | 512 KB | manageable |
| 2,048 | 8 MB | fine |
| 8,192 | 128 MB | tight |
| 32,768 | 2 GB | problematic |
| 131,072 | 32 GB | exceeds most GPUs |

This $O(n^2)$ memory wall is the reason Flash Attention ([[ch-07]]) exists — it reformulates the computation to avoid materializing the full score matrix, reducing memory to $O(n)$ while maintaining exact computation. Understanding the quadratic cost here makes the motivation for Flash Attention immediate.

### Why This Matters for Architecture Decisions

The quadratic cost creates a three-way tension that drives much of the architecture research in this course:

1. **Longer contexts enable better modeling** — more context means better predictions, especially for tasks requiring long-range dependencies.
2. **Quadratic cost makes long contexts expensive** — doubling the context quadruples the attention cost.
3. **Approximations sacrifice quality** — linear attention variants ([[ch-15]], [[ch-22]]) reduce cost but often lose the sharp, selective routing that makes attention effective.

Every architecture in Phase 4-5 of this course takes a position on this trilemma. DeepSeek-V3's MLA ([[ch-19]]) compresses KV representations by 93%. Gemma 3 ([[ch-20]]) uses 5:1 local/global attention ratios. Jamba ([[ch-21]]) hybridizes attention with SSMs. Mamba-2 ([[ch-22]]) eliminates attention entirely. Understanding *why* they make these choices starts here, with the quadratic cost.

---

## 6. Multi-Head Attention

### The Motivation

A single attention head computes one set of attention weights — one pattern of information routing per layer. But language requires simultaneous tracking of multiple relationship types: syntactic structure ("the verb agrees with its subject three clauses back"), semantic role ("the agent of this action"), coreference ("which noun does 'it' refer to?"), and positional proximity.

Multi-head attention addresses this by running $h$ independent attention computations in parallel, each with its own learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

where each head is:

$$\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

with $W_Q^i \in \mathbb{R}^{d_{model} \times d_k}$, $W_K^i \in \mathbb{R}^{d_{model} \times d_k}$, $W_V^i \in \mathbb{R}^{d_{model} \times d_v}$, and $W_O \in \mathbb{R}^{hd_v \times d_{model}}$.

### The Dimension Arithmetic

This is a design choice that's easy to overlook but important:

- $d_{model} = 512$, $h = 8$ heads
- $d_k = d_v = d_{model} / h = 64$ per head

Total parameter count for multi-head attention with $d_k = 64, h = 8$ is **identical** to single-head attention with $d_k = 512$. The projections $W_Q, W_K, W_V$ are each $512 \times 512$ in both cases — multi-head attention simply partitions the dimensions across heads. The computational cost is the same. You get multiple attention patterns for free (in terms of FLOPs).

The output projection $W_O \in \mathbb{R}^{512 \times 512}$ is an additional parameter matrix that lets the model learn how to combine information from different heads. This is the only "extra" cost of multi-head vs single-head.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:sans-serif;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:20px; font-weight:bold;">Multi-Head Attention Architecture</div>
<div style="display:flex; flex-direction:column; gap:20px; align-items:center;">

<div style="display:flex; gap:8px; align-items:center;">
<div style="background:#0f3460; padding:10px 20px; border-radius:8px; color:#e94560; font-weight:bold; font-size:13px;">Input X</div>
<div style="color:#888; font-size:11px;">(n x 512)</div>
</div>

<div style="color:#e94560; font-size:16px;">|</div>
<div style="color:#888; font-size:11px;">Linear projections (W_Q, W_K, W_V per head)</div>
<div style="color:#e94560; font-size:16px;">|</div>

<div style="display:flex; gap:8px; flex-wrap:wrap; justify-content:center;">
<div style="background:#16213e; border:1px solid #e94560; padding:12px 14px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">Head 1</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Q,K,V</div>
<div style="color:#888; font-size:10px;">64-dim</div>
<div style="color:#aaa; font-size:10px; margin-top:4px;">syntactic?</div>
</div>
<div style="background:#16213e; border:1px solid #e94560; padding:12px 14px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">Head 2</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Q,K,V</div>
<div style="color:#888; font-size:10px;">64-dim</div>
<div style="color:#aaa; font-size:10px; margin-top:4px;">semantic?</div>
</div>
<div style="background:#16213e; border:1px solid #e94560; padding:12px 14px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">Head 3</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Q,K,V</div>
<div style="color:#888; font-size:10px;">64-dim</div>
<div style="color:#aaa; font-size:10px; margin-top:4px;">positional?</div>
</div>
<div style="background:#16213e; border:1px solid #333; padding:12px 14px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#888; font-weight:bold; font-size:12px;">...</div>
<div style="color:#666; font-size:10px; margin-top:4px;">x 8</div>
</div>
<div style="background:#16213e; border:1px solid #e94560; padding:12px 14px; border-radius:8px; text-align:center; min-width:80px;">
<div style="color:#e94560; font-weight:bold; font-size:12px;">Head 8</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Q,K,V</div>
<div style="color:#888; font-size:10px;">64-dim</div>
<div style="color:#aaa; font-size:10px; margin-top:4px;">coref?</div>
</div>
</div>

<div style="color:#e94560; font-size:16px;">|</div>
<div style="color:#888; font-size:11px;">Concatenate all heads (8 x 64 = 512), then linear projection W_O</div>
<div style="color:#e94560; font-size:16px;">|</div>

<div style="display:flex; gap:8px; align-items:center;">
<div style="background:#0f3460; padding:10px 20px; border-radius:8px; color:#e94560; font-weight:bold; font-size:13px;">Output</div>
<div style="color:#888; font-size:11px;">(n x 512)</div>
</div>

</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center; border-top:1px solid #333; padding-top:12px;">
8 heads x 64 dims = 512 dims. Same total parameters as single-head attention with d_k=512.<br/>
Head labels ("syntactic", "semantic", etc.) are illustrative — heads learn specialized roles through training, not by design.
</div>
</div>

### What Different Heads Actually Learn

Vaswani et al.'s Table 2 ablation study on head count is revealing:

| Heads ($h$) | $d_k$ | Parameters | EN-DE BLEU |
|---|---|---|---|
| 1 | 512 | same | 25.8 |
| 4 | 128 | same | 26.3 |
| 8 | 64 | same | 25.8 (base) |
| 16 | 32 | same | 25.7 |
| 32 | 16 | same | 24.7 |

The sweet spot is around 4-16 heads. Too few heads limit the diversity of attention patterns. Too many heads (32) with very small $d_k = 16$ degrade quality — each head has too few dimensions to form meaningful representations.

Subsequent analysis of trained Transformers (Clark et al. 2019, Voita et al. 2019) found that heads specialize:

- **Positional heads** attend to the previous or next token (local context)
- **Syntactic heads** track subject-verb agreement, dependency arcs
- **Rare-word heads** attend to infrequent tokens that carry high information
- **Separator/delimiter heads** attend to punctuation and structural markers

### Head Redundancy

A critical finding: **many heads are redundant.** Voita et al. (2019) showed that in a 6-layer, 8-head Transformer, only a small subset of heads are truly important. Pruning 60-80% of heads at inference time (by setting their outputs to zero) causes minimal performance degradation on many tasks.

This redundancy has architectural implications explored in [[ch-07]]: Multi-Query Attention (MQA) shares keys and values across all heads, reducing KV-cache memory by $h\times$ with modest quality loss. Grouped-Query Attention (GQA) is the compromise — sharing within groups of heads. DeepSeek's Multi-head Latent Attention (MLA) goes further, compressing KV representations into a low-rank latent space. All of these exploit the insight that heads are partially redundant.

---

## 7. Self-Attention vs. Cross-Attention

The original Transformer uses attention in three distinct configurations:

1. **Encoder self-attention:** Queries, keys, and values all come from the encoder's own representations. Every encoder position attends to every other encoder position. This builds rich, context-aware representations of the input.

2. **Decoder self-attention (masked):** Same as encoder self-attention, but with a causal mask that prevents position $i$ from attending to positions $j > i$. This enforces the autoregressive property from [[ch-01]] — each token can only see its predecessors.

3. **Encoder-decoder cross-attention:** Queries come from the decoder; keys and values come from the encoder. This is the direct descendant of Bahdanau attention — it lets the decoder focus on relevant parts of the encoded input at each generation step.

For decoder-only LLMs ([[ch-04]]), only masked self-attention survives. The encoder and cross-attention are eliminated entirely. Understanding this simplification is key to understanding the modern LLM architecture.

The causal mask is implemented by setting the upper triangle of the score matrix to $-\infty$ before softmax:

$$S_{ij} = \begin{cases} q_i \cdot k_j / \sqrt{d_k} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax, the $-\infty$ entries become 0 — position $i$ assigns zero weight to all future positions. This is what makes teacher-forced parallel training compatible with autoregressive modeling: the causal mask ensures that the prediction at position $t$ depends only on positions $1, \ldots, t$, even though all positions are computed simultaneously.

---

## Core Insights from the Literature

### Insight 1: Attention Was Motivated by a Failure of Compression, Not by a Theory of Computation
**Paper:** Sutskever et al. (2014) ([[seq2seq|paper]]), Bahdanau et al. (2014) ([[bahdanau-attention|paper]])

The fixed-length bottleneck was an engineering limitation, not a deliberate design choice. Sutskever's seq2seq paper *reversed input sentences* as a workaround — a hack that happened to shorten the effective dependency distance. Bahdanau's attention mechanism was the principled fix: let the decoder dynamically access all encoder states instead of relying on a single compressed vector. The lesson is that attention was not invented from first principles as "the right way to route information" — it was discovered as the solution to a specific failure mode. Many of the best architectural innovations in deep learning follow this pattern: observe a failure, trace it to a structural bottleneck, add a mechanism to bypass the bottleneck. **Guideline:** When an architecture underperforms, look for information bottlenecks first. The fix is usually a more flexible routing mechanism, not more parameters.

### Insight 2: The Scaling Factor Is Not Optional — It's What Makes Dot-Product Attention Work
**Paper:** Vaswani et al. (2017) ([[attention-is-all-you-need|paper]]), Table 2 ablations

Vaswani et al. explicitly tested unscaled dot-product attention and found it underperformed additive attention (Bahdanau's formulation) at large $d_k$. The $1/\sqrt{d_k}$ scaling is what made the simpler dot-product formulation competitive — without it, softmax saturates and attention degenerates into hard (nearly one-hot) selection. This means the scaling factor isn't just a normalization convenience; it's a *necessary condition* for the attention mechanism to function as a soft, differentiable selection. **Guideline:** Whenever you introduce a dot-product similarity in a network (attention, retrieval, contrastive learning), check the variance of the logits. If they grow with dimension, you need to scale — otherwise gradients will vanish through the softmax.

### Insight 3: Multi-Head Attention Provides Diversity for Free
**Paper:** Vaswani et al. (2017), analysis by Voita et al. (2019)

Multi-head attention with $h$ heads of dimension $d_k = d_{model}/h$ has the same parameter count and FLOPs as single-head attention with dimension $d_{model}$. The heads are a "free" reparameterization that enables the model to maintain multiple simultaneous attention patterns. However, most heads turn out to be redundant — you can prune 60-80% of them post-training without major quality loss. This redundancy is not waste; it's a form of ensembling that provides training stability and robustness. But it does mean that at inference time, there is substantial room to compress. **Guideline:** Design architectures with head redundancy during training (it helps optimization), but plan for head compression at inference (MQA, GQA, MLA). Training and serving have different optimal configurations.

### Insight 4: Attention Weights Are Interpretable — But Only Partially
**Paper:** Bahdanau et al. (2014) ([[bahdanau-attention|paper]]), Clark et al. (2019)

Bahdanau's attention weights cleanly corresponded to linguistic alignment. This interpretability was a major selling point. But in deep multi-head self-attention, the picture is muddier: attention weights show what positions are correlated, not what information flows through the residual stream. A head might attend strongly to position $j$ but the value vector from $j$ could project to near-zero in the output. Attention maps are necessary but not sufficient for understanding information flow. **Guideline:** Use attention weights for initial hypothesis generation about model behavior, but verify with probing classifiers or activation patching. Don't treat attention maps as ground truth for what the model "knows" or "uses."

---

## Key Takeaways

1. **Attention was born from a specific failure.** The fixed-length bottleneck of seq2seq models caused performance to collapse on long sentences. Attention is the dynamic routing mechanism that solved it.

2. **Scaled dot-product attention is a soft dictionary lookup.** $Q$ queries, $K$ keys, $V$ values — the dot product measures relevance, softmax normalizes, and the output is a weighted value blend. The $1/\sqrt{d_k}$ scaling is essential, not cosmetic.

3. **The cost is quadratic.** $O(n^2 d)$ time and $O(n^2)$ memory in sequence length. This is manageable for $n < d_{model}$ but becomes the dominant bottleneck for long contexts — the motivation for Flash Attention, sparse attention, and SSM alternatives.

4. **Multi-head attention is a free reparameterization** that enables diverse attention patterns at no additional FLOPs. But most heads are redundant, creating opportunity for inference-time compression (MQA, GQA, MLA in [[ch-07]]).

5. **Self-attention, masked self-attention, and cross-attention are three configurations of the same mechanism.** Decoder-only LLMs use only masked self-attention — the simplest form.

6. **The quadratic cost forces a three-way tradeoff** between context length, compute cost, and approximation quality. Every modern architecture takes a different position on this tradeoff — understanding it here is prerequisite for Chapters 7, 15, 16, and 19-23.

---

## References

- Sutskever, Vinyals, Le. "Sequence to Sequence Learning with Neural Networks" (2014). [[seq2seq|paper]]
- Bahdanau, Cho, Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate" (2014). [[bahdanau-attention|paper]]
- Vaswani et al. "Attention Is All You Need" (2017). [[attention-is-all-you-need|paper]]
- Alammar, Jay. "The Illustrated Transformer" (2018). [[alammar-illustrated-transformer|blog]]
- Clark et al. "What Does BERT Look At? An Analysis of BERT's Attention" (2019).
- Voita et al. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting" (2019).
- Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014).
