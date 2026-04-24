# Chapter 22: Case Study — Mamba-2

<!-- scope: Mamba-2 / SSD — pure SSM architecture, structured state space duality, hardware-aware implementation
     deps: [[ch-15]]
     see-also: [[ch-21]]
-->

## Overview

Mamba-1 ([[mamba|paper]]) proved that selective state space models could match Transformers on language modeling at the 3B scale, with 5x inference throughput and linear-time sequence processing. But it left two open questions. First, *why* do SSMs work so well — is there a deeper mathematical relationship between selective SSMs and attention, or are they fundamentally different computational primitives? Second, can the implementation get faster? Mamba-1's hardware-aware parallel scan was clever but left significant performance on the table compared to the matmul-heavy kernels that GPUs are optimized for.

Mamba-2 ([[mamba-2|paper]], Dao & Gu, 2024) answers both questions. The theoretical contribution is **Structured State Space Duality (SSD)**: a proof that SSMs and a specific class of attention (linear attention with causal masking) compute exactly the same mathematical object — a structured semiseparable matrix. The practical contribution is a new architecture that exploits this duality to run 2-8x faster than Mamba-1, by reformulating the selective SSM as a structured matrix multiplication that maps directly onto GPU tensor cores.

This chapter dissects the SSD framework, walks through the architecture changes from Mamba-1 to Mamba-2, examines Codestral Mamba as industrial validation, and confronts the limitations that still prevent pure SSMs from replacing Transformers everywhere.

---

## 1. The Theoretical Foundation: Structured State Space Duality

### SSMs and Attention Compute the Same Matrix

The core claim of the SSD paper is striking: under specific constraints, the output of a selective SSM and the output of linear attention are identical. Both compute a matrix-vector product $y = Mx$ where $M$ is a **semiseparable matrix**.

A matrix $M$ is semiseparable (of rank $N$) if every submatrix contained entirely within its lower-triangular part has rank at most $N$. This is precisely the structure that falls out of both computations:

**From the SSM side**, the discrete-time state space recurrence

$$h_t = A_t h_{t-1} + B_t x_t, \qquad y_t = C_t h_t$$

produces an input-output mapping that can be written as $y = Mx$ where each entry $M_{ij}$ (for $i \geq j$) equals:

$$M_{ij} = C_i^\top \left(\prod_{k=j+1}^{i} A_k\right) B_j$$

This is a product of three low-rank factors — $C_i^\top$ (a row vector), a product of state transition matrices, and $B_j$ (a column vector). The rank of any lower-triangular submatrix of $M$ is bounded by the state dimension $N$.

**From the attention side**, linear attention (attention without the softmax nonlinearity) with causal masking computes:

$$y_i = \sum_{j \leq i} \left(\frac{q_i^\top k_j}{\text{normalization}}\right) v_j$$

which can also be written as $y = Mx$ where $M_{ij} = q_i^\top k_j$ for $i \geq j$. This is a rank-$d$ semiseparable matrix, where $d$ is the head dimension.

The duality: **the SSM's state dimension $N$ plays the same role as linear attention's head dimension $d$**. Both control the rank of the semiseparable matrix $M$. An SSM with state dimension $N = 64$ computes the same class of input-output mappings as linear attention with head dimension $d = 64$.

### What Semiseparable Means Computationally

The semiseparable structure is not just a theoretical curiosity — it determines which algorithms are available for computing $y = Mx$:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Two Dual Computation Paths for the Same Matrix</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Path</th>
<th style="text-align:left; padding:8px;">Algorithm</th>
<th style="text-align:right; padding:8px;">Time</th>
<th style="text-align:right; padding:8px;">Memory</th>
<th style="text-align:left; padding:8px;">Best When</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Recurrence (SSM)</td>
<td style="padding:8px;">Sequential scan</td>
<td style="text-align:right; padding:8px;">O(TN)</td>
<td style="text-align:right; padding:8px;">O(N)</td>
<td style="padding:8px;">Long sequences, autoregressive generation</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Quadratic (Attention)</td>
<td style="padding:8px;">Matrix multiply</td>
<td style="text-align:right; padding:8px;">O(T²N)</td>
<td style="text-align:right; padding:8px;">O(T²)</td>
<td style="padding:8px;">Short sequences, prefill, training</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
T = sequence length, N = state dimension. The crossover point depends on hardware: on modern GPUs with fast matmul units, the quadratic path can be faster for T up to several thousand.
</div>
</div>

This is the actionable insight: **you do not have to commit to one algorithm**. For the prefill phase (processing the entire prompt), the quadratic matmul-based path is faster because it maps onto tensor cores. For autoregressive generation (one token at a time), the linear recurrence is faster because it avoids materializing any $T \times T$ matrix. Mamba-2 can switch between them because both compute the same result.

### What SSD Does *Not* Claim

The duality holds for **linear** attention — attention without the softmax nonlinearity. Standard softmax attention produces a *dense* lower-triangular matrix (every entry is nonzero and the matrix has full rank), which is not semiseparable. The softmax is what gives Transformers their ability to form sharp, sparse attention patterns — attending strongly to a few positions while ignoring everything else.

This distinction matters. When people say "Transformers are SSMs," the precise statement is: *Transformers with linear attention are SSMs*. Standard softmax Transformers are strictly more expressive for any finite state dimension $N$, because they can represent full-rank attention matrices that no fixed-$N$ SSM can capture. The limitations discussed in Section 6 trace directly to this gap.

---

## 2. From Mamba-1 to Mamba-2: The Architecture Changes

### The Key Simplification: Scalar State Transitions

Mamba-1 used a diagonal state transition matrix $A = \text{diag}(a_1, \ldots, a_N)$, where each of the $N$ state dimensions had its own decay rate. Mamba-2 constrains this further to a **scalar-times-identity** structure:

$$A_t = \alpha_t \cdot I_N$$

where $\alpha_t$ is a single scalar (derived from the input) and $I_N$ is the $N \times N$ identity matrix. Every state dimension decays at the same rate at each timestep.

This looks like a significant expressivity loss. Why would you want this? Because the scalar structure makes the semiseparable matrix $M$ **structured** — its entries factor in a way that enables decomposition into block matrix multiplications. The product of state transitions between positions $j$ and $i$ simplifies to:

$$\prod_{k=j+1}^{i} A_k = \left(\prod_{k=j+1}^{i} \alpha_k\right) \cdot I_N$$

This scalar product can be precomputed as a cumulative product, and the entire matrix $M$ can be expressed as:

$$M_{ij} = C_i^\top B_j \cdot \prod_{k=j+1}^{i} \alpha_k$$

The $C_i^\top B_j$ term is a rank-$N$ outer product (analogous to the $q_i^\top k_j$ term in linear attention), and the cumulative product of $\alpha_k$ values acts as a position-dependent decay mask (analogous to causal masking). This factored form maps directly onto GPU matrix multiplication hardware.

### Multi-Head SSM

Mamba-2 introduces a **multi-head** structure analogous to multi-head attention. Instead of a single SSM processing the full hidden dimension, the model uses $H$ heads, each with its own state dimension $N$ and its own $B$, $C$ projections:

$$\text{head}_h: \quad h_t^{(h)} = \alpha_t^{(h)} \cdot h_{t-1}^{(h)} + B_t^{(h)} x_t^{(h)}, \quad y_t^{(h)} = C_t^{(h)\top} h_t^{(h)}$$

Each head operates on a slice of the input dimension $d/H$ and maintains a state of dimension $N$. The outputs are concatenated, paralleling MHA. The number of heads is a tunable hyperparameter that trades off between per-head expressivity (larger $N$ per head) and diversity of state dynamics (more heads).

### The Mamba-2 Block

The overall block structure is similar to Mamba-1 but simplified:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Mamba-2 Block Architecture</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
<div style="background:#16213e; padding:10px 60px; border-radius:8px; color:#e0e0e0; font-size:13px; font-weight:bold;">Input x (d_model)</div>
<div style="color:#e94560; font-size:16px;">&darr;</div>
<div style="display:flex; gap:30px;">
<div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
<div style="background:#0f3460; padding:8px 16px; border-radius:6px; color:#4ecdc4; font-size:12px; font-weight:bold;">Linear &uarr; (expand)</div>
<div style="color:#4ecdc4; font-size:14px;">&darr;</div>
<div style="background:#0f3460; padding:8px 16px; border-radius:6px; color:#4ecdc4; font-size:12px; font-weight:bold;">1D Conv</div>
<div style="color:#4ecdc4; font-size:14px;">&darr;</div>
<div style="background:#e94560; padding:10px 16px; border-radius:6px; color:#fff; font-size:12px; font-weight:bold;">SSD Layer (multi-head)</div>
<div style="color:#4ecdc4; font-size:14px;">&darr;</div>
<div style="background:#0f3460; padding:8px 16px; border-radius:6px; color:#4ecdc4; font-size:12px; font-weight:bold;">Norm</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
<div style="background:#0f3460; padding:8px 16px; border-radius:6px; color:#ffd93d; font-size:12px; font-weight:bold;">Linear &uarr; (gate)</div>
<div style="color:#ffd93d; font-size:14px;">&darr;</div>
<div style="background:#0f3460; padding:8px 16px; border-radius:6px; color:#ffd93d; font-size:12px; font-weight:bold;">SiLU</div>
<div style="color:#ffd93d; font-size:60px; line-height:0.7;">&vellip;</div>
</div>
</div>
<div style="color:#ffd93d; font-size:13px; font-weight:bold;">&otimes; element-wise multiply</div>
<div style="color:#e94560; font-size:16px;">&darr;</div>
<div style="background:#0f3460; padding:8px 16px; border-radius:6px; color:#4ecdc4; font-size:12px; font-weight:bold;">Linear &darr; (project)</div>
<div style="color:#e94560; font-size:16px;">&darr;</div>
<div style="background:#16213e; padding:10px 60px; border-radius:8px; color:#e0e0e0; font-size:13px; font-weight:bold;">Output (d_model)</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
No separate MLP block. The gated structure combines SSM processing with nonlinear gating in a single block.<br>
The SSD layer is the multi-head selective SSM with scalar state transitions.
</div>
</div>

Key differences from Mamba-1:

1. **SSD replaces the selective SSM** — the scalar-identity $A$ constraint enables the structured matrix computation
2. **Multi-head structure** — analogous to multi-head attention, with tunable head count
3. **Normalization after SSD** — a normalization layer between the SSM output and the gating, improving training stability
4. **No separate MLP** — like Mamba-1, the gated architecture subsumes the FFN's role. Each Mamba-2 block is simultaneously a "sequence mixing" and "channel mixing" operation

### Chunk-Wise Computation: The Best of Both Worlds

The practical implementation uses a **hybrid chunk-wise algorithm** that combines both dual computation paths:

1. Divide the sequence into chunks of size $C$ (typically 64-256 tokens)
2. **Within each chunk**: use the quadratic (matmul) form — compute the $C \times C$ attention-like matrix and multiply. This is efficient because $C$ is small and matmul is fast on GPU tensor cores
3. **Across chunks**: use the linear (recurrence) form — propagate the SSM state from one chunk to the next. This avoids materializing any cross-chunk attention matrix

The total cost is $O(T \cdot C \cdot N)$ for within-chunk computation plus $O(T \cdot N)$ for cross-chunk state propagation. Since $C$ and $N$ are constants, the overall complexity remains $O(T)$ — linear in sequence length. But the constant factor is dramatically smaller than Mamba-1's parallel scan because the within-chunk computation maps onto highly optimized matmul kernels.

This is where the SSD duality pays off practically. You are not choosing between "SSM mode" and "attention mode" — you are using *both simultaneously* at different granularities, exploiting the mathematical equivalence to pick whichever is faster at each scale.

[Interactive visualization: SSD duality diagram](figures/ssd-duality.html)

---

## 3. Hardware-Aware Implementation: Why 2-8x Faster

### The GPU Matmul Advantage

Modern GPUs (A100, H100) have specialized tensor core units designed for matrix multiplication. These units achieve peak throughput only when executing dense matmul operations with specific tile sizes and data layouts. Mamba-1's parallel scan algorithm, while theoretically $O(T)$, used a custom kernel that did not fully exploit tensor cores — it was fundamentally a *scan* operation, not a *matmul* operation.

Mamba-2's SSD reformulation converts the core computation into a structured matrix multiplication:

$$Y_{\text{chunk}} = (L \odot (CB^\top)) \cdot X_{\text{chunk}}$$

where $L$ is a lower-triangular mask derived from the cumulative products of $\alpha_t$ values, $C$ and $B$ are the output and input projections, and $\odot$ denotes element-wise multiplication. The parenthesized term $(L \odot (CB^\top))$ is the within-chunk "attention matrix" — a $C \times C$ matrix that can be computed via standard matmul and then applied to the input chunk via another matmul.

This is precisely the kind of operation that tensor cores excel at. The result: **2-8x wall-clock speedup over Mamba-1** at equivalent model quality.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Mamba-2 vs Mamba-1: Speed Comparison</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Configuration</th>
<th style="text-align:right; padding:8px;">Mamba-1</th>
<th style="text-align:right; padding:8px;">Mamba-2</th>
<th style="text-align:right; padding:8px;">Speedup</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Seq 2K, state N=16</td>
<td style="text-align:right; padding:8px;">1.0x</td>
<td style="text-align:right; padding:8px;">2.0-3.0x</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">2-3x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Seq 2K, state N=64</td>
<td style="text-align:right; padding:8px;">1.0x</td>
<td style="text-align:right; padding:8px;">4.0-6.0x</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">4-6x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Seq 8K, state N=64</td>
<td style="text-align:right; padding:8px;">1.0x</td>
<td style="text-align:right; padding:8px;">6.0-8.0x</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">6-8x</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Speedup scales with state dimension N because larger N benefits more from matmul reformulation.<br>
Source: Dao & Gu, "Transformers are SSMs" (2024), Table 1.
</div>
</div>

### Why the Speedup Scales with State Dimension

Mamba-1's parallel scan has complexity proportional to $N^2$ per step (the state transition involves $N \times N$ diagonal matrix operations). The SSD reformulation converts these into matmul operations of size $C \times C$ with $N$-dimensional inner products — and matmul efficiency *improves* as the inner dimension grows (up to tensor core tile sizes). Larger $N$ means more arithmetic intensity per byte loaded from memory, pushing the operation further into the compute-bound regime where GPUs excel.

This has an important architectural implication: **Mamba-2 can afford larger state dimensions than Mamba-1**. Mamba-1 typically used $N = 16$ because larger states were too slow. Mamba-2 can practically use $N = 64$ or $N = 128$, giving the model more memory capacity per layer without sacrificing speed. More state capacity means the fixed-size recurrent state can represent richer summaries of past context.

[Interactive visualization: O(n) vs O(n^2) scaling comparison](figures/scaling-comparison.html)

---

## 4. The Pure SSM Bet: Zero Attention Layers

### Why This Is Radical

Every competitive LLM architecture since GPT-1 has included attention layers. Even hybrid architectures like Jamba ([[ch-21]]) use attention in some layers — the design question has been *how much* attention, not *whether* to include it. Mamba-2 is a bet that you can build a competitive language model with **zero attention layers**.

The architectural implications are fundamental:

**No KV cache.** Transformers cache key and value tensors for every past position, creating a memory cost that grows linearly with sequence length. Mamba-2 maintains a fixed-size state $h_t \in \mathbb{R}^{H \times N}$ regardless of how many tokens have been processed. For a model with $H = 64$ heads and $N = 64$ state dimension, the total state is $64 \times 64 = 4096$ floating-point values per layer. Compare to a GQA Transformer with 8 KV heads and $d_k = 128$: at sequence length 4K, the KV cache is $2 \times 8 \times 128 \times 4096 = 8M$ values per layer — roughly 2000x more memory.

**Constant-time generation.** Each decoding step in Mamba-2 costs exactly the same regardless of context length. In a Transformer, each step requires attending to all cached positions, so the cost per step grows linearly with context length (and the total cost of generating $T$ tokens from a prompt of length $P$ is $O(T \cdot P)$ for attention alone). For Mamba-2, the total cost is $O(T)$ — independent of prompt length after the initial prefill.

**Linear-time prefill.** Processing the initial prompt takes $O(T)$ time rather than $O(T^2)$ for standard attention (or $O(T^2 d^2/M)$ for Flash Attention, which is still quadratic in $T$). For very long prompts (100K+ tokens), this is where the wall-clock savings become most dramatic.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Memory During Autoregressive Generation</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Architecture</th>
<th style="text-align:right; padding:8px;">State at 1K ctx</th>
<th style="text-align:right; padding:8px;">State at 32K ctx</th>
<th style="text-align:right; padding:8px;">State at 256K ctx</th>
<th style="text-align:left; padding:8px;">Growth</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Transformer (GQA-8)</td>
<td style="text-align:right; padding:8px;">256 KB/layer</td>
<td style="text-align:right; padding:8px;">8 MB/layer</td>
<td style="text-align:right; padding:8px;">64 MB/layer</td>
<td style="padding:8px; color:#e94560;">O(T) per layer</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Mamba-2 (H=64, N=64)</td>
<td style="text-align:right; padding:8px;">16 KB/layer</td>
<td style="text-align:right; padding:8px;">16 KB/layer</td>
<td style="text-align:right; padding:8px;">16 KB/layer</td>
<td style="padding:8px; color:#4ecdc4;">O(1) per layer</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Transformer: 2 x 8 KV heads x 128 dim x seq_len x 2 bytes. Mamba-2: 64 heads x 64 state x 2 bytes = fixed 8 KB (FP16).<br>
At 256K context, Transformer KV cache is ~4000x larger than Mamba-2's state.
</div>
</div>

### The Tradeoff: Fixed State vs Unbounded Cache

The constant-memory property is both Mamba-2's greatest strength and its fundamental limitation. A Transformer's KV cache grows with sequence length because it stores *exact* representations of every past token. Mamba-2's fixed-size state must *compress* all past information into $H \times N$ values — it is a lossy summary.

For a Transformer, retrieving the exact value of the 500th token from a 100K-token context is trivial: look it up in the KV cache. For Mamba-2, that information must have survived compression through thousands of state updates. Whether it did depends on whether the model learned to preserve it — and a fixed-size state cannot preserve arbitrarily many distinct facts.

This is the fundamental information-theoretic argument: a Transformer's effective memory capacity scales with $T$ (sequence length), while an SSM's scales with $N$ (state dimension). For $N \ll T$, the SSM must discard information. The question is whether the *selectivity mechanism* (input-dependent $B$, $C$, $\alpha$) is smart enough to keep what matters and discard what doesn't.

---

## 5. Codestral Mamba: Industrial Validation

### Mistral's Bet on Pure SSM

In mid-2024, Mistral AI released **Codestral Mamba** — a 7B-parameter code generation model built entirely on the Mamba-2 architecture. This was the first deployment of a pure SSM model by a major AI lab for a practical, commercial use case.

The choice of code generation was deliberate. Code has several properties that play to SSM strengths:

1. **Long contexts are common.** Repository-level code understanding requires processing thousands of lines. Linear-time scaling makes this practical.
2. **Local structure dominates.** Code has strong local dependencies (variable scoping, block structure, adjacent-line relationships) that SSM's inherently local processing handles well.
3. **Retrieval demands are structured.** Code retrieval (finding a function definition, a variable declaration) tends to follow syntactic patterns that a trained SSM can learn to track in its state, unlike arbitrary natural-language retrieval.

### Codestral Mamba's Results

Codestral Mamba demonstrated competitive performance with Transformer-based code models of similar size:

- Matched or approached CodeLlama 7B on standard code benchmarks (HumanEval, MBPP)
- Excelled on long-context code understanding tasks where the linear-time scaling provided tangible speed advantages
- Supported a 256K token context window — processing entire repositories in a single pass
- Inference throughput significantly exceeded Transformer-based alternatives at long context lengths

The significance is not that Codestral Mamba was the best code model at 7B — Transformer-based alternatives with more training data and RLHF tuning scored higher on some benchmarks. The significance is that a pure SSM architecture was *competitive at all* on a practical, commercially deployed task. It proved that "zero attention" was not a death sentence for real-world utility.

### What Code Generation Teaches About SSM Fitness

The fact that Mistral chose code (not general chat, not reasoning, not retrieval-augmented generation) reveals where they believed the architecture's strengths aligned:

- **Code is more predictable** than open-domain text. The entropy of code token sequences is lower, which means the fixed-size state has an easier compression job.
- **Code has explicit scope boundaries** (functions, classes, blocks) that naturally signal when the SSM should reset or update its state — the selectivity mechanism can learn these patterns.
- **Code quality is measurable** via execution tests, providing clean signal for model evaluation without the subjectivity of natural-language quality assessment.

---

## 6. Limitations: Where Pure SSMs Fall Short

### In-Context Learning Weakness

Transformers excel at in-context learning (ICL) — the ability to learn new patterns from examples provided in the prompt. Given a few (input, output) pairs, a Transformer can often infer the pattern and apply it to new inputs. This capability relies on the attention mechanism's ability to form precise *cross-position comparisons*: the model attends from the current query to specific past examples, retrieves their structure, and applies it.

SSMs process the sequence through a fixed-size state that summarizes everything seen so far. For ICL to work, the state must encode not just *what* examples were seen but the *relational structure* between inputs and outputs across multiple examples — a much harder compression problem. Empirically, pure SSMs show weaker ICL performance than Transformers, especially when:

- The number of in-context examples is large (more information to compress)
- The mapping from inputs to outputs is complex (requiring precise retrieval of example structure)
- The task is truly novel (not a pattern the model encountered during training)

### Precise Retrieval Failures

The "needle-in-a-haystack" test — finding a specific piece of information embedded in a long context — exposes a fundamental SSM weakness. A Transformer can attend directly to the position containing the target information. An SSM must have preserved that information in its state through all subsequent timesteps.

For short-to-medium contexts, trained SSMs perform reasonably on retrieval tasks because the state dimension is large enough to preserve the relevant information. But as context grows and more intervening information passes through the state, retrieval accuracy degrades — the state has finite capacity and must overwrite old information to accommodate new input.

This is not a training problem that better data or longer training can fix. It is a *capacity* limitation inherent to fixed-size recurrent state. The only remedies are:

1. Increase the state dimension $N$ (but this increases computation proportionally)
2. Use a hybrid architecture with some attention layers for precise retrieval ([[ch-21]])
3. Accept the limitation and target applications where approximate recall suffices

### The Copying Problem

A seemingly simple task — copying an input sequence verbatim — reveals another SSM limitation. Transformers can trivially copy by attending from each output position to the corresponding input position. For SSMs, copying requires encoding the entire input sequence in the fixed-size state and then decoding it sequentially. For sequences longer than the state dimension, lossless copying is information-theoretically impossible.

In practice, trained SSMs can copy short-to-medium sequences by learning efficient encodings, but they fail on sufficiently long sequences in ways that Transformers do not. This matters for practical tasks like code refactoring (copying existing code with modifications) and long-form summarization (preserving specific quotes from the source).

### The Expressivity Gap: Softmax vs Linear

The SSD framework shows that SSMs are equivalent to *linear* attention. The softmax nonlinearity in standard attention provides two capabilities that linear attention lacks:

1. **Sharpening.** Softmax can concentrate probability mass on a small number of positions, creating sparse, precise attention patterns. Linear attention produces inherently smoother, more diffuse patterns — it cannot "ignore everything except position 42."

2. **Normalization-dependent computation.** The softmax denominator creates dependencies between attention weights — increasing attention to one position necessarily decreases attention to others. This competitive dynamic enables winner-take-all computation that linear attention (and therefore SSMs) cannot replicate.

These are not limitations that scaling will overcome. They are structural properties of the computational primitive. A larger SSM state dimension $N$ increases the *rank* of the semiseparable matrix but does not make it dense or normalized like softmax attention.

---

## 7. Core Insights from the Literature

### Insight 1: SSMs and linear attention are mathematical duals, not competing paradigms
**Paper:** Dao & Gu, "Transformers are SSMs" ([[mamba-2|paper]])

The SSD framework proves that the distinction between SSMs and linear attention is one of *algorithm*, not *computation*. Both compute the same structured semiseparable matrix; they differ only in whether you evaluate that matrix via sequential recurrence or parallel matmul. This unification is more than aesthetic — it enables choosing the optimal algorithm for each hardware context (recurrence for generation, matmul for training/prefill), and it shows exactly what SSMs can and cannot express (anything a rank-$N$ semiseparable matrix can represent). **Guideline:** When evaluating SSM vs attention architectures, the real question is not "which paradigm is better" but "what rank of semiseparable matrix does the task require, and is softmax's sharpening necessary?"

### Insight 2: Hardware dictates architecture more than theory does
**Paper:** Dao & Gu, "Transformers are SSMs" ([[mamba-2|paper]])

Mamba-2's 2-8x speedup over Mamba-1 comes entirely from reformulating the same computation as structured matrix multiplications that map onto GPU tensor cores. The mathematical expressivity is *reduced* (scalar $A$ vs diagonal $A$), but the wall-clock speed is dramatically better because the computation matches the hardware's strengths. This is the same lesson as Flash Attention ([[ch-07]]): the bottleneck is not theoretical complexity but how well the algorithm maps onto real hardware. **Guideline:** When designing sequence models, optimize for tensor core utilization before optimizing for theoretical expressivity. An algorithm that is 3x faster on actual hardware can afford 3x more layers or 3x larger state, often recovering any expressivity loss.

### Insight 3: Fixed-size state is a feature *and* a fundamental limitation
**Paper:** Gu & Dao, "Mamba" ([[mamba|paper]])

The O(1) memory footprint during generation is what makes SSMs attractive for long-context deployment — no KV cache means no memory wall at 256K+ tokens. But the same fixed-size state is why SSMs struggle with precise retrieval and in-context learning: they cannot store arbitrary facts about arbitrary positions. This is an information-theoretic constraint, not an engineering limitation. Increasing the state dimension $N$ helps but cannot fully close the gap because the compression ratio ($T/N$) grows with sequence length. **Guideline:** Deploy pure SSMs when the task prioritizes throughput and long-context processing over precise information retrieval. For tasks requiring exact recall from arbitrary positions, use hybrid architectures or accept the retrieval accuracy tradeoff.

### Insight 4: Code generation is the ideal proving ground for pure SSMs
**Source:** Codestral Mamba (Mistral AI, 2024)

Mistral's choice to deploy a pure Mamba-2 model for code generation — not general chat, not reasoning — reveals where the architecture's strengths genuinely align. Code has lower entropy than natural language, has explicit scope boundaries that map onto the selectivity mechanism, and has measurable quality via execution tests. The success of Codestral Mamba at 256K context validates that pure SSMs are not merely theoretical curiosities but practical tools for the right problem class. **Guideline:** When evaluating whether an SSM architecture fits your use case, ask: does the task have strong local dependencies, relatively low entropy, and tolerance for approximate (rather than exact) long-range retrieval? If yes, pure SSM may outperform Transformers on throughput while matching on quality.

---

## Key Takeaways

1. **SSD unifies SSMs and linear attention.** Both compute structured semiseparable matrices. The SSM recurrence and the attention matmul are dual algorithms for the same computation, not competing paradigms. The state dimension $N$ in SSMs corresponds to the head dimension $d$ in linear attention.

2. **The scalar $A$ constraint trades expressivity for speed.** Mamba-2 constrains the state transition to $\alpha_t \cdot I$ (scalar-times-identity), losing Mamba-1's per-dimension decay rates but enabling 2-8x faster structured matrix multiplication on GPU tensor cores.

3. **Chunk-wise computation gets the best of both paths.** Within chunks: quadratic matmul (fast on tensor cores). Across chunks: linear recurrence (avoids materializing large matrices). Overall complexity remains $O(T)$.

4. **No KV cache means constant-memory generation.** Mamba-2's fixed-size state is O(1) per layer regardless of context length — roughly 4000x smaller than a GQA Transformer's KV cache at 256K tokens. This is the primary deployment advantage.

5. **Pure SSMs cannot replicate softmax attention's sharpening.** The equivalence is with *linear* attention, which cannot form sparse, winner-take-all patterns. This explains persistent weaknesses in in-context learning, precise retrieval, and sequence copying.

6. **Codestral Mamba validates pure SSM for code.** Code generation's local structure, low entropy, and explicit scope boundaries make it an ideal fit for SSM's strengths while avoiding its retrieval weaknesses.

7. **Hardware-algorithm co-design dominates theoretical elegance.** Mamba-2's speedup comes from matching the algorithm to tensor core matmul, not from theoretical complexity improvements. The same lesson applies across the entire sequence modeling stack: profile the hardware before optimizing the math.

---

## References

- [[mamba-2|Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024) (paper)]] — SSD framework and Mamba-2
- [[mamba|Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) (paper)]] — Mamba-1, selective SSMs
- [[flash-attention|Dao et al., "FlashAttention" (2022) (paper)]] — IO-aware exact attention, motivating hardware-algorithm co-design
- Codestral Mamba — Mistral AI (2024), pure Mamba-2 model for code generation, 7B parameters, 256K context
