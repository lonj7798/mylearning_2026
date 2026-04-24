# Chapter 15: State Space Models and Alternatives to Attention

<!-- scope: SSMs (S4, Mamba, Mamba-2), linear attention, hybrid architectures (Jamba) — escaping O(n^2)
     deps: [[ch-02]]
     see-also: [[ch-21]], [[ch-22]]
-->

## Overview

Every attention variant in [[ch-07]] accepts a premise: the core operation is softmax attention, and the job is to make it cheaper. This chapter asks a different question — what if you replace attention entirely?

The motivation is fundamental. Softmax attention computes pairwise interactions between all positions, producing an $N \times N$ matrix that costs $O(N^2)$ in both time and memory. Flash Attention ([[flash-attention|paper]]) reduces the memory to $O(N)$ by never materializing the full matrix, but the $O(N^2)$ FLOPs remain. For a 1M-token genomics sequence or a multi-document retrieval context, quadratic scaling is a wall that no IO optimization can remove.

State space models (SSMs) offer a fundamentally different computation: a fixed-size recurrent state that processes each token in $O(1)$ time, giving $O(N)$ total cost. The catch is that a fixed-size state compresses the entire history into a finite-dimensional vector — unlike the KV cache, which stores every past token explicitly. The history of SSMs from 2021 to 2024 is the story of making that compression smart enough to compete with attention on language.

This chapter traces three generations of that effort. S4 (2021) showed that structured state spaces with careful initialization could match Transformers on long-range benchmarks. Mamba (2023) made the state space parameters input-dependent, giving SSMs the content-based reasoning they lacked, and achieved the first SSM results competitive with Transformers on language modeling. Mamba-2 (2024) revealed that SSMs and linear attention are mathematical duals — two views of the same semiseparable matrix — enabling a 2-8x speedup through algorithmic flexibility. Finally, Jamba (2024) demonstrated that the best practical architecture may not be pure SSM or pure attention, but a hybrid that uses each where it excels.

The core tension throughout: **attention remembers everything but pays quadratic cost; SSMs pay linear cost but must learn what to forget.**

---

## 1. The Recurrence That Precedes Everything: Classical SSMs

Before Mamba, before S4, the state space model is a concept from control theory. A continuous-time linear dynamical system maps an input signal $x(t)$ to an output $y(t)$ through a hidden state $h(t)$:

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

where $A \in \mathbb{R}^{N \times N}$ is the state transition matrix, $B \in \mathbb{R}^{N \times 1}$ maps input to state, and $C \in \mathbb{R}^{1 \times N}$ maps state to output. The state dimension $N$ controls the system's memory capacity.

To use this in a neural network operating on discrete tokens, you discretize with a step size $\Delta$:

$$\bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

The discrete recurrence becomes:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

This is just a linear RNN. Each step takes $O(N)$ time (matrix-vector multiply with the $N \times N$ state matrix), and the total cost for a length-$L$ sequence is $O(LN)$ — linear in sequence length. There is no $L \times L$ attention matrix.

**The convolution view.** Because $\bar{A}$, $\bar{B}$, $C$ are constant (input-independent), you can unroll the recurrence:

$$y_t = C\bar{A}^t\bar{B}x_0 + C\bar{A}^{t-1}\bar{B}x_1 + \cdots + C\bar{B}x_t$$

This is a convolution with kernel $\bar{K}_t = C\bar{A}^t\bar{B}$. During training (when the full sequence is available), you can compute the entire output in $O(L \log L)$ time via FFT — faster than unrolling the recurrence step by step. During inference (autoregressive generation), you use the recurrence form: $O(1)$ per step, no growing cache.

This dual view — convolution for training, recurrence for inference — is the key computational advantage of classical SSMs over Transformers, which have no efficient recurrent form.

### Why Classical SSMs Failed at Language

The problem is the state transition matrix $A$. It is fixed — the same matrix governs how every token's information decays and mixes in the state, regardless of what that token is. This means a classical SSM cannot selectively remember important tokens and forget irrelevant ones. It applies the same exponential decay to the word "not" as to the word "the."

For continuous signals like audio waveforms, this is acceptable — the statistics are relatively stationary. For language, where a single token can invert meaning, it is fatal. Classical SSMs (including S4) consistently underperformed Transformers on language modeling despite strong results on synthetic benchmarks like the Long Range Arena.

---

## 2. S4: Structured State Spaces and HiPPO Initialization

The Structured State Space for Sequence Modeling (S4, Gu et al. 2021) made two contributions that turned SSMs from a theoretical curiosity into a practical architecture.

### HiPPO: The Right Initialization for A

The state transition matrix $A$ governs how information decays over time. Most random initializations cause either exponential blowup or rapid forgetting. HiPPO (High-order Polynomial Projection Operators) provides a principled initialization where the state $h_t$ maintains an optimal polynomial approximation of the input history.

The HiPPO-LegS matrix is:

$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

This specific matrix has the property that the $n$-th component of the state $h_t$ approximates the $n$-th Legendre polynomial coefficient of the input history up to time $t$. In other words, the state maintains a compressed representation of the entire input history, with resolution that degrades gracefully for older inputs — exactly the behavior you want for sequence modeling.

### Structured Computation

The second problem: even with a good $A$, the recurrence $h_t = \bar{A}h_{t-1} + \bar{B}x_t$ requires materializing the $N \times N$ matrix $\bar{A}$, which is expensive for large state dimensions. S4 constrains $A$ to be a **diagonal plus low-rank (DPLR)** matrix, enabling the convolution kernel $\bar{K}$ to be computed in $O(N + L)$ time via a Cauchy kernel formulation.

The practical result: S4 achieved the first state-of-the-art results on the Long Range Arena benchmark (including the Path-X task at sequence length 16K, which no Transformer had solved). But on language modeling, S4 still lagged behind Transformers — because the core limitation remained: **fixed parameters cannot do content-based reasoning**.

---

## 3. Mamba: Selective State Spaces ([[mamba|paper]])

Gu and Dao (2023) identified the exact bottleneck: prior SSMs treat every input token identically because $A$, $B$, $C$, and $\Delta$ are all input-independent. The fix is conceptually simple — make these parameters functions of the input.

### The Selection Mechanism

In Mamba, the parameters $B$, $C$, and $\Delta$ become input-dependent:

$$B_t = s_B(x_t), \qquad C_t = s_C(x_t), \qquad \Delta_t = \text{softplus}(s_\Delta(x_t))$$

where $s_B$, $s_C$, $s_\Delta$ are learned linear projections. The discretized recurrence becomes:

$$\bar{A}_t = \exp(\Delta_t A), \qquad \bar{B}_t = \Delta_t B_t$$
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
$$y_t = C_t h_t$$

The discretization step $\Delta_t$ is the critical control knob. When $\Delta_t$ is large, $\bar{A}_t \approx 0$ and $\bar{B}_t$ is large — the model **resets** its state and writes the current input strongly. When $\Delta_t$ is small, $\bar{A}_t \approx I$ and $\bar{B}_t \approx 0$ — the model **preserves** its existing state and largely ignores the current input. This gives the SSM the ability to selectively remember or forget based on content.

See [SSM State Evolution Animation](figures/ssm-state-evolution.html) for a step-by-step visualization of how $\Delta_t$ controls selective memory.

### The Price of Selection: Losing the Convolution View

Input-dependent parameters break the convolution trick. When $\bar{A}_t$ varies per timestep, you cannot precompute a single convolution kernel — each step has its own transition matrix. The $O(L \log L)$ FFT-based training is gone.

This is a serious problem. The recurrence is inherently sequential: $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, and so on. Naively, you must process tokens one at a time — terrible for GPU utilization during training.

### Hardware-Aware Parallel Scan

Mamba's second contribution is an efficient GPU algorithm for computing the selective recurrence. The key insight, directly inspired by Flash Attention's IO-aware approach ([[flash-attention|paper]]):

1. **Parallel scan.** The recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ has the algebraic structure of a prefix sum with an associative binary operator. This means it can be parallelized using a scan algorithm in $O(\log L)$ parallel steps.

2. **SRAM residency.** The expanded state $h_t \in \mathbb{R}^{B \times L \times D \times N}$ (batch x length x model dim x state dim) is too large to materialize in HBM. Mamba's fused kernel keeps the state in GPU SRAM, computing the scan without round-tripping through HBM — exactly the same IO-awareness principle behind Flash Attention.

3. **Kernel fusion.** The discretization, scan, and output multiplication are fused into a single CUDA kernel, avoiding intermediate memory allocation.

The result: Mamba's selective scan runs at comparable speed to the optimized convolution of S4, despite the theoretically harder computation.

### The Mamba Block

The full Mamba block eliminates both attention and the separate MLP (feed-forward) block found in Transformers:

```
Input x
  |
  +---> Linear projection (expand D -> E*D)
  |         |
  |         v
  |     1D Convolution (kernel size 4)
  |         |
  |         v
  |     SiLU activation
  |         |
  |         v
  |     Selective SSM (the core scan)
  |         |
  +---> Linear projection (D -> E*D) --> SiLU
  |         |
  +-------- * (element-wise gate)
            |
            v
        Linear projection (E*D -> D)
            |
            v
        Output
```

The gating branch (right path) acts as a multiplicative gate, similar to the gating in SwiGLU feed-forward networks. The 1D convolution provides local context that helps the SSM's selection mechanism. There is no normalization within the block beyond the outer residual stream's RMSNorm.

### Results: The First SSM Competitive on Language

Mamba-3B outperformed Transformers of the same size and matched Transformers at twice the size on language modeling perplexity. Crucially, it also matched Transformer quality on downstream tasks (HellaSwag, PIQA, WinoGrande, ARC).

**Inference throughput** tells the real story. During autoregressive generation:
- Mamba operates as a true RNN: $O(1)$ time per step, constant memory
- No KV cache — the entire "memory" is the fixed-size state vector $h_t \in \mathbb{R}^{D \times N}$
- At sequence length 1M, Mamba achieves **5x throughput** over an equivalent Transformer

The throughput advantage grows with sequence length because the Transformer's KV cache grows linearly while Mamba's state is constant.

### What Mamba Cannot Do

The fixed-size state is both Mamba's advantage and its fundamental limitation. A state of dimension $N$ (typically 16) can store at most $D \times N$ floats of information about the entire history. Compare this to a Transformer's KV cache, which stores $2 \times d_k$ floats *per token per layer* — explicitly retaining every past token's representation.

This means Mamba struggles with tasks requiring **exact retrieval from arbitrary positions**:
- Needle-in-a-haystack: finding a specific fact buried in a long context
- In-context learning with many examples: the state must compress all examples
- Precise copying: reproducing a specific substring from the input

The selection mechanism helps — it can learn to write important tokens strongly into the state — but the finite capacity means information must eventually be overwritten. This is the fundamental capacity-vs-efficiency tradeoff that motivates hybrid architectures.

---

## 4. Mamba-2 and Structured State Space Duality ([[mamba-2|paper]])

Dao and Gu (2024) made a theoretical discovery with deep practical implications: SSMs and attention are not rival architectures. They are two computational views of the same underlying mathematical object.

### Semiseparable Matrices: The Unifying Structure

Consider the output of a selective SSM unrolled across a sequence. Position $t$ produces:

$$y_t = \sum_{s=0}^{t} C_t \left(\prod_{r=s+1}^{t} \bar{A}_r\right) \bar{B}_s \, x_s$$

This can be written as a matrix-vector product $y = Mx$ where $M$ is a lower-triangular matrix with entry:

$$M_{ts} = C_t \left(\prod_{r=s+1}^{t} \bar{A}_r\right) \bar{B}_s \qquad \text{for } t \geq s$$

This matrix $M$ is **semiseparable**: every submatrix contained entirely in the lower-triangular part has rank at most $N$ (the state dimension). This is a well-studied matrix class in numerical linear algebra.

Now consider causal linear attention (attention without the softmax nonlinearity):

$$y_t = \sum_{s=0}^{t} (q_t^\top k_s) \, v_s$$

This is also a matrix-vector product with a lower-triangular matrix — one where entry $M_{ts} = q_t^\top k_s$, which has rank at most $d_k$ (the head dimension). When $d_k$ is small, this matrix is also semiseparable.

**The duality:** Both SSMs and linear attention produce semiseparable output matrices. The SSM computes this matrix via its recurrence (the "linear" form). Linear attention computes it via the outer product (the "quadratic" form). They are two algorithms for the same mathematical operation.

See [SSM vs Attention Comparison](figures/ssm-vs-attention.html) for a side-by-side visualization of these dual computation paths.

### Dual Computation Paths

The SSD framework gives Mamba-2 algorithmic flexibility:

| Regime | Better Algorithm | Why |
|--------|-----------------|-----|
| Short sequences | Quadratic form (attention-like) | Leverages GPU matrix-multiply units (tensor cores) |
| Long sequences | Linear form (recurrence) | $O(LN)$ beats $O(L^2)$ when $L \gg N$ |
| Mixed (practical) | **Chunk-wise hybrid** | Quadratic within chunks, recurrence across chunks |

The chunk-wise algorithm divides the sequence into chunks of size $C$. Within each chunk, the quadratic form is used (a small $C \times C$ matrix multiply, perfect for tensor cores). Across chunks, the recurrent form propagates the state. This is analogous to Flash Attention's tiling, but exploiting the *algebraic* structure rather than just the *memory hierarchy*.

### The Scalar-Identity Simplification

Mamba-2's key architectural change: the state transition matrix $A_t$ is constrained to be a **scalar times the identity matrix** ($A_t = a_t \cdot I$). In Mamba-1, $A$ was a diagonal matrix with $N$ independent values. The scalar constraint reduces this to a single value per timestep.

This sounds like a severe restriction, but it enables the chunked computation to be expressed as a standard matrix multiply — directly using tensor core hardware that achieves peak throughput. The tradeoff: each SSM "head" has less expressive state dynamics. The compensation: Mamba-2 introduces **multiple SSM heads** (analogous to multi-head attention), each with its own scalar $a_t$, recovering expressivity through parallelism rather than per-head complexity.

The result: **2-8x speedup** over Mamba-1 with equivalent or slightly better language modeling quality.

### What the Duality Means for Architecture Design

The SSD framework reframes the SSM-vs-attention debate. The question is not "which is better?" but rather "which computation path is faster for this hardware and sequence length?" A model designer can:

1. Use the recurrent form when inference is autoregressive (constant time per step)
2. Use the quadratic form when the full sequence is available and fits in SRAM
3. Use the chunked hybrid when the sequence is long but training parallelism matters

This flexibility also means SSM advances directly inform attention research, and vice versa. Any improvement to semiseparable matrix computation benefits both families.

---

## 5. Linear Attention: What You Lose Without Softmax

Linear attention replaces the softmax nonlinearity with a kernel decomposition:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \qquad \longrightarrow \qquad \frac{\phi(Q)\phi(K)^\top V}{\phi(Q)\phi(K)^\top \mathbf{1}}$$

where $\phi$ is a feature map. The key trick: by associativity, you can compute $\phi(K)^\top V$ first (a $d \times d$ matrix, independent of sequence length), then multiply by $\phi(Q)$. This changes the complexity from $O(L^2 d)$ to $O(Ld^2)$ — linear in sequence length.

This is exactly the "quadratic form" of the SSD framework applied to attention. Linear attention and SSMs are computing the same semiseparable matrix product, just with different parameterizations of the entries.

### The Softmax Gap

Removing softmax is not free. Softmax attention has two properties that linear attention loses:

1. **Sharp selection.** Softmax produces a sparse-ish distribution: a few keys get most of the weight, and the rest are effectively ignored. The exponential nonlinearity amplifies score differences. Linear attention produces smoother, more uniform weights — it cannot focus as sharply on specific positions.

2. **Normalization stability.** Softmax normalizes attention weights to sum to 1, preventing the output magnitude from depending on sequence length. Linear attention must explicitly normalize (the denominator $\phi(Q)\phi(K)^\top \mathbf{1}$), and this normalization can be numerically unstable, especially with certain feature maps.

The practical consequence: linear attention consistently underperforms softmax attention on tasks requiring precise retrieval or sharp contextual selection. Weng ([[weng-transformer-family|blog]]) catalogs the Performer (Choromanski et al. 2020), which uses random orthogonal features for $\phi$, achieving linear complexity but with meaningful quality degradation on language tasks.

### The MiniMax Cautionary Tale

The most instructive data point comes from industry. MiniMax deployed a linear-attention model (MiniMax-01) in production and found that it degraded multi-turn conversation quality and reasoning sufficiently to justify reverting to quadratic attention in subsequent releases, despite the efficiency penalty. As Raschka ([[raschka-attention-variants|blog]]) notes, this retreat from linear attention is a cautionary tale: the efficiency gains must be weighed against the quality cost on the specific tasks your model must handle.

Linear attention is not dead — it lives on inside SSMs (via the SSD duality) and in hybrid architectures where it handles the "easy" layers. But as a wholesale replacement for softmax attention in language models, it has not proven viable.

---

## 6. Hybrid Architectures: The Jamba Approach ([[jamba|report]])

If SSMs excel at efficient long-range processing but struggle with precise retrieval, and attention excels at retrieval but costs $O(N^2)$, the natural question is: why not use both?

### Jamba's Architecture

AI21's Jamba (2024) is the first large-scale hybrid Transformer-Mamba model. Its architecture interleaves Mamba and attention layers at a **1:7 ratio** — only 4 out of 32 layers use attention, with the remaining 28 using Mamba SSM layers.

| Component | Value |
|-----------|-------|
| Total parameters | 52B |
| Active parameters | 12B (MoE routing) |
| Attention layers | 4 (out of 32) |
| Mamba layers | 28 (out of 32) |
| MoE experts | 16 per MoE layer, top-2 routing |
| Context length | 256K tokens |

The attention layers use GQA ([[ch-07]]), and MoE is applied on alternating layers (replacing the MLP), with 16 experts and top-2 routing. The architecture was explicitly designed to fit on a single 80GB GPU.

### Why the Hybrid Works

The key finding from Jamba's ablations: **pure Mamba fails at in-context learning, but a small fraction of attention layers fixes it.**

Pure Mamba models struggle with format adherence and in-context learning — tasks that require the model to precisely recall and reproduce patterns from the prompt. This aligns with the fundamental limitation: Mamba's fixed-size state cannot guarantee exact retrieval of arbitrary positions.

Adding just 4 attention layers (1 per 8-layer block) provides enough explicit token-to-token matching to handle retrieval tasks, while the 28 Mamba layers handle the bulk of sequential processing at linear cost. The attention layers act as "retrieval checkpoints" in an otherwise recurrent architecture.

### The KV Cache Advantage

Because only 4 layers maintain a KV cache (versus 32 in a pure Transformer), Jamba's cache is dramatically smaller:

| Model | KV Cache at 256K Context |
|-------|-------------------------|
| Llama-2 7B (32 attention layers) | 128 GB |
| Mistral 7B (32 attention layers, SWA) | 32 GB |
| Mixtral 12.9B (32 attention layers, SWA) | 32 GB |
| **Jamba 12B active (4 attention layers)** | **4 GB** |

A 32x reduction versus a standard Transformer at the same context length. The Mamba layers contribute zero to the KV cache — their state is a fixed-size vector regardless of sequence length.

### Throughput Results

On a single A100 80GB GPU with 8K context, Jamba achieves **3x throughput** over Mixtral at equivalent active parameter count. At 128K context on 4x A100s, the advantage holds at 3x. This comes from both the reduced KV cache (less memory bandwidth consumed loading cached values) and the linear-time Mamba layers.

### Practical Finding: RMSNorm Stabilization

A critical training detail: scaling Mamba layers to large model sizes causes training loss spikes. Jamba discovered that applying **RMSNorm within Mamba layers** prevents these instabilities. This is analogous to the normalization challenges that plagued early deep Transformer training — the SSM recurrence can amplify or diminish activations across many steps without explicit normalization.

### No Positional Encoding

Jamba uses no explicit positional encoding (no RoPE, no sinusoidal). Mamba's recurrent structure provides implicit positional information — the state naturally evolves differently based on position in the sequence. Ablations confirmed that adding RoPE to the attention layers provided no benefit, so it was omitted for simplicity.

---

## 7. The Complexity Landscape

See [SSM vs Attention Comparison](figures/ssm-vs-attention.html) for an interactive comparison of these complexity profiles.

| Architecture | Training Cost | Inference (per step) | Memory (inference) | Exact Retrieval |
|-------------|---------------|---------------------|-------------------|-----------------|
| Softmax Attention | $O(L^2 d)$ | $O(Ld)$ (load KV cache) | $O(L)$ KV cache | Yes |
| Flash Attention | $O(L^2 d)$ FLOPs, $O(L)$ memory | $O(Ld)$ | $O(L)$ KV cache | Yes (exact) |
| Linear Attention | $O(Ld^2)$ | $O(d^2)$ | $O(d^2)$ fixed state | Weak |
| Classical SSM (S4) | $O(L \log L)$ via FFT | $O(N)$ | $O(N)$ fixed state | No |
| Selective SSM (Mamba) | $O(LN)$ via scan | $O(N)$ | $O(N)$ fixed state | Weak |
| Hybrid (Jamba) | Mixed | Mixed | $O(L)$ for attn layers + $O(N)$ for SSM layers | Attn layers: yes |

The fundamental tradeoff is clear: linear-time methods achieve their efficiency by compressing history into a fixed-size state, which necessarily loses information. Quadratic-time attention retains all history explicitly, paying the cost in memory and bandwidth. Hybrid architectures try to have both — linear cost for most processing, quadratic cost only where precise retrieval is needed.

---

## Core Insights from the Literature

### Insight 1: Content-based reasoning is what separates SSMs from attention — and selectivity closes the gap
**Paper:** Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" ([[mamba|paper]])

The Mamba paper identifies the precise reason prior subquadratic architectures (linear attention, gated convolutions, classical SSMs) failed on language: they cannot perform content-based reasoning because their parameters are input-independent. Making $B$, $C$, and $\Delta$ functions of the input gives the SSM the ability to selectively propagate or forget information — the same operation that softmax attention performs implicitly through its score-weighted aggregation. The selection mechanism is not a hack bolted onto SSMs; it is the minimal change needed to make linear-time models competitive with attention on discrete, content-rich modalities like language. **Guideline:** When evaluating any subquadratic architecture, check whether it can condition its information routing on input content. If parameters are fixed regardless of what token is being processed, the architecture will fail on language.

### Insight 2: SSMs and attention are mathematical duals, not competing paradigms
**Paper:** Dao and Gu, "Transformers are SSMs" ([[mamba-2|paper]])

The SSD framework proves that selective SSMs and linear attention both compute structured semiseparable matrices — the same mathematical object, accessible through either a recurrent form ($O(LN)$) or a quadratic form ($O(L^2)$). This is not a loose analogy; it is an exact algebraic equivalence. The practical implication is algorithmic flexibility: choose the faster computation path based on hardware and sequence length, and improvements to either family transfer to the other. **Guideline:** When the state dimension $N$ is small relative to sequence length $L$, the recurrent path is faster. When $L$ is short enough that $L^2$ fits comfortably in SRAM, the quadratic path wins because it maps to tensor core matrix multiplies. The chunk-wise hybrid is the default for training.

### Insight 3: A small fraction of attention layers rescues SSMs from retrieval failure
**Paper:** AI21 Labs, "Jamba" ([[jamba|report]])

Jamba's ablations show that pure Mamba models fail at in-context learning and format adherence — tasks requiring exact token retrieval from the context. But adding attention to just 1 in 8 layers (a 1:7 ratio) fully recovers this capability while keeping 87.5% of the layers at linear cost. The attention layers function as "retrieval checkpoints" that compensate for the SSM's finite-capacity state. This 1:7 ratio reduces KV cache by 32x versus a pure Transformer. **Guideline:** When designing hybrid SSM-attention architectures, start with a small attention fraction (1:7 to 1:4) and increase only if retrieval benchmarks (needle-in-a-haystack, multi-shot ICL) require it. Each additional attention layer adds KV cache cost.

### Insight 4: SSM training stability requires explicit normalization at scale
**Paper:** AI21 Labs, "Jamba" ([[jamba|report]])

Jamba discovered that Mamba layers exhibit training loss spikes when scaled to large models, and that applying RMSNorm within the Mamba block prevents this. This parallels the normalization challenges in deep Transformers (Pre-LN vs Post-LN, [[ch-02]]) — any deep recurrence that compounds activations across many steps needs explicit normalization to prevent magnitude drift. **Guideline:** When scaling SSM layers beyond ~7B parameters, add RMSNorm within the SSM block (not just in the residual stream). This is a necessary adaptation that the original Mamba paper did not encounter at its 3B scale.

---

## Key Takeaways

1. **Classical SSMs are linear RNNs with a dual convolution view.** The continuous-time formulation discretized with step $\Delta$ gives a recurrence for inference ($O(1)$ per step) and a convolution for training ($O(L \log L)$). HiPPO initialization ensures the state maintains a principled approximation of input history.

2. **Mamba's selection mechanism is the key innovation.** Making $B$, $C$, $\Delta$ input-dependent gives SSMs content-based reasoning — the ability to selectively remember and forget. This breaks the convolution view but is solved by a hardware-aware parallel scan algorithm.

3. **SSMs and linear attention are mathematical duals.** The SSD framework (Mamba-2) shows both compute semiseparable matrices. This enables choosing the fastest computation path for the hardware: recurrence for long sequences, quadratic form for short, chunked hybrid in practice.

4. **Linear attention loses sharp selection.** Without softmax's exponential nonlinearity, attention weights become smoother and less discriminative. This is why linear attention degrades on retrieval and reasoning tasks. The MiniMax retreat from production linear attention is the clearest empirical signal.

5. **Hybrid architectures (Jamba) are the pragmatic answer.** A 1:7 attention-to-SSM ratio gives 32x KV cache reduction while preserving retrieval quality. The attention layers serve as retrieval checkpoints; the SSM layers handle efficient sequential processing.

6. **The core tradeoff is memory capacity vs. cost.** Attention explicitly stores every past token ($O(L)$ memory, $O(L)$ per-step cost). SSMs compress history into a fixed state ($O(N)$ memory, $O(1)$ per-step cost). No architectural trick eliminates this fundamental tradeoff — you can only choose where on the spectrum to operate.

7. **SSM scaling requires normalization.** Training loss spikes in large Mamba models are prevented by RMSNorm within the SSM block. This is a practical requirement not apparent from small-scale experiments.

---

## References

- [[mamba|Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) (paper)]] — selective SSMs, hardware-aware scan
- [[mamba-2|Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024) (paper)]] — SSD framework, Mamba-2
- [[jamba|AI21 Labs, "Jamba: A Hybrid Transformer-Mamba Language Model" (2024) (report)]] — hybrid SSM-attention architecture
- [[weng-transformer-family|Weng, "The Transformer Family Version 2.0" (2023) (blog)]] — linear attention variants, Performer, sparse attention taxonomy
- [[flash-attention|Dao et al., "FlashAttention" (2022) (paper)]] — IO-aware exact attention, parallel with Mamba's SRAM-residency approach
- Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (2021) — S4, HiPPO initialization
- Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (2020) — HiPPO theory
- Choromanski et al., "Rethinking Attention with Performers" (2020) — random feature linear attention
