# Excerpt: Mamba's Selection Mechanism

Source: [[mamba|paper]] — Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

---

## The Core Problem: Why Fixed Parameters Fail on Language

The Mamba paper opens with a precise diagnosis of why every subquadratic architecture before it failed on language — not a vague claim about "efficiency vs quality" but an identification of the specific missing capability.

From the abstract:

> Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning.

The diagnosis is precise: prior subquadratic architectures fail on language because they process every token the same way. A classical SSM applies the same state transition matrix $A$ regardless of whether the current token is "not" (which inverts meaning) or "the" (which carries almost no information). Attention, by contrast, routes information based on content — the softmax scores are functions of the actual query and key vectors.

## The Fix: Input-Dependent Parameters

> First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token.

The selection mechanism makes three parameters input-dependent:

- **B_t = s_B(x_t)**: The input projection matrix. Controls *what* gets written into the state from the current token.
- **C_t = s_C(x_t)**: The output projection matrix. Controls *what* gets read from the state for the current output.
- **delta_t = softplus(s_delta(x_t))**: The discretization step. Controls *how much* the state transitions — the critical forget/remember knob.

The discretization step $\Delta_t$ deserves special attention. When discretized:

$$\bar{A}_t = \exp(\Delta_t A)$$

- Large $\Delta_t$: $\bar{A}_t \to 0$, meaning the old state is nearly erased. $\bar{B}_t = \Delta_t B_t$ is large, so the new input is written strongly. **The model resets and writes.**
- Small $\Delta_t$: $\bar{A}_t \to I$, meaning the old state is preserved almost unchanged. $\bar{B}_t \approx 0$, so the new input is barely registered. **The model preserves and ignores.**

## The Price: Losing Convolution Efficiency

> Even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode.

Input-dependent parameters make the system non-linear and time-varying. The convolution kernel $\bar{K}_t = C\bar{A}^t\bar{B}$ no longer exists as a fixed sequence — each timestep has its own $\bar{A}_t$, $\bar{B}_t$, $C_t$. The FFT-based $O(L \log L)$ training path is gone.

Mamba compensates with a parallel scan algorithm that keeps the state in GPU SRAM (following Flash Attention's IO-aware principle), achieving comparable speed to S4's convolution despite the theoretically harder computation.

### The Hardware-Aware Scan in Detail

The parallel scan exploits the associativity of the recurrence operator. The operation $(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2, a_2 b_1 + b_2)$ — corresponding to composing two affine state transitions — is associative, enabling a prefix sum computation in $O(\log L)$ parallel steps on $L$ processors.

The critical optimization is **SRAM residency**. The expanded state tensor has shape $(B, L, D, N)$ — batch x length x model dim x state dim. At batch 64, length 2K, $D = 2048$, $N = 16$, this is 4 GB in FP32 — far too large for HBM round-trips at every scan step. Mamba's fused CUDA kernel:

1. Loads a block of inputs $(x_t, \Delta_t, B_t, C_t)$ from HBM into SRAM
2. Computes the discretization ($\bar{A}_t$, $\bar{B}_t$) in SRAM
3. Runs the scan recurrence entirely in SRAM
4. Writes only the final output $y_t$ back to HBM

No intermediate states are materialized in HBM. This is the same IO-awareness principle that makes Flash Attention fast — the bottleneck is not compute but memory traffic, and the solution is to keep hot data in fast SRAM.

## The Result

> As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

Mamba-3B matches Transformer-6B quality while running at **5x throughput** during inference, with constant memory regardless of sequence length (no KV cache).

## The Fundamental Limitation

> The fixed-size state (dimension N) limits the model's ability to recall information from arbitrarily far in the past — unlike Transformers which can attend to any position in the KV cache. This is a fundamental capacity-vs-efficiency tradeoff.

A state of dimension $N$ (typically 16) stores $D \times N$ floats for the entire history. A Transformer's KV cache stores $2 \times d_k$ floats *per token*. For a 10K token sequence, the Transformer explicitly retains ~2.5M floats per layer; Mamba compresses the same history into ~80K floats. The selection mechanism helps prioritize, but information must eventually be overwritten.

This limitation directly motivates hybrid architectures like Jamba ([[jamba|report]]).

---

## Key Equations

**Classical SSM (fixed parameters):**
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t$$

**Selective SSM (input-dependent):**
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t h_t$$

where $\bar{A}_t = \exp(\Delta_t A)$, $\bar{B}_t = \Delta_t B_t$, and $\Delta_t = \text{softplus}(s_\Delta(x_t))$.

**The Mamba block (no attention, no MLP):**
```
x -> Linear(D, ED) -> Conv1D(k=4) -> SiLU -> Selective SSM -> gate(*) -> Linear(ED, D) -> y
x -> Linear(D, ED) -> SiLU ─────────────────────────────────/
```

The gating branch provides a multiplicative control signal analogous to SwiGLU in Transformer FFNs. The 1D convolution provides local context to help the selection mechanism. The entire block replaces both attention and MLP — a simpler architecture with fewer distinct components.
