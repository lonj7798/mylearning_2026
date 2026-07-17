<!-- chapter: ch-04
     track: attention
     kind: content
     title: Attention Is a Memory Problem: O(N²) and Why the Kernel Decides
     deps: [[ch-03]]
     sources: [[self-attention-no-n2-memory]], [[online-softmax]]
-->

# Chapter 4 — Attention Is a Memory Problem: O(N²) and Why the Kernel Decides

> **Core insight.** Standard multi-head attention materializes an N×N score matrix and its softmax as activations: O(N²) bytes per head per layer, which grows quadratically with sequence length and dominates GPU memory — not FLOPs — as the primary limiting factor at long context. Rabe & Staats 2021 prove this is not mathematically necessary: exact self-attention can be computed in O(1) extra memory per query by streaming over keys with a running softmax normalizer, never writing the full matrix. The choice of attention *kernel* — not the model architecture — determines whether training costs O(N²) or O(N) activation memory per layer.

> **Guideline.** If your training job OOMs at long sequence length, the first lever is always the attention kernel, not gradient checkpointing or batch size. Confirm PyTorch SDPA did not silently fall back to the MATH backend (which allocates the full N×N matrix). If it did, force FlashAttention or xFormers. The online-softmax recurrence (Milakov & Gimelshein 2018) is the mathematical primitive that makes O(N)-memory tiling numerically stable; understanding it tells you exactly why the kernel can tile safely and when it cannot.

---

## 1. Standard Attention: The O(N²) Memory Budget

### 1.1 The Vanilla Forward Pass, Step by Step

Multi-head self-attention for a single head with query matrix Q ∈ ℝ^(N×d), key matrix K ∈ ℝ^(N×d), and value matrix V ∈ ℝ^(N×d) computes:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

Spelled out as an execution sequence:

```
Step 1.  S = Q Kᵀ           # shape: N×N  (the score matrix)
Step 2.  S_scaled = S / √d   # elementwise divide; same shape
Step 3.  P = softmax(S_scaled, dim=-1)  # row-wise softmax; shape N×N
Step 4.  O = P · V           # shape: N×d  (the output)
```

Steps 1-3 produce two tensors of shape N×N that must be held in memory simultaneously: `S_scaled` (for the backward pass through softmax) and `P` (for the matmul in Step 4). At bf16 (2 bytes per element), one N×N activation tensor for one head costs:

```
bytes = N² × 2
```

For N = 32,768 (32k tokens), that is 32768² × 2 = **2,147,483,648 bytes ≈ 2 GB per head**.

With H heads per layer and L layers, the total activation footprint just from attention scores is:

```
attention_activations = 2 × H × L × N² × dtype_bytes
                        ↑ stores both S_scaled and P
```

For a 70B model (L=80, H=64) at N=32k, bf16:

```
= 2 × 64 × 80 × (32768²) × 2 ≈ 549 TB
```

That number is absurd, which is precisely the point: standard attention at long sequence does not fit on any GPU and does not fit in any training cluster. The N² term is not a coefficient detail — it is a qualitative regime change.

### 1.2 Why Memory, Not FLOPs, Is the Wall

The standard narrative focuses on O(N²) *compute*. But [[self-attention-no-n2-memory]] makes a sharper claim:

> "device memory rather than compute capability is often the limiting factor on modern accelerators"

The FLOPs for standard attention are also O(N²), but modern GPUs have enough compute throughput that the bottleneck is *memory bandwidth* (reading/writing the N×N matrix from HBM) and *memory capacity* (fitting the activation tensors that must be saved for the backward pass). You can train a longer sequence with fewer FLOPs by recomputing activations — but with standard attention, even *recomputing* requires re-materializing the N×N matrix, so memory never drops below O(N²) unless you change the algorithm.

This is the argument that sets up the rest of the attention track: the fundamental problem is the existence of the N×N materialized tensor, and fixing it requires an algorithmic change to *how* softmax is computed.

---

## 2. Online Softmax: The Mathematical Key

### 2.1 Classical Softmax Needs Three Passes

Given a length-N score row x = [x₁, x₂, ..., xₙ], standard numerically-stable softmax proceeds:

```
Pass 1:  m = max(x₁, ..., xₙ)            # find global max for stability
Pass 2:  d = Σᵢ exp(xᵢ − m)              # compute partition function
Pass 3:  pᵢ = exp(xᵢ − m) / d  for all i  # normalize
```

Each pass reads the entire row from memory. This mandates that the full row be stored — which for attention scores means the full N×N matrix row must be resident in HBM during attention computation. As [[online-softmax]] documents: "Three-pass classical softmax: (1) read all x to find max m; (2) read all x again to compute sum d; (3) read all x a third time to output." That is 3× memory bandwidth over the full score vector just for softmax.

### 2.2 The One-Pass Online Recurrence

Milakov & Gimelshein 2018 ([[online-softmax]]) derive a recurrence that computes numerically-stable softmax in a single streaming pass, maintaining only two scalars — a running maximum m and a running sum d:

```
Initialize: m₀ = -∞,  d₀ = 0

For each element xₖ  (k = 1 … N):
    m_new = max(m_old, xₖ)
    d_new = exp(m_old − m_new) · d_old + exp(xₖ − m_new)
    m_old ← m_new,  d_old ← d_new

Final output:  pᵢ = exp(xᵢ − m_final) / d_final  for all i
```

The rescaling term `exp(m_old − m_new)` is the key: when m_new > m_old (i.e., we found a larger element), all previously accumulated contributions are retroactively rescaled downward by exactly the right factor, maintaining numerical equivalence to the subtract-max trick — at no additional storage cost. At each step, only m and d need to be in registers.

The measured payoff is significant: [[online-softmax]] reports "Softmax alone accelerates up to 1.3×; fused Softmax+TopK accelerates up to 5× on GPU." But the training-memory payoff is more important than the raw speedup:

> "the one-pass recurrence is the key that lets FlashAttention tile an attention computation over blocks of Q and K without ever writing the full N×N score matrix to HBM. Without online softmax, each tile would need to read back previously written partial results to renormalize — requiring O(N²) HBM writes." ([[online-softmax]])

With the online recurrence, the running (m, d) scalars fit in registers and the attention output is accumulated in SRAM; backward-pass activation storage for attention collapses from O(N²) to O(N).

### 2.3 Why the Recurrence Is Numerically Exact

It is worth being precise: this is not an approximation. The online recurrence produces bit-identical results to the classical three-pass computation (up to floating-point associativity). The only structural requirement is that each xₖ is seen exactly once in order, which is exactly what streaming over a blocked attention tile provides. This exactness is what allows [[self-attention-no-n2-memory]] to claim its streaming attention is "exact, not approximate."

---

## 3. Self-Attention Without O(N²) Memory — Rabe & Staats 2021

### 3.1 The Core Observation

Rabe & Staats 2021 ([[self-attention-no-n2-memory]]) frame the problem precisely:

> **Core Insight.** Exact self-attention can be computed with O(1) extra memory per query by deferring the softmax normalization: accumulate weighted values and a running normalizer in a single outer-loop pass over K/V blocks, never materializing the full N×N score matrix.

The mechanism is a direct application of the online softmax recurrence to the attention computation. Instead of materializing S = QKᵀ, compute the output one query at a time:

```python
# Pseudocode for O(1)-per-query streaming attention
# Q: (N, d),  K: (N, d),  V: (N, d)

for i in range(N):                     # outer loop: queries
    q_i = Q[i]                         # (d,)
    m_i, d_i = -inf, 0.0              # running max, running sum
    o_i = zeros(d)                     # running output accumulator

    for j in range(N):                 # inner loop: keys (streamed)
        s_ij = dot(q_i, K[j]) / sqrt(d)   # scalar score
        m_new = max(m_i, s_ij)
        d_i = exp(m_i - m_new) * d_i + exp(s_ij - m_new)
        o_i = exp(m_i - m_new) * o_i + exp(s_ij - m_new) * V[j]
        m_i = m_new

    O[i] = o_i / d_i                  # normalize output
```

Memory at any moment: q_i (d scalars), m_i and d_i (2 scalars), o_i (d scalars), and the current K[j] / V[j] row (2d scalars). Total: O(d) = O(1) extra memory per query, independent of N.

### 3.2 The Three Variants

[[self-attention-no-n2-memory]] proves three exact memory regimes:

| Variant | Memory | Mechanism |
|---------|--------|-----------|
| O(1) per-query | Single query outer loop, full K/V inner loop; stores only running scalars | Pure streaming; serial over queries |
| O(log N) for full self-attention | One scalar index per query | Theoretical minimum for arbitrary self-attention |
| O(√N) practical accelerator variant | Chunk Q and K at size √N | Exploits GPU/TPU tile parallelism; SRAM fits both chunks |

The O(√N) variant is the practically important one: it chunks both Q and K at block size √N, fitting both blocks into SRAM simultaneously and enabling efficient matrix-level GEMM operations within each chunk — the operation that GPUs are built for. This is the variant that xFormers implements and that FlashAttention converges toward.

### 3.3 Measured Memory Reduction

At N = 16,384 (a realistic long-context sequence length):

- **Inference memory: reduced 59× vs standard attention**
- **Backpropagation memory: reduced 32× vs standard attention**
- **Runtime overhead: within a few percent of the baseline**

From [[self-attention-no-n2-memory]]:

> "Training-memory angle: eliminates the activation tensor that normally grows as O(N²·B·H) bytes (batch × heads × N² attention scores stored for the backward pass). Backward-pass memory for attention drops from O(N²) to O(N·d) — a qualitative regime change for long-sequence training."

The phrase "qualitative regime change" is exactly right: at N=16k, the difference between O(N²) and O(N·d) is roughly 16384/512 ≈ 32× for a head dimension of 512 — matching the measured 32× backprop reduction. As N doubles, O(N²) quadruples; O(N) only doubles. The gap compounds.

### 3.4 The Backward Pass: Checkpointing Instead of Storing

A naive backward pass would store all intermediate attention scores to compute gradients, eliminating the memory advantage. Rabe & Staats address this with selective checkpointing:

> "The paper applies selective checkpointing over chunk-summarization functions — gradients are recomputed on-the-fly, never storing the N×N matrix during backprop." ([[self-attention-no-n2-memory]])

This is the same recompute-vs-store tradeoff as [[ch-03]]'s gradient checkpointing, but applied specifically to the attention kernel's internal states rather than to inter-layer activations. The key quantities saved are the per-query (m, d) scalars — the logsumexp — which are O(N) and allow recomputing the attention weights on the backward pass without ever writing the N×N matrix.

---

## 4. Why the Kernel Decides

The central framing of this chapter: the *algorithm* for computing softmax(QKᵀ/√d)V determines the memory regime, but the *kernel* is what implements that algorithm on hardware.

```
Model architecture:   defines Q, K, V, head count, head dimension
       ↓
Attention algorithm:  standard (O(N²)) vs streaming (O(N))
       ↓
Kernel:               PyTorch MATH / FlashAttention / xFormers / SDPA dispatch
       ↓
Actual GPU memory:    N² bytes vs N·d bytes per head per layer
```

The model does not control which path is taken. The kernel does. Critically, as [[ch-05]] will show:

- **PyTorch SDPA MATH backend** falls back silently to the O(N²) path — allocating the full score matrix — when the input configuration is not supported by a fast kernel (wrong dtype, non-contiguous layout, causal mask format mismatch).
- **FlashAttention 1/2/3** implement the online-softmax tiling on SRAM and never write the N×N matrix to HBM.
- **xFormers `memory_efficient_attention`** is a CUTLASS FMHA kernel implementing the Rabe & Staats O(N) algorithm; it is PyTorch SDPA's EFFICIENT_ATTENTION backend.

The decision point is not in your model code. It is in which backend executes. Understanding the math here — why online softmax enables O(N) tiling — tells you exactly what properties a kernel must have to avoid the O(N²) footprint: it must be able to stream over K/V blocks, accumulate the output with a running (m, d) normalizer, and never write a full attention row to a buffer larger than O(d).

---

## 5. Putting It Together: The Memory Picture at Long Context

To make the O(N²) cost concrete, consider training a 7B model (L=32, H=32, d_head=128) at various sequence lengths with standard vs streaming attention. Activation memory from attention alone, per layer, one head, bf16:

```
N=2,048:   2048²  × 2 B = 8  MB  per head   →  32 heads × 32 layers = 8   GB
N=8,192:   8192²  × 2 B = 128 MB per head   →  32 heads × 32 layers = 128 GB
N=32,768:  32768² × 2 B = 2  GB  per head   →  32 heads × 32 layers = 2   TB
```

(These are the attention-activation numbers; total activation memory includes feedforward layers too, and gradient checkpointing from [[ch-03]] can trade some of this for recompute.)

With O(N) streaming attention (Rabe & Staats / FlashAttention), the N×N term disappears. The activation footprint from attention becomes O(N·d·H·L·B), which for the same model at N=32k, d=128, H=32, L=32, B=1:

```
= 32768 × 128 × 32 × 32 × 1 × 2 B = 8.6 GB
```

The contrast — 2 TB vs 8.6 GB — at N=32k and B=1 is the regime change in practice. The kernel choice at 32k context is not an optimization; it is the difference between a job that runs and one that does not.

---

## Core Insights from the Literature

**From [[self-attention-no-n2-memory]] (Rabe & Staats 2021):**  
Self-attention does not need O(N²) memory. The equivalence between "materializing the full score matrix" and "computing standard attention" is an implementation assumption, not a mathematical requirement. The information-theoretic minimum for exact self-attention with a single query is O(d): only the running normalizer (m, d) and the output accumulator. This changes the design space for long-context training from "can we fit the matrix?" to "which kernel implements streaming correctly?"

**From [[online-softmax]] (Milakov & Gimelshein 2018):**  
The one-pass online recurrence — m_new = max(m_old, xₖ), d_new = exp(m_old − m_new)·d_old + exp(xₖ − m_new) — is the enabling primitive. It is exact, not approximate. Its significance in the training-memory context is not the 1.3× softmax speedup but that it allows a kernel to tile the attention computation over SRAM-sized blocks of Q and K without requiring a second HBM read to renormalize. Without this recurrence, tiling would require storing all partial scores to normalize later — defeating the memory savings.

**The substrate forces streaming; the kernel chooses whether to use it:**  
HBM bandwidth (not compute) gates long-context throughput. The O(N²) score matrix is an HBM-write/read bottleneck: writing 2 GB per head per layer to HBM, then reading it back for the output matmul, consumes the same bandwidth that limits MFU. The online softmax recurrence eliminates this roundtrip by keeping the normalizer in registers — the substrate (HBM bandwidth vs SRAM size vs register file) exactly explains why the streaming approach wins.

**Exactness matters for trust:**  
Both the Rabe & Staats O(1) variant and the FlashAttention kernels that build on [[online-softmax]] are *exact* in the sense of producing bit-consistent results with standard attention (up to floating-point associativity, same as any fused kernel). This is not approximate attention (like Linformer or Longformer). The N×N computation still happens; it just happens in SRAM tiles rather than being materialized in HBM. This distinction is why FlashAttention can be used as a drop-in replacement with no model-accuracy regression.

---

## Key Takeaways

- Standard attention materializes S = QKᵀ and P = softmax(S) as O(N²) activations per head; at N=32k and a 70B model, this is physically impossible to store.
- The Milakov & Gimelshein 2018 one-pass recurrence — (m, d) running scalars updated per element — eliminates the need for a second pass over the full score row, making single-pass numerically-stable softmax possible with O(1) extra state.
- Rabe & Staats 2021 apply this to self-attention: streaming over K/V blocks with the running (m, d) normalizer reduces attention activation memory from O(N²) to O(1) per query, O(√N) for the practical GPU-parallel variant. Measured at N=16k: 59× inference memory reduction, 32× backprop reduction.
- The backward pass uses selective checkpointing of per-query logsumexp scalars to recompute attention weights on the fly, never storing the N×N matrix during backprop.
- The kernel — not the model — determines which regime applies. This is the central design axis [[ch-05]] (FlashAttention) and [[ch-06]] (SDPA/xFormers/SageAttention/Ring/Paged) will map out in detail.
- At N=32k on a 7B model, the difference between standard and streaming attention is roughly 2 TB vs 8.6 GB in attention-only activation memory: not an optimization, a prerequisite for the job to run at all.

---

## References

- Markus N. Rabe and Charles Staats. "Self-attention Does Not Need O(n²) Memory." arXiv:2112.05682, December 2021. https://arxiv.org/abs/2112.05682 — [[self-attention-no-n2-memory]]
- Maxim Milakov and Natalia Gimelshein. "Online normalizer calculation for softmax." arXiv:1805.02867, May 2018. https://arxiv.org/abs/1805.02867 — [[online-softmax]]

**Sibling chapters:** [[ch-03]] (gradient checkpointing and activation budgets), [[ch-05]] (FlashAttention 1/2/3 built on this math), [[ch-06]] (the full attention kernel zoo and their training-memory profiles), [[ch-09]] (capstone: long-context MoE budget).
