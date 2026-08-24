<!-- chapter: ch-05
     track: attention
     kind: content
     title: FlashAttention (1/2/3): IO-Aware Exact Attention
     deps: [[ch-04]]
     sources: [[flash-attention-1]], [[flash-attention-2]], [[flash-attention-3]]
-->

# Chapter 5 — FlashAttention (1/2/3): IO-Aware Exact Attention

> **Core insight.** Attention is slow and memory-hungry not because of arithmetic complexity, but because of HBM bandwidth: the N×N score matrix is written to HBM and read back three separate times in a naive implementation (Q·Kᵀ → softmax → ·V). FlashAttention eliminates all three of those HBM round-trips by tiling Q, K, V into blocks that fit in SRAM and maintaining a numerically exact online softmax recurrence entirely on-chip — the N×N matrix is never materialized, yielding O(N) activation memory and a 2–3× wall-clock speedup that comes from IO elimination, not from computing fewer FLOPs.

> **Guideline.** Enable FlashAttention (via `torch.nn.functional.scaled_dot_product_attention` with the `FLASH_ATTENTION` backend, or the `flash-attn` package directly) for every training run where sequence length exceeds ~512 tokens. On A100s, prefer FA2 for ~2× throughput over FA1 at identical O(N) memory. On H100s, prefer FA3 for 740 TFLOPs/s (FP16) or ~1.2 PFLOPs/s (FP8). Never let training silently fall back to the `MATH` backend — it allocates the full O(N²) score matrix and is ~38× slower; audit `torch.backends.cuda.flash_sdp_enabled()` and watch for silent fallback on unsupported dtypes/masks.

---

## 1. The Problem: HBM Is the Bottleneck

The naive attention algorithm for sequence length N and head dimension d performs these HBM accesses ([[flash-attention-1]]):

```
# Standard attention — what actually runs without FA
S = Q @ K.T              # shape (N, N) — WRITE to HBM: Θ(N²) bytes
P = softmax(S)           # READ S from HBM, WRITE P: Θ(N²) bytes again
O = P @ V                # READ P from HBM: Θ(N²) bytes a third time
# Total HBM traffic: Θ(Nd + N²)
```

The GPU's on-chip SRAM (~20 MB on A100) is ~19 TB/s; the off-chip HBM is ~1.5–2 TB/s — a 10× bandwidth gap. When N=2048, a single layer's score matrix for a batch-32, 32-head, d=64 setup costs:

```
32 (batch) × 32 (heads) × 2048² (scores) × 2 (bytes/fp16) = 8 GB per layer
```

That 8 GB must be written and read three times per forward pass. At 32 layers, you're moving 768 GB of N×N buffers through a 2 TB/s pipe each step — that is the real bottleneck, not tensor core throughput.

**FlashAttention's answer:** tile Q, K, V so each tile fits in SRAM; accumulate the output O and softmax statistics there; never write the score matrix to HBM at all.

> **Interactive companion:** [figures/flash-tiling.html](figures/flash-tiling.html) — step through the tiling loop tile by tile, showing which data lives in SRAM vs HBM and how the online softmax running statistics (m, l) are updated at each step.

---

## 2. FlashAttention-1: Tiling + Online Softmax (Dao et al., 2022)

### 2.1 Tile sizes

Block sizes are set to fit two Q/K/V tiles simultaneously in SRAM ([[flash-attention-1]]):

```
Br = ⌈M / 4d⌉    # Q tile rows   (M = SRAM size, e.g. 20 MB)
Bc = min(⌈M / 4d⌉, d)  # K/V tile columns
```

Each (Br × Bc) pair is a self-contained attention sub-problem that can be computed without touching HBM mid-computation.

### 2.2 Online softmax recurrence

The mathematical primitive that makes single-pass tiling numerically stable is the **online softmax** from Milakov & Gimelshein 2018. For each new tile of K scores arriving as a vector x_new:

```
m_new = max(m_old, max(x_new))           # running row maximum
d_new = exp(m_old - m_new) * d_old       # rescale old denominator
      + sum(exp(x_new - m_new))           # add new tile's contribution
O_new = (d_old / d_new) * exp(m_old - m_new) * O_old  # rescale accumulated output
      + exp(x_new - m_new) * V_tile / d_new            # add new tile's contribution
```

After all K/V tiles are processed, `O_new` is the exact softmax-weighted value — equal to what you would get if you had materialized the full N×N matrix and applied softmax globally. The recurrence is the key: without it, a second HBM read to renormalize would be unavoidable.

**HBM traffic:** FlashAttention reduces total HBM accesses from Θ(Nd + N²) to **Θ(N²d²/M)** — for large enough M relative to Nd, this is far fewer round-trips. In practice, the N×N write/read cycles are eliminated entirely.

### 2.3 Memory complexity: O(N) activations

FlashAttention's forward pass stores only ([[flash-attention-1]]):

| Tensor | Shape | Cost |
|--------|-------|------|
| Output O | (N, d) | O(Nd) |
| Softmax row-max m | (N,) | O(N) |
| Softmax denominator l | (N,) | O(N) |
| Input Q, K, V | (N, d) each | O(Nd) — already accounted in activations |

The N×N score matrix S and probability matrix P are **never stored**. For the concrete example above (batch 32, 32 heads, N=2048, d=64) this drops the per-layer activation memory from 8 GB to near-zero for the attention weights — only the O(N) logsumexp statistics m and l (each a 32×32×2048 float tensor ≈ 8 MB total) are retained.

### 2.4 Backward pass: recomputation is cheaper than storage

The backward pass needs S and P to compute gradients. Rather than storing them (the 8 GB/layer cost), FlashAttention **recomputes** them from the stored Q, K, V, m, l ([[flash-attention-1]]):

```
# Backward: recompute S and P on-the-fly from stored (Q, K, V, l, m)
# Cost: ~33% extra FLOPs vs a hypothetical "store everything" backward
# Saving: 8 GB/layer of N×N buffers that would otherwise stay resident
#         in GPU memory across the entire forward pass
```

This is the same recomputation-vs-storage tradeoff as gradient checkpointing ([[ch-03]]), applied specifically to the attention weight tensors. The extra FLOPs are cheaper than the HBM traffic those stored tensors would incur.

### 2.5 Measured results (FA1)

- BERT-large (N=512): 15% end-to-end speedup over MLPerf 1.1 baseline
- GPT-2 (N=1024): **3× speedup**
- Long-range Arena (N=1K–4K): 2.4× speedup
- Block-sparse FA: enables 64K-token sequences; first transformers to achieve Path-256 (63.1% accuracy)

GPU MFU: 25–40% — the warp-level work partitioning is the bottleneck (fixed in FA2).

---

## 3. FlashAttention-2: Warp Work Partitioning (Dao, 2023)

FA1 reached only 25–40% MFU because different warps were racing to write shared memory for the softmax rescaling step: in FA1's "split-K" warp layout, 4 warps each own a K/V strip but must **synchronize** to compute the shared softmax denominator — this inter-warp communication costs shared-memory round-trips. FA2 fixes this structurally ([[flash-attention-2]]).

### 3.1 Three algorithmic improvements

**1. Fewer non-matmul FLOPs — Q-outer loop reordering**

FA1 iterates: `for each Q tile → for each K/V tile → rescale`. The rescaling (O(d) operations per tile pair) uses non-matmul instructions, which run at ~1/16th the throughput of tensor-core matmul on A100. FA2 restructures to batch rescaling across all K/V tiles per Q tile, cutting the number of non-matmul operations proportionally.

**2. Parallelism over sequence dimension**

FA1 couldn't parallelize within a single batch×head because causal masking made column strips data-dependent. FA2 assigns **disjoint row ranges of Q to different thread blocks** — each block is independent and runs in parallel, directly boosting SM occupancy, especially for small-batch or few-head configurations.

**3. Warp-per-output-row partitioning**

FA1: 4 warps each own K/V strips → requires inter-warp communication for the softmax denominator.  
FA2: each warp owns **a partition of Q output rows** → produces a complete output row without touching shared memory for softmax. Zero inter-warp communication for the softmax step.

### 3.2 Performance

| Metric | FA1 | FA2 | Ratio |
|--------|-----|-----|-------|
| A100 MFU (attention) | 25–40% | 50–73% | ~2× |
| Peak TFLOPs/s (A100) | ~110 | **225** | ~2× |
| Memory (activation) | O(N) | O(N) | same |
| Wall-clock vs FA1 | — | **~2×** | — |

The activation memory regime is unchanged from FA1: same O(N) logsumexp storage, same backward recomputation. The gain is **pure throughput** from better warp utilization. At fixed memory budget, 2× faster attention means 2× more tokens per hour, or 2× longer sequences within the same wall-clock step time.

---

## 4. FlashAttention-3: Hopper-Specific Async + FP8 (Shah et al., 2024)

FA2 is synchronous: each warp issues a GEMM, waits for it to complete, runs softmax, issues the next GEMM. On Hopper H100, this leaves the Tensor Memory Accelerator (TMA) async pipeline idle — the GPU can overlap data loading with computation, but FA2 never asks it to. FA3 restructures around a **producer-consumer warp specialization** ([[flash-attention-3]]).

### 4.1 Three Hopper-specific techniques

**1. Producer-consumer ping-pong pipeline (TMA async)**

```
# FA3 warp layout on H100
Producer warps:  drive TMA async loads → double-buffer tiles in shared memory
Consumer warps:  drive tensor-core GEMM (wgmma) on the already-loaded tile
# Overlap: while consumer runs GEMM on tile i, producer loads tile i+1
# Result: consumer never stalls waiting for data
```

This is only possible on Hopper because TMA is an H100-exclusive unit — there is no equivalent on A100 (FA2 design is correct for A100).

**2. Interleaved block-wise GEMM + softmax**

Within the consumer warp group, the two-GEMM attention sequence `(S = QKᵀ, then P·V)` is pipelined: softmax rescaling of the partial P happens while the next GEMM stage is issued to tensor cores. Softmax latency is hidden behind matmul throughput.

**3. FP8 block quantization + incoherent processing**

- **Per-block quantization:** quantize small tiles within each attention tile (not per-tensor), bounding the dynamic range each step must cover.
- **Incoherent processing:** apply a random Hadamard rotation before quantization, spreading outlier activations uniformly across all elements so the FP8 range is used efficiently.
- Result: **2.6× lower numerical error** vs naive per-tensor FP8 attention.

### 4.2 Measured performance on H100 SXM5

| Precision | TFLOPs/s | Utilization | vs FA2 |
|-----------|----------|-------------|--------|
| FP16 | **740** | 75% | ~2× |
| FP8 | **~1200** | ~75% | — |

FA2 on H100 achieves only ~370 TFLOPs/s (~40% utilization) because the synchronous kernel design cannot exploit TMA pipelining. FA3 nearly doubles utilization.

### 4.3 Training-memory profile of FA3

FA3 does **not** change the O(N) activation memory regime established by FA1/2 ([[flash-attention-3]]). The memory significance is at the bandwidth level:

- TMA delivers tiles to SRAM faster with less warp stall → SRAM is used more continuously → less HBM bandwidth wasted on stall-induced re-fetching.
- FP8 halves the bytes per attention tile vs FP16 → the HBM bandwidth consumed per attention tile is halved → longer sequences fit within the same HBM bandwidth budget per step.

The O(N) logsumexp storage and backward recomputation strategy are unchanged from FA1.

---

## 5. The Training-Memory Ledger Impact

The single biggest long-context training memory lever is the elimination of O(N²) attention activations. To make this concrete, consider a model with L=32 layers, B=8 batch, H=32 heads, N=8192, d=64, FP16:

```
# Naive attention — score matrix per layer per forward pass:
  B × H × N × N × 2 bytes = 8 × 32 × 8192 × 8192 × 2 = 34 GB per layer
  × 32 layers = 1.1 TB (impossible on any current GPU)

# FlashAttention — logsumexp statistics only:
  B × H × N × 2 (for m and l) × 4 bytes (fp32) = 8 × 32 × 8192 × 2 × 4 = 134 MB per layer
  × 32 layers = 4.3 GB
```

FlashAttention reduces the attention activation footprint from terabytes to gigabytes at N=8192. This is why FA is a prerequisite for any long-context fine-tuning or pretraining — not an optimization but a requirement.

The interaction with [[ch-03]] gradient checkpointing is additive: checkpointing saves the other activations (MLP, layer norms, residuals), while FA saves the attention scores. At N≥4096, FA's saving dominates even aggressive checkpointing.

---

## 6. FA1 → FA2 → FA3 Progression Summary

| Version | Key innovation | MFU (A100) | MFU (H100) | Memory | Hardware target |
|---------|---------------|-----------|-----------|--------|----------------|
| FA1 (2022) | Tiling + online softmax, O(N) memory | 25–40% | — | O(N) | Any CUDA GPU |
| FA2 (2023) | Warp-per-row, Q-outer loop, seq parallelism | 50–73% | 40% | O(N) | A100 (Ampere) |
| FA3 (2024) | TMA async, producer-consumer, FP8 | — | 75% FP16, ~75% FP8 | O(N) | H100 (Hopper) |

**What is invariant** (forced by the substrate):
- Online softmax recurrence — required by streaming, cannot be avoided without a second HBM pass
- O(N) activation footprint — the mathematical consequence of tiling; all three versions share this
- Backward recomputation from logsumexp — cheaper than storing N×N at any bandwidth regime

**What is variant** (free design choice that evolved across versions):
- Warp/thread-block work partition — FA1 used split-K (synchronization overhead); FA2 switched to split-Q (independent per-row)
- Data loading strategy — FA2 relies on software-managed prefetch; FA3 uses TMA hardware units (Hopper-only)
- Numeric precision — FA1/2 are FP16/BF16 only; FA3 adds FP8 with incoherent processing for error control

---

## 7. The Online Softmax Recurrence — Derivation Sketch

For readers who want to verify the recurrence is numerically stable (a key exam question):

Standard softmax of row x ∈ ℝᴺ:
```
softmax(x)_i = exp(x_i) / Σⱼ exp(x_j)
```

Split x into two halves x = [a; b]:
```
m_a = max(a)
m_b = max(b)
m   = max(m_a, m_b)

# Numerically stable: subtract global max before exp
softmax([a; b])_i = exp(x_i - m) / [Σⱼ∈a exp(aⱼ - m) + Σⱼ∈b exp(bⱼ - m)]
                  = exp(x_i - m) / [exp(m_a - m)·Σⱼ exp(aⱼ - m_a) + exp(m_b - m)·Σⱼ exp(bⱼ - m_b)]
```

This is the recurrence: maintain running (m, d) where `d = Σⱼ exp(xⱼ - m)`, and when new scores arrive, update both m and d. No second pass is needed because the rescaling factor `exp(m_old - m_new)` is exact and computed in O(1) per tile. The same O accumulation follows.

The original proof that this is possible for arbitrary splits appears in Rabe & Staats (2021) — [[ch-04]] covers this as the mathematical foundation. FA1 implements it as the first practical hardware-efficient realization.

---

## Core Insights from the Literature

**1. IO is the right cost model for attention** ([[flash-attention-1]]). Standard attention performs Θ(Nd + N²) HBM accesses; FA performs Θ(N²d²/M). The 8 GB/layer score matrix at N=2048 means standard attention is I/O-bound, not compute-bound — tensor core FLOPs are sitting idle waiting for data. Any optimization that reduces HBM traffic without reducing output correctness is a strict win.

**2. Warp-level work partitioning is where GPU utilization is won or lost** ([[flash-attention-2]]). FA1's 25–40% MFU is not a theoretical limit of the tiling algorithm — it is an implementation flaw. Reorganizing which warp owns which data (split-K → split-Q) nearly doubles throughput without changing the mathematical algorithm at all. This is a lesson that applies to kernel design generally: algorithmic correctness and hardware efficiency are separate problems.

**3. Hopper requires a new programming model, not just a faster chip** ([[flash-attention-3]]). H100 has 2× the FP16 FLOPs/s of A100, but FA2 on H100 achieves only ~370 TFLOPs/s vs FA3's 740 TFLOPs/s — because the synchronous kernel design cannot use TMA. Hardware generations often invalidate software patterns; a new kernel generation is required, not a parameter tune.

**4. Backward recomputation is strictly better than N×N storage at long sequences** ([[flash-attention-1]]). The HBM bandwidth cost of storing and re-reading the N×N matrix dwarfs the FLOPs cost of recomputing it from Q, K, V, l, m. This is the principle behind [[ch-03]]'s gradient checkpointing applied at the finest granularity: single-layer, single-head attention weight tensors.

---

## Key Takeaways

- FlashAttention's speedup comes from **HBM read/write elimination**, not FLOP reduction. The N×N score matrix is the target.
- The **online softmax recurrence** (maintain running m, d, O across tiles) is the mathematical primitive that makes single-pass tiling exact — without it, a second HBM pass is unavoidable.
- All three versions share **O(N) activation memory** and **backward recomputation** from (l, m). What changed across versions is warp partitioning (FA2) and async pipelining + FP8 (FA3).
- At N=8192, naive attention needs ~34 GB/layer just for score matrices; FA reduces this to ~134 MB/layer — a 250× reduction. This is the **primary long-context training memory lever**, dominating even aggressive gradient checkpointing.
- FA2 on A100: **225 TFLOPs/s, 50–73% MFU**. FA3 on H100: **740 TFLOPs/s (FP16), ~1.2 PFLOPs/s (FP8), 75% MFU**.
- Silent fallback to the `MATH` backend in PyTorch SDPA allocates the full O(N²) score matrix and runs ~38× slower — audit this in every training setup.

---

## References

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. https://arxiv.org/abs/2205.14135 ([[flash-attention-1]])
- Tri Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. https://arxiv.org/abs/2307.08691 ([[flash-attention-2]])
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608, July 2024. https://arxiv.org/abs/2407.08608 ([[flash-attention-3]])

**Related chapters:** [[ch-03]] (gradient checkpointing — the same recompute-vs-store tradeoff), [[ch-04]] (online softmax math + Rabe & Staats O(N) theory), [[ch-06]] (the broader attention kernel zoo: SDPA backends, xFormers, SageAttention, Ring Attention, PagedAttention)
