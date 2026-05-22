<!-- chapter: ch-11
     phase: kernels-runtime
     title: FlashAttention Lineage (1/2/3 + FlashDecoding + FlashInfer)
     sources: [[flashattention]], [[flashattention-2]], [[flashattention-3]], [[flashdecoding]], [[flashinfer]], [[xformers-memory-efficient-attention]]
     back: [[attention-complexity]] (ch-02)
     forward: ch-12 CUDA Graphs, ch-14 speculative decoding (kernel reuse)
-->

# Chapter 11 — FlashAttention Lineage (1/2/3 + FlashDecoding + FlashInfer)

> **Core insight.** Standard attention is *FLOP-efficient* but *memory-traffic-inefficient*: the L×L score matrix is too big for SRAM, so it spills to HBM and the kernel becomes bandwidth-bound. FlashAttention reframes attention as an **IO problem**: tile Q, K, V; stream blocks through SRAM; never materialize the L×L matrix in HBM. The result is *exact* attention with linear memory and 2–5× wall-clock speedup. The four-year lineage (FA1 → FA2 → FA3 → FlashDecoding → FlashInfer) is one design idea reapplied as the hardware substrate and the workload shifted: HBM bandwidth (FA1), GPU occupancy (FA2), Hopper async/FP8 (FA3), decode parallelism (FlashDecoding), serving-format flexibility (FlashInfer).
>
> **Guideline.** For Ampere training/prefill, use FA2. For Hopper training/prefill, use FA3. For long-context decode, use FlashDecoding (split-K). For serving with paged/ragged KV across model variants, use FlashInfer as the dispatch layer. Never use the naive PyTorch `F.scaled_dot_product_attention` fallback for long context — at 32k it's 30× slower than FA2 and uses 100× more memory.

---

## Why this chapter exists

Attention is the operator that defines transformer cost. From ch-02 ([[attention-complexity]]) we have:

- prefill: O(L²·d) compute, O(L²) intermediate memory (the scores matrix)
- decode: O(L·d) compute per token, O(L·d) intermediate memory (one row of scores)

The compute scales gracefully. The intermediate memory does not. A 32k-token sequence at fp16 produces a 2 GB scores matrix — bigger than the model's per-layer activation for a 70B model. Materializing that matrix to HBM, then reading it back for softmax, then reading it back again for the V multiply, is **three HBM round-trips** for data that is touched once.

GPUs are bandwidth-starved for this access pattern. An H100 has ~3 TB/s HBM bandwidth and ~67 TFLOPS BF16 — a 22:1 ratio. Attention as written (materialize → softmax → matmul) is throttled by HBM, not by the tensor cores. The tensor cores sit at <20% utilization.

FlashAttention (Dao 2022) is the kernel that fixes this. Its descendants — FA2, FA3, FlashDecoding, FlashInfer — are four years of follow-on engineering: keep the IO-aware framing, retarget for the next hardware bottleneck.

This chapter walks the lineage in order.

---

## 1. Standard attention — why it's bandwidth-bound

The reference implementation:

```python
# inputs: Q, K, V each [B, H, L, d]
S = Q @ K.transpose(-2, -1) / sqrt(d)   # [B, H, L, L]  ← materialized in HBM
P = softmax(S, dim=-1)                  # [B, H, L, L]  ← materialized in HBM
O = P @ V                               # [B, H, L, d]  ← output
```

Memory traffic per attention layer (single head, batch 1, ignoring constants):

| Step | HBM read | HBM write |
|---|---|---|
| `S = QK^T` | Q (L·d) + K (L·d) | S (L²) |
| `P = softmax(S)` | S (L²) | P (L²) |
| `O = PV` | P (L²) + V (L·d) | O (L·d) |
| **Total** | **3L² + 3L·d** | **2L² + L·d** |

The L² terms dominate for L ≥ 1024. At L=8192, d=128, fp16:

- HBM reads: 3·(8192²) · 2 = 384 MiB per head
- with 32 heads, per layer = 12 GiB, per 80-layer model = 1 TiB of HBM traffic for one prefill pass

At 3 TB/s that's ~330 ms in HBM alone — and 70 TFLOPS × ~ms means tensor cores would do the math in <10 ms. The kernel is **97% idle, waiting on HBM**.

This is the gap FlashAttention closes.

---

## 2. FlashAttention 1 (Dao 2022) — IO-aware tiling + online softmax

[[flashattention]] (Dao, Fu, Ermon, Rudra, Re, NeurIPS 2022, arXiv 2205.14135) introduces three ideas:

### (a) Tile Q, K, V; never materialize S in HBM

Partition Q into blocks of size `B_r × d` rows, and K, V into blocks of size `B_c × d`. Choose `B_r`, `B_c` so that one (Q-block, K-block, V-block, partial S, partial O) fits in SRAM (~100 KB per SM on A100).

```text
For each block of Q (rows i):
    Load Q_i into SRAM
    Initialize O_i = 0, m_i = -inf, ℓ_i = 0  (output, running max, running sum)
    For each block of K, V (cols j):
        Load K_j, V_j into SRAM
        Compute S_ij = Q_i @ K_j^T / sqrt(d)         # in SRAM
        Compute block softmax with online recurrence (see (b))
        Accumulate O_i += P_ij @ V_j                  # in SRAM
    Write O_i to HBM
```

The L×L scores matrix is never written to HBM. Each Q-block streams through every K-block, accumulating the output incrementally.

### (b) Online softmax — exact, numerically stable

The hard part: softmax requires the global max and sum over the entire row of S. With tiling, you only see one block of S at a time. The recurrence:

```text
For each new block j:
    m_new = max(m_old, max(S_ij))
    ℓ_new = exp(m_old - m_new) · ℓ_old + sum(exp(S_ij - m_new))
    O_new = (exp(m_old - m_new) · ℓ_old / ℓ_new) · O_old
          + (exp(S_ij - m_new) / ℓ_new) · V_j
```

Each step **rescales** the running output and normalizer by `exp(m_old − m_new)` whenever a new max appears. Mathematically equivalent to global softmax; numerically as stable as the standard log-sum-exp trick.

### (c) IO complexity analysis

Standard attention: **Θ(L²·d + L·d)** HBM accesses.
FlashAttention with SRAM size M: **Θ(L²·d² / M)** HBM accesses.

For typical (L=4096, d=128, M=100 KB): FlashAttention does ~9× fewer HBM transfers. **The kernel becomes compute-bound.**

### (d) Headline numbers

- **GPT-2 training**: 3× faster than the cuDNN baseline; 2× faster than the Megatron-LM kernel.
- **BERT-large MLM**: 15% wall-clock training speedup.
- **Long Range Arena** (path-X, L=16k): exact attention now fits in memory; previously required approximate kernels.
- **Memory**: linear in L instead of quadratic — enables 64k context training on a single A100 for the first time.

The paper also introduces **block-sparse FlashAttention** for approximate attention (skip blocks below a mask), but dense FA1 is the more important contribution.

See [[excerpts/flashattention]] for the online softmax recurrence and the IO complexity proof.

---

## 3. FlashAttention 2 (Dao 2023) — better parallelism, fewer non-matmul FLOPs

[[flashattention-2]] (Dao, arXiv 2307.08691) is the "make FA1 actually approach GEMM efficiency" follow-on. Three improvements:

### (a) Lower non-matmul FLOPs

FA1's online-softmax rescaling does ~3× more non-matmul work than necessary. The reordering (move the `1/ℓ_new` division outside the loop, only rescale once per output) cuts non-matmul ops by 2×. Non-matmul ops are slow because they don't use tensor cores — every saved non-matmul op is a real wall-clock win.

### (b) Parallelize across sequence-length (not just batch × head)

FA1 parallelizes one thread block per `(batch, head)`. At small batch or long sequence, this leaves SMs idle: a 4-GPU H100 cluster running batch=1, head=32 uses 32 thread blocks across 432 SMs — 92% idle.

FA2 also parallelizes across **the sequence dimension** of Q. Each thread block now handles a sub-range of Q's rows. Occupancy is restored at long context.

### (c) Better warp-level work partitioning

FA1 has all warps in a block do similar work, requiring frequent shared-memory communication. FA2 specializes warps:

- Some warps load Q tiles.
- Some warps load K/V tiles.
- Some warps do the matmul + softmax math.
- Communication via fewer shared-memory barriers.

### Headline numbers

- **2× faster than FA1** on A100 (50–73% of FP16 peak GEMM throughput).
- **Up to 225 TFLOPS/sec on A100** — the first attention kernel to come close to GEMM peak.
- Supports head dimensions up to 256 (required for newer models like Llama-3-8B's d=128 GQA).
- Supports MQA and GQA natively — important for production models where `kv_heads < n_heads`.

### Why FA2 is the production default on Ampere

FA2 is what `xformers.ops.memory_efficient_attention` and PyTorch's `F.scaled_dot_product_attention` dispatch to on Ampere (A100/A10/A40/RTX 4090). Most serving stacks (vLLM, SGLang, TGI) use it for prefill.

See [[excerpts/flashattention-2]] for the work-partitioning diagrams and the GEMM-peak measurement.

---

## 4. FlashAttention 3 (Shah et al. 2024) — Hopper async, WGMMA, FP8

[[flashattention-3]] (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, arXiv 2407.08608) retargets the kernel for Hopper (H100/H200). Hopper has four hardware features FA2 doesn't exploit:

| Hopper feature | What it does | FA2 status |
|---|---|---|
| **TMA** (Tensor Memory Accelerator) | async HBM↔SRAM with one instruction | unused (uses sync `ldmatrix`) |
| **WGMMA** (Warp Group MMA) | 256-thread warp groups for matmul, async | unused (uses `mma.sync`) |
| **FP8 tensor cores** | 2× the FLOPS of FP16 | unused (FP16 only) |
| **Distributed shared memory** | SMs can read each other's SRAM | unused |

FA3 rebuilds the kernel around all four.

### (a) Producer-consumer warp specialization

```text
Warp group 0 (producer): TMA loads K, V tiles asynchronously
Warp group 1 (consumer): WGMMA computes S = QK^T and P @ V
Warp group 2 (consumer): softmax + correction math
```

Producer and consumer overlap via shared-memory mbarriers. **The kernel hides the entire HBM latency behind tensor-core math.**

### (b) Matmul-softmax interleaving

The non-matmul softmax work is FA2's remaining bottleneck. FA3 issues the *next* block's matmul (WGMMA, async) while the *current* block's softmax is still running. WGMMA is non-blocking; softmax fills the gap.

### (c) FP8 with block scaling

FP8 attention naively loses ~1 bit of accuracy because the per-tile dynamic range is large. FA3 applies:

- **Block quantization**: scale per tile of Q and K, not per tensor.
- **Incoherent processing**: random Hadamard rotation pre-FP8-cast smears outliers across the tile.

Result: FP8 attention with **error close to FP16** rather than the ~2× error of naive FP8 cast.

### Headline numbers

- **1.5–2.0× faster than FA2 on H100** in FP16 (740 TFLOPS, ~75% of peak).
- **~1.2 PFLOPS in FP8** — first attention kernel to break the petaflop mark on a single GPU.
- ~2.6× lower numerical error than baseline FP8 attention.

### Why FA3 matters for serving

H100 deployments running FA2 leave half the GPU on the table. FA3 reclaims that. For chat workloads (short prefill, lots of decode), the prefill speedup is less load-bearing; for long-context (RAG, document QA), FA3 in prefill cuts TTFT roughly in half on H100.

See [[excerpts/flashattention-3]] for the warp-specialization diagram and the FP8 accuracy results.

---

## 5. FlashDecoding — the decode case, where FA2/3 alone underutilize

[[flashdecoding]] (Dao, Haziza, Massa, Sizov, 2023, PyTorch blog) addresses a different problem: **decode-time attention with batch size 1 and query length 1**.

At decode, Q is one token per sequence. FA2 parallelizes one thread block per `(batch, head)`. At batch=1, head=32, that's **32 thread blocks** for a GPU with 132 SMs (H100). 76% of the SMs idle.

This is the killer for **long-context single-sequence decode** — exactly the chat-with-long-history workload that's most common in production.

### The split-K trick

FlashDecoding partitions the KV cache (the K and V matrices, length `L_kv`) into chunks along the sequence dimension. Each chunk gets its own thread block:

```text
For each chunk k of K, V (length L_chunk):
    Compute partial output O_k and log-sum-exp lse_k = log(sum(exp(S_k)))
After all chunks finish:
    Combine: O = sum_k(exp(lse_k - lse_global) · O_k) where
             lse_global = logsumexp(lse_k for all k)
```

The combination step uses the same online-softmax identity as FA1: rescale partial outputs by their relative max. Mathematically exact; computationally distributable across many SMs.

### Numbers

- **Up to 8× faster** generation at very long context (>32k) on long-CoT decode.
- Scales linearly with KV length while batch is small; degenerates to FA2 once batch is large enough to fill the GPU.

### When to use FlashDecoding vs FA2 at decode

| Decode workload | Best kernel |
|---|---|
| batch >= 32, any context | FA2 / FA3 (already saturates SMs) |
| batch 1–8, context >= 8k | **FlashDecoding** |
| batch 1–8, context < 1k | FA2 (SM saturation matters less) |
| MQA/GQA single sequence | FlashDecoding (KV bandwidth-bound) |

Production serving stacks dispatch automatically: vLLM uses FA2 or FlashDecoding based on `(batch_size, kv_length)`. FlashInfer (next section) generalizes this dispatch.

See [[excerpts/flashdecoding]] for the split-K formula and the SM-occupancy measurement.

---

## 6. FlashInfer — the serving-aware kernel library

[[flashinfer]] (Ye et al., MLSys 2025, arXiv 2501.01005) is the kernel library production serving frameworks actually call. It's not a single kernel; it's a **dispatch + JIT engine** that picks the right attention variant for the serving system's actual state.

### Why a serving-aware library is needed

Production attention has variants FA2/FA3/FlashDecoding don't all cover natively:

1. **Paged KV cache** ([[pagedattention]], ch-06): KV blocks are non-contiguous; the kernel must walk a block table.
2. **Ragged batches**: requests in the batch have different sequence lengths; padding wastes compute.
3. **GQA / MQA**: kv_heads < n_heads; one K/V serves many Qs.
4. **Cascade attention**: shared prefix attention computed once, then per-request suffix attention combined.
5. **Custom score modifiers**: ALiBi bias, attention sinks, query-aware pruning.
6. **CUDA graph compatibility** (ch-12): kernel shapes must be static for graph capture.

FlashInfer provides all of these as **composable operators with JIT specialization**: at engine init, it compiles the exact kernel variant for `(model architecture, attention config, KV layout)`.

### Three core APIs

```python
# Prefill: full attention over a prompt batch
flashinfer.prefill.batch_prefill_with_paged_kv_cache(
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
    causal=True, sm_scale=1/sqrt(d), pos_encoding="rope"
)

# Decode: one-token-per-sequence attention with paged KV
flashinfer.decode.batch_decode_with_paged_kv_cache(
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len,
    sm_scale=1/sqrt(d)
)

# Cascade: shared-prefix attention combined with per-request suffix
flashinfer.cascade.merge_states([prefix_state, suffix_state])
```

`kv_indptr / kv_indices` is the paged-KV addressing structure from PagedAttention ([[pagedattention]]).

### CUDA-graph-compatible scheduling

The serving engine pre-creates kernel plans for likely `(batch, max_kv_len)` shapes and reuses them across iterations. This lets the engine wrap attention in a CUDA graph (ch-12) without recompiling per iteration.

### Integration

- **vLLM**: FlashInfer is one attention backend (set `VLLM_ATTENTION_BACKEND=FLASHINFER`); the other is `FLASH_ATTN` (direct FA2/3).
- **SGLang**: FlashInfer is the default attention backend.
- **MLC-Engine**: uses FlashInfer.

For the learner: FlashInfer is where the FlashAttention lineage **meets the serving system's actual constraints** (paged KV, dynamic batches, CUDA graphs). FA1/2/3 are kernels; FlashInfer is a kernel *engine*.

See [[excerpts/flashinfer]] for the paged-attention API and the JIT-specialization mechanism.

---

## 7. xFormers — the operator-dispatch library

[[xformers-memory-efficient-attention]] (Meta, 2021+; Rabe & Staats 2021 for the math) provides `xformers.ops.memory_efficient_attention`. It's a **PyTorch-level dispatch wrapper** over multiple backends (CUTLASS, FlashAttention, custom kernels).

Why it matters: xFormers was the first production-quality wrapper around memory-efficient attention. It pre-dates the FA1 paper and uses an O(n)-memory algorithm of similar shape. It dispatches to FA2 on Ampere when available, FA3 on Hopper, and falls back to a CUTLASS kernel otherwise.

For inference: xFormers is convenient for *model code* (`F.scaled_dot_product_attention` in modern PyTorch dispatches similarly), but it's a layer below serving — it doesn't handle paged KV or ragged batches. Serving engines wrap attention with FlashInfer (or call FA directly), not xFormers.

---

## 8. The lineage as one chart

```text
                Standard attention (Vaswani 2017)
                            │ HBM-bound, L² intermediate
                            ▼
              FA1 (Dao 2022): IO-aware tiling + online softmax
                            │ exact, linear memory, 3× speedup
                            ▼
              FA2 (Dao 2023): seq-dim parallel + less non-matmul
                            │ 2× FA1, ~75% GEMM peak on A100
                            ▼
              FA3 (Shah 2024): Hopper async + WGMMA + FP8
                            │ 1.5–2× FA2 on H100, ~1.2 PFLOPS in FP8
                            │
            ┌───────────────┼──────────────────────┐
            ▼               ▼                       ▼
     FlashDecoding   FlashInfer (Ye 2025)    xFormers (Meta 2021)
     (Dao 2023)      paged + ragged +        operator dispatch
     split-K decode  cascade + CUDA-graph    over backends
     8× at long ctx  serving-engine native
```

The intellectual through-line: **the L² intermediate must not touch HBM**. Every kernel in the lineage preserves that property; they differ in how the SRAM-resident computation is scheduled onto the hardware.

---

## 9. Choosing a kernel — production decision table

| Workload | Best kernel | Notes |
|---|---|---|
| Hopper prefill, FP16/BF16 | FA3 | 1.5–2× FA2 |
| Hopper prefill, FP8 | FA3 FP8 | NVFP4 hardware will extend this |
| Ampere prefill | FA2 | the production default since 2023 |
| Decode, batch ≥ 32 | FA2/FA3 | SM-saturating already |
| Decode, batch < 8, long context | FlashDecoding | the split-K case |
| Serving engine with paged KV | FlashInfer | dispatches to FA backends |
| Serving with shared prefixes | FlashInfer cascade | one shared-prefix kernel |
| Model definition code | `F.scaled_dot_product_attention` | PyTorch native dispatch to FA |
| Diffusion / non-LLM | xFormers MEA | mature, well-tested |
| Pre-Ampere GPU | xFormers CUTLASS backend | FA needs Ampere+ |

The rule of thumb: use **FlashInfer** if you're building a serving engine; use **FA2/3 directly** if you're writing a model trainer; use **xFormers** or PyTorch SDPA if you want one line of code and don't care about serving-specific features.

---

## 10. What this lineage enables downstream

- **Long context (ch-08)**: FA1 made 32k+ context feasible; FA2 made it fast; FlashDecoding made it interactive at decode.
- **Speculative decoding (ch-14, ch-15)**: the draft + target verification step reuses the same attention kernels — FlashInfer's cascade attention is optimized for it.
- **Paged KV (ch-06)**: PagedAttention's contribution is the *block-table layout*; FlashInfer is what makes it kernel-efficient.
- **CUDA graphs (ch-12, next chapter)**: FlashInfer's static-shape design is a precondition for graph capture in serving.
- **MoE inference (ch-13)**: MoE attention is the same kernel — the routing happens in the FFN; attention is one of the few unchanged parts.

---

## Connections and what's next

- **Back to [[attention-complexity]] (ch-02)** — the O(L²·d) compute and O(L²) memory that FA flattens.
- **Back to [[kv-cache-memory-formula]] (ch-03)** — KV cache size is what FlashDecoding's split-K parallelizes over.
- **Back to [[pagedattention]] (ch-06)** — paged KV is FlashInfer's primary input format.
- **Forward to [[cuda-graphs-inference]] (ch-12)** — FlashInfer is designed for CUDA-graph capture; the two compose.
- **Forward to ch-14 / ch-15** — speculative decoding reuses these kernels; FlashInfer cascade attention is the spec-dec-optimized variant.

## Further reading

- [[flashattention]] — Dao et al. 2022; the foundational IO-aware paper.
- [[flashattention-2]] — Dao 2023; the GEMM-peak follow-on.
- [[flashattention-3]] — Shah et al. 2024; the Hopper retargeting.
- [[flashdecoding]] — Dao et al. 2023; split-K for decode.
- [[flashinfer]] — Ye et al. 2025; serving-aware kernel engine.
- [[xformers-memory-efficient-attention]] — Meta 2021+; the operator dispatch library.

## Companion visualization

**[figures/flashattention-tiling.html](figures/flashattention-tiling.html)** — interactive walkthrough of the tile-and-stream pattern, the online-softmax recurrence (with a slider for tile size), and a side-by-side HBM traffic comparison vs standard attention.
