<!-- chapter: ch-06
     track: attention
     title: The Attention Kernel Zoo: SDPA, xFormers, SageAttention, Ring, Paged
     deps: [[ch-05]]
     sources: [[pytorch-sdpa]], [[xformers-mem-efficient]], [[sage-attention]], [[ring-attention]], [[paged-attention]]
-->

# Chapter 6 — The Attention Kernel Zoo: SDPA, xFormers, SageAttention, Ring, Paged

> **Core insight.** The O(N) memory regime established by FlashAttention in [[ch-05]] is not a single fixed path: PyTorch `scaled_dot_product_attention` is a runtime dispatcher to four distinct backends, each with different hardware constraints, and silent fallback to the math backend when constraints are unmet is the most common source of unexpected OOM in codebases that believe they have enabled FlashAttention. Beyond that dispatcher, the kernel family branches into quantized inference acceleration (SageAttention: INT8 QKᵀ), cross-device sequence sharding (Ring Attention: O(L/D) per GPU), and inference-serving KV-cache management (PagedAttention: paged physical blocks) — techniques that solve fundamentally different problems, only some of which are relevant to training memory.

> **Guideline.** In a training loop: pin a backend explicitly with `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` and handle the `RuntimeError` rather than relying on silent dispatch; use Ring Attention (context parallelism, `--context-parallel-size D`) when even O(N) per-device KV memory overflows HBM; treat SageAttention and PagedAttention as inference tools and understand why their memory tricks do not transfer to training. When the model forbids standard attention entirely — such as a GDN linear-attention MoE — the entire kernel zoo becomes moot and the activation memory problem requires a different analysis (see [[ch-09]] capstone hook).

---

## 1. The Substrate: Why a Kernel Zoo Exists at All

[[ch-04]] established that attention's n×n score matrix is the dominant activation-memory consumer at long sequences. [[ch-05]] showed that FlashAttention tiles computation in SRAM and recomputes the score matrix on backward, reducing activation memory from O(N²) to O(N). But a single kernel does not cover all hardware, all dtypes, and all attention bias patterns. The result is a zoo of implementations that all target the same mathematical operation — softmax(QKᵀ/√d)V — but differ in:

- Which hardware generations and dtypes they support
- What constraint violations cause them to refuse (and what happens when they do)
- Whether they work during training (need backward pass) or only during inference
- Whether they shard computation across devices (require a ring topology)

The governing tradeoff: a more capable kernel makes stronger assumptions about the input. When those assumptions are violated, either the kernel fails gracefully (raising an error) or silently degrades (falling back to the 38× slower math backend). The most dangerous outcome is silent degradation with no error message.

> **Interactive companion:** [figures/kernel-memory.html](figures/kernel-memory.html) — a side-by-side visualization of per-device memory cost vs. sequence length for all five kernels, with toggleable configuration (batch size, head dim, device count for Ring Attention), showing precisely where each kernel's O(N²) vs O(N) vs O(L/D) curves diverge.

---

## 2. PyTorch SDPA: The Dispatcher — [[pytorch-sdpa]]

**What it is.** `torch.nn.functional.scaled_dot_product_attention` (SDPA) is not itself an attention kernel. It is a dispatch layer that selects one of four backends at runtime based on hardware, dtype, head dimension, and input properties. Every Hugging Face transformer that calls `F.scaled_dot_product_attention` runs through this dispatcher.

**The four backends** ([[pytorch-sdpa]]):

| Backend | Algorithm | Memory | Constraints |
|---|---|---|---|
| `MATH` | Pure PyTorch C++; materializes full N×N score matrix | O(N²) | Always available |
| `FLASH_ATTENTION` | FlashAttention-2 kernel | O(N) | CUDA, fp16/bf16, head_dim ≤ 128, no arbitrary bias |
| `EFFICIENT_ATTENTION` | xFormers CUTLASS FMHA (Rabe & Staats O(N)) | O(N) | Broader: custom bias, larger head_dim |
| `CUDNN_ATTENTION` | cuDNN SDPA graph; autograd-compatible | O(N) | Requires supported cuDNN + CUDA versions |

**Performance gap.** The [[pytorch-sdpa]] excerpt gives a concrete benchmark: ~87,478 µs for MATH vs. ~2,274 µs for optimized backends on identical inputs. That is a ~38× difference, attributable entirely to HBM bandwidth cost of the N×N score matrix. In memory terms, at n=4096, batch=16, 32 heads the math backend allocates ~8 GB of attention activations per layer; FlashAttention reduces this to ~200 MB — an order-of-magnitude regression with no warning if the fallback triggers silently.

**The silent fallback trap.** Without explicit backend selection, any of the following causes PyTorch to silently revert to MATH with no error or warning:
- Unsupported dtypes (fp32, int8)
- head_dim > 128 (FlashAttention constraint)
- Attention bias tensors with non-standard shapes
- `is_causal=True` combined with explicit mask tensors in some PyTorch versions
- Odd sequence lengths in certain cuDNN configurations

The right pattern ([[pytorch-sdpa]]):

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Explicit: raises RuntimeError if flash unavailable — the safe failure mode
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Raising a `RuntimeError` when the backend is unavailable is the safe outcome. The dangerous outcome is the default dispatch silently falling through to MATH. Every production training codebase should pin the backend and handle the error explicitly. Relying on "I set use_flash_attention=True somewhere" without verifying the dispatcher is not sufficient.

**Connection to [[ch-05]].** SDPA's FLASH_ATTENTION backend is the route through which FA2's O(N) regime reaches user code. The EFFICIENT_ATTENTION backend is the xFormers route, discussed next. CUDNN_ATTENTION wraps the cuDNN graph API. All three deliver O(N) activation memory; MATH is the O(N²) trap.

---

## 3. xFormers memory_efficient_attention — [[xformers-mem-efficient]]

**What it is.** Meta's production implementation of the Rabe & Staats (2021) O(N) streaming attention algorithm, fused into a CUTLASS FMHA kernel. It is PyTorch SDPA's `EFFICIENT_ATTENTION` backend — the fallback O(N) path when FlashAttention's hardware constraints are not met.

**The algorithm.** [[self-attention-no-n2-memory]] (Rabe & Staats 2021, arXiv:2112.05682) proved three exact memory variants:

- **O(1) per-query**: single outer loop over Q rows, streaming K/V inner loop; accumulate weighted values and running normalizer in SRAM only — never writes the N×N score matrix to HBM.
- **O(log N) for full self-attention**: the theoretical minimum for arbitrary self-attention.
- **O(√N) practical accelerator variant**: chunks both Q and K at size √N, exploiting GPU/TPU tile parallelism while keeping tiles in SRAM.

At N=16,384, Rabe & Staats measured 59× inference memory reduction and 32× backpropagation memory reduction vs. standard attention ([[self-attention-no-n2-memory]]). The xFormers CUTLASS FMHA implements this streaming algorithm with hardware-specific dispatch: Ampere (sm80) CUTLASS kernel, FlashAttention-3 on Hopper via `_get_use_fa3` toggle, Composable Kernel for AMD ROCm ([[xformers-mem-efficient]]).

**Why it exists alongside FA2.** The two kernels differ in coverage vs. peak throughput:

- FA2/FA3: higher throughput on A100/H100 for standard configurations (fp16/bf16, head_dim ≤ 128, standard causal mask, ~60–70% of theoretical peak on A100).
- xFormers FMHA: broader support — arbitrary attention bias tensors (per-head ALiBi, relative position biases, `BlockDiagonalMask`, `PagedBlockDiagonalGappyKeysMask`), larger head_dim, older hardware. On A100 runs at roughly 60–70% of FA2 speed for standard causal attention, but is the correct choice when FA2 cannot be used since the only alternative is the 38× slower MATH backend ([[xformers-mem-efficient]]).

**Training-memory angle.** Both xFormers FMHA and FA2 deliver O(N) activation memory — the same regime change from [[ch-05]]. Choosing between them is a throughput question, not a memory question. The practical rule: if FA2 is available and your configuration satisfies its constraints, prefer it; otherwise fall through to EFFICIENT_ATTENTION, which is far preferable to MATH.

---

## 4. SageAttention — [[sage-attention]]

**What it is.** A quantized attention kernel (Jintao Zhang et al., ICLR 2025, arXiv:2410.02367) that replaces the two attention matrix multiplications with mixed-precision operations to cut memory bandwidth and improve throughput. **Inference only** — no backward-pass implementation exists.

**Quantization scheme** ([[sage-attention]]). The two matrix multiplications in attention have different numerical properties:

- **QKᵀ matmul**: quantize Q and K to **INT8** before the multiply. The paper shows INT8 accuracy dominates FP8 (both E4M3 and E5M2) for this operation, and INT8 matmul is 2× faster than FP8 matmul on RTX4090/RTX3090. The catch: K has per-channel outliers that would ruin naive INT8 quantization.
- **P·V matmul**: keep P (the softmax output) and V in **FP16** with FP16 accumulators. FP16 is "2× faster than INT8 for this stage while preserving accuracy" — so this matmul is *not* quantized.

**K channel smoothing.** The key insight for making INT8 work on K. K has "distinct channel-wise outliers" — each token's key vector is a large shared bias plus a small token-wise signal. Apply:

```
γ(K) = K − mean(K)    # mean across the token dimension
```

This does not change attention output because softmax is invariant to uniform additive shifts in the logits:

```
σ(q · (K − mean(K))ᵀ) = σ(q · Kᵀ)
```

The shared bias cancels in softmax. Overhead: <0.2% of runtime. After smoothing, the quantization range of K is dramatically tighter, making INT8 accurate ([[sage-attention]]).

**Measured results on RTX4090** ([[sage-attention]]):
- 341 TOPS (52% of theoretical INT8 peak)
- **2.1× vs. FlashAttention2**, **2.7× vs. xFormers**
- Cosine similarity vs. full precision: 1.0; Relative L1: 0.019

SageAttention2 (arXiv:2411.10958) extends to INT4 with per-warp quantization and Smooth Q/Smooth V centering, achieving **3.1× vs. FA2** at RTX4090.

**Why this is inference-only — and why that matters for training.** SageAttention has no backward pass. INT8 quantization introduces small errors in the forward pass (Relative L1 ≈ 0.019) that are acceptable for inference but would accumulate and bias gradients during training. More fundamentally, the training memory problem is about storing activations for the backward pass — the N×N score matrix. SageAttention's quantization reduces memory bandwidth in the forward pass but does not eliminate the backward-pass activation storage need. FA1/FA2/FA3's recomputation trick (storing only logsumexp, recomputing scores on backward) is orthogonal and solves the training problem; SageAttention's INT8 trick solves the inference throughput problem ([[sage-attention]]).

**The right mental model:** SageAttention is a faster attention *evaluation* at the cost of ~2% numerical error. FlashAttention is an exactly equivalent attention *algorithm* that avoids N×N materialization. They operate at different levels of the stack.

---

## 5. Ring Attention / Context Parallelism — [[ring-attention]]

**What it is.** A multi-device attention algorithm (Hao Liu, Matei Zaharia, Pieter Abbeel, ICLR 2024, arXiv:2310.01889) that shards the sequence across D devices arranged in a logical ring, enabling sequences D times longer than single-device FlashAttention can support. This is the algorithmic foundation for what production frameworks call "context parallelism" (CP).

**The problem it solves.** [[ch-05]]'s FlashAttention eliminates the N×N score matrix, giving O(N) per-device KV activation memory. But when N is very large (e.g., 128k or 1M tokens), even O(N · d) per-device memory can exceed 40 GB HBM. FlashAttention does not help here — it has already eliminated all the redundant N×N storage. Ring Attention addresses this wall.

**Ring topology and communication** ([[ring-attention]]). D devices are arranged in a logical ring. Each device owns a contiguous slice of the sequence, length L/D. Each "round" of the algorithm:

1. Device i computes blockwise attention between its local Q slice and the K/V slice it currently holds.
2. Simultaneously, device i sends its K/V slice to device i+1 in the ring while receiving the prior device's K/V slice from device i−1.
3. After D rounds, every query has attended to every key in the full sequence.

The communication (K/V ring rotation) is pipelined behind the GEMM compute. Since compute time ≥ communication time at typical head dimensions, the rotation adds zero latency to the critical path: "no additional communication and computation overheads" ([[ring-attention]]).

**Memory scaling.** Per-device KV activation memory is O(L/D · d) — constant in L for fixed D. Total achievable context length scales as L ∝ D. The paper demonstrates "up to device count times longer" sequences than single-device memory-efficient transformers, at "millions of tokens context size." Exactness is preserved: D rounds complete so every query attends to every key; the blockwise softmax accumulation uses the same online normalization recurrence as FlashAttention ([[ring-attention]]).

**Connection to Megatron-LM CP.** Ring Attention is the algorithmic basis for what later production frameworks call "context parallelism." Megatron-LM's `--context-parallel-size` flag implements the same ring-KV communication pattern. In the world-size factorization covered in [[ch-07]]:

```
world_size = TP × PP × CP × DP
```

CP > 1 means Ring Attention is active. Each increase of CP by 2× halves the per-device KV memory at the cost of D ring-rotation communication rounds.

**When to use.** Ring Attention is the right lever when: sequence length is so large that even O(N · d) per-device KV memory overflows HBM after FlashAttention is already enabled. At that point, the only options without approximation are: reduce batch size (which may make training inefficient), or shard the sequence with CP. The tradeoff is all-to-all communication overhead for the ring rotation, which is hidden behind compute at standard head dimensions but becomes visible at very small head dims.

**Capstone hook.** The [[ch-09]] capstone involves a GDN linear-attention MoE. Linear attention replaces the softmax kernel with an associative kernel that enables O(N) time and O(d) memory natively — making the entire ring-attention discussion irrelevant for that architecture. When linear attention is used, CP provides no benefit because the KV memory is already O(d) not O(N). The capstone must handle this distinction.

---

## 6. PagedAttention — [[paged-attention]]

**What it is.** A KV-cache memory management scheme for inference serving (Woosuk Kwon et al., SOSP 2023, arXiv:2309.06180) that applies OS virtual-memory paging to the KV cache, reducing internal fragmentation from 60–80% waste to under 4%. **Not a training technique.** Its inclusion here is contrastive: to draw the training-vs-serving memory boundary precisely.

**The inference-serving problem** ([[paged-attention]]). During autoregressive decoding, prior systems (FasterTransformer, Orca) pre-allocate a contiguous buffer of `max_seq_len × d_model × 2 × num_layers × 2 (K+V)` bytes per request at request start. For a 1024-token max context on a 13B model: ~1.7 GB per request. Most of this is wasted because the sequence is still short. Internal fragmentation (unused suffix of a contiguous buffer) plus external fragmentation (requests of different lengths cannot share space) wastes 60–80% of KV HBM.

**PagedAttention solution.** KV cache divided into fixed-size **physical blocks** (~16 tokens/block, stored contiguously). Each request has a **block table** mapping logical block index → physical block index. Blocks are allocated on demand; shared across requests for common prefixes (prefix caching). Beam search beams share physical blocks until they diverge (copy-on-write), collapsing beam-search KV memory from O(beam_width × length) to O(length + small diverged suffix). Result: **≤4% fragmentation**, **2–4× throughput improvement vs. FasterTransformer and Orca** at the same latency ([[paged-attention]]).

**Why this is not a training technique.** The distinction is in what "KV cache" means in each context ([[paged-attention]]):

- **Training**: the K and V tensors for a batch are computed in the forward pass and may be stored for the backward pass (as activations), but they are not *accumulated across decoding steps*. Each training step is a complete forward-backward over the full sequence. There is no growing KV buffer to page.
- **Inference serving**: autoregressive decoding accumulates K and V tensors from all prior tokens. The KV cache grows step-by-step and persists across decoding steps for each active request. This is what PagedAttention manages.

FlashAttention's trick (recomputing the score matrix on backward, storing only logsumexp) is a *training activation* optimization. PagedAttention is a *serving KV accumulation* optimization. These solve orthogonal problems. A model can use both: FlashAttention kernels during training, PagedAttention during serving. Confusing them is a common error when learners encounter "KV cache" discussions in both contexts.

---

## 7. Cross-Implementation Synthesis

### Memory behavior comparison table

| Kernel | Setting | Memory complexity | N×N materialized | Backward pass | Primary constraint |
|---|---|---|---|---|---|
| SDPA/MATH | Training + Inference | O(N²) | Yes — full score matrix in HBM | Yes | None (always available) |
| SDPA/FLASH_ATTENTION | Training | O(N) | No — recomputed on backward | Yes | CUDA, fp16/bf16, head_dim ≤ 128, no arbitrary bias |
| SDPA/EFFICIENT_ATTENTION (xFormers) | Training | O(N) | No — streaming accumulation | Yes | Broader: custom bias, large head_dim |
| SDPA/CUDNN_ATTENTION | Training | O(N) | No | Yes | cuDNN version requirements |
| SageAttention INT8 | Inference only | O(N) forward | No — streaming | **No** | Inference only; ~2% numerical error |
| Ring Attention / CP | Training | O(L/D) per device | No — blockwise | Yes | Multi-device ring; communication overhead |
| PagedAttention | Inference serving only | O(KV cache, paged) | N/A | **No** | KV accumulation management, not training |

### What is invariant vs. variant

**Invariant** (forced by the substrate): Any exact attention kernel must compute softmax(QKᵀ/√d)V. The N² work is irreducible in compute. What is reducible is *HBM memory* — whether the N×N score matrix is written to HBM at all. The online softmax recurrence (Milakov & Gimelshein 2018, [[online-softmax]]) is the universal primitive that permits single-pass streaming: `m_new = max(m_old, x_k)`, `d_new = exp(m_old − m_new) · d_old + exp(x_k − m_new)`. Every O(N) kernel (FA1/FA2/FA3, xFormers FMHA, Ring Attention blockwise) uses this recurrence.

**Variant** (free design choices that separate the implementations):
- *Granularity of work partitioning*: FA1 splits K/V across warps (requires shared-memory sync for softmax denominator); FA2 splits Q across warps (each warp owns output rows independently, eliminating sync) — this doubles MFU from 25–40% to 50–73% on A100.
- *Hardware specialization*: FA3 adds Hopper TMA producer-consumer warp specialization to reach 740 TFLOPs/s (75% utilization) in FP16 and ~1.2 PFLOPs/s in FP8. xFormers FMHA uses CUTLASS 2.x for broader coverage at lower peak throughput.
- *Precision*: SageAttention quantizes QKᵀ to INT8 (with K-channel smoothing) for inference; FA3 uses FP8 block quantization with incoherent processing (Hadamard rotation before quantization, 2.6× lower error than naive FP8) for training.
- *Scope of sharding*: Ring Attention extends the O(N) regime across D devices so per-device memory is O(L/D), solving the problem that single-device FlashAttention cannot address.
- *Dispatch policy*: PyTorch SDPA adds an indirection layer with silent fallback semantics — introducing the silent-MATH-fallback OOM trap as a consequence of prioritizing backward compatibility over fail-fast behavior.

The fundamental design question the zoo answers: given a fixed mathematical computation (exact softmax attention), what is the minimal amount of data that must pass through slow HBM memory, and how does that minimum change when you add hardware constraints, multi-device topology, or inference-vs-training context?

---

## Core Insights from the Literature

1. **The silent fallback is the most dangerous default in transformer training.** [[pytorch-sdpa]] documents that without explicit `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, unsupported inputs cause PyTorch to silently fall to the MATH backend — 38× slower and O(N²) memory — with no warning. The correct contract is: explicit backend selection raises `RuntimeError` on unsupported inputs; the default dispatch swallows the error. Codebases that "enable FlashAttention" via a config flag without verifying actual dispatch are silently OOMing on edge-case inputs.

2. **The online softmax recurrence is the universal primitive.** Every O(N) attention kernel — FA1, FA2, FA3, xFormers FMHA, Ring Attention's blockwise accumulation — relies on Milakov & Gimelshein's single-pass recurrence ([[online-softmax]]). Without it, tiling would require a second HBM read for renormalization, reintroducing O(N²) bandwidth. The recurrence is four arithmetic operations per element: it is what makes single-pass streaming numerically exact.

3. **Training memory and inference memory are orthogonal problems.** [[paged-attention]] and [[sage-attention]] both reduce memory in important ways, but neither is relevant to training. PagedAttention pages a *persisted* KV cache that accumulates across decoding steps — a structure that does not exist in training. SageAttention accelerates the *forward-pass evaluation* of attention but has no backward pass — it cannot participate in a training loop. [[flash-attention-1]]'s backward recomputation (store logsumexp, recompute scores from Q, K, V) is the only known exact technique that reduces training activation memory for the attention layer.

4. **Sequence sharding (Ring Attention) is the only solution when O(N) per-device memory itself OOMs.** FlashAttention's O(N) regime is not the end of the story when N is very large. [[ring-attention]] shows that per-device memory scales as O(L/D) with D devices in a ring, enabling sequences D× longer with no approximation and no added latency (communication pipelined behind GEMM). This is the algorithmic basis for Megatron-LM's context parallelism (`--context-parallel-size`), which [[ch-07]] covers in the full parallelism taxonomy.

---

## Key Takeaways

- `torch.nn.functional.scaled_dot_product_attention` is a dispatcher, not a kernel. Pin backends explicitly with `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` in production code; silent fallback to MATH = silent OOM.
- Four backends: MATH (O(N²), always available), FLASH_ATTENTION (O(N), strict constraints), EFFICIENT_ATTENTION/xFormers (O(N), broader coverage), CUDNN_ATTENTION (O(N), cuDNN-dependent). The ~38× performance gap between MATH and optimized backends is entirely HBM bandwidth cost of the N×N score matrix.
- xFormers `memory_efficient_attention` implements Rabe & Staats 2021 (arXiv:2112.05682): O(√N) practical variant, 59× inference memory and 32× backpropagation memory reduction at N=16,384. It is the right choice when FA2's constraints (head_dim, bias shape) cannot be met.
- SageAttention: INT8 for QKᵀ (with K-channel smoothing: subtract per-channel mean, cancels in softmax), FP16 for P·V, 2.1× over FA2 at RTX4090. **Inference only** — no backward pass, not applicable to training.
- Ring Attention: D devices in a ring, per-device KV memory O(L/D), zero communication overhead (rotation pipelined behind GEMM), exact (D rounds, every Q attends every K). This is Megatron-LM's `--context-parallel-size` = CP in `world_size = TP × PP × CP × DP`.
- PagedAttention: fixed-size physical blocks (~16 tokens) allocated on demand, ≤4% fragmentation vs 60–80% waste in prior systems, 2–4× serving throughput. **Inference serving only** — no training analog. The "KV cache" it pages is a persisted structure that grows across decoding steps, which does not exist during training.
- For the [[ch-09]] capstone: GDN linear attention has O(d) KV memory natively; Ring Attention / CP provides no benefit. The entire FlashAttention kernel family is moot — but the SDPA dispatch and its OOM trap still apply to any standard attention layers in the same model.

---

## References

- PyTorch SDPA Tutorial — https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html ([[pytorch-sdpa]])
- facebookresearch/xformers — https://xformers.org/ · https://github.com/facebookresearch/xformers ([[xformers-mem-efficient]])
- Markus N. Rabe and Charles Staats. "Self-attention Does Not Need O(n²) Memory." arXiv:2112.05682 (2021) — https://arxiv.org/abs/2112.05682
- Jintao Zhang et al. "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration." ICLR 2025, arXiv:2410.02367 — https://arxiv.org/abs/2410.02367 ([[sage-attention]])
- Jintao Zhang et al. "SageAttention2." arXiv:2411.10958 — https://arxiv.org/abs/2411.10958
- Hao Liu, Matei Zaharia, Pieter Abbeel. "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024, arXiv:2310.01889 — https://arxiv.org/abs/2310.01889 ([[ring-attention]])
- Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023, arXiv:2309.06180 — https://arxiv.org/abs/2309.06180 ([[paged-attention]])
- Maxim Milakov and Natalia Gimelshein. "Online normalizer calculation for softmax." arXiv:1805.02867 (2018) — https://arxiv.org/abs/1805.02867 ([[online-softmax]])
- Tri Dao et al. "FlashAttention." NeurIPS 2022, arXiv:2205.14135 — https://arxiv.org/abs/2205.14135 ([[flash-attention-1]])
- Tri Dao. "FlashAttention-2." ICLR 2024, arXiv:2307.08691 — https://arxiv.org/abs/2307.08691 ([[flash-attention-2]])
- Jay Shah et al. "FlashAttention-3." arXiv:2407.08608 (2024) — https://arxiv.org/abs/2407.08608 ([[flash-attention-3]])

---

## Questions

1. The [[pytorch-sdpa]] excerpt states the MATH fallback is ~38× slower and allocates the full O(N²) score matrix, yet the fallback happens silently. What specific input property — not a bug, but a legitimate architectural choice in some models — would trigger this fallback even in a codebase that "enabled FlashAttention," and how would you detect it before hitting an OOM in production?

2. [[sage-attention]] achieves 2.1× over FA2 at RTX4090 by quantizing QKᵀ to INT8, yet FA2 on A100 already reaches 50–73% MFU by eliminating the N×N HBM write. Are these two speedup claims compatible? What does each one optimize — and why does SageAttention's technique not transfer to A100 training runs?

3. Ring Attention guarantees "no additional communication and computation overheads" because the KV rotation is pipelined behind the GEMM. Describe the condition under which this pipelining breaks down and the communication *does* become a bottleneck — and what parameter you would tune to recover.

4. The Rabe & Staats paper offers three exact memory variants: O(1) per-query, O(log N), and O(√N). xFormers FMHA implements the O(√N) variant for GPU/TPU tile parallelism. Given what you know from [[ch-04]] about the online softmax recurrence, explain mechanically why the O(1) variant cannot be efficiently parallelized across warps in the same way the O(√N) tile variant can.

5. PagedAttention cuts KV fragmentation from 60–80% waste to ≤4% using physical blocks of ~16 tokens. A naive re-implementation using blocks of 1 token would achieve 0% fragmentation. Why is block size of ~16 tokens the practical choice rather than 1 token, and which two hardware constraints drive that number?

6. The [[ch-09]] capstone model uses GDN linear attention. Given that linear attention has O(d) KV memory natively (not O(N)), trace which entries in the comparison table in §7 become irrelevant, which remain relevant, and what new memory problem (if any) replaces the O(N) activation concern for that architecture.
