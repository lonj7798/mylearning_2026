# insights — `training-memory` source library index

One-line core insight per excerpt for the **GPU Memory in LLM Training** course ([[outline]]).
"Feeds" = chapters whose `read.md` cites the excerpt via `[[slug]]` (primary chapter first).

## Cluster A — Ledger / precision / activations (ch-01..03)

| Excerpt | Core Insight (one line) | Feeds |
|---------|-------------------------|-------|
| [[transformer-math-101]] | Training memory = four additive buckets (params + optimizer + gradients + activations) with exact bytes/param formulas; dominant bucket rotates between activations and optimizer states. | ch-01, ch-02, ch-03, ch-08, ch-09 |
| [[ultrascale-playbook]] | Mixed-precision AdamW has a hard 16 N-byte static floor, but activations become the largest burden once batch/seq grow — selective recompute breaks that wall (70% saving, 2.7% compute). | ch-01, ch-02, ch-08, ch-09 |
| [[ml-engineering-memory]] | AdamW mixed precision costs ≥18 bytes/param (6 weights + 8 optimizer + 4 gradients) before activations; the activation bucket explodes non-linearly with seq length. | ch-01, ch-08, ch-09 |
| [[mixed-precision-training]] | fp16 training needs three interventions — fp32 master weights, loss scaling, fp32 accumulation — to avoid silent underflow divergence; defines the three-copy precision layout. | ch-01, ch-02, ch-03 |
| [[fp8-training]] | FP8 (Hopper-only) cuts optimizer memory 16→6 B/param and total training memory 28–39% vs BF16 via E4M3 forward / E5M2 gradients + per-tensor dynamic scaling. | ch-02 |
| [[liger-fused-ce]] | Fused chunked linear+cross-entropy never materializes the `(B·T × V)` logit tensor, removing the largest transient spike (~60–80% activation reduction) exactly. | ch-01, ch-02, ch-08, ch-09 |
| [[gradient-checkpointing-chen]] | Checkpointing √n activations and recomputing the rest drops activation memory O(n)→O(√n) for exactly one extra forward pass (~33% compute). | ch-03, ch-08 |
| [[selective-recompute-korthikanti]] | Recompute only large-cheap-to-recompute activations (attention scores), keep small-costly ones (MLP/LN); with sequence parallelism gives 5× activation saving at <4% compute. | ch-03, ch-08, ch-09 |

## Cluster B — Attention kernels (ch-04..06)

| Excerpt | Core Insight (one line) | Feeds |
|---------|-------------------------|-------|
| [[self-attention-no-n2-memory]] | Exact attention needs only O(1) extra memory per query by deferring softmax normalization — never materialize the n×n score matrix (59× inference, 32× backprop saving at n=16k). | ch-04, ch-06 |
| [[online-softmax]] | Single-pass softmax via running (max m, sum d) scalars is the algorithmic primitive that lets FlashAttention tile attention without writing the full score row to HBM. | ch-04, ch-06 |
| [[flash-attention-1]] | Attention is HBM-bandwidth-bound, not FLOP-bound; tiling + online softmax keeps it in SRAM so the n×n matrix never touches HBM — O(n²)→O(n) activations, exact. | ch-05, ch-06 |
| [[flash-attention-2]] | Fixes FA1's warp work-partitioning (Q-across-warps, not split-K) to lift MFU 25–40%→50–73% (~2×) at the identical O(N) memory footprint. | ch-05, ch-06 |
| [[flash-attention-3]] | Exploits Hopper async (TMA + warp-group MMA ping-pong) + FP8 block quant with incoherent processing: 740 TFLOPs/s FP16, same O(N) memory regime. | ch-05, ch-06 |
| [[pytorch-sdpa]] | `scaled_dot_product_attention` dispatches to four backends; a silent MATH fallback reallocates the O(n²) matrix (~38× slower) — always select the backend explicitly. | ch-06 |
| [[xformers-mem-efficient]] | Meta's CUTLASS implementation of Rabe–Staats streaming attention; the SDPA `EFFICIENT_ATTENTION` backend and O(N) fallback when FlashAttention's constraints aren't met. | ch-06 |
| [[sage-attention]] | INT8 QKᵀ + FP16 P·V with per-channel K smoothing gives 2.1× over FA2 — but is inference-only (no backward); included as a training contrast anchor. | ch-06 |
| [[ring-attention]] | Shards the sequence across D devices in a ring and rotates KV behind the GEMM at zero extra comm cost — per-device KV memory O(L/D); foundation of context parallelism. | ch-06, ch-09 |
| [[paged-attention]] | OS-style paging of the inference KV cache eliminates 60–80% fragmentation; inference-only — no training analog (contrast: KV cache ≠ training activations). | ch-06 |

## Cluster C — Parallelism / formulas / OOM (ch-07..09)

| Excerpt | Core Insight (one line) | Feeds |
|---------|-------------------------|-------|
| [[zero-memory-optimization]] | Partition optimizer states / gradients / parameters across DP ranks (ZeRO-1/2/3) to cut per-GPU model state 16Ψ→16Ψ/N at ≤1.5× DDP communication; activations untouched. | ch-07, ch-08, ch-09 |
| [[megatron-tp-sp]] | Tensor parallelism splits weight matrices within a node; sequence parallelism shards the replicated LN/dropout region — together a t× activation cut `(sbh/t)(34+5as/h)`. | ch-03, ch-07, ch-08, ch-09 |
| [[pipeline-parallelism-1f1b]] | PP stages layers across GPUs (divides weight memory); 1F1B holds O(p) microbatch activations vs GPipe's O(m) at the same `(p-1)/m` bubble. | ch-07, ch-08, ch-09 |
| [[pytorch-fsdp]] | FSDP = ZeRO-3 native in PyTorch: flat-param all-gather per transformer-layer unit, reduce-scatter gradients; memory `16Ψ/N + 2·P_unit`, comm 3Ψ vs DDP 2Ψ. | ch-07, ch-08, ch-09 |
| [[deepspeed-moe-ep]] | Expert parallelism shards experts across EP ranks (weight/EP_size) with all-to-all dispatch+combine; transient all-to-all buffers must be budgeted separately from steady-state. | ch-07, ch-08, ch-09 |
| [[memory-calculator-notes]] | Peak GPU memory = five additive terms (W+O+G+A+logit spike) that peak at different times; ZeRO-3 divides W+G+O by N, activations/logit spike are world-invariant. | ch-08, ch-09 |
| [[training-oom-failure-modes]] | Every CUDA OOM reads out requested/free/phase from the traceback; follow estimate→smoke→read→lever, fixing the one overflowing component (W/G/O/A/L) minimally. | ch-08, ch-09 |
