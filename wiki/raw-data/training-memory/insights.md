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

## Cluster D — Transformer fundamentals (ch-extra, prerequisite sitting before ch-04)

| Excerpt | Core Insight (one line) | Feeds |
|---------|-------------------------|-------|
| [[transformer-block-tensor-ledger]] | A block is an ordered chain of ~16 named tensors and backward needs a derivable subset (matmul→input, softmax→output, dropout→mask, add→nothing); that list summed *is* Korthikanti's `34 = 11 attn + 19 MLP + 4 LN` plus `5as/h`. | ch-extra, ch-03, ch-04 |
| [[pre-ln-vs-post-ln]] | Post-LN puts a norm on the residual highway (`O(1/√ℓ)` gradients, 4,000-step warmup); Pre-LN moves it inside the branch leaving an identity path (`O(1)`) — identical activation bytes, entirely a trainability win. | ch-extra, ch-02 |
| [[residual-stream-memory-backbone]] | The model is one `[B,T,h]` stream that sub-layers only add to; that fixed-width backbone is why activations are `L ×` per-layer and why a checkpoint costs 33.55 MB against 1.64 GB of interior tensors (49×). | ch-extra, ch-03 |
| [[kv-cache-mechanism]] | Causal masking makes past K/V immutable, so a cache is memoization over an already-pure function; it turns `N(N+1)/2` token-forward-passes into `N` (515× fewer FLOPs at N=1024, d=8192) — and exists only because decode is a sequential loop. | ch-extra, ch-04, ch-06 |
| [[kv-cache-memory-formula]] | `KV bytes = 2·B·s·L·n_kv_heads·d_head·bytes` (the 2 = K and V); Llama-3-70B = 327,680 B/token → 2.50 GiB at 8k, 40.0 GiB at 128k; the same algebra names training K/V *activations*, not a cache. | ch-extra, ch-04, ch-06 |
| [[gqa-mqa-mla-kv-heads]] | MHA/GQA/MQA differ only in `n_kv_heads` (divisor `H_q/H_kv`, exactly 8× for Llama-3-70B); MLA drops the factor 2 entirely (`L·(d_c+d_rope)·b` = 68.6 KiB/token, 56.9×) — but training only gets `3H_q/(H_q+2H_kv)` ≈ 2.4×. | ch-extra, ch-04, ch-06 |
| [[train-vs-infer-kv-boundary]] | **Training has no KV cache at all** — teacher forcing computes all `s` positions in one parallel pass, so there is no loop to amortize; K/V exist only as activations (same bytes, one-step lifetime, deletable by recompute). | ch-extra, ch-03, ch-06, ch-09 |
| [[attention-permutation-equivariance]] | Bare self-attention is exactly permutation-equivariant (`Attn(PX)=P·Attn(X)`, the `Pᵀ` cancelling against `V`), so position must be injected — and only logit-additive schemes (ALiBi/T5 RPE) pay for it, needing a `B·H·T·T` object = 4.000 GiB at B=1,H=32,T=8192 bf16 per layer. | ch-extra, ch-04 |
| [[sinusoidal-absolute-encoding]] | Sinusoidal PE is a bank of `d/2` clocks at `ω_i = 10000^{−2i/d}` whose position-shift map is a pure rotation; as a non-trainable buffer it costs `4·L·d` B (128.00 MiB at L=8192,d=4096) with zero optimizer state, while the learned table it competes with costs 16 B/param (GPT-3: 384.00 MiB). | ch-extra, ch-01, ch-04 |
| [[rope-rotary-position-embedding]] | Rotating each 2D pair of Q and K by `mθ_i` is the *unique* map making `⟨R_m q, R_n k⟩` depend only on `m−n`; it costs a 4.00 MiB fp32 cos/sin cache at L=8192,d_head=128, zero params, and — being applied *before* the kernel — leaves FlashAttention's O(T) footprint intact. | ch-extra, ch-04, ch-06 |
| [[qkv-scaled-dot-product]] | Attention is a differentiable dictionary lookup `softmax(QKᵀ/√d_k)V`; three projections exist because `XXᵀ` is symmetric and (post-norm) diagonal-dominant, so one shared projection collapses attention to the identity — and `11sbh` vs `5as²b` cross over at only `s = 11h/(5a)` = 282 tokens. | ch-extra, ch-04 |
| [[sqrt-dk-scaling-variance]] | `Var(q·k) = d_k` *exactly* (Vaswani footnote 4), so unscaled scores saturate softmax and the Jacobian `p(1−p)` collapses — at `d_k=64` the scaling buys ~586× more gradient; fold the scale into `Q` or a second `(B,a,N,N)` tensor appears. | ch-extra, ch-04 |
| [[causal-mask-neg-inf]] | Additive `−∞` upper triangle before softmax is what makes teacher-forced parallel training identical to sequential decoding; use `finfo(dtype).min` (not `-1e9`, unrepresentable in fp16), and let a causal kernel apply it implicitly — the mask alone is 2.15 GB bf16 at N=32k. | ch-extra, ch-04, ch-05 |
| [[multi-head-split-concat-wo]] | `h` heads of `d_head = d_model/h` cost identical params (`4·d_model²`) and identical FLOPs (the `h` cancels), but the `(B,h,N,N)` score tensor scales **linearly in `h`** — multi-head is free in compute, never in memory; Table 3(A) verified: h=1 → 24.9 BLEU vs h=8/16 → 25.8. | ch-extra, ch-04, ch-05 |
