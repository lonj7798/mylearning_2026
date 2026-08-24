# Training Has No KV Cache — The Train/Serve Memory Boundary
<!-- slug: train-vs-infer-kv-boundary · type: doc · source: wiki:course/llm-inference:wiki/courses/llm-inference/ch-03/excerpts/prefill-vs-decode.md -->

**Core Insight.** **There is no KV cache during training. None. Zero bytes.** Teacher forcing means the entire
target sequence is already known, so training runs **one parallel forward pass over all `s` positions** and the
causal mask makes position `t`'s output depend only on positions `≤ t`. There is no autoregressive loop, therefore
no repeated work, therefore nothing to amortize, therefore no cache. The KV cache is a pure inference-time data
structure that exists only because decoding is a sequential loop of `N` forward passes.

**Guideline.** In a training memory budget, never allocate a "KV cache" line item. K and V *do* exist during
training — as ordinary activations saved for backward — and they belong in the activation bucket, where
gradient checkpointing can delete them. Any training OOM analysis that includes a KV-cache term is double-counting.

## Technical Details

- **Why teacher forcing removes the loop.** Training feeds the ground-truth sequence `x_1..x_s` and computes
  cross-entropy at every position simultaneously. Position `t` predicts `x_{t+1}` using the causally masked
  attention over `x_1..x_t`. All `s` predictions come out of **one** forward pass, so K and V for every position
  are computed exactly once. Inference cannot do this: token `t+1` is not known until token `t` has been sampled,
  which forces `N` separate forward passes — the only reason memoization pays off ([[kv-cache-mechanism]]).
- **The exact side-by-side.**
  | | **Training (teacher forcing)** | **Inference (autoregressive decode)** |
  |---|---|---|
  | Forward passes per sequence | **1** | 1 prefill + `N` decode steps |
  | Are K/V recomputed? | Never — computed once | Never — read from cache |
  | Why K/V are held | backward needs them for `∂L/∂Q, ∂L/∂K, ∂L/∂V` | next decode step attends over them |
  | Lifetime | forward → backward of **one** optimizer step, then freed | whole request, freed at request end |
  | Grows during use? | No — allocated at full `s` in one shot | **Yes** — +1 token per sequence per step |
  | Can it be removed? | **Yes** — recompute under gradient checkpointing | No — only compressed (GQA/MLA/quant) or paged |
  | Scales with | micro-batch × seq-len × layers | concurrent requests × context × layers |
  | Ledger bucket | **activations** ([[transformer-math-101]]) | separate persistent KV pool |
  | Optimisation target | FlashAttention, recompute, SP/CP | GQA/MLA, KV quant, PagedAttention |
- **The trap: identical algebra, different object.** The bytes of K and V held during a training forward are
  `2 · B · s · L · n_kv_heads · d_head · b` — *character for character* the inference KV-cache formula
  ([[kv-cache-memory-formula]]). Worked: Llama-3-8B geometry, `B=1, s=8192, bf16`
  → `2·32·1·8192·8·128·2 = 1,073,741,824 B = 1.00 GiB`, the same number as the inference KV cache for one
  8k-context request. Llama-3-70B at the same settings: `2,684,354,560 B = 2.50 GiB`. Same number, opposite
  lifetime. This is precisely why the two get conflated.
- **In training, Q is stored too — and Q usually dominates.** FlashAttention's backward saves `Q, K, V, O` and
  the logsumexp row statistics; GQA shrinks only K and V. Per layer at `B=1, s=8192, d_head=128, bf16` for
  Llama-3-70B (`H_q=64, H_kv=8`): `Q = 128 MiB, K = 16 MiB, V = 16 MiB` → 160 MiB/layer, 12.5 GiB over 80 layers.
  See [[gqa-mqa-mla-kv-heads]] for the `3·H_q/(H_q + 2·H_kv)` divisor.
- **Prefill is the closest inference analog to a training forward.** Prefill computes Q, K, V for all `S`
  positions in parallel, costs `~7·B·S·d² + 2·B·S²·d` FLOPs, and is **compute-bound** (arithmetic intensity
  ~3100 FLOPs/byte on H100, knee at `1979e12/3.35e12 = 590`). Decode costs `~7·B·d² + 2·B·t·d` per step and is
  **bandwidth-bound** (~0.3 FLOPs/byte at `B=1`). A training step behaves like prefill — compute-bound, no cache,
  everything in parallel — with a backward pass bolted on. That is the right mental bridge: **training ≈ prefill
  + backward, forever; serving ≈ prefill once, then a long bandwidth-bound decode tail.**
- **PagedAttention has no training analog.** vLLM's block/page manager (16 tokens per block, per-sequence block
  table, copy-on-write prefix sharing, >96% KV utilisation vs 20–38% for contiguous allocators) solves *KV pool
  fragmentation*, which does not exist in training because the allocation is one contiguous shape known before
  the step begins ([[paged-attention]]).
- **Boson / Lina TMR hook.** GDN linear-attention layers are hard-asserted to `CP=1` in the capstone
  ([[ch-09]]). Their inference-time state is a fixed-size recurrent state per head (a `d_head × d_head` matrix in
  standard linear attention), **independent of sequence length** — so the `2·L·n_kv_heads·d_head·s·b` scaling
  argument does not apply to those layers at all. On the training side that changes nothing about this boundary:
  GDN blocks still store their per-position activations for backward, and the 32k-sequence activation bill is
  still a checkpointing problem, not a cache problem.
- **Training-memory angle:** This is the load-bearing negative result of the whole cluster. When budgeting a
  training GPU, the six ledger items are weights, gradients, optimizer states, **activations**, the loss-head
  logit spike, and CUDA/NCCL overhead ([[transformer-math-101]], [[memory-calculator-notes]]) — KV cache is not
  among them. K and V appear only inside "activations," where FlashAttention removes the `O(s²)` score matrix
  ([[flash-attention-1]]) and recomputation removes the rest ([[selective-recompute-korthikanti]]). Conversely,
  when you later serve the model you trained, the KV cache appears as a brand-new memory class that no training
  run ever paid for — which is why a model that trains fine on a node can fail to serve on the same node.

## Citation
Amey Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," 2023,
arXiv:2308.16369 · Yinmin Zhong et al., "DistServe: Disaggregating Prefill and Decoding," OSDI 2024,
arXiv:2401.09670 · Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with
PagedAttention," SOSP 2023, arXiv:2309.06180 · Tri Dao et al., "FlashAttention," NeurIPS 2022, arXiv:2205.14135.
Synthesized from `course/llm-inference:wiki/courses/llm-inference/ch-03/excerpts/prefill-vs-decode.md`,
`.../ch-06/excerpts/pagedattention.md`, and `wiki/raw-data/training-memory/excerpts/paged-attention.md`.
