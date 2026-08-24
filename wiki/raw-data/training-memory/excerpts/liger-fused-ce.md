# Liger-Kernel: Fused Linear Cross-Entropy
<!-- slug: liger-fused-ce · type: module · source: https://github.com/linkedin/Liger-Kernel -->

**Core Insight.** Standard cross-entropy loss materializes a full `(B·T × V)` logit tensor before computing the loss — a spike that reaches 1+ GB at modest batch/sequence/vocab sizes — and Liger's fused kernel eliminates it by chunking along the token dimension and fusing the linear projection with the loss computation inside a single Triton kernel, reducing peak activation memory by ~80%.

**Guideline.** Replace the final `nn.Linear` + `CrossEntropyLoss` pair with `LigerFusedLinearCrossEntropyLoss` for any model with a large vocabulary (≥32K tokens). The kernel is exact (no approximation); backward is handled inside the kernel. Apply early in any memory-constrained training run — it costs nothing in accuracy and removes the most acute single-step spike.

## Technical Details

- **The spike:** For a standard forward pass with BF16 precision, logit tensor = `B·T × V × 2 bytes`. Example: seq=16,384 tokens, vocab=32,000 → `16384 × 32000 × 2 = 1.05 GB` for logits alone, appearing transiently before the loss reduce.
- **Chunking mechanism:** The fused kernel processes tokens in chunks of size ≤ `MAX_FUSED_SIZE` (platform-specific; typically 65,536÷2 on CUDA, 2,048 on NPU). For the same example, the largest temporary allocation is `2,048 × 32,000 × 2 = 131 MB` — 8× smaller per chunk.
- **Three in-kernel strategies to avoid full materialization:**
  1. Incremental projection: hidden states processed chunk-by-chunk through the linear head.
  2. In-place gradient accumulation: gradient w.r.t. `weight` accumulates into `grad_weight` across chunks rather than being computed from the full logit matrix.
  3. No separate logit buffer: the loss scalar and `dL/d(hidden)` are computed within the chunk loop.
- **Reported memory reductions:**
  - General LLM training (pretraining/SFT): **~60% overall memory reduction**
  - Post-training alignment (DPO, ORPO, CPO): **up to 80% reduction** (because these methods double the forward pass over the same vocab head)
- **Exact semantics:** No approximations; numerics are identical to standard PyTorch CE. Backward pass is integrated — calling `.backward()` on the fused loss returns correct `d_hidden` and `d_weight`.
- **Training-memory angle:** The logit-materialization spike is the largest transient memory event in a standard training step for high-vocab models; it sits outside the static 16–18 B/param floor and can trigger OOM even after careful static memory budgeting. Liger removes this spike entirely by never constructing the `B·T × V` tensor.

## Citation
Austin Liu, Bofei Gao, Charles Goddard, Jiacong He, Zhiqiang Shen, et al. "Liger Kernel: Efficient Triton Kernels for LLM Training." arXiv:2410.10989, 2024.
GitHub: https://github.com/linkedin/Liger-Kernel · Docs: https://linkedin.github.io/Liger-Kernel/
