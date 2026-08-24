# xFormers memory_efficient_attention
<!-- slug: xformers-mem-efficient · type: doc · source: https://xformers.org/ -->

**Core Insight.** xFormers' `memory_efficient_attention()` is Meta's production implementation of the Rabe & Staats O(N) streaming attention algorithm, fused into a CUTLASS kernel, and serves as PyTorch SDPA's `EFFICIENT_ATTENTION` backend — the fallback O(N) path when FlashAttention's hardware constraints are not met.

**Guideline.** Prefer `memory_efficient_attention` over standard attention for any configuration where FlashAttention is unavailable (older GPUs, custom attention bias, head_dim > 128) — it delivers the same O(N) activation memory savings as FlashAttention with broader hardware coverage, at the cost of somewhat lower throughput than FA2/FA3 on A100/H100.

## Technical Details
- **Algorithm**: implements the Rabe & Staats 2021 streaming attention (arXiv 2112.05682) — single outer loop over Q rows, streaming K/V blocks, accumulating output and running normalizer in SRAM; never writes the n×n score matrix to HBM. Memory complexity: O(N·d), not O(N²).
- **Kernel backend**: CUTLASS 2.x FMHA kernel targeting Ampere (sm80) and older; dispatches to FlashAttention3 on H100 (Hopper) via `_get_use_fa3` toggle. For AMD ROCm hardware, dispatches to Composable Kernel.
- **Auto-dispatch**: `memory_efficient_attention(q, k, v, attn_bias=None, scale=None)` automatically selects the fastest available FMHA kernel for the given GPU and input configuration. Handles `LowerTriangularMask` (causal), `BlockDiagonalMask`, and `PagedBlockDiagonalGappyKeysMask` (KV paging for inference).
- **Integration**: PyTorch SDPA's `SDPBackend.EFFICIENT_ATTENTION` calls this kernel directly. Used in HuggingFace Diffusers, fairseq, and LLaMA reference implementations.
- **Coverage vs. FlashAttention-2**: supports broader dtype/device combinations and arbitrary attention bias tensors (unlike FA2 which restricts bias to specific shapes); useful for architectures with per-head ALiBi, relative position biases, or custom masks.
- **Training-memory angle:** same O(N) activation memory regime as FlashAttention — eliminates the n×n score matrix from both forward activation storage and backward-pass buffers. The practical difference from FA2 is throughput: on A100, xFormers FMHA runs at roughly 60–70% of FA2 speed for standard causal attention, but is the correct choice when FA2's constraints (head_dim ≤ 128, no bias tensor) cannot be met, since the alternative is the 38× slower MATH backend.

## Citation
facebookresearch/xformers. "Hackable and optimized Transformers building blocks." GitHub/xformers.org, 2024. https://xformers.org/ · https://github.com/facebookresearch/xformers
