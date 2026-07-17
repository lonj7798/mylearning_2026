# PyTorch scaled_dot_product_attention — SDPA and SDPBackend
<!-- slug: pytorch-sdpa · type: doc · source: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html -->

**Core Insight.** `torch.nn.functional.scaled_dot_product_attention` is a single function call that dispatches to one of four fused kernel backends at runtime based on hardware capability and input properties; when none of the fast backends qualifies, it silently falls back to the C++ math implementation — which is 30–40× slower and allocates the full O(n²) attention matrix in HBM.

**Guideline.** Always explicitly select a backend via `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` in production training code and verify it succeeds; relying on the default dispatch risks an undetected math-fallback that silently doubles or triples activation memory and drops throughput off a cliff.

## Technical Details
- **Four backends** (as of PyTorch 2.x):
  - `SDPBackend.MATH` — pure PyTorch C++ fallback; computes full n×n score matrix, stores it in HBM; O(n²) memory; always available.
  - `SDPBackend.FLASH_ATTENTION` — calls FlashAttention kernel (requires CUDA, fp16/bf16, head_dim ≤ 128, no custom attention bias in some versions); O(N) memory.
  - `SDPBackend.EFFICIENT_ATTENTION` — calls xFormers memory_efficient_attention; broader hardware/dtype support than Flash, O(N) memory.
  - `SDPBackend.CUDNN_ATTENTION` — routes through cuDNN's SDPA graph; autograd-compatible; available on supported cuDNN + CUDA versions.
- **Selection API:**
  ```python
  from torch.nn.attention import SDPBackend, sdpa_kernel
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
      out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
  ```
  Raises `RuntimeError` if the requested backend is unavailable for the given inputs — which is the safe failure mode; the silent fallback only happens when using the default (no `sdpa_kernel` context).
- **Performance gap**: benchmark shows ~87,478 µs for MATH vs. ~2,274 µs for optimized backends on identical inputs — ~38× difference, attributable entirely to HBM bandwidth cost of the n×n score matrix.
- **Silent math fallback risk**: without explicit backend selection, unsupported dtypes, head dimensions, or attention bias shapes cause PyTorch to silently drop to MATH without warning. This is the most common source of unexpected OOM or slow training in codebases that "enabled FlashAttention" but did not verify.
- **Training-memory angle:** the difference between MATH and FLASH_ATTENTION backends is the difference between O(n²) and O(n) activation memory for the attention layer. At n=4096, batch=16, 32 heads, this is the difference between ~8 GB and ~200 MB of attention activations per layer. The silent fallback therefore represents a potential order-of-magnitude regression in activation memory with no warning.

## Citation
PyTorch Documentation. "Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA)." PyTorch Tutorials, 2024. https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
