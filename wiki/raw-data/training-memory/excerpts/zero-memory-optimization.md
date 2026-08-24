# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
<!-- slug: zero-memory-optimization · type: paper · source: https://arxiv.org/abs/1910.02054 -->

**Core Insight.** ZeRO eliminates memory redundancy in data-parallel training by partitioning optimizer states, gradients, and parameters across ranks — cutting per-GPU memory from 16Ψ bytes (DDP) to 16Ψ/N bytes (ZeRO-3) with only 1.5× communication overhead vs. DDP's all-reduce.

**Guideline.** Use ZeRO-1 (Pos) as a free optimization whenever you have ≥4 DP ranks. Move to ZeRO-2 (Pos+g) when optimizer memory is insufficient. Use ZeRO-3 (Pos+g+p) only when ZeRO-2 still OOMs — the all-gather overhead per layer is real. Never run vanilla DDP on a model where 16P > GPU_memory × N.

## Technical Details

- **Baseline mixed-precision Adam formula:** `2Ψ + 2Ψ + KΨ = 16Ψ bytes` where Ψ = number of parameters, K=12 for Adam (fp32 master params 4Ψ + fp32 momentum 4Ψ + fp32 variance 4Ψ; fp16 params 2Ψ + fp16 grads 2Ψ).
- **ZeRO stage memory formulas (per GPU, Nd ranks):**
  - ZeRO-1 (Pos): `4Ψ + KΨ/Nd` — only optimizer states sharded, params+grads replicated → ~4× saving at large Nd
  - ZeRO-2 (Pos+g): `2Ψ + (2+K)Ψ/Nd` — grads also sharded → ~8× saving at large Nd
  - ZeRO-3 (Pos+g+p): `16Ψ/Nd` — everything sharded → linear Nd scaling
- **7.5B model @ Nd=64 concrete numbers:** DDP=120 GB/GPU, ZeRO-1≈31 GB, ZeRO-2≈17 GB, ZeRO-3≈1.9 GB.
- **Communication overhead:** ZeRO-1 and ZeRO-2 have same communication volume as DDP (2Ψ). ZeRO-3 adds 1.5× (3Ψ total: all-gather Ψ before forward + reduce-scatter Ψ during backward + all-gather Ψ before backward).
- **"ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware"** — Rajbhandari et al. SC 2020.
- **Training-memory angle:** ZeRO-3 is the primary lever for fitting weights+optimizer into GPUs; it moves optimizer state, gradient, and parameter memory off the per-GPU working set, but *activations and logit-buffer spike are not touched* by ZeRO — those require separate treatment (activation checkpointing, sequence parallelism).

## Citation
Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC '20 (Supercomputing). https://arxiv.org/abs/1910.02054
