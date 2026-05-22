---
chapter: ch-19
course: model-quantization
phase: read
excerpt_of: "Machete: Hopper-Native W4A16 GEMM (Wilkinson, Neural Magic / Red Hat, October 2024)"
source_url: https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel
created_at: "2026-05-21"
---

# Excerpt: Machete — Hopper-native W4A16 with TMA + WGMMA

**Authors:** Lucas Wilkinson (Neural Magic / Red Hat)
**Year:** 2024 (released Oct 14, 2024)
**URL:** https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel; vLLM source `csrc/quantization/machete/`
**Raw-data source:** [[raw-data/machete-kernel]]

---

## The diagnosis: Marlin leaves 37% on the floor on Hopper

Marlin was designed before Hopper shipped and uses Ampere's `cp.async` + `mma.sync`. On H100 those instructions still work but **bypass the new WGMMA tensor-core path and the Tensor Memory Accelerator**.

Machete is the from-scratch redesign that uses Hopper's async tensor cores (WGMMA) and TMA. It recovers that 37% and pushes W4A16 GEMM to **FP16 parity at batch ≥ 128**.

---

## What changed (the 37% gap)

| Aspect | Marlin (Ampere) | Machete (Hopper) |
|--------|-----------------|-------------------|
| Tensor-core ISA | `mma.sync` (sync) | **WGMMA** (async, larger tiles) |
| Global → shared copy | `cp.async` (per-thread arithmetic) | **TMA** (one descriptor, built-in swizzle) |
| Consumer tile size | Ampere-MMA sized | Hopper-WGMMA wider |
| Layout description | hand-derived | **CUTE layout algebra** |

### WGMMA vs mma.sync

WGMMA is **async** — it has much larger tile sizes (64×N×16 for FP16) and frees the issuing warp to do other work while the tensor core runs. mma.sync stalls the warp until completion.

### TMA vs cp.async

TMA issues **one descriptor to copy a whole tile**, with swizzling and out-of-bounds handling built in. cp.async needs per-thread address arithmetic. TMA also coalesces accesses more efficiently.

### Larger consumer tiles

Hopper's WGMMA shapes are wider than Ampere's MMA, so Machete uses larger thread-block tiles, **amortizing the per-tile bookkeeping**.

### CUTE layout algebra

The 4-bit weight pre-shuffle is described as a CUTE layout — future hardware changes can be re-targeted by editing the CUTE layout, not by hand-deriving bit-twiddling. This makes Machete forward-portable to Blackwell-era tensor-core layouts.

---

## Warp specialization (the canonical Hopper pattern)

Same skeleton as Flash-Attention 3 and [[deepseek-v3-fp8]]'s FP8 GEMM:

- **Producer warps:** drive TMA copies of weight + activation tiles into shared memory.
- **Consumer warps:** issue WGMMA against the loaded tiles, accumulate in FP16 registers, write out via `stmatrix`.
- `mbarrier` synchronizes the two groups; double-buffered shared memory hides load latency.

---

## Weight format

- Same packed-INT4 format used by [[marlin-kernel]] / GPTQ / AWQ.
- **Different pre-shuffle** (Hopper's WGMMA expects a different lane-to-thread mapping than Ampere's MMA).
- The pre-shuffle is described as a CUTE layout — re-targeting to a future tensor-core layout is an algebra change, not a rewrite.
- Group size 128 (matching GPTQ default).

---

## Coverage

| Mode | Bits | Notes |
|------|------|-------|
| W4A16 | 4/16 | primary; GPTQ + AWQ |
| W8A16 | 8/16 | secondary |
| compressed-tensors | various | NeuralMagic unified format |

---

## Performance

| Configuration | Speedup vs Marlin |
|---------------|-------------------|
| Llama-70B, 1× H100, geomean | **+29%** |
| Llama-405B, 4× H100, geomean | **+42%** |
| Batch ≥ 128 | **matches FP16 cuBLAS** |

The headline: **W4A16 serving has no throughput cost in the high-concurrency regime** that matters for production.

---

## vLLM integration

- vLLM 0.6.2+ **auto-selects Machete over Marlin when CUDA capability ≥ 9.0** (Hopper).
- Same `--quantization gptq_marlin` / `--quantization awq_marlin` CLI flags trigger Machete on H100.
- Compatible with all GPTQ / AWQ checkpoints already in the hub — no re-quantization needed.

---

## Why Machete matters as a kernel-engineering exemplar

Three lessons that generalize beyond W4A16:

1. **Tensor-core ISA matters more than algorithm.** The same GPTQ checkpoint moved from "good" (Marlin) to "production-throughput" (Machete) purely by switching to WGMMA.
2. **TMA + WGMMA + warp specialization is the Hopper pattern.** Flash-Attention 3, DSV3's FP8 GEMM, and Machete all share this skeleton. If you're writing a new Hopper-targeted kernel, start here.
3. **CUTE layout algebra is the forward-portability trick.** Describing the bit layout as algebra rather than hand-derived means the same kernel can re-target to Blackwell with code changes localized to the layout description.

---

## Connections

- [[marlin-kernel]] / [[excerpts/marlin-kernel]] — the Ampere predecessor.
- [[gptq]] / [[awq]] / ch-08-09 — algorithms whose checkpoints Machete serves.
- [[vllm-quant]] — vLLM's quantization integration where Machete is the Hopper default backend.
- [[tensorrt-llm-quant]] — NVIDIA's competing W4A16 kernel for Hopper; Machete is open-source.
- [[deepseek-v3-fp8]] — same warp-specialized producer/consumer skeleton.
- [[frantar-alistarh-ist-austria]] / [[neural-magic]] — Marlin came from IST-Austria, Machete from Neural Magic (now Red Hat).
- [[ch-19]] — parent synthesis.
