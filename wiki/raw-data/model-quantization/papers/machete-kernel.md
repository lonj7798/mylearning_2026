<!-- scope: Machete W4A16 GEMM kernel for NVIDIA Hopper (Wilkinson / Neural Magic, 2024)
     deps: [[marlin-kernel]], [[gptq]], [[awq]]
     see-also: [[vllm-quant]], [[tensorrt-llm-quant]]
-->

# Machete: Hopper-Native W4A16 GEMM
- **Core Insight:** Marlin's Ampere-tuned pipeline leaves ~37 % of Hopper's peak compute on the floor because it doesn't use WGMMA or TMA; Machete is the from-scratch redesign that uses Hopper's async tensor cores (WGMMA) and the Tensor Memory Accelerator (TMA), recovering that 37 % and pushing W4A16 GEMM to FP16 parity at batch ~128.
- **Guideline:** On H100/H200, use Machete (vLLM 0.6.2+) for W4A16 / W8A16 GPTQ/AWQ serving; stay on [[marlin-kernel]] on A100. Expect ~29 % geomean speedup on Llama-70B (single H100) and ~42 % on Llama-405B (4×H100).
- **Authors:** Lucas Wilkinson (Neural Magic / Red Hat)
- **Year:** 2024 (released Oct 14, 2024)
- **URL:** https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel • vLLM source `csrc/quantization/machete/`
- **Relevant topics:** W4A16 GEMM, Hopper, WGMMA, TMA, warp specialization, CUTE layout algebra, vLLM

## Abstract
Marlin was designed before Hopper shipped and uses Ampere's `cp.async` + `mma.sync` tensor cores; on H100 those instructions still work but bypass the new WGMMA tensor-core path and the Tensor Memory Accelerator. Machete is a Hopper-native W4A16 mixed-input GEMM written with CUTLASS / CUTE: it issues WGMMA group-matrix-multiply-accumulate, uses TMA for asynchronous tile loads with built-in coalescing, warp-specializes between data-mover and compute warps, and resolves the 4-bit→FP16 packing offline via CUTE-described layout algebra so the dequant is a single shift-and-cast inside the compute warp. On Hopper, this recovers ~37 % of peak compute that Marlin left unused and brings W4A16 GEMM to FP16 throughput at batch ≥ 128. It is the kernel behind vLLM's `--quantization gptq_marlin` (auto-routed to Machete) on Hopper.

## Key Contributions
- First production-grade W4A16 GEMM for Hopper to use **WGMMA** (group matrix-multiply-accumulate) — the asynchronous tensor-core API introduced with H100.
- Uses **TMA** (Tensor Memory Accelerator) to issue async global→shared copies of large tiles with built-in bounds + swizzle, removing the per-thread `cp.async` setup overhead Marlin paid.
- Warp specialization between producer (TMA-driven) and consumer (WGMMA) warps with `mbarrier` sync — same pattern as DeepSeek-V3's FP8 GEMM and Flash-Attention 3.
- **CUTE layout algebra** to describe the 4-bit weight pre-shuffle: future hardware changes can be re-targeted by editing the CUTE layout, not the hand-derived bit-twiddling.
- ~37 % closer to peak on Hopper W4A16 than Marlin; ~29 % geomean speedup on Llama-70B single-H100, ~42 % geomean on Llama-405B four-H100.
- Competitive with FP16 GEMM at batch ≥ 128 — meaning W4A16 serving has no throughput cost in the regime that matters for high-concurrency serving.

## Key Figures/Tables to Study
- The pipeline diagram showing TMA producer warp + WGMMA consumer warp + mbarrier coordination — same skeleton as Flash-Attention-3.
- Throughput table: Machete vs Marlin vs FP16 cuBLAS on H100 across batch sizes 1, 8, 32, 128, 512.
- The end-to-end vLLM throughput speedup table on Llama-70B and Llama-405B.

## Technical Details

### What Machete added over Marlin (the 37 % gap)
- **WGMMA vs MMA.sync:** WGMMA is async, has much larger tile sizes (64×N×16 for FP16), and frees the issuing warp to do other work while the tensor core runs.
- **TMA vs cp.async:** TMA issues one descriptor to copy a whole tile, with swizzling and out-of-bounds handling built in; cp.async needs per-thread address arithmetic.
- **Larger consumer tiles:** Hopper's WGMMA shapes are wider than Ampere's MMA, so Machete uses larger thread-block tiles, amortizing the per-tile bookkeeping.

### Weight format
- Same packed-INT4 format used by [[marlin-kernel]] / GPTQ / AWQ, but a different pre-shuffle (Hopper's WGMMA expects a different lane-to-thread mapping than Ampere's MMA).
- The pre-shuffle is described as a CUTE layout — re-targeting to a future tensor-core layout is an algebra change, not a rewrite.
- Group size 128 (matching GPTQ default).

### Warp specialization
- **Producer warps:** drive TMA copies of weight + activation tiles into shared memory.
- **Consumer warps:** issue WGMMA against the loaded tiles, accumulate in FP16 registers, write out via stmatrix.
- mbarrier synchronizes the two groups; double-buffered shared memory hides load latency.

### Coverage
- **W4A16** (GPTQ-style and AWQ-style 4-bit weights, FP16 activations) — primary use case.
- **W8A16** (8-bit weights) — secondary.
- **compressed-tensors** format from Neural Magic's library — supported.

### Performance numbers (from the Red Hat / NM blog)
| Configuration | Speedup vs Marlin |
|---------------|-------------------|
| Llama-70B, 1× H100, geomean | +29 % |
| Llama-405B, 4× H100, geomean | +42 % |
| Batch ≥ 128 | matches FP16 cuBLAS |

### vLLM integration
- vLLM 0.6.2+ auto-selects Machete over Marlin when CUDA capability ≥ 9.0 (Hopper).
- Same `--quantization gptq_marlin` / `--quantization awq_marlin` CLI flags trigger Machete on H100.
- Compatible with all GPTQ / AWQ checkpoints already in the hub — no re-quantization needed.

## Connections
- [[marlin-kernel]] — the Ampere predecessor; Machete is the Hopper-native rewrite.
- [[gptq]] / [[awq]] — quantization algorithms whose checkpoints Machete serves.
- [[vllm-quant]] — vLLM's quantization integration where Machete is the Hopper default backend.
- [[tensorrt-llm-quant]] — NVIDIA's competing W4A16 kernel for Hopper; Machete is open-source.
- [[deepseek-v3-fp8]] — same warp-specialized producer/consumer skeleton as DSV3's FP8 GEMM and Flash-Attention 3.
- [[frantar-alistarh-ist-austria]] / [[neural-magic]] — Marlin came from IST-Austria, Machete from Neural Magic (now Red Hat).
