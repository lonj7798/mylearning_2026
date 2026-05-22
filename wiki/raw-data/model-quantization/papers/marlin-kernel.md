<!-- scope: MARLIN W4A16 GEMM kernel for LLM serving (Frantar et al. 2024)
     deps: [[gptq]], [[awq]]
     see-also: [[machete-kernel]], [[vllm-quant]], [[tensorrt-llm-quant]]
-->

# MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs
- **Core Insight:** A W4A16 GEMM can hit near-FP16 throughput at batch sizes up to ~32 — not just batch 1 — by combining asynchronous global-to-shared loads, a warp-specialized dequant pipeline, double-buffered shared memory, and a static weight reshuffle so the 4-bit weights land directly in the tensor-core register layout without on-the-fly permutation.
- **Guideline:** Use Marlin (vLLM / SparseML / NeuralMagic stack) as the default W4A16 kernel on Ampere; use [[machete-kernel]] on Hopper. Expect ~4× over FP16 at batch 1-8 and roughly FP16 parity by batch 64; below batch 32, almost no inference cost penalty over the FP16 baseline.
- **Authors:** Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2408.11743 • https://github.com/IST-DASLab/marlin
- **Relevant topics:** W4A16 GEMM, Ampere/Hopper, async copy, warp specialization, double buffering, GPTQ deployment

## Abstract
Mixed-input GEMM kernels — INT4 weights × FP16 activations — historically had a critical-batch-size problem: they crush batch-1 latency but stop helping (and often hurt) as soon as the workload becomes compute-bound rather than memory-bound. Marlin closes that gap by treating the kernel as an explicit memory + compute pipeline: cp.async copies global weights into shared memory while the prior block dequants and tensor-core-MMAs in parallel, double buffers swap with no stalls, and a one-time weight pre-shuffle puts the 4-bit weights into the exact lane layout each warp's MMA needs so dequant is just a shift + sign-extend. The result is ~4× FP16 throughput maintained up to batch 16-32, and continuing speedup up to batch ~64 on A100. Marlin is the production kernel behind vLLM's GPTQ/AWQ serving on Ampere.

## Key Contributions
- A GPTQ-compatible W4A16 GEMM that keeps near-peak speedup through the critical batch size range (16-32) where prior kernels (e.g. ExLlama, AutoGPTQ's CUDA kernel) collapsed to FP16 throughput.
- **Async global-to-shared copies** via `cp.async` (Ampere) so weight loads happen in the shadow of the dequant + MMA pipeline.
- **Warp specialization**: some warps do memory movement, others do dequant + MMA; warps cooperate via shared memory with `mbarrier`-style sync.
- **Double-buffered shared memory:** while warp A consumes buffer 0, warp B fills buffer 1; the swap is a flag-flip, no stall.
- **Weight pre-shuffle**: a static offline reordering of the 4-bit weights so that after the bit-shifts inside the dequant warp, each tensor-core thread already has its operand in the right register slot — eliminates per-call permute instructions.
- Extended to **2:4 sparsity + W4** in a follow-up; integrated with vLLM 0.6+ for production deployment.
- End-to-end speedup: up to **2.8×** for vLLM-served Llama-70B on A100, near-peak utilization at moderate batch.

## Key Figures/Tables to Study
- **Roofline figure**: shows the kernel staying on the memory-bandwidth-bound line through batch 16, then transitioning gracefully into the compute-bound region without a discontinuity (other kernels show a sharp drop).
- **Pipeline diagram**: the three-stage pipeline (load → dequant → MMA) with double-buffered shared memory and warp specialization.
- **Throughput table**: TFLOPS achieved at batch 1, 4, 8, 16, 32, 64 for Marlin vs ExLlama vs AutoGPTQ vs FP16 cuBLAS on A100.

## Technical Details

### Quantization format consumed
- Marlin consumes the **GPTQ packed-int4** weight format with **group_size 128** (the de-facto standard since the GPTQ paper).
- Per-group scale and zero-point in FP16; weights packed 8 × INT4 per 32-bit word.
- The static pre-shuffle changes the in-memory layout but preserves the bit content — i.e. you can take any GPTQ-quantized checkpoint and re-pack it for Marlin offline.

### Pipeline structure (per CTA)
1. **Stage A (loader warps):** `cp.async` 4-bit weight tile (128 K rows × N cols, packed) and FP16 activation tile from global memory into shared memory buffer *i*.
2. **Stage B (compute warps):** in parallel, dequant the FP16 weights from shared buffer *1-i* via shift + sign-extend + per-group scale multiply, then issue tensor-core MMA on the FP16 weights × FP16 activations.
3. **Stage C (barrier):** mbarrier sync, swap buffers, repeat.

The double buffer means stage A on iteration *k+1* runs in parallel with stage B on iteration *k* — load latency is hidden by compute.

### Why the critical batch range matters
At batch 1, the GEMM is purely memory-bound and any 4× weight compression gives 4× speedup. As batch grows, FLOPs grow linearly but weight bytes stay constant — at some batch the GEMM crosses into compute-bound. Pre-Marlin kernels couldn't keep the tensor cores fed past that crossover; their FLOPS/s collapsed. Marlin's warp-specialized pipeline keeps both memory and compute units saturated through batch ~32, which is the range that matters for production serving.

### Hardware targets
- **Ampere (A100, A6000):** the original target; uses `cp.async` for global→shared.
- **Hopper (H100):** Marlin runs but doesn't use TMA/WGMMA — see [[machete-kernel]] for the Hopper-native successor.
- **Sparse Marlin:** extends to NVIDIA 2:4 sparsity, achieving another ~1.6× on top.

### Integration
- vLLM: `--quantization gptq_marlin` or `awq_marlin` automatically rewrites a GPTQ/AWQ checkpoint into the Marlin layout and uses the Marlin kernel.
- Also exposed through SparseML / NeuralMagic deepsparse / TensorRT-LLM (community port).

## Connections
- [[gptq]] — the quantization algorithm whose 4-bit checkpoint format Marlin consumes.
- [[awq]] — same format with an AWQ-style activation-aware scale; vLLM uses `awq_marlin` to serve AWQ checkpoints with Marlin's kernel.
- [[machete-kernel]] — the Hopper-native successor that uses TMA + WGMMA; Marlin remains the Ampere default.
- [[vllm-quant]] — production integration of Marlin in vLLM's serving stack.
- [[tensorrt-llm-quant]] — NVIDIA's competing W4A16 kernel; Marlin matches or beats it in published benchmarks on Ampere.
- [[frantar-alistarh-ist-austria]] — the IST-Austria group that produced GPTQ, Marlin, AQLM, SparseGPT.
