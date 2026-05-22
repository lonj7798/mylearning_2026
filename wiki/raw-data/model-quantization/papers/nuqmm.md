<!-- scope: nuQmm / LUT-GEMM — LUT-based non-uniform weight-quant GEMM kernel for sub-4-bit LLMs
     deps: [[int4]], [[lq-nets]]
     see-also: [[squeezellm]], [[gptq]], [[autogptq]]
-->

# LUT-GEMM (nuQmm): Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models
- **Core Insight:** When weights are non-uniformly quantized (BCQ / binary-coding format), GEMM can be expressed as a sum of `b` binary {−1,+1} matrix-vector products precomputed into a LUT indexed by 8 bits of activation at a time — eliminating dequantization entirely and beating GPTQ-style INT4 kernels.
- **Guideline:** For weight-only sub-4-bit LLM inference, use a LUT-GEMM-style kernel with binary-coded weights (1–4 bits encoded as `Σ α_i B_i`, B_i ∈ {−1,+1}) and 8-bit activation chunking; group_size=32–128 balances accuracy vs LUT-table size.
- **Authors:** Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, Dongsoo Lee
- **Year:** 2022 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2206.09557
- **Relevant topics:** LUT-based GEMM, binary-coding quantization, sub-4-bit weight-only kernel, dequant-free inference

## Abstract
LUT-GEMM (originally "nuQmm") targets the inference-memory bottleneck of LLMs by removing the dequantization step from low-bit weight-only GEMM. Weights are stored in binary-coding (BCQ) format `W ≈ Σ_{i=1..b} α_i ⊙ B_i` with B_i ∈ {−1, +1}. For each input row segment, the kernel precomputes a 256-entry LUT of inner products against the activation chunk; the multi-bit weight contribution is then a sequence of LUT look-ups and α-weighted accumulations. The kernel achieves a 2.1× speedup over OPTQ (GPTQ) at 3-bit on OPT-175B and supports flexible group-wise scaling.

## Key Contributions
- **LUT-based dequant-free GEMM**: replaces FP dequant + INT8 multiply with a sequence of 8-bit-indexed table lookups; especially well-suited to memory-bound LLM decode.
- Supports arbitrary bit-widths (1, 2, 3, 4) via binary-coded representation `Σ α_i B_i`, mapping naturally to b LUT layers.
- Per-group α with flexible group size G (typical 32–128) → tunable accuracy/throughput tradeoff.
- Demonstrates **3-bit OPT-175B** at 2.1× the throughput of GPTQ-INT4 kernels on a single A100.

## Key Figures/Tables to Study
- **Figure 4:** kernel diagram — pre-LUT construction from activation × {−1,+1} columns, then b layers of LUT lookups.
- **Table 3:** speedup vs GPTQ-INT4 kernels at 3/4-bit on OPT-30B/175B.

## Technical Details

### Binary-coded quantization (BCQ) format
For weight tile `W ∈ R^{d × G}` with group size G and b bits:
```
W ≈ Σ_{i=1..b} α_i · B_i,   B_i ∈ {−1, +1}^{d×G},   α_i ∈ R^{d}
```
α's are found by alternating optimization on the calibration set (as in LQ-Nets / BCQ literature). 1-bit = sign quant; b-bit = b binary planes summed.

### LUT-GEMM kernel
For one input activation row `x ∈ R^{G}` and a weight tile in BCQ form:
1. Split `x` into 8-bit-wide chunks (or any aligned chunk size).
2. For each binary plane `B_i`, the inner product `B_i · x` over each chunk depends only on the chunk's sign pattern → precompute a 256-entry LUT keyed by the 8-bit unsigned sign-pattern index.
3. The full output is `y = Σ_i α_i · LUT_i[chunk_pattern]`.

No FP dequantization: weights live as packed binary planes in DRAM, the LUT is in shared memory, accumulation is in FP16/FP32.

### Group-wise scale α
Per-row α per group of G consecutive input dims. Smaller G → higher accuracy, more LUT-table reads. Typical G = 32 or 128.

### Hyperparameters
| Knob | Value |
|------|-------|
| Bits b | 1, 2, 3, 4 |
| Group size G | 32–128 |
| Chunk size | 8 bits (one byte of activation) |
| Hardware target | A100 / H100 (CUDA shared-mem LUT) |
| Calibration | weight-only, no activation calibration |

## Connections
- Same family (binary-code weight quant): [[lq-nets]].
- INT-based GEMM kernel rival: [[marlin-kernel]] (W4A16 Marlin from IST-Austria), [[machete-kernel]].
- Non-uniform LUT-quant cousin with sensitivity weighting: [[squeezellm]].
- Used as a baseline kernel for [[autogptq]] and llama.cpp k-quant lineage [[gguf-k-quants]].
