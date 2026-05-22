<!-- scope: early FP8 LLM inference studies showing FP8 PTQ parity with FP16
     deps: [[fp8-formats-paper]], [[fp8-e4m3]]
     see-also: [[zeroquant-fp]], [[smoothquant]], [[fp8-lm]]
-->

# FP8 LLM Inference — early studies (2022–2023)
- **Core Insight:** Per-tensor E4M3 weight and activation quantization is essentially lossless for LLM inference up to 175B parameters, with the wider dynamic range of FP8 absorbing the outliers that defeat INT8 — making FP8 the simplest possible "drop-in" replacement for FP16 inference on H100-class hardware.
- **Guideline:** For H100 deployment, prefer E4M3 weights + E4M3 activations with per-tensor (calibrated) scales; only fall back to INT8 + SmoothQuant when targeting non-FP8 hardware. Use static (calibration-set) scale rather than dynamic per-token to avoid kernel-launch overhead.
- **Authors:** Various — NVIDIA Applied Deep Learning Research, Hugging Face Optimum / Intel Neural Compressor practitioners
- **Year:** 2022–2023
- **URL:** NVIDIA H100 FP8 inference blog https://developer.nvidia.com/blog/nvidia-h100-tensor-core-gpu-architecture-deep-dive/ ; OCP FP8 spec https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1 ; Hugging Face FP8 deep-dive https://huggingface.co/blog/hf-bitsandbytes-integration (companion thread)
- **Relevant topics:** FP8 PTQ inference, H100 tensor cores, per-tensor scale calibration, FP8 vs INT8

## Abstract
A cluster of early industry studies (NVIDIA, Hugging Face, Intel Neural Compressor) showed in 2022–2023 that FP8 (specifically E4M3 from [[fp8-formats-paper]]) is essentially a free lunch for LLM inference on H100-class hardware. Unlike INT8 — which requires outlier handling ([[llm-int8]]) or equivalent-transformation preprocessing ([[smoothquant]]) — FP8's larger dynamic range absorbs activation outliers directly. The practical PTQ recipe is: collect per-tensor absmax on a small calibration set, set `s = max(|X|) / 448`, cast to E4M3, run native tensor-core GEMM. Quality matches BF16 across OPT, BLOOM, LLaMA, and GPT-3 series up to 175B without algorithmic tricks.

## Key Findings (consolidated across early reports)
- **Per-tensor static FP8 PTQ** is sufficient — no per-token dynamic scale, no SmoothQuant-style migration needed, no outlier sidecar.
- FP8 weights and FP8 activations independently come within 0.1 ppl of FP16; combined W8A8-FP within 0.2 ppl.
- Quantization granularity: **per-tensor** for activations, **per-row** for weights, scales in FP32.
- On H100, FP8 GEMM is 2× BF16 throughput; on B100/B200, FP8 is 2× FP8-Hopper and stacks with NVFP4.

## Calibration Recipe
1. Run ~128–512 calibration sequences through the FP16 model.
2. Per linear layer, record per-tensor absmax of input X and per-row absmax of W.
3. Set `s_X = max|X| / 448`, `s_W = max|W_row| / 448`.
4. Cast `X_fp8 = round_E4M3(X / s_X)`, `W_fp8 = round_E4M3(W / s_W)`.
5. Inference: `Y = s_X · s_W · TensorCoreGEMM_FP8(X_fp8, W_fp8^T)` with FP32 accumulation.

The 448 is the largest E4M3-representable value (see [[fp8-e4m3]]).

## Technical Details

### Format used
E4M3, exponent bias = 7, no infinities, single NaN — provides ~6× the dynamic range of INT8 (`[2⁻⁹, 448]` vs `[1, 127]`) at slightly worse precision in the mid-range.

### Scale handling
- **Static / calibrated**: scale is fixed after calibration; no runtime cost. Used by TensorRT-LLM ([[tensorrt-llm-quant]]).
- **Dynamic per-tensor**: scale recomputed each forward; trivial CUDA cost.
- **Per-token / per-channel**: rarely needed — FP8's range usually makes finer granularity unnecessary.

### Comparison to INT8
| Aspect | INT8 | FP8 (E4M3) |
|--------|------|-------------|
| Dynamic range | [1, 127] | [2⁻⁹, 448] (~6× wider) |
| Outlier handling | needs SmoothQuant / decomposition | usually none needed |
| Calibration | per-token or per-tensor + outlier scan | per-tensor (static) suffices |
| H100 throughput | 2× BF16 | 2× BF16 |
| B100 throughput | 2× BF16 | 2× BF16 (+ NVFP4 lane available) |

### Caveats
- Some LLaMA-2/3 layers' input embeddings have isolated outliers > 448 → needs per-token scale on the embedding output.
- KV cache in FP8 is straightforward; INT8 KV needs more care.

## Connections
- The spec: [[fp8-formats-paper]], [[fp8-e4m3]], [[fp8-e5m2]].
- FP8 training (the orthogonal axis): [[fp8-lm]], [[transformer-engine]], [[deepseek-v3-fp8]].
- INT8 alternatives that FP8 obviates: [[llm-int8]], [[smoothquant]].
- Practitioner blog reference: [[hf-fp8-deep-dive]].
- FP8 LLM PTQ formal study: [[zeroquant-fp]].
- Hardware vendor blogs: [[nvidia-h100-fp8]], [[amd-mi300-fp8]].
