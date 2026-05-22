<!-- scope: NVIDIA NVFP4 — block-scaled FP4 with FP8 block scale + FP32 tensor scale; the Blackwell production 4-bit format
     deps: [[fp4-e2m1]], [[mx-formats]], [[fp8-e4m3]]
     see-also: [[nvfp4-training]], [[microscaling-formats]]
-->

# NVFP4 (NVIDIA Blackwell-Era 4-bit Float)
- **Core Insight:** NVFP4 takes MXFP4's block-scaled FP4 and *doubles the block-scale resolution*: block size 16 (vs MXFP4's 32) and FP8 E4M3 block scale (vs MXFP4's E8M0 power-of-2-only scale), plus an additional FP32 per-tensor scale on top — buying ~0.5 PPL of additional quality at the cost of 0.25 → 0.5 extra bits/element.
- **Guideline:** When deploying on NVIDIA Blackwell, choose NVFP4 over MXFP4 if you need the absolute best 4-bit quality and have proprietary-format tolerance; choose MXFP4 if you need cross-vendor portability or are bandwidth-bound (NVFP4 has 50% more scale overhead).
- **Authors:** NVIDIA (introduced with Blackwell architecture announcement, 2024)
- **Year:** 2024 (Blackwell announcement); 2025 (production deployment in TensorRT-LLM, vLLM)
- **URL:** https://developer.nvidia.com/blog/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing/ ; https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html (Transformer Engine NVFP4 API)
- **Relevant topics:** NVFP4, Blackwell, FP4, two-level block scale, production 4-bit training

## Abstract
NVFP4 is NVIDIA's production 4-bit floating-point format introduced with the Blackwell GPU architecture (B100/B200, 2024). It uses the same E2M1 element format as MXFP4 but with two key differences: (1) **finer block size** — 16 elements per block instead of MXFP4's 32; (2) **higher-precision block scale** — FP8 E4M3 (4 exp + 3 mantissa) instead of MXFP4's E8M0 (exponent-only). NVFP4 additionally tracks an **FP32 per-tensor scale** to cover the case when block-scale dynamic range itself overflows. Together these give ~0.5 PPL better quality than MXFP4 on Llama-class LLMs, at the cost of 50% more scale overhead (4.5 effective bits/element vs MXFP4's 4.25). NVFP4 is the NVIDIA-recommended 4-bit format for both inference and training on Blackwell.

## Key Contributions
- Two-level block scale: FP8 E4M3 per 16-element block + FP32 per-tensor scale.
- Finer block size (16 vs MXFP4's 32) → better outlier containment.
- Mantissa-bearing block scale → continuous (not power-of-2-only) per-block magnitude.
- Production deployment in Transformer Engine, TensorRT-LLM, vLLM.
- Demonstrates 4-bit *pretraining* viability on Blackwell at trillion-token scale (see [[nvfp4-training]]).

## Key Figures/Tables to Study
- **NVFP4 vs MXFP4 quality vs bit-budget**: NVFP4 wins by ~0.5 PPL on Llama-7B/13B at 4-bit weight, near-parity at higher precision.
- **Two-level scale diagram**: FP32 tensor scale × FP8 block scale × E2M1 element value = decoded FP value.

## Technical Details

### NVFP4 number structure
```
NVFP4 tensor = (s_tensor, [(S_b, [P_b,0, P_b,1, ..., P_b,15]) for b in blocks])
```
- **s_tensor**: FP32 per-tensor scale; one value for the whole tensor.
- **S_b**: FP8 E4M3 block scale; one value per 16-element block.
- **P_b,i**: E2M1 element value; 16 per block.

### Decoded value
```
value(P_b,i) = s_tensor · decode_e4m3(S_b) · decode_e2m1(P_b,i)
```
where decode_e4m3 returns the FP value of the E4M3 code (see [[fp8-e4m3]]) and decode_e2m1 returns the FP value of the E2M1 code (one of {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}; see [[fp4-e2m1]]).

### Element format: E2M1 (same as MXFP4)
- 1 sign + 2 exp + 1 mantissa
- 16 codes: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- See [[fp4-e2m1]] for the full table.

### Block scale format: FP8 E4M3
- 1 sign + 4 exp + 3 mantissa, bias 7
- Range 0 to 448 (sign bit unused for scale → effectively unsigned).
- 8 mantissa values per exponent step → continuous magnitudes (vs MXFP4's E8M0 power-of-2-only).
- See [[fp8-e4m3]] for full layout.

### Tensor scale: FP32
- Stored once per tensor.
- Provides "headroom" when the FP8 block scale itself overflows.
- Recipe: choose s_tensor such that the max block-scale magnitude after division by s_tensor fits in E4M3's 448 max.

### Bit budget
```
4 (element)
+ 8/16 (FP8 block scale per 16 elements)
+ 32/N_total (negligible FP32 tensor scale)
≈ 4.5 effective bits/element
```
vs MXFP4: 4 + 8/32 = 4.25 effective bits/element.

### Comparison with MXFP4
| | MXFP4 | NVFP4 |
|---|---|---|
| Element format | E2M1 | E2M1 |
| Block size | 32 | 16 |
| Block scale | E8M0 (8b, exp-only) | FP8 E4M3 (8b, exp+mant) |
| Per-tensor scale | none | FP32 |
| Effective bits/elem | 4.25 | 4.5 |
| Quality (Llama-7B PPL gap from BF16) | ~0.3 | ~0.15 |
| Hardware | Blackwell, MI355X, Gaudi 3 | Blackwell only |
| Standardization | OCP | NVIDIA-proprietary |

### Why finer block size + mantissa-bearing scale helps
- **Block 16 vs 32**: outliers in a single 16-element neighborhood are less likely to dominate a smaller block's scale; effectively halves the "outlier blast radius."
- **E4M3 vs E8M0 scale**: E8M0 can only express scales of the form 2^k → a block whose true max-magnitude is 1.4 must round its scale up to 2 (wasting half the FP4 range) or down to 1 (clipping 40% of the block). E4M3's mantissa lets you store scale = 1.5 → no rounding loss on the scale.

### Use in training
NVFP4 is the basis for native FP4 pretraining on Blackwell:
- Forward: weights and activations in NVFP4; tensor-core matmul in FP4 with FP32 accumulator.
- Backward: gradients in NVFP4 (or E5M2-element NVFP4 variant for wider gradient range).
- Master weights: FP32.
- Stochastic rounding on weight update (preserves expectation; see [[stochastic-rounding]]).

### Use in inference
- **W4A4 (weights and activations both NVFP4)**: production target for serving on Blackwell; ~2× throughput over FP8.
- **W4A16**: weights NVFP4, activations BF16; for backward-compatibility with non-Blackwell deployments.
- **KV-cache in NVFP4**: 4× memory savings over FP16 KV cache; viable with per-token/per-channel partitioning.

### Hardware support
- **NVIDIA B100 / B200 (Blackwell)**: native NVFP4 tensor cores at ~9 PFLOPS (dense), ~18 PFLOPS sparse.
- **TensorRT-LLM**: NVFP4 inference path with Transformer Engine.
- **vLLM**: NVFP4 weight loading and Marlin-style kernels announced.
- Pre-Blackwell hardware: NVFP4 weights can be stored but require dequant-to-BF16 emulation.

### Limitations
- Proprietary format → not portable to AMD / Intel hardware (those use MXFP4).
- 0.25 extra bits/element vs MXFP4 → marginal memory cost.
- Two-level scale dequant logic is slightly more complex than MX's single-level.

## Connections
- [[fp4-e2m1]] — the underlying element format.
- [[mx-formats]] — the cross-vendor competing standard.
- [[fp8-e4m3]] — used as the NVFP4 block scale.
- [[microscaling-formats]] — Rouhani 2023, the conceptual ancestor.
- [[nvfp4-training]] — production FP4 pretraining recipes on Blackwell.
- [[transformer-engine]] — NVIDIA's FP4/FP8 software stack.
