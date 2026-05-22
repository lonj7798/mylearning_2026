<!-- scope: Atom — W4A4 + KV4 PTQ with sub-channel reorder + dynamic activation quant for LLM serving
     deps: [[smoothquant]], [[gptq]]
     see-also: [[qserve]], [[awq]], [[kivi]]
-->

# Atom: Low-bit Quantization for Efficient and Accurate LLM Serving
- **Core Insight:** End-to-end LLM serving throughput is bandwidth-bound, so pushing both weights *and* activations *and* the KV cache to 4-bit is what matters; mixed-precision *sub-channel* reorder isolates the small fraction of outlier channels into a separate INT8 path while keeping the bulk at INT4, recovering accuracy with negligible kernel overhead.
- **Guideline:** When deploying for inference throughput on Ampere/Hopper, use Atom: W4A4 with sub-channel reorder (top-K outlier channels promoted to INT8) + per-token dynamic INT4 activation quant + per-head INT4 KV cache; achieves 7.7× FP16 / 2.5× INT8 throughput at same latency.
- **Authors:** Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, Baris Kasikci (UW + CMU)
- **Year:** 2024 (MLSys 2024)
- **URL:** https://arxiv.org/abs/2310.19102
- **Relevant topics:** W4A4 + KV4, sub-channel reorder, dynamic activation quant, LLM serving throughput

## Abstract
Atom is a low-bit LLM serving stack delivering full W4A4 + KV4 quantization with throughput gains of 7.7× over FP16 and 2.5× over INT8 at the same latency target on A100. The accuracy loss is bounded by combining (1) sub-channel reordering and mixed-precision handling of outlier channels, (2) per-token dynamic quantization for activations, and (3) per-head INT4 KV-cache quantization with grouped scaling. Atom ships as kernels integrated into a custom serving runtime.

## Key Contributions
- Sub-channel reorder: pre-permutation that groups outlier channels together so they can be routed to a small INT8 sub-matmul while the bulk runs INT4.
- Per-token dynamic activation quantization (no static calibration scales — recomputed per inference call) — preserves accuracy across distribution shifts.
- Per-head, per-token KV-cache quantization at INT4 with group size 128.
- Fused CUDA kernels: dequant + INT4-INT4 GEMM + outlier INT8 path, single launch.
- Demonstrates 7.7× FP16 / 2.5× INT8 throughput on A100 for OPT and LLaMA serving.

## Key Figures/Tables to Study
- **Figure 4:** Sub-channel reorder schematic — outlier channels gathered into a dense INT8 block.
- **Figure 6:** End-to-end throughput bar chart — Atom vs SmoothQuant vs LLM.int8() vs FP16.
- **Table 3:** PPL on LLaMA-7B/13B — Atom W4A4 vs SmoothQuant W4A4 vs FP16.

## Technical Details

### Sub-channel reorder + mixed precision
For each weight matrix W ∈ ℝ^{C_out × C_in}:
1. From calibration, identify the top-K input channels by maximum activation magnitude (typically K ≈ 128, ~3% of channels).
2. Permute these K channels to the front: W = [W_outlier | W_normal], x = [x_outlier; x_normal].
3. Quantize W_outlier and x_outlier to INT8; W_normal and x_normal to INT4.
4. Compute `y = W_outlier · x_outlier (INT8) + W_normal · x_normal (INT4)` and sum.

The INT8 path takes only K/C_in fraction of the total FLOPs, so the dequantization and INT8 matmul cost is amortized — negligible runtime overhead.

### Per-token dynamic activation quantization
At every inference step, compute the per-token absmax of the activation tensor and rescale to INT4:
`scale_t = max_i |x_{t,i}| / 7`,  `x̂_{t,i} = round(x_{t,i} / scale_t)`
No static calibration scales — robust to distribution shift across prompts.

### KV cache at INT4
- K: per-head per-token, group size 128 along channel dim.
- V: per-head per-token, group size 128 along channel dim.
Stored as INT4 packed; dequantized on-the-fly inside the attention kernel.

### Weight quant (W4)
GPTQ-style Hessian-aware INT4 with group size 128. After sub-channel reorder is applied, the outlier columns are stored separately at INT8.

### Kernel
Single fused CUDA kernel per linear: load INT4 weight tile + INT4 activation tile → dequant in registers → tensor-core matmul → accumulate; outlier path runs in a small parallel SM block. Uses CUTLASS Mixed-precision tile patterns.

### Why the throughput numbers
Memory bandwidth = N_weights × bits_per_weight. INT4 = 4× FP16. Activation read also 4×. KV cache read 4×. End-to-end memory ≈ 4× smaller → ~4× throughput, with the rest of the gain coming from larger achievable batch size at the same KV-cache footprint.

## Connections
- Sibling W4 + KV4 serving systems: [[qserve]] (W4A8KV4, Hopper-optimized).
- Activation-aware predecessor: [[smoothquant]], [[awq]].
- Outlier-channel isolation lineage: [[llm-int8]] (similar idea at INT8).
- KV-quant siblings: [[kivi]], [[kvquant]].
- Weight quantizer: [[gptq]] used as W4 baseline.
