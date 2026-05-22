<!-- scope: QoQ — quattuor-octō-quattuor (W4A8KV4) progressive quantization for LLM serving
     deps: [[awq]], [[smoothquant]], [[kivi]]
     see-also: [[atom]], [[qserve]]
-->

# QoQ / QServe: W4A8KV4 with Progressive Group Quantization
- **Core Insight:** The bottleneck for high-batch LLM serving is dequant-overhead per GEMM, not memory bandwidth — by quantizing weights in two stages (4-bit per-group → 8-bit per-channel scale that the tensor core consumes natively) the dequant inside the GEMM main loop collapses to a single low-bit op, unlocking W4A8 throughput that matches FP16 at high batch while still using 4-bit weight storage.
- **Guideline:** For high-concurrency serving where prior 4-bit serving stacks saturate, use the W4A8KV4 (QoQ) recipe: progressive 4→8 weight quantization, INT8 activations via SmoothAttention/SmoothQuant scales, and 4-bit per-channel K + 4-bit per-channel V KV cache.
- **Authors:** Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han (MIT Han Lab)
- **Year:** 2024-2025
- **URL:** https://arxiv.org/abs/2405.04532 (QServe paper) • https://github.com/mit-han-lab/qserve
- **Relevant topics:** W4A8KV4 serving, progressive quantization, KV-cache quant, batched throughput, kernel design

## Abstract
W4A16 inference (Marlin / Machete / TinyChat) is excellent at low batch but the dequant overhead inside the main loop dominates at large batches. QoQ (Quattuor-Octō-Quattuor — 4/8/4 for weights/activations/KV) is the MIT Han Lab response: weights are stored as 4-bit per-group but the dequant inside the GEMM main loop only has to expand to 8-bit (via a per-channel INT8 scale layer applied at load time), so the tensor-core sees an INT8×INT8 GEMM that is roughly 2× FP16 throughput. Activations are INT8 via a SmoothAttention-extended SmoothQuant calibration; the KV cache is INT4 per-channel for both K and V. QServe is the serving system that ships the recipe with custom CUTLASS kernels and continuous batching; it reports 1.2-3.5× throughput over vLLM-FP16 at frontier batch sizes on Llama-3-70B.

## Key Contributions
- **Progressive group quantization**: weights stored at INT4 per-group (G=128 channel groups), but the per-group FP16 scale is decomposed into an INT8 *per-channel* scale + a small FP16 *per-group* correction. The main GEMM loop reads the INT4 weight and the INT8 per-channel scale, multiplies them into an INT8 operand in registers, and feeds INT8×INT8 to the tensor core. The per-group correction is applied as a cheap epilogue.
- **SmoothAttention**: extends SmoothQuant from the FFN to the attention QKV projections by migrating outlier scales through K/V along with W; needed because attention activations have a distinctly different outlier pattern from FFN inputs.
- **W4A8KV4**: the deployed quantization tuple — 4-bit weights, 8-bit activations, 4-bit KV cache.
- **Custom CUTLASS kernels**: INT8×INT8 main loop with the per-group epilogue fused; 2× FP16 throughput at high batch on A100/H100.
- **Continuous batching + paged KV** integrated; serving system released as QServe.

## Key Figures/Tables to Study
- The "progressive dequant" figure: shows the two-stage weight reconstruction (INT4 → INT8 in main loop, INT8 → FP16 in epilogue) and why it changes the inner-loop arithmetic.
- The throughput table on Llama-3-8B / 70B comparing QServe (W4A8KV4) to vLLM-FP16, TensorRT-LLM-W4A16, and Atom W4A4 at various batch sizes.
- The SmoothAttention scale-migration diagram (analogous to the SmoothQuant diagram for FFN).

## Technical Details

### Format
- **Weights:** INT4, per-group (G=128). Each group also carries an INT8 per-channel scale (computed at quantization time) and a small FP16 per-group correction.
- **Activations:** INT8, per-token dynamic scale (SmoothQuant style) computed at runtime.
- **KV cache:** INT4 K + INT4 V, per-channel scale (asymmetric per-head treatment from KIVI-style empirical observation that K is channel-heterogeneous, V less so).

### Kernel structure (main loop)
```
for each K tile:
  load INT4 weight tile + INT8 per-channel scale
  dequant in-register: int8_w = int4_w * int8_scale  (single MUL)
  load INT8 activation tile
  tensor_core_mma(int8_w, int8_a, int32_acc)
epilogue:
  multiply int32_acc by (per-group FP16 correction * per-token FP16 act scale)
  cast to FP16 output
```
The expensive per-group + per-token scale combination only happens once per output tile in the epilogue; the main loop is pure INT8 GEMM.

### SmoothAttention
- Stock SmoothQuant migrates outlier scale from FFN activations into the FFN weight; this fails for attention QKV because the outlier pattern is per-head, not per-channel.
- SmoothAttention pre-computes a per-head migration scale s_h that scales K by 1/s_h and W_K by s_h, leaving Q · K^T mathematically identical but with the activation outliers absorbed into the weight (which is INT4 anyway).
- Same migration done for V.

### Throughput numbers (Han Lab paper)
| Model | Hardware | Recipe | Throughput vs FP16 vLLM |
|-------|----------|--------|--------------------------|
| Llama-3-8B | A100 | W4A8KV4 | 1.5-2.4× |
| Llama-3-70B | A100 | W4A8KV4 | 2.4-3.5× |
| Llama-2-70B | L40S | W4A8KV4 | 2-3× |

### Comparison to W4A16 and W4A4
- W4A16 (Marlin/Machete) wins at batch 1; QoQ wins at batch ≥ 16-32 because INT8 activation is twice the tensor-core throughput.
- W4A4 (Atom) wins on theoretical throughput but loses accuracy at INT4 activations; QoQ's INT8 activations preserve quality.

## Connections
- [[awq]] — the activation-aware weight quantization line; QoQ inherits AWQ's per-channel scale idea for weights.
- [[smoothquant]] — the SmoothQuant outlier migration is extended to attention here.
- [[atom]] — sibling W4A4 + KV4 approach; trades activation precision for higher tensor-core throughput.
- [[qserve]] — same MIT Han Lab system; this page covers the algorithm, qserve.md covers the serving stack.
- [[kivi]] — KV-cache quant lineage that motivates the asymmetric K/V treatment.
- [[han-song-mit]] — the Song Han lab page that produced AWQ, SmoothQuant, QoQ, SmoothAttention.
