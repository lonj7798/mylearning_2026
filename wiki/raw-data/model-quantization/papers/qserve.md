<!-- scope: QServe (QoQ) — W4A8KV4 serving with progressive group quantization and register-level dequant on Hopper
     deps: [[smoothquant]], [[gptq]], [[awq]]
     see-also: [[atom]], [[marlin-kernel]], [[kivi]]
-->

# QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving
- **Core Insight:** Naive INT4 dequantization on GPUs incurs 20–90% runtime overhead because dequant happens at the wrong memory hierarchy level (HBM ↔ SMEM); QServe co-designs the quant scheme (progressive *group-then-channel* W4) and the kernel (register-level dequant on Hopper tensor cores) so the dequant cost is amortized inside the GEMM.
- **Guideline:** For Hopper (H100/H200) production LLM serving, prefer W4A8KV4 (QServe / QoQ) over W4A4: the A8 activation path keeps tensor-core utilization high while the W4 weight path delivers HBM-bandwidth savings; SmoothAttention prevents KV4 accuracy loss.
- **Authors:** Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han (MIT HAN Lab)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.04532
- **Relevant topics:** W4A8KV4, progressive group quantization, register-level dequant, SmoothAttention, Hopper kernel

## Abstract
QServe presents the QoQ (Quattuor-Octo-Quattuor: 4-8-4) quantization algorithm and a co-designed serving system. Existing INT4 methods suffer 20–90% overhead dequantizing weights or partial sums on GPUs. QoQ introduces (1) progressive group quantization for weights, (2) SmoothAttention to mitigate KV4 accuracy loss from attention dot-products, and (3) compute-aware weight reordering. The system reaches 1.2–3.5× throughput vs prior W4A8 / INT4 serving systems across Llama-3-8B and Qwen1.5-72B on A100/L40S.

## Key Contributions
- W4A8KV4 quantization regime: 4-bit weights, 8-bit activations, 4-bit KV cache — the sweet spot for Hopper tensor cores.
- Progressive group quantization: weights are first per-channel quantized to INT8 then per-group sub-quantized to INT4, so register-level dequant produces INT8 (not FP) operands the INT8 tensor-core can consume directly.
- SmoothAttention: SmoothQuant-style equivalent transformation applied to attention scores to mitigate the impact of KV4 quantization on softmax stability.
- Compute-aware weight reordering: permutes weight tiles to match tensor-core fragment layout, eliminating shared-memory bank conflicts.
- Fused attention with KV4 + dynamic activation quant.

## Key Figures/Tables to Study
- **Figure 3:** Progressive group quant pipeline (INT4 → INT8 register-level dequant → INT8 tensor-core GEMM).
- **Figure 5:** SmoothAttention insertion in the attention block.
- **Table 4:** Throughput comparison QServe vs TensorRT-LLM W8A8 vs Atom W4A4 on L40S/A100/H100.

## Technical Details

### Progressive group quantization (the key idea)
- **Stage 1 (per-channel INT8):** Quantize each output channel of W to INT8 with a single per-channel scale s_c ∈ FP16. Store as INT8 W_s.
- **Stage 2 (per-group INT4):** Within each group of g=128 weights along the input axis, quantize the INT8 values to INT4 with a per-group scale s_g ∈ INT8 (note: scale is itself an integer, not FP). Store as INT4 W_g.
- **At inference:** dequantize INT4 → INT8 entirely *in registers* (multiply by s_g, an INT8×INT8 → INT16 op cheap on tensor cores); feed INT8 operand into the INT8 tensor-core GEMM with A8 activation. No FP dequant in the critical path.

The reason this matters: prior W4A8 (e.g. Marlin) dequantizes INT4 → FP16 in registers and feeds FP16 into the tensor core. The FP dequant adds 2–3 instructions per element and pushes register pressure. QoQ keeps everything integer until the GEMM accumulator.

### SmoothAttention
Attention `softmax(QK^T/√d)V` is sensitive to KV4 because INT4 K introduces noise that gets amplified by softmax. QoQ applies a SmoothQuant-style learnable per-head scaling:
`Q' = Q · s,  K' = K / s`
so QK^T is unchanged but K' has reduced dynamic range, making K4 quantization gentler. s is calibrated to minimise softmax KL divergence.

### Compute-aware weight reorder
Weight tiles are pre-permuted to match the Tensor Memory Accelerator (TMA) layout on Hopper, avoiding bank conflicts during the shared-memory → register load.

### Activation A8 dynamic per-token quant
Per-token absmax to INT8 — like SmoothQuant but without the static calibration scale.

### KV4 layout
- K: per-head, per-token, INT4 with per-group scale.
- V: per-head, per-token, INT4 with per-group scale.
Fused into attention kernel that dequantizes on-the-fly.

### Throughput results
- Llama-3-8B: 1.2× over TensorRT-LLM W8A8 on H100, 2.4× on L40S.
- Qwen1.5-72B: 3.5× over Atom W4A4 on A100 (Atom is hurt by softmax instability from A4).

## Connections
- Direct W4A8 ancestor for the dequant strategy: [[marlin-kernel]] (W4A16 GEMM); QServe pushes to W4A8.
- Activation-side smoothing lineage: [[smoothquant]] → SmoothAttention here.
- Weight-quant calibration: [[gptq]] / [[awq]] both used as W4 baseline.
- Sibling W4A4KV4 serving system: [[atom]].
- KV-quant siblings: [[kivi]], [[kvquant]].
- HAN-Lab quant lineage: [[awq]], [[smoothquant]], [[squeezellm]].
