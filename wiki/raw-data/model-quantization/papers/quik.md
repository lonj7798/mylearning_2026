<!-- scope: QUIK — end-to-end W4A4 PTQ with outlier preservation in higher precision
     deps: [[gptq]], [[smoothquant]]
     see-also: [[atom]], [[quarot]], [[omniquant]]
-->

# QUIK: Towards End-to-End 4-Bit Inference on Generative Large Language Models
- **Core Insight:** True end-to-end W4A4 inference is achievable on LLMs if you accept a small fraction of outlier weights and activations stay at INT8 — a "mostly-4-bit" GEMM with a small dense INT8 sidecar that handles outlier rows/cols, dispatched through a custom GPU kernel.
- **Guideline:** When the goal is real 4-bit *throughput* (not just memory), use QUIK with W4A4 plus an outlier mask (~0.5–1% of channels at INT8) and the QUIK kernel; combine with 2:4 sparsity for an additional ~2× weight memory savings.
- **Authors:** Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, Dan Alistarh
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.09259
- **Relevant topics:** W4A4 PTQ, end-to-end 4-bit inference, outlier preservation, 2:4 sparsity, IST-Austria kernel

## Abstract
QUIK is one of the first PTQ pipelines to deliver actual end-to-end W4A4 inference speedup (3.4× over FP16) on LLaMA / OPT / Falcon, by combining GPTQ-style weight quantization with INT4 activation quantization plus an outlier-preserving INT8 sidecar for the small set of weight rows / activation columns whose dynamic range is too extreme for INT4. The authors also demonstrate composition with 2:4 structured sparsity for additional weight memory savings. The custom CUDA kernel performs the W4×A4 inner product on tensor cores and the small W8×A8 outlier inner product, fused.

## Key Contributions
- End-to-end (weights AND activations) 4-bit PTQ for LLMs with realised speedup, not just memory.
- Outlier-preserving INT8 sidecar with **structured** (column/row) granularity that suits tensor-core kernels.
- Custom W4A4 + W8A8 fused GEMM kernel with 3.4× end-to-end speedup vs FP16 on A100.
- Demonstrated composition with 2:4 structured sparsity → additional ~2× weight memory.

## Key Figures/Tables to Study
- **Figure 2:** the QUIK GEMM diagram — 4-bit dense path + 8-bit outlier path.
- **Table 3:** OPT/LLaMA-2 W4A4 perplexity vs SmoothQuant + GPTQ — QUIK is the only one viable at A4.
- **Table 6:** end-to-end throughput on A100 — 3.4× FP16 for LLaMA-2 7B.

## Technical Details

### Outlier identification
For each linear layer, identify outlier *channels* (input dims) by per-channel activation max (calibration data) and outlier *rows* (output dims) by weight sensitivity. Top ~0.5–1% kept at INT8; rest at INT4.

### Mixed-precision GEMM
For `Y = X · W^⊤` with X ∈ R^{B×C_in}, W ∈ R^{C_out×C_in}:
- Partition C_in into `C_in = C_in^{4bit} ∪ C_in^{8bit}`.
- Partition C_out into `C_out = C_out^{4bit} ∪ C_out^{8bit}`.
- Four sub-matmuls:
  ```
  Y_44 = X_4 · W_44^⊤  (W4A4 tensor core)
  Y_48 = X_4 · W_48^⊤  (W8A4 lane)
  Y_84 = X_8 · W_84^⊤  (W4A8 lane)
  Y_88 = X_8 · W_88^⊤  (W8A8 tensor core)
  ```
- Sum the four partial results.

In practice the 8-bit slices are tiny (0.5–1% of channels) → the W4A4 path carries the work.

### Weight quantization
GPTQ-based on the W4 partition, RTN on the W8 partition. Group_size=128, percdamp=0.01.

### Activation quantization
Per-token absmax INT4 on the X_4 slice; per-token INT8 on the X_8 outlier columns. The outlier-column choice is fixed at calibration time → no runtime branching.

### 2:4 sparsity composition
Optional: enforce 2-out-of-4 structured zeros on the W_44 block (NVIDIA Ampere/Hopper sparse tensor cores) for an additional ~2× weight memory and a small throughput uplift on sparse-supported kernels.

### Hyperparameters
| Knob | Value |
|------|-------|
| Weight bits | 4 (dense), 8 (outlier rows) |
| Activation bits | 4 (dense), 8 (outlier columns) |
| Outlier fraction | 0.5–1% per axis |
| Group size | 128 |
| Calibration | 128 × 2048 |
| Sparsity (optional) | 2:4 structured |
| Kernel | custom CUDA, fused W4A4 + W8A8 |

## Connections
- Same lab predecessors: [[gptq]], [[marlin-kernel]] (W4A16 kernel).
- W4A4 + KV4 sibling with sub-channel reorder: [[atom]].
- Outlier-preserving weight-only cousins: [[spqr]], [[owq]], [[squeezellm]].
- Rotation-based successor that obviates the outlier sidecar: [[quarot]].
- Learnable W4A4 alternative: [[omniquant]].
