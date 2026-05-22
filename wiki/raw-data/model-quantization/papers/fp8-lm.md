<!-- scope: FP8-LM — Microsoft framework for FP8 mixed-precision LLM training (gradients + optimizer + comms)
     deps: [[fp8-formats-paper]], [[bf16]]
     see-also: [[transformer-engine]], [[deepseek-v3-fp8]], [[megatron-fp8]]
-->

# FP8-LM: Training FP8 Large Language Models
- **Core Insight:** Most LLM training tensors — gradients, optimizer states (m, v), all-reduce traffic — tolerate FP8 just as well as the matmul inputs do; pushing FP8 beyond the GEMM into the optimizer + communication layer gives a second 30%+ memory reduction and a second large speedup, with no quality loss vs BF16.
- **Guideline:** For from-scratch LLM pretraining on H100, adopt FP8 progressively in three levels: (L1) FP8 GEMM (E4M3 forward, E5M2 backward), (L2) add FP8 master-weight gradient + FP8 distributed all-reduce, (L3) add FP8 optimizer states; expect ~40% memory cut and ~75% speedup over Megatron-BF16 at GPT-175B.
- **Authors:** Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, Ruizhe Wang, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, Peng Cheng
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.18313
- **Relevant topics:** FP8 training, mixed-precision, gradient quantization, FP8 optimizer states, FP8 all-reduce, MS-AMP

## Abstract
FP8-LM is the first end-to-end FP8 LLM pretraining framework. Beyond FP8 matmul (which NVIDIA Transformer Engine already supports), FP8-LM pushes FP8 into three additional layers: (1) FP8 gradients with per-tensor scales managed by a "scaling factor predictor" to avoid underflow; (2) FP8 distributed all-reduce communication that halves cross-GPU traffic; (3) FP8 Adam optimizer states (m, v) with paired FP8 master scales. The framework is opt-in in three levels (L1: GEMM only; L2: + grad + comms; L3: + optimizer), each preserving training-loss equivalence with BF16. On H100, GPT-175B reaches 39% lower memory and 75% faster wall-clock than Megatron-LM-BF16, and 37% faster than NVIDIA Transformer Engine.

## Key Contributions
- Three-level FP8 training framework with monotone memory/speed gains and no quality regression.
- **Scaling factor management for FP8 gradients** — predictive per-tensor scales to keep gradient magnitudes inside the E5M2 representable range.
- **FP8 all-reduce** — halves DP/TP communication volume.
- **FP8 Adam states** — m in E4M3 (high-precision, low-range), v in E5M2 (wide-range, low-precision), each with its own master scale.
- Open-sourced as **MS-AMP** (https://github.com/Azure/MS-AMP) and integrated into Megatron-DeepSpeed.

## Key Figures/Tables to Study
- **Figure 1:** the three FP8 levels and what each one quantizes.
- **Figure 5:** GPT-175B loss curves — FP8-L3 overlays BF16 within noise.
- **Table 3:** memory + throughput at 7B, 13B, 175B — 39% memory cut, 75% faster than Megatron-BF16.

## Technical Details

### Level 1 — FP8 GEMM (baseline)
Same as NVIDIA Transformer Engine ([[transformer-engine]]):
- Forward weights and activations in E4M3 with per-tensor scale.
- Backward gradients in E5M2 with per-tensor scale.
- Accumulation in FP32 inside tensor cores.
- Per-tensor scales updated each step via running max with delayed scaling.

### Level 2 — FP8 gradients + FP8 communication
Gradients in E5M2 with predicted per-tensor scale; the predictor maintains an EWMA of the absmax of the previous step's gradient and bumps it preemptively if a near-overflow is detected.

FP8 all-reduce:
- Pre-reduce: convert local FP16/BF16 gradients to E4M3 with shared per-bucket scale.
- Network transport: FP8 tensors halve the bandwidth.
- Post-reduce: dequant to FP32 for the optimizer step.
- Lossy compression artifact is absorbed by the optimizer-state EMA.

### Level 3 — FP8 Adam optimizer states
Standard Adam keeps `(m, v)` in FP32 — ~16 GB/B params extra. FP8-LM stores:
- **m (first moment)** in **E4M3** (range modest, precision matters) with per-tensor FP32 scale.
- **v (second moment)** in **E5M2** (range very wide because gradient² spans many orders, precision matters less) with FP32 scale.

Update step is done in FP32 transiently:
```
m_fp32 = s_m · m_fp8
v_fp32 = s_v · v_fp8
m_fp32 = β1 · m_fp32 + (1 − β1) · g
v_fp32 = β2 · v_fp32 + (1 − β2) · g²
m_fp8, s_m = quant_E4M3(m_fp32)
v_fp8, s_v = quant_E5M2(v_fp32)
```
Net Adam state memory: `2 bytes/param + 8 bytes scales/tensor` ≈ 4× smaller than FP32.

### Master weights
Master weights remain in BF16 (or FP32) — quantization for these is deferred to follow-up work (and to [[deepseek-v3-fp8]]).

### Hyperparameters (recipe)
| Knob | Value |
|------|-------|
| Forward format | E4M3 |
| Backward grad format | E5M2 |
| All-reduce format | E4M3 (per-bucket scale) |
| Adam m | E4M3 |
| Adam v | E5M2 |
| Master weights | BF16 |
| Scale update | delayed (1-step EMA + overflow probe) |
| Hardware | H100 (E4M3, E5M2 native tensor cores) |
| Memory / speed | 39% memory ↓, 75% wall-clock speedup vs Megatron-BF16 at 175B |

## Connections
- The spec: [[fp8-formats-paper]], [[fp8-e4m3]], [[fp8-e5m2]].
- Forward-only FP8 (GEMM only, no optimizer): [[transformer-engine]].
- Successor that pushes FP8 deeper (into the *master weights*): [[deepseek-v3-fp8]].
- FP8 inference companion: [[zeroquant-fp]], [[fp8-llm-inference]].
- MoE-aware FP8 training: [[megatron-fp8]] integration.
