# FP8-LM: Training FP8 Large Language Models
<!-- slug: fp8-training · type: paper · source: https://arxiv.org/abs/2310.18313 -->

**Core Insight.** FP8 training (H100/Hopper-only) cuts optimizer-state memory from 16 B/param to 6 B/param and overall training memory by 28–39% versus BF16 mixed precision, by using E4M3 (high precision, narrow range) for forward-pass tensors and E5M2 (wide range) for gradients, coupled with per-tensor dynamic loss scaling for distributed all-reduce.

**Guideline.** FP8 is only available on NVIDIA H100+ (Hopper architecture). On A100 or older, do not attempt FP8; use BF16 + fp32 masters instead. When FP8 is available, apply E4M3 to weights and activations (forward), E5M2 to backward gradients; use the global minimum scaling factor across GPUs for distributed gradient sync.

## Technical Details

- **Two FP8 formats (OCP standard):**
  - **E4M3** — 1 sign + 4 exponent + 3 mantissa bits; range ±448; higher precision; used for weights and activations (forward pass).
  - **E5M2** — 1 sign + 5 exponent + 2 mantissa bits; range ±57,344; wider range; used for gradients (backward pass).
- **Per-tensor scaling:** A dynamic scaling factor μ adjusts throughout training. If overflow (>0.001% threshold): μ → μ/2. If underflow: μ → μ × 2 after 1,000 training steps. For distributed all-reduce, global minimum scaling factor `s_g' = min(s'₁, s'₂, ..., s'_n)` across GPUs eliminates per-tensor sync overhead.
- **FP8 optimizer memory layout (6 B/param total):**
  - Master weights: 2 B (FP16 with scaling)
  - Gradients: 1 B (FP8)
  - First-order moment: 1 B (FP8)
  - Second-order moment: 2 B (FP16)
  - Versus BF16 AdamW baseline: 16 B/param → **2.6× optimizer-state reduction**
- **Overall memory reduction vs BF16 mixed precision:**
  - GPT-7B: 29% reduction
  - GPT-13B: 28% reduction
  - GPT-175B: **39% reduction**
- **Throughput gains (H100):** GPT-175B runs **75% faster** than BF16 Megatron-LM; 37% faster than NVIDIA Transformer Engine. GPT-7B: 38% faster.
- **Sequence length benefit:** FP8 enables GPT-175B at seq=4,096 where BF16 is limited to seq=2,048 on the same H100 cluster.
- **Hardware constraint (critical):** Requires NVIDIA H100 (Hopper) with native FP8 tensor core support. Not available on A100, V100, or consumer GPUs.
- **Training-memory angle:** FP8 compresses the optimizer-state bucket (normally 8 B/param in fp32 AdamW) to 3 B/param for moments + 2 B for masters = 5–6 B total, freeing ~40% of the static 16 N floor. On H100, this shifts the binding constraint from optimizer states back to activations for long-sequence regimes.

## Citation
Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Wang, Chen Li, Jia Chen, Zheng Zhang, Han Hu, Peng Cheng, Deyuan Chen. "FP8-LM: Training FP8 Large Language Models." arXiv:2310.18313, 2023. https://arxiv.org/abs/2310.18313
