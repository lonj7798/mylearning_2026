<!-- scope: mixed precision training; fp16 vs bf16 vs fp8; loss scaling
     deps: [[adam]]
     see-also: [[gradient-clipping]], [[batch-vs-layer-norm]]
-->

# Mixed Precision Training (Micikevicius 2017) and the bf16 / fp8 successors
- **Core Insight:** Storing weights, activations, and gradients in 16-bit while keeping a master fp32 weight copy and an fp32 accumulator preserves accuracy at half the memory and 2–8x the throughput.
- **Guideline:** Use **bf16** for all LLM training in 2025 (no loss scaling needed); use fp16 only on Volta-era GPUs that lack bf16; reserve fp8 for the matmul path with fp32 master weights and per-tensor scaling.
- **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA + Baidu)
- **Year:** 2017
- **URL:** https://arxiv.org/abs/1710.03740
- **Relevant topics:** numerical precision, training throughput, hardware utilization

## Abstract
The paper introduces a recipe for training deep networks in IEEE-754 half-precision (fp16) without loss in accuracy across image classification, speech, GAN, and NMT models. Three techniques make this work: (1) keep an fp32 *master copy* of the weights that the optimizer updates; (2) *scale the loss* before backprop so small fp16 gradients do not underflow; (3) accumulate matmul *outputs* in fp32 even when inputs are fp16. With these tricks, models match fp32 baselines while doubling effective batch size and 2–6x training throughput on Volta hardware.

## Key Contributions
- The first systematic recipe for fp16 training that works across modalities — previously fp16 was considered too unstable for deep networks.
- **Loss scaling**: multiply loss by `S` (e.g. 1024) so gradients are scaled into representable range; unscale before optimizer step.
- **fp32 master weights**: the optimizer step happens in fp32; weights are downcast to fp16 for the next forward pass. Critical because Adam's `m`, `v`, and small weight updates underflow in fp16.
- **fp32 accumulation in matmul**: tensor cores compute `C += A @ B` with fp16 inputs but fp32 accumulator — without this, sum-of-products of long sequences saturates fp16.
- Demonstrated zero accuracy loss on ImageNet (ResNet-50), Big LSTM (1B word LM), GNMT, and DCGAN.

## Key Figures/Tables to Study
- **Figure 2** (gradient histogram for Multibox SSD): shows most fp32 gradients land in fp16's underflow region without loss scaling — visceral motivation for the loss-scale trick.
- **Table 1** (accuracy comparison): demonstrates parity with fp32 baselines across many models.
- **Section 3.2** (loss scaling — choosing the scale factor): the dynamic-loss-scaling algorithm that's now built into every framework.

## Technical Details

**Three precision formats — exponent / mantissa / range**:
| Format | Bits | Exp | Mantissa | Range | Notes |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | Reference baseline; master weights |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | Narrow range; **needs loss scaling** |
| bf16 | 16 | **8** | 7 | ~1e-38 to 3e38 | Same range as fp32; lower precision; **no loss scaling needed** |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | Forward path (activations/weights) |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | Backward path (gradients) |

**The fp16 recipe (Micikevicius 2017)**:
```
# Forward
w_fp16 = cast(w_fp32, fp16)
loss = forward(x_fp16, w_fp16)
scaled_loss = S * loss

# Backward (produces fp16 grads, scaled by S)
scaled_grads = backward(scaled_loss)

# Optimizer step in fp32
grads_fp32 = cast(scaled_grads, fp32) / S
clip_grad_norm_(grads_fp32, max_norm)        # unscale before clip!
adamw_step(w_fp32, grads_fp32, lr, ...)
```
**Dynamic loss scaling**: start `S = 2^15`. If any gradient is inf/NaN, skip the step and halve `S`. Every `N` (e.g. 2000) successful steps, double `S`. This auto-tunes to the gradient distribution.

**bf16 (the modern default)**: same exponent range as fp32, so loss scaling is **not needed**; gradients never underflow. Cost: only 7 mantissa bits → ~1% relative precision. For Adam/AdamW the optimizer-state precision matters more than weight precision; bf16 weights with fp32 optimizer state is the universal recipe (Llama, GPT-NeoX, Mistral, Qwen, DeepSeek all use this).

**fp8 (H100 / B100, 2023+)**: only the matmul tensor cores are fp8; surrounding ops, normalization, residual stream, and softmax stay in bf16/fp32. Per-tensor (or per-block, e.g. DeepSeek-V3 1×128 / 128×128 blocks) scaling factors are tracked and applied around each matmul. Speedup ~2x over bf16 on H100. Stability tricks needed:
- E4M3 forward, E5M2 backward (different range needs).
- "Delayed scaling": the scale for tensor `t` is computed from amax history, lagging by 1 step.
- Per-tensor amax monitoring; many production runs (DeepSeek-V3) use mixed bf16/fp8 — keeping outlier-prone layers in bf16.

**Stability tricks (universal)**:
- Keep LayerNorm/RMSNorm in fp32 (reduction-heavy; small numerical errors compound).
- Keep softmax computation in fp32 (exponentials).
- Keep cross-entropy loss in fp32.
- Master weights and optimizer state always fp32.

**Common pitfalls**:
- Mixing fp16 and bf16 in the same run (e.g. fp16 forward, bf16 grads) → silent divergence.
- Forgetting to unscale before grad clipping → clipping threshold is off by `S`.
- Logging loss in fp16 → loss curves look quantized/jaggy; log in fp32.
- Using a tiny `eps` in AdamW under fp16 → division by zero. Bump `eps` to `1e-5` or use bf16.

## Connections
- **[[adam]]**: optimizer state must remain fp32 — bf16 `v_hat` underflows on small gradients within ~100 steps.
- **[[gradient-clipping]]**: ordering is non-negotiable: `unscale → clip → step`.
- **[[batch-vs-layer-norm]]**: norm layers are the most reduction-sensitive ops; **always** compute in fp32 even under fp8 training.
- **DeepSeek-V3** ([[deepseek-v3]]): the canonical 2024 fp8 training recipe; demonstrates 1×128 / 128×128 block-wise fp8 with selective bf16 fallback.
- **Llama-3 70B / 405B**: trained in bf16 throughout, fp32 master + AdamW.
- **Karpathy's recipe** ([[karpathy-training-neural-net-recipe]]): recommends starting in fp32 for debugging; switch to mixed precision only after training is stable.
