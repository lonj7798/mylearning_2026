<!-- scope: DeepSeek-V3 FP8 native training recipe (technical report §3.3)
     deps: [[fp8-formats-paper]], [[fp8-lm]]
     see-also: [[transformer-engine]], [[deepseek-v3]], [[nvfp4-training]]
-->

# DeepSeek-V3 FP8 Mixed-Precision Training
- **Core Insight:** Frontier-scale FP8 pretraining became stable by combining fine-grained per-block scaling (1×128 activation tiles, 128×128 weight blocks), online quantization, and FP32 promotion every 4 WGMMA instructions inside Tensor Cores to defeat FP8 accumulation drift.
- **Guideline:** When training in FP8 at frontier scale, never trust per-tensor scales — use per-row/per-block scales chosen online from the actual activation amax, keep a BF16 master weight and FP32 optimizer state, and promote partial sums to FP32 inside the GEMM, not just outside it.
- **Authors:** DeepSeek-AI (Liang et al.)
- **Year:** 2024 (V3 report) — training run completed Dec 2024
- **URL:** https://arxiv.org/abs/2412.19437 (§3.3 FP8 Training)
- **Relevant topics:** FP8 native training, E4M3/E5M2, per-block scaling, fine-grained accumulation, BF16 master weights, low-precision optimizer, MoE FP8

## Abstract
DeepSeek-V3's technical report dedicates Section 3.3 to a from-scratch FP8 mixed-precision framework that pretrains a 671B-total / 37B-active MoE on 14.8T tokens — the first publicly documented frontier-scale FP8 run with no loss spikes. Most prior FP8 work (Transformer Engine's per-tensor delayed scaling, FP8-LM) failed at this scale because (a) activation outliers blow out per-tensor scales and (b) H800's FP8 Tensor Core promotes partial sums in a narrow internal accumulator, losing precision after a few WGMMA steps. DSV3 fixes both with **fine-grained quantization** (per-tile / per-block scales chosen online) and a **promote-to-FP32-every-N-WGMMA** GEMM scheme that keeps mathematical fidelity while paying ~no kernel cost. The result is roughly 2× wall-clock vs BF16 with relative loss error < 0.25 % through all 14.8T tokens.

## Key Contributions
- Fine-grained, **online** quantization: per-tile (1×128) for activations, per-block (128×128) for weights — recomputed every step, no amax history needed.
- FP32 partial-sum promotion every **4 WGMMA** instructions inside the H800 Tensor Core, working around the limited FP22 accumulator that otherwise drifts after ~32 mantissa bits of additions.
- E4M3 used uniformly for forward, weight gradients, and activation gradients (instead of the Transformer-Engine convention of E4M3-fwd / E5M2-bwd) — feasible because per-block scaling reclaims the dynamic range that E4M3's 4-bit exponent would otherwise lose.
- BF16 master weights + FP32 first/second moments; weight update is in BF16 so optimizer math doesn't suffer from FP8 round-off.
- FP8 cached activations and FP8 cross-node all-to-all dispatch communication in the MoE expert-parallel scheme — major memory and bandwidth savings.
- Empirically demonstrates < 0.25 % relative loss gap vs a BF16 baseline across the entire 2.664M-H800-hour pretrain.

## Key Figures/Tables to Study
- **Figure 6 (DSV3 report):** the overall FP8 framework — which tensors live in BF16, FP8, FP32; where casts happen.
- **Figure 7:** the GEMM diagram showing FP8 inputs, FP32 partial sums promoted every 4 WGMMA, and the per-block scales applied at accumulation time.
- **Table 5:** loss / benchmark comparison between FP8 and BF16 small-scale runs — the validation that the recipe matches BF16 to within 0.25 %.

## Technical Details

### Quantization granularity
- **Activations:** per-token, per-128-channel tile → tile shape **1 × 128**, one E4M3 scale per tile, picked online from the tile's amax.
- **Weights:** **128 × 128** blocks, one scale per block, picked online during the forward pass.
- **Element format:** E4M3 throughout (4-bit exponent, 3-bit mantissa, 1-bit sign).
- **Why this granularity:** activation outliers in LLMs are heavily channel-localized (the SmoothQuant observation); a 1×128 tile keeps the scale tight enough that a single outlier channel doesn't crush the rest. Per-tensor FP8 (Transformer Engine's default) lost > 3 % loss at the same training budget.

### FP32 fine-grained accumulation
- The H800 Tensor Core's internal FP8 → FPx accumulator has roughly **14-bit mantissa precision** (FP22-like), so after ~32 successive multiply-adds the partial sum starts losing the smaller contributions.
- DSV3 promotes the partial sum to FP32 in CUDA-core registers **every 4 WGMMA instructions** (each WGMMA does a fixed-size FP8 matmul tile). The cost is a few extra FP32 adds per group; the gain is bit-exact accumulation across the full K dimension.
- Combined with the per-block scales, the dequantized output is essentially identical to a BF16 GEMM at the same matrix.

### Master weights, optimizer, communication
- **Master weights:** BF16, kept in HBM. Used to compute the FP8 weights for each step's forward.
- **Optimizer state:** FP32 first moment, BF16 second moment (AdamW). Update is done in BF16 with FP32 promotion of the m/v reads.
- **Activation cache for backward:** stored in FP8 → roughly **half the activation memory** vs BF16.
- **MoE all-to-all dispatch:** the token-routing all-to-all is sent as FP8. Combine step is BF16 (gradient stability).
- **Sensitive ops kept in BF16/FP32:** embedding lookup, LayerNorm/RMSNorm, softmax in attention, the router gate, the loss head.

### What is *not* FP8
- Embedding table, RMSNorm scales, the routing gate logits, attention softmax, and the cross-entropy loss head all stay BF16/FP32 — small parameter cost, but they are the most precision-sensitive operations in the model.

### Hyperparameters
| Knob | Value |
|------|-------|
| Element format | E4M3 (all forward + backward + grad) |
| Activation tile | 1 × 128 |
| Weight block | 128 × 128 |
| Master-weight dtype | BF16 |
| Optimizer m / v dtype | FP32 / BF16 |
| WGMMA promote interval | 4 |
| FP8 comm | dispatch only; combine in BF16 |
| Loss gap vs BF16 | < 0.25 % relative |

### Why E4M3 everywhere (instead of E5M2 for backward)
Per-tensor FP8 needs E5M2 on the backward pass because gradients have a much wider dynamic range than activations. DSV3's per-block scales already absorb the dynamic range *inside the scale*, so the element format only has to cover the local dispersion within the 128-element block — which E4M3 (with its extra mantissa bit) does more accurately than E5M2.

## Connections
- [[fp8-formats-paper]] — the joint NVIDIA/Arm/Intel spec for E4M3 / E5M2; DSV3 implements E4M3 + scales.
- [[fp8-lm]] — Microsoft's earlier per-tensor FP8 recipe; DSV3 shows per-tensor doesn't scale and replaces it with per-block.
- [[transformer-engine]] — NVIDIA's reference FP8 library (delayed scaling, amax history); DSV3 instead does *online* per-block scaling.
- [[nvfp4-training]] — Blackwell-era successor that hard-codes a 16-element block scale into the Tensor Core, generalizing the DSV3 approach to FP4.
- [[deepseek-v3]] — the parent model report; §3.3 is the FP8 chapter, §3.2 is DualPipe, §3.4 is the inference deployment.
- [[microscaling-formats]] — the OCP MX format family; DSV3's per-block FP8 is an in-house variant of the same idea.
