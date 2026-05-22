---
chapter: ch-17
course: model-quantization
phase: read
excerpt_of: "DeepSeek-V3 Technical Report §3.3 FP8 Training (Liang et al., DeepSeek-AI 2024)"
source_url: https://arxiv.org/abs/2412.19437
created_at: "2026-05-21"
---

# Excerpt: DeepSeek-V3 FP8 — per-block scaling at frontier scale

**Authors:** DeepSeek-AI (Liang et al.)
**Year:** 2024 (V3 report; training completed Dec 2024)
**URL:** https://arxiv.org/abs/2412.19437 (§3.3 FP8 Training)
**Raw-data source:** [[raw-data/deepseek-v3-fp8]]

---

## The recipe — one table

| Knob | Value |
|------|-------|
| Element format | E4M3 (all forward + backward + grad) |
| Activation tile | 1 × 128 (per-token × 128-channel) |
| Weight block | 128 × 128 |
| Scaling | per-block, online (no amax history) |
| Master-weight dtype | BF16 |
| Optimizer m / v dtype | FP32 / BF16 |
| WGMMA promote interval | 4 |
| FP8 comm | dispatch only; combine in BF16 |
| **Loss gap vs BF16** | **< 0.25 % relative** |
| Wall-clock vs BF16 | ~2× |
| Scale (671B MoE / 14.8T tokens) | ~2.664M H800-hours, no loss spikes |

DSV3 is the first publicly documented frontier-scale FP8 training run. Two design decisions explain why per-tensor FP8 (TE / FP8-LM) could not scale and per-block could.

---

## Why per-tensor failed at 70B+

Activation outliers in LLMs are heavily channel-localized — the same observation [[smoothquant]] made for inference PTQ. A single outlier channel sets the per-tensor amax, which sets the per-tensor scale, which means every other channel's quantization grid is wasted on absorbing that one outlier's magnitude. At 175B with TE-style FP8, this cost > 3 % loss vs BF16. At 671B-active-37B MoE, this is untenable.

The fix: shrink the scope of the scale. A 1×128 activation tile means each row gets 128 channels' worth of amax, not the whole tensor. An outlier channel pushes its own tile's scale but cannot crush other tiles. Per-block scales are picked **online** from the tile's actual amax — no history buffer, no delay, no synchronization bubble.

This is the SmoothQuant observation generalized from PTQ into training.

---

## Why E4M3 everywhere (instead of E5M2 for backward)

The standard FP8 convention ([[fp8-formats-paper]], [[transformer-engine]]) is **E4M3 forward / E5M2 backward** — gradients have a wider dynamic range than activations, so E5M2's 5-bit exponent is needed.

DSV3 uses **E4M3 for forward, weight-grad, and activation-grad**. Why? Because the per-block scale already absorbs the dynamic range — the FP8 element only has to carry the residual *within the 128-element block*. E4M3's extra mantissa bit (3 vs 2) gives better precision for that residual than E5M2's wider exponent does.

In other words: once you have a tight per-block scale, you don't need a wide-range element format. The wide range moves into the scale.

---

## FP32 partial-sum promotion inside the GEMM

The Hopper / H800 Tensor Core's internal FP8 → FPx accumulator has roughly **14-bit mantissa precision** (FP22-like). After ~32 successive multiply-adds, the partial sum starts losing the smaller contributions. This is *silent* — no overflow flag, no NaN — the gradient just drifts.

DSV3 promotes the partial sum to **FP32 in CUDA-core registers every 4 WGMMA instructions** (each WGMMA does a fixed-size FP8 matmul tile). The cost is a few extra FP32 adds per group; the gain is bit-exact accumulation across the full K dimension.

This is a kernel-level trick that has to be cooperatively scheduled with the WGMMA producer/consumer warp pattern. Without it, per-block scaling helps but accumulation drift still bites at frontier scale.

---

## What stays out of FP8

The pattern: small-parameter-count but high-precision-sensitivity ops stay in BF16/FP32.

- **Embedding lookup** — BF16 (the embedding table itself, ~0.6 % of params).
- **RMSNorm scales** — BF16.
- **Routing gate logits** (MoE) — BF16.
- **Attention softmax** — FP32 inside, BF16 outside.
- **Cross-entropy loss head** — BF16/FP32.
- **MoE combine step** — BF16. (Dispatch leg is FP8.)
- **Master weights** — BF16.
- **Optimizer:** FP32 first moment + BF16 second moment.

The win is the giant FFN GEMMs in 256 experts — that's where 90+ % of FLOPs live.

---

## FP8 MoE communication

The MoE token-routing all-to-all is sent as FP8 (per-bucket scale). This is the part that gives DSV3 *another* large speedup beyond the GEMM itself: cross-node bandwidth is roughly halved on the dispatch.

The combine step (on the way back, gradient-carrying) stays BF16. Sending the combine in FP8 was empirically unstable for gradient flow.

---

## Comparison to TE DelayedScaling

| Aspect | TE DelayedScaling | DSV3 per-block |
|--------|-------------------|----------------|
| Scale scope | per-tensor | per-1×128 tile / 128×128 block |
| Scale source | rolling amax history (1024 steps) | online, current step |
| Forward format | E4M3 | E4M3 |
| Backward format | E5M2 | E4M3 |
| Sync cost | none (history-based) | per-tile reduction (small) |
| 175B loss vs BF16 | ~0 % when stable | < 0.25 % at 671B |
| Scale-collapse from outliers | severe at frontier | absorbed by per-tile scope |

TE 2.x has since added per-block scaling recipes (`MXFP8BlockScaling`, `NVFP4`), generalizing the DSV3 approach to first-class library support.

---

## Connections

- [[fp8-formats-paper]] — the E4M3 / E5M2 spec DSV3 implements (with E4M3-only convention).
- [[transformer-engine]] — the per-tensor predecessor DSV3 measured itself against.
- [[fp8-lm]] — Microsoft's earlier per-tensor FP8 recipe (extends FP8 to optimizer + comm).
- [[smoothquant]] / ch-09 — the channel-localized-outlier observation that DSV3 generalizes from inference PTQ to training.
- [[nvfp4-training]] / [[excerpts/nvfp4-training]] — the Blackwell successor that pushes per-block scaling one bit-width lower.
- [[ch-17]] — parent synthesis.
