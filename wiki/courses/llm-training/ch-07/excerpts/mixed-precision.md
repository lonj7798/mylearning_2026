---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/mixed-precision.md
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
---

# Excerpt: Micikevicius 2017 — mixed precision as the dominant NaN surface

**Source library:** `wiki/raw-data/llm-training/classics/mixed-precision.md`
**Paper:** Micikevicius et al. 2017, *"Mixed Precision Training"* (NVIDIA + Baidu).

---

## Why this source anchors ch-07

Every NaN the chapter catalogs in §1 lives on the mixed-precision surface. fp16's exponent range is five bits — top 6.5e4, bottom 6e-8 — and *every* arithmetic path that leaves those bounds produces an inf/NaN that then propagates through the loss. The source's Figure 2 (the Multibox SSD gradient histogram) is the visceral motivation: a large fraction of an unmodified fp32 gradient distribution falls below fp16's underflow threshold, which is why loss scaling is necessary and why forgetting to persist the scaler state on resume is a catastrophic ch-06 bug.

Ch-07's diagnostic tree leans on this source three times:

- §1a: softmax overflow — fp16's 6.5e4 ceiling vs bf16's fp32-class range.
- §1c: Adam's `eps` under fp16 — division-by-zero in the optimizer step.
- §2: the plateau branch fp16-specific variant — unscale-before-clip ordering bug.

And it provides the rule that makes bf16 the 2025 default: *"Use bf16 for all LLM training in 2025 (no loss scaling needed); use fp16 only on Volta-era GPUs that lack bf16."* Under bf16 most of ch-07 §1 simply cannot fire.

---

## Three precision formats — the range table

From the source (Technical Details):

| Format | Bits | Exp | Mantissa | Range | Notes |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | Reference; master weights |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | Narrow; **needs loss scaling** |
| bf16 | 16 | **8** | 7 | ~1e-38 to 3e38 | fp32 range; lower precision; no scaling |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | Forward path |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | Backward path |

The critical number for ch-07 §1a is fp16's **65504 ceiling**. Any logit whose `exp` is near or above this value overflows fp16's softmax. For an attention score `Q·K/√d_k` this happens when the query and key magnitudes aren't controlled — typically after a weight-norm drift of ~5–10% from init. OLMo 1's spike phenotype ([[excerpts/olmo-2]]) was exactly this overflow; QK-Norm's job is to bound `Q` and `K` so the product can't exceed log(65504) ≈ 11 in the relevant direction.

bf16's same-exponent-as-fp32 property means the ceiling is 3e38 — effectively unreachable in training. But the 7-bit mantissa means `exp(89)` already *saturates to a meaningless value* because the fractional precision is gone. So bf16 converts an fp16 NaN into a bf16 silent precision loss — which is worse in some ways (no NaN alarm) but better in others (no training-destroying inf to propagate). Ch-07 §2's divergence branch covers this: "softmax / logits overflowing but bf16 is hiding it" is a real mode, detectable only by logging `logits.abs().max()` (the ch-07 §7 checklist includes this).

---

## The fp16 recipe — and the scaler-state ch-06 ordering

From the source:

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

Notice the comment: `# unscale before clip!`. This is the ch-07 §2 fp16 plateau variant and the [[excerpts/gradient-clipping]] ordering rule rolled together. If you clip before unscaling, your clip threshold is effectively `S · c ≈ 2^15` in fp32-space — unreachable under any realistic gradient distribution — and `clipped_fraction` is structurally 0, hiding the spike signal.

The source's dynamic-loss-scaling algorithm is the subject of ch-06 §5.2's scaler-drop-on-resume bug:

> *"Start `S = 2^15`. If any gradient is inf/NaN, skip the step and halve `S`. Every `N` (e.g. 2000) successful steps, double `S`. This auto-tunes to the gradient distribution."*

The state that must be persisted across a resume is therefore *two numbers*: current `S` and steps-since-last-overflow counter. Drop either and the scaler re-warms from `2^15`, giving you 2000 steps of skipped overflow detection if your run had stabilized at `2^18` ([[excerpts/mixed-precision]] shows this explicitly in the ch-06 excerpt). The run looks fine in the tokens/sec metric; token-efficiency is quietly 25% worse for 2000 steps.

---

## Why bf16 won — and what it doesn't fix

From the source:

> *"bf16 (the modern default): same exponent range as fp32, so loss scaling is not needed; gradients never underflow. Cost: only 7 mantissa bits → ~1% relative precision. For Adam/AdamW the optimizer-state precision matters more than weight precision; bf16 weights with fp32 optimizer state is the universal recipe."*

Operationally bf16 eliminates:

1. The loss-scaler state from the checkpoint surface (ch-06 §5.2).
2. The unscale-before-clip ordering bug (ch-07 §2).
3. Gradient underflow as a silent failure (ch-07 §1c when combined with Adam's `v̂`).
4. Most fp16 softmax overflow NaNs (ch-07 §1a).

It *does not* eliminate:

1. Softmax precision loss — `exp(89)` is still garbage in bf16; no NaN, but wrong. Ch-07 §2's bf16-hides-overflow divergence branch.
2. `v̂` underflow when `eps` is too small — `v̂` of 1e-40 is still fine in bf16 (fp32 master), but adding `1e-8` in an `a / (sqrt(v) + eps)` where `sqrt(v) = 1e-20` gives `eps`-dominated denominator. This is [[excerpts/adam]]'s "bump `eps` to 1e-5" rule.
3. `log(0)` in hand-rolled CE or KL — bf16 makes small numbers representable, but `log(0)` is `-inf` in every format. Ch-07 §1b.

So bf16 is a *lot* cheaper than fp16 on the ch-07 surface, but it is not a free pass.

---

## The fp32 invariants — what must never be downcast

From the source:

> *"Stability tricks (universal): Keep LayerNorm/RMSNorm in fp32 (reduction-heavy; small numerical errors compound). Keep softmax computation in fp32 (exponentials). Keep cross-entropy loss in fp32. Master weights and optimizer state always fp32."*

These are the fp32 invariants every stable 2025 LLM trainer enforces. The kernel-level mechanism: PyTorch's `torch.autocast` defers autocasting for reduction ops by default, but only for operators in its allowlist; custom kernels (FlashAttention-custom-path, Liger, XFA) may override the list. Ch-07 §4b's cross-sample attention leakage has a sibling bug here: a custom attention override that silently casts the softmax to bf16 produces precision loss rather than cross-sample leakage, but both are invisible because neither produces a NaN.

The source's common-pitfalls list names one ch-07 §1c item directly:

> *"Using a tiny `eps` in AdamW under fp16 → division by zero. Bump `eps` to 1e-5 or use bf16."*

This is the optimizer-step NaN branch of ch-07 §1c's arithmetic diagnostic table.

---

## The fp8 surface — 2025's new NaN geography

The source covers fp8 briefly because H100+ training started in 2023:

> *"fp8 (H100 / B100, 2023+): only the matmul tensor cores are fp8; surrounding ops, normalization, residual stream, and softmax stay in bf16/fp32. Per-tensor (or per-block, e.g. DeepSeek-V3 1×128 / 128×128 blocks) scaling factors are tracked and applied around each matmul. ... Stability tricks needed: E4M3 forward, E5M2 backward (different range needs); delayed scaling: the scale for tensor `t` is computed from amax history, lagging by 1 step; per-tensor amax monitoring."*

fp8 re-introduces the loss-scaler class of bug at a finer granularity. Each tensor has its own scale factor, each scale must be persisted at checkpoint, each must be re-loaded before the first forward post-resume. A lab running DeepSeek-V3-style 1×128 block-scaled fp8 has a checkpoint surface where the per-block scales are part of the state; miss any and the ch-06 §5.2 scaler-drop bug returns tenfold. Ch-07's chapter is written for the bf16 baseline; fp8 has its own layer of stability machinery that future chapters will re-enter.

---

## What to take from Micikevicius for ch-07

1. **fp16's 6.5e4 ceiling is the source of most §1a NaNs.** bf16's 3e38 ceiling eliminates them at the cost of 7 mantissa bits.
2. **Unscale before clip.** Under fp16 the ordering matters; under bf16 there is no scaler and no ordering bug.
3. **Loss-scaler state is an item in the ch-06 checkpoint list.** Dropping it silently burns 2000 steps of token-efficiency.
4. **fp32 invariants are non-negotiable for norm / softmax / CE / optimizer state.** Custom kernels that violate these produce silent precision loss, not NaN alarms.
5. **fp8 re-introduces the scaler bug surface at per-tensor granularity.** Ch-07 is written for bf16; fp8 requires extra checkpoint discipline.

---

## Connections

- [[excerpts/gradient-clipping]] — the unscale → clip → step ordering is shared between the two sources; fp16 binds them.
- [[excerpts/adam]] — `eps` placement under fp16; the optimizer-step NaN path.
- [[excerpts/fsdp-sft]] — bf16 params + fp32 optimizer-state as the FSDP recipe; fp8 is an extension.
- [[excerpts/karpathy-training-neural-net-recipe]] — "start simple, use fp32 until training is stable" maxim aligns with this source's "fp32 first" implicit recommendation.
- [[ch-07]] — §1a (softmax overflow), §1b (log(0) clamp), §1c (/0 in advantage), §2 (fp16 plateau variant), §7 (`isfinite(loss)` assertion).
