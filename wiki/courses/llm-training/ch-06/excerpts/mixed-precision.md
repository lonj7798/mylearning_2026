---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/mixed-precision.md
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
---

# Excerpt: Mixed Precision Training — loss-scaler state and why optim-state stays fp32

**Source library:** `wiki/raw-data/llm-training/classics/mixed-precision.md`
**Paper:** Micikevicius et al. 2017, "Mixed Precision Training" (NVIDIA + Baidu)

---

## Why this source anchors ch-06 §3 and §5.2

Ch-06 makes two claims that live or die on this paper:

1. The ninth row of the checkpoint state table — **loss-scaler state** — appears only under fp16 training and is the most frequently dropped item in hand-rolled fp16 checkpoints.
2. **Master fp32 weights must be persisted, period.** Even under bf16/fp8 compute, the optimizer's internal storage is fp32 and re-deriving it from the bf16 weights on resume loses 16 mantissa bits of accumulated progress.

The Micikevicius paper is the canonical source for both claims.

---

## The three precision formats — what "fp16 underflow" actually means

From the source (lines 30-38):

| Format | Bits | Exp | Mantissa | Range | Notes |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | Reference baseline; master weights |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | Narrow range; **needs loss scaling** |
| bf16 | 16 | 8 | 7 | ~1e-38 to 3e38 | Same range as fp32; lower precision; **no loss scaling needed** |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | Forward path (activations/weights) |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | Backward path (gradients) |

Notice the fp16 range row: `6e-8` as the smallest representable normal. The source's Figure 2 (referenced line 25) showed that **most fp32 gradient values in Multibox SSD fell below `6e-8`** — they would underflow to zero if naively stored in fp16. This is the visceral motivation for loss scaling: multiply the loss by `S = 2^15` before backprop, shifting the gradient distribution up by 15 bits into fp16's representable range.

bf16 has the same exponent as fp32 (8 bits), so it covers the fp32 range end-to-end. No underflow, no loss scaling. The cost is mantissa bits (7 vs 10), i.e. ~1% relative precision instead of fp16's ~0.1%. For adaptive optimizers where the `v` accumulator is a *running average* of squared gradients, the relative-precision hit is empirically negligible. This is why bf16 won: it removes the entire loss-scaler control surface.

---

## The fp16 recipe — where the scaler state is born

From the source (lines 40-54):

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

Notice four moves in order: (1) cast weights to fp16, (2) scale the loss by `S`, (3) backward produces scaled grads in fp16, (4) in the optimizer, cast to fp32 and *divide by `S`* before clipping and stepping.

Step (4) is where ch-06 §5.2's "scaler-state drop" fails silently. The divisor `S` is not a static constant — it is **dynamic**:

> **Dynamic loss scaling**: start `S = 2^15`. If any gradient is inf/NaN, skip the step and halve `S`. Every `N` (e.g. 2000) successful steps, double `S`. This auto-tunes to the gradient distribution.

So `S` has state:

- Current value (an fp32 number, though powers of 2 in practice).
- Steps-since-last-increase counter (int).
- Growth interval `N` (int, typically 2000).
- Backoff factor (0.5, constant).
- Growth factor (2.0, constant).

PyTorch's `torch.cuda.amp.GradScaler` exposes this as `scaler.state_dict()`. It returns something like:

```python
{
    "scale":          65536.0,    # current S
    "growth_factor":  2.0,
    "backoff_factor": 0.5,
    "growth_interval": 2000,
    "_growth_tracker": 1847,       # successful steps since last growth
}
```

The **growth tracker** is the silent-failure trigger. If you drop scaler state on resume and the framework re-initializes the scaler at `S = 2^15`, you have lost:

- The current scale (your run had stabilized at, say, `S = 2^18` → 8× underscaling).
- The growth tracker (you restart the 2000-step ramp from zero).

Ch-06 §5.2's operational consequence:

> Loss looks fine; token-efficiency is 25% worse for 2000 steps. This is why bf16 won: no scaler, no bug.

The 25% figure is a heuristic, not a measurement — it reflects that at the re-warmed scale, your first few thousand gradient updates are silently 8× smaller than the optimizer was "calibrated" for, so effective LR is 8× tighter until `S` doubles back to its prior value. No exception fires. The run continues. You lose ~2000 steps of quality compute per resume, and you will not see it in the loss curve.

---

## Why optimizer state stays fp32 across resumes

From the source (line 57):

> **bf16 (the modern default)**: same exponent range as fp32, so loss scaling is **not needed**; gradients never underflow. Cost: only 7 mantissa bits → ~1% relative precision. **For Adam/AdamW the optimizer-state precision matters more than weight precision;** bf16 weights with fp32 optimizer state is the universal recipe (Llama, GPT-NeoX, Mistral, Qwen, DeepSeek all use this).

Notice the **emphasized clause**. It is the single most important line for ch-06 §5.4's "optimizer state partially loaded" failure. Adam/AdamW maintain:

- `m_t = β1 · m_{t-1} + (1-β1) · g_t`
- `v_t = β2 · v_{t-1} + (1-β2) · g_t²`
- `w_t = w_{t-1} - lr · m_hat / (sqrt(v_hat) + eps)`

All three arithmetic operations accumulate over thousands of steps. The `v_t` update especially — `(1-β2)` with `β2 = 0.95` or `0.999` means each new `g_t²` contributes only 1–5% per step, and the running value is dominated by a tail of thousands of past gradients. In bf16 with 7 mantissa bits, this accumulation *drifts by rounding* within ~100 steps — the source's footnote on [[adam]] makes this explicit: *"bf16 `v_hat` underflows on small gradients within ~100 steps."*

The fix is to keep `m`, `v`, and the **master fp32 weight copy** all in fp32 permanently. The working memory is bf16 (params, activations, gradients for the communication path), but the optimizer's state-of-record is fp32. A resume that reconstructs the master from bf16 weights loses 16 mantissa bits of accumulated Adam progress — not visible at step `k+1`, but visible at step `k+10000` when the run has drifted noticeably from the published trajectory.

This is also why ch-06 §1's table row *"Master fp32 weights | fp32 | sharded | resume rounds bf16 → fp32 and loses 7-mantissa-bit progress; silent ~0.1% perplexity drift"* lists a ~0.1% perplexity drift as the observable cost. Perplexity is a log-space metric; 0.1% is the kind of delta that is below most lab's noise threshold for pretraining monitoring. Nobody notices until the downstream eval diverges a month later.

---

## Stability tricks that are also resume-sensitive

From the source (lines 64-68):

> **Stability tricks (universal)**:
> - Keep LayerNorm/RMSNorm in fp32 (reduction-heavy; small numerical errors compound).
> - Keep softmax computation in fp32 (exponentials).
> - Keep cross-entropy loss in fp32.
> - Master weights and optimizer state always fp32.

Only the last bullet is *checkpoint-state*; the others are runtime choices. But they interact with checkpointing in a subtle way: if you change the norm-fp32 policy between save and load (for example, "I will drop LayerNorm-fp32 to claw back some memory"), the resumed run produces slightly different logits for the same input. Ch-06 §3's bit-exact resume check catches this — the test fails — but only if you are *running* the test. Many labs only bit-exact-test on the save-load-step-identity path, not on the "I changed the precision config" path.

---

## Logging in fp32 — the practitioner note ch-06 inherits

From the source (line 73):

> - Logging loss in fp16 → loss curves look quantized/jaggy; log in fp32.

Ch-06 §4 turns this into the explicit rule:

> **Log in fp32, always.** [[mixed-precision]] warns against logging loss in fp16 — the curve looks quantized-jaggy because fp16 has 10 mantissa bits. `loss.item()` upcasts implicitly; explicit `loss.detach().float().item()` documents intent.

Notice the word "implicitly." `loss.item()` does upcast via Python's double conversion, so the *value* is fine. The reason to write `.float().item()` explicitly is instrumentation discipline: future-you reading the code six months later needs to know the intent. This is a Karpathy-style "documents intent" rule — see [[excerpts/karpathy-training-neural-net-recipe]] on the general pattern of making implicit correctness explicit.

---

## The ordering rule — unscale → clip → step

From the source (line 72):

> - Forgetting to unscale before grad clipping → clipping threshold is off by `S`.

The ordering `unscale → clip → step` is the single most error-prone sequence in fp16 training. At `S = 2^15`, a naive `clip_grad_norm_(model.parameters(), 1.0)` sees gradients 32768× larger than their true scale; the clip-by-1.0 rescales *everything* down, destroying the direction signal (gradient clipping preserves direction only within its scaled regime). The fix is framework-provided: `scaler.unscale_(optimizer)` materializes unscaled gradients in place, after which the standard clip works.

On resume, if the scaler state is missing and `S` resets to `2^15` while the model's *actual* gradient magnitudes are calibrated to `S = 2^18`, the unscale call divides by the wrong factor. The first 2000 post-resume steps run with 8× inflated effective gradients — a silent regime shift visible only as a slight uptick in grad-norm logs. See [[excerpts/gradient-clipping]] for how the `pre_clip_grad_norm` metric catches this.

---

## Why ch-06 emphasizes bf16

The source's summary line (line 57 again):

> **bf16 (the modern default)**: same exponent range as fp32, so loss scaling is **not needed**; gradients never underflow.

From a checkpointing perspective, bf16 eliminates the entire ninth row of the state table. No dynamic scale, no growth tracker, no unscale-before-clip ordering. The checkpoint shrinks by zero bytes (the scaler state is ~100 bytes) but the *failure surface* shrinks substantially. Every fp16 resume bug in ch-06 §5.2 is bypassed by switching to bf16. This is the "why bf16 won" claim — it is a *operational* win (fewer silent failure modes on resume), not just a numerical one.

fp8 training (DeepSeek-V3) keeps the bf16-style "no scaler" property for the optimizer state but re-introduces *per-tensor* scale tracking for the fp8 matmuls. Those scales are also resume-sensitive — if you drop the amax history on resume, the delayed scaling uses a cold-started amax for ~100 steps, during which fp8 overflow/underflow is undetected. DeepSeek-V3 checkpoints the amax history explicitly for this reason.

---

## Connections

- [[excerpts/gradient-clipping]] — the `unscale → clip → step` ordering that fails under dropped scaler state; `pre_clip_grad_norm` is the signal that catches it.
- [[excerpts/fsdp-sft]] — sharded fp32 optimizer state (`12P`) is 6× the bf16 weight size; this excerpt explains *why* fp32.
- [[excerpts/early-stopping-and-checkpointing]] — classical checkpoint table row "Loss-scaler state (fp16)" is the ninth item; this excerpt explains what the state actually contains.
- [[excerpts/olmo-2]] / [[excerpts/llama-3]] — both run bf16, so their checkpoints have no scaler row; their resume paths are simpler by construction.
- [[excerpts/karpathy-training-neural-net-recipe]] — "start in fp32 for debugging, switch to mixed precision only after training is stable" is the methodology that exposes scaler-state bugs at tiny scale.
- [[ch-06]] — §1 table row 9 (loss-scaler), §3 (bit-exact resume), §5.2 (scaler-state drop), §5.4 (master-fp32 partial load).
