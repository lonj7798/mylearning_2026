---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "Mixed Precision Training (Micikevicius et al. 2017)"
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
---

# Excerpt: Mixed-precision training — fp16, bf16, and fp8

**Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA + Baidu)
**Year:** 2017
**Venue:** ICLR 2018
**URL:** https://arxiv.org/abs/1710.03740
**arXiv ID:** 1710.03740
**Raw-data source:** [[raw-data/mixed-precision]]

---

## The three-ingredient recipe

The paper's contribution is not a new algorithm but a *combination* of three numerical tricks that together make fp16 training match fp32 accuracy:

1. **Master fp32 weights.** Store the authoritative weights in fp32. Downcast to fp16 only for the forward pass; upcast after the gradient to apply optimizer updates in fp32.
2. **Loss scaling.** Multiply the loss by a scalar `S` (typically `S = 2^k` for some small integer `k`) before backprop so gradients land inside fp16's representable range; divide by `S` before the optimizer step.
3. **fp32 accumulation in matmul.** Tensor-core matmul computes `C += A @ B` with fp16 operands but an fp32 accumulator register.

All three are necessary; removing any one produces a silently-broken run.

---

## Why fp16 needs loss scaling — the histogram argument

Figure 2 of the paper is the single most important plot in numerical-precision training. It shows a histogram of gradient magnitudes during Multibox SSD training in fp32, overlaid with fp16's representable range. The result: most gradients are in the `10⁻¹¹` to `10⁻³` range, but fp16's smallest normal number is `~6 × 10⁻⁸`. Over half of all gradient values would underflow to zero if simply cast to fp16 — the network would be silently training with two-thirds of its gradients erased.

**The loss-scaling derivation.** If the loss `L` is multiplied by a scalar `S` before backprop, by linearity of differentiation every gradient in the graph is also scaled by `S`:

```math
\frac{\partial (S \cdot \mathcal{L})}{\partial \theta} = S \cdot \frac{\partial \mathcal{L}}{\partial \theta}
```

Choose `S = 2^k` large enough to shift the gradient histogram into fp16's representable region. Compute backward in fp16 (now the values fit), then *unscale* by dividing by `S` in fp32 before the optimizer step. The unscale must happen before gradient clipping (see [[excerpts/gradient-clipping]]) because the clip threshold is defined on unscaled gradients.

**Why `S = 2^k` specifically.** Multiplication by a power of two in floating-point is *exact* — it only modifies the exponent, not the mantissa. Any other choice introduces a rounding error in the scale step itself, on top of the rounding error from fp16 backward. Dynamic loss scaling (built into every framework today) runs the algorithm: start `S = 2^15`; if any gradient is `inf`/`NaN`, skip this step and halve `S`; every `N=2000` successful steps, double `S`. This auto-tracks the gradient histogram as training dynamics change.

---

## Why bf16 does NOT need loss scaling — the exponent argument

bf16 and fp16 both occupy 16 bits, but allocate them differently:

| Format | Bits | Exponent | Mantissa | Range | Relative precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | ~7 decimal digits |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | ~3–4 decimal digits |
| bf16 | 16 | **8** | 7 | ~1e-38 to 3e38 | ~2–3 decimal digits |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | ~1 decimal digit |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | <1 decimal digit |

bf16 has the **same exponent as fp32**, so it has the same dynamic range: gradients that fit in fp32 also fit in bf16 without any scaling. The cost is the 7-bit mantissa — roughly 1% relative precision per operation. In practice this is tolerable because (a) the optimizer state is still fp32, absorbing the precision loss over EMA steps, and (b) most LLM operations are averages over hundreds of thousands of tokens, where 1% per-sample noise averages out.

**The one-line summary**: fp16 is *precise* but *narrow*; bf16 is *wide* but *imprecise*. LLMs are exponent-limited (gradients range across many orders of magnitude) not mantissa-limited, so bf16 wins. This is why every frontier lab since ~2022 trains in bf16 and why loss scaling has quietly disappeared from modern training code.

---

## The full recipe, written out

```
# FORWARD
w_bf16 = cast(w_fp32, bf16)                       # master fp32, cast down for forward
loss = forward(x_bf16, w_bf16)                    # bf16 activations, bf16 weights

# BACKWARD (bf16: no scaling; fp16: multiply loss by S first)
grads_bf16 = backward(loss)

# OPTIMIZER (always fp32)
grads_fp32 = cast(grads_bf16, fp32)
clip_grad_norm_(grads_fp32, max_norm=1.0)         # see [[excerpts/gradient-clipping]]
adamw_step(w_fp32, grads_fp32, lr, ...)           # see [[excerpts/adam]]
```

Notice: the fp32 master copy of the weights never leaves the host. Every forward pass produces a fresh bf16 cast. This doubles the weight storage cost (fp32 master + bf16 working copy) — a cost paid happily in exchange for numerical stability.

---

## Why optimizer state must remain fp32

Two reasons:

1. **AdamW's `v̂` underflows in bf16.** The second moment `v = β₂v + (1-β₂)g²` involves `g²`. For small gradients in the `10⁻³` range, `g² ≈ 10⁻⁶`; multiplied by `(1-β₂) = 0.05`, you get `~5 × 10⁻⁸` per step. After hundreds of such accumulation steps, `v` is in the `10⁻⁵` range — which bf16 can represent, but each `(1-β₂)g²` update is ~10 mantissa bits below `v`'s magnitude and gets rounded away. The EMA **stops updating** for any parameter with small historical gradients. This is the "stale v̂" failure mode.
2. **Small-update cancellation.** In AdamW, `θ ← θ - η · (m̂/√v̂ + λθ)`. For a pretraining run with `η ≈ 1e-4`, typical updates are in `10⁻⁵` to `10⁻⁷` relative to the parameter. bf16's 7-bit mantissa means relative precision is `~10⁻²`; updates that small get rounded *to zero* in bf16. You can verify this with `w + 1e-6*δw` in bf16 — the result equals `w`.

The standard pattern — bf16 weights + fp32 master + fp32 optimizer state — is thus not a conservative choice; it is the *only* correct choice. Every frontier model (Llama-3, Mistral, Qwen, DeepSeek-V3) follows it.

---

## fp8 (H100-era) — the 2024 frontier

H100 tensor cores support fp8 matmul with 2× throughput over bf16. The recipe, as refined by the Transformer Engine and DeepSeek-V3:

- **E4M3 forward, E5M2 backward.** Forward activations/weights use E4M3 (4-bit exponent, wider mantissa, tighter range) because activation values are typically bounded. Backward gradients use E5M2 (5-bit exponent, narrower mantissa, wider range) because gradients span many orders of magnitude.
- **Per-tensor or per-block scaling factors.** Each tensor carries a scalar `s` such that `tensor_fp8 · s ≈ tensor_fp32`. Scale factors are tracked via an `amax` history (the running max of absolute values) with a 1-step delay ("delayed scaling").
- **Selective bf16 fallback.** DeepSeek-V3 uses 1×128 and 128×128 block-wise scaling, and keeps outlier-prone layers (LayerNorm, softmax, residual adds) in bf16.
- **Non-matmul ops stay in bf16 or fp32.** Only the matmul path is fp8.

The payoff: ~2× speedup over bf16 at a similar loss curve. The cost: new failure modes, particularly at layers with heavy-tailed activation distributions.

---

## The universal stability rules

Regardless of whether you use fp16, bf16, or fp8, the following ops **must** remain in higher precision:

1. **LayerNorm / RMSNorm** — reductions over `d_model` amplify round-off. Always fp32.
2. **Softmax** — exponentials span many orders of magnitude. Always fp32.
3. **Cross-entropy loss** — log-probabilities of rare tokens underflow quickly. Always fp32.
4. **Master weights and optimizer state** — as derived above.

Getting any of these wrong produces quiet divergence, not a loud crash. The symptom is a loss curve that looks roughly right but is ~1–3% worse than a correctly-implemented baseline, and you only notice at evaluation time.

---

## Common pitfalls

- **Logging loss in bf16**: the curve looks jaggy/quantised because bf16 can only represent ~128 distinct values in any given decade. Log in fp32 (convert before `.item()`).
- **Mixing fp16 and bf16 in the same run**: e.g. fp16 forward, bf16 grads. Silent divergence.
- **Forgetting to unscale before clip**: clipping threshold is off by `S`. See [[excerpts/gradient-clipping]].
- **`eps=1e-8` in AdamW under fp16**: `√v̂` underflows, division by ~0 produces NaN. Bump `eps` to `1e-5` or switch to bf16.
- **fp8 without outlier handling**: a single large activation corrupts the per-tensor amax, and the next step's scaling factor is too large — everything rounds to zero.

---

## Historical timeline

- **2017**: Micikevicius et al. — fp16 + loss scaling + fp32 master.
- **2019–2020**: GPT-2 / Megatron — fp16 production training at scale.
- **2021**: bf16 becomes widely available (TPU v3, A100); Google shifts its pretraining to bf16.
- **2022–2023**: bf16 becomes universal; Llama, Mistral, OLMo all bf16 by default.
- **2023–2024**: fp8 on H100; Transformer Engine ships; DeepSeek-V3 trains at 671B in fp8.
- **2024–2026**: fp8 moving from "research recipe" to "default for H100+".

---

## Connections

- [[excerpts/adam]] — the optimizer state discussion; `β₂=0.95` and `ε=1e-8` assume fp32 optimizer state.
- [[excerpts/gradient-clipping]] — the `unscale → clip → step` ordering is the canonical mixed-precision footgun.
- [[excerpts/weight-init]] — poorly-initialised networks produce activation distributions that don't fit fp16's range; init sensitivity rises in mixed precision.
- [[excerpts/lr-schedules]] — warmup is especially important under fp16 because bias-corrected `v̂` at step 1 is noisier and more prone to representing as `0` or `inf`.
- [[ch-01]] — parent chapter for training fundamentals.
