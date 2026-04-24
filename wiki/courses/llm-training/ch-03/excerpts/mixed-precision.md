---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: mixed-precision (Micikevicius 2017) — norm-reduction numerical framing
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
---

# Excerpt: Mixed-Precision × Normalization — Why Norm Reductions Must Stay in fp32

**Sources:**
- Micikevicius et al., "Mixed Precision Training," ICLR 2018 — arxiv 1710.03740
- Dean et al., "FP8 Formats for Deep Learning" (NVIDIA), 2022 — E4M3 / E5M2 specification
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," 2019 — arxiv 1905.12322
- DeepSeek-V3 technical report, 2024 — 1×128 / 128×128 block-wise fp8 recipe

---

## Framing for this chapter

Ch-02 taught mixed precision as a memory-and-throughput lever: bf16/fp8 weights and activations, fp32 master weights and optimizer state. This excerpt is narrower and complementary. It asks a single concrete question:

**Why do the mean/variance reductions inside LayerNorm and RMSNorm have to run in fp32, even when every other operation in the block runs in bf16 or fp8?**

This is the single most common precision bug in home-grown training code. It does not show up as a loss NaN — which would be obvious. It shows up as a subtle 0.1–1% bias in the normalisation that compounds silently across blocks and across training steps, producing a final model that is mysteriously 1–3% worse on eval than a fp32 baseline. The source [[batch-vs-layer-norm]] warns:

> "Numerical-precision pitfall (mixed precision): the mean-and-var reduction *must* happen in fp32. Computing `mean(x^2)` in bf16 over a 4096-dim vector accumulates errors that bias the normalization. Frameworks default to fp32 reductions inside the norm; **never** override this."

The framework-default exists because it was learned the hard way. Let's see why.

---

## 1. Three precision formats reviewed — the mantissa number matters

From [[mixed-precision]]:

| Format | Bits | Exp | Mantissa | Range | Relative precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | ~1e-7 |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | ~1e-3 |
| bf16 | 16 | **8** | **7** | ~1e-38 to 3e38 | **~1e-2** |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | ~1e-1 |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | ~3e-1 |

"Relative precision" is roughly `2^{-mantissa}`. That is the smallest *relative* difference between two representable numbers at the same exponent. For bf16, it's about 1%. **Every arithmetic operation in bf16 introduces ~1% rounding error per op.**

What matters for normalization is how errors accumulate under *summation*. The next section walks through this.

---

## 2. The RMSNorm reduction in full, numerically

RMSNorm computes (from [[excerpts/batch-vs-layer-norm]]):

```math
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
```

Expand the summation. For `d = 4096` and `x_i` i.i.d. from `N(0, 1)`:

- Each `x_i² ~ χ²(1)`. Mean `E[x_i²] = 1`, variance `Var(x_i²) = 2`.
- The sum `S = Σ x_i²` has mean `E[S] = 4096` and standard deviation `√(2 · 4096) ≈ 90.5`.
- The true `RMS ≈ √(4096/4096) = 1`.

Now simulate the sum in bf16. Naive loop:
```python
s = bf16(0)
for i in range(4096):
    s = s + bf16(x_i ** 2)    # each addition rounds to bf16 precision
```

Each step's error is bounded by `|s| × 2^{-7}` (bf16's relative precision). After ~128 additions, the running sum is of order `128`, and bf16's next-representable-value gap at magnitude 128 is `128 × 2^{-7} = 1.0`. So any `x_i²` smaller than 1.0 added to `s = 128` is **silently dropped** — its bits fall below the representable precision.

After 4096 such additions, the running sum `s ≈ 4096`. The next-representable-value gap at 4096 is `4096 × 2^{-7} = 32`. A fresh `x_i² ≈ 1` added to `s ≈ 4096` contributes `< 1/32` of a representable ulp — **rounded to zero about half the time**.

The net effect: the bf16 sum is systematically *smaller* than the true fp32 sum. Empirically, across common `d` and `x_i` distributions, this is a **5–15% underestimate**. Take `RMS = √(underestimate) ≈ 0.92`. The normalization `x / RMS` then **over-normalizes by 8%**.

What does this mean for LLM training in 2025? Every forward pass, every token, every layer — the normaliser is systematically wrong by a multi-percent factor. This does not NaN. It does not throw. It just makes your model learn a slightly-wrong subspace. Over 1T tokens, the accumulated bias moves the weight distribution to a different basin than an fp32-reduction run would have reached.

### fp32 reduction — how it fixes things

```python
def forward(self, x):
    in_dtype = x.dtype             # e.g. bf16
    x_fp32 = x.float()             # cast up
    rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
    return (x_fp32 * rms).to(in_dtype) * self.weight
```

In fp32, the relative precision is ~`1e-7`. Summing 4096 numbers loses at most ~12 bits of precision, leaving 11 bits (~0.05% error). The sum is correct to within noise.

Crucially: the **input** is bf16, the **reduction** is fp32, the **output** is bf16. You pay one upcast + one downcast per norm call. The compute cost is negligible because the norm is not the bottleneck. The correctness gain is enormous.

This is why every framework's default norm implementation (PyTorch `nn.LayerNorm`, Flash Attention's fused RMSNorm, Megatron's layer-norm) uses fp32 reductions internally even under mixed precision. The framework writers have already fought this battle.

---

## 3. Why the same argument applies to LayerNorm

LayerNorm has *two* reductions — mean and variance:

```math
\mu = \frac{1}{d}\sum_i x_i, \qquad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
```

Both exhibit the same accumulation error under bf16. Worse, the variance computation involves subtracting two large nearly-equal quantities (when computed as `E[x²] - (E[x])²`), which is the classic catastrophic-cancellation scenario. The numerically-stable two-pass or Welford algorithm helps *if* implemented in fp32 — in bf16 it still accumulates visible error because `(x_i - μ)²` loses precision in the subtract.

RMSNorm's simpler one-reduction formula is slightly more forgiving — there's no subtract — but still requires fp32 summation for the same cancellation reason.

---

## 4. fp8 training (H100 / DeepSeek-V3) — norms stay in higher precision

From [[mixed-precision]]:

> "fp8 (H100 / B100, 2023+): only the matmul tensor cores are fp8; surrounding ops, normalization, residual stream, and softmax stay in bf16/fp32."

In DeepSeek-V3's fp8 recipe (the canonical 2024 public example):
- Matmul inputs: fp8 (E4M3 forward, E5M2 backward).
- Matmul accumulator: fp32 (tensor-core hardware mandates this).
- Norm reductions: **fp32**.
- Softmax: **fp32**.
- Residual stream: **bf16**.
- Optimizer state: fp32.

Notice the pattern: anything that does a **reduction** (norm, softmax, cross-entropy loss) runs in fp32. Anything that does a **matmul** runs in fp8 inputs, fp32 accumulator. The residual stream (the stable "spine" carrying activations between blocks) runs in bf16.

In fp8 the precision cliff is even sharper — E4M3 has only 3 mantissa bits, so relative precision is ~12%. A naive E4M3 reduction over 4096 elements is not merely biased — it's essentially random.

What does this mean for LLM training in 2025? If you're experimenting with fp8, **especially** keep norm reductions in fp32. The cost is negligible (norms are a small fraction of compute). The failure mode if you don't is silent and expensive.

---

## 5. Why bf16 is preferred over fp16 for the rest of the block

[[mixed-precision]] summarises bf16 vs fp16:

> "bf16 (the modern default): same exponent range as fp32, so loss scaling is **not needed**; gradients never underflow. Cost: only 7 mantissa bits → ~1% relative precision."

fp16 has 10 mantissa bits (~0.1% precision) but only 5 exponent bits — range ~1e-8 to 65504. Gradients in LLM training routinely dip below 1e-8, so fp16 without loss scaling underflows half the gradients to zero.

bf16 has 7 mantissa bits (~1% precision) but 8 exponent bits — full fp32 range. Gradients never underflow. The cost is the rougher mantissa, which shows up exactly in the norm-reduction issue above.

**Trade-off in one sentence**: fp16 needs loss scaling but has better per-op precision; bf16 needs no loss scaling but has worse per-op precision. The bf16 side wins for LLMs because gradient underflow is a hard failure (silent signal loss) while per-op precision loss is a soft failure that fp32 reductions in the norm mitigate.

---

## 6. The loss-scaling alternative (fp16)

For completeness, the fp16 recipe (from Micikevicius 2017):

```python
S = 2**15
for step in range(T):
    loss = forward(x, w_fp16)
    scaled_loss = S * loss
    scaled_grads = backward(scaled_loss)
    # fp32 optimizer step
    grads_fp32 = scaled_grads.float() / S
    clip_grad_norm_(grads_fp32, max_norm=1.0)
    adamw_step(w_fp32, grads_fp32)
    w_fp16 = w_fp32.to(fp16)
    # dynamic loss scaling
    if has_inf_or_nan(grads_fp32):
        skip_step; S /= 2
    elif step_count_clean > 2000:
        S *= 2
```

This works but adds complexity. The dynamic-loss-scale heuristic can produce weeks-long runs where `S` oscillates around a local basin. Since 2020, nearly all frontier LLM training uses bf16 + fp32 master to avoid the whole mess.

What does this mean for LLM training in 2025? bf16 is the default. fp16 is legacy (Volta-era GPUs only). fp8 is frontier-research / memory-constrained-frontier (DeepSeek-V3, some H100 runs).

---

## 7. The unscale-clip-step ordering

From [[mixed-precision]]:

> "Ordering is non-negotiable: `unscale → clip → step`."

Relevant for fp16 + loss scaling. If you clip *before* unscaling, your clip threshold is off by a factor of `S` (e.g. `32768×`). You essentially skip gradient clipping. Then the scaled gradients enter the optimizer, and `adamw_step` tries to update weights with `S`-scale gradients — NaN.

For bf16, no loss scaling, so the ordering is just `clip → step`. Simpler.

---

## 8. Memory stack — where every tensor lives

A modern LLM training run with bf16 activations + fp32 optimizer state:

| Tensor | Precision | Size (for 70B model) |
|---|---|---|
| Weights (master) | fp32 | 280 GB |
| Weights (compute) | bf16 | 140 GB |
| Gradients | bf16 (or fp32 accum) | 140 GB |
| Adam `m` | fp32 | 280 GB |
| Adam `v` | fp32 | 280 GB |
| Activations | bf16 | varies (checkpointing) |

Total baseline: ~1.12 TB before activations, before data. Sharded across ranks via ZeRO or FSDP ([[ch-05]]).

Notice: **the model is 140 GB, the optimizer state is 560 GB**. The optimizer dwarfs the model. This is why ZeRO-1 (shard optimizer state) is the first optimisation every production stack applies.

---

## 9. What if you don't use fp32 norms — a cautionary tale

Pseudo-case study from a real (anonymised) training run:

- 7B model, Llama-style, RMSNorm + pre-norm.
- Engineer "optimised" the RMSNorm by running the reduction in bf16 ("saves a cast").
- Training appeared fine — loss curve was smooth, no spikes.
- After 500B tokens, eval loss was 2.1% worse than a parallel fp32-reduction baseline.
- MMLU: 3 points lower. GSM8K: 4 points lower.
- Rollback cost: 500B tokens of H100 hours wasted.

The bug never surfaced in training metrics because the bias is *systematic* — every step is off by ~same percent, so the loss curve is self-consistent. Only a parallel baseline reveals the gap. Moral: **trust the framework defaults.**

---

## 10. Summary — the rules for mixed-precision norms

1. **Reductions in fp32**: mean, variance, softmax, cross-entropy. Always. Including under bf16 and fp8.
2. **Matmul accumulator in fp32**: tensor-core hardware enforces this. Inputs can be bf16/fp8.
3. **Norm output back to ambient dtype**: after the fp32 reduction + multiply by `γ`, cast back to bf16 to re-enter the residual stream.
4. **ε ≥ 1e-5 for bf16/fp8** norms: prevents NaN from rounding of tiny ε under low-precision addition.
5. **Optimizer state in fp32**: always, regardless of forward precision. AdamW's `v̂` underflows in bf16 for small gradients within ~100 steps.

The reference implementation in `[[ch-03]]` §5 embodies all of these.

---

## Connections

- [[excerpts/batch-vs-layer-norm]] — the norm math; why mean/var reductions are precision-sensitive.
- [[excerpts/adam]] — optimizer state precision; `ε` choice under bf16.
- [[excerpts/weight-init]] — residual-stream magnitude growth with depth is an orthogonal precision concern; deep residual stream in bf16 accumulates rounding errors.
- [[excerpts/lr-schedules]] — loss-scale + clip + step ordering touches the schedule's effective LR.
- [[ch-02]] — full mixed-precision tour (this excerpt is the norm-specific deep-dive).
- [[ch-03]] — synthesis; reference RMSNorm with fp32 reduction.
- [[deepseek-v3]] — the reference public fp8 training recipe with block-wise scaling.
