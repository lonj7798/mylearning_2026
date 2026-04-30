---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/classics/mixed-precision.md
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
---

# Excerpt: Mixed-Precision Training under Sharding

**Paper:** Mixed Precision Training
**Authors:** Micikevicius, Narang, Alben, Diamos, Elsen, Garcia, Ginsburg, Houston, Kuchaiev, Venkatesh, Wu (NVIDIA + Baidu, 2017)
**arXiv:** 1710.03740

**This excerpt focuses on how mixed precision interacts with FSDP sharding — specifically the `MixedPrecision(param_dtype, reduce_dtype, buffer_dtype)` contract, the role of fp32 master weights under sharded optimizer state, and why loss scaling differs under DDP vs FSDP.** The general fp16/bf16/fp8 recipe is covered in ch-02.

---

## The three-dtype axis under FSDP

The Micikevicius 2017 recipe has three independent precision axes (quoted from Technical Details):

1. **Master weights** in fp32 (optimizer updates these).
2. **fp16 compute weights** — cast from master for each forward.
3. **fp32 accumulation** in matmul (fp16 inputs, fp32 accumulator).

FSDP's `MixedPrecision` dataclass exposes these as:

```python
MixedPrecision(param_dtype=torch.bfloat16,
               reduce_dtype=torch.bfloat16,
               buffer_dtype=torch.bfloat16)
```

The mapping from 2017 recipe to FSDP 2023:

| Original recipe            | FSDP axis                   |
|----------------------------|-----------------------------|
| Master weights (fp32)      | Not a `MixedPrecision` field; FSDP keeps optimizer state in fp32 *unconditionally* under FULL_SHARD. |
| Compute weights (fp16/bf16)| `param_dtype` — the dtype of weights after AllGather. |
| fp32 matmul accumulation   | Not a `MixedPrecision` field; handled by the Tensor Core kernel. |
| Gradient reduction dtype   | `reduce_dtype` — the dtype of the ReduceScatter op. |
| Non-param buffers          | `buffer_dtype` — RoPE caches, attention masks. |

**Notice:** `param_dtype` and `reduce_dtype` are independent for a reason. High-scale pretraining runs sometimes set `param_dtype=bf16, reduce_dtype=fp32` — keeping the forward/backward bandwidth at bf16 while paying 2× bandwidth on the ReduceScatter to avoid numerical drift on the most-reduction-sensitive tensor (the gradient). At SFT scale this is overkill; bf16/bf16 is standard.

---

## fp32 master weights under FSDP — the `12P` optimizer-state line

Quoted from the Technical Details:

> "Master weights and optimizer state always fp32."

In the FSDP memory table ([[excerpts/fsdp-sft]]):

```math
\text{Optimizer state} = 12P = \underbrace{4P}_{m_t \text{ fp32}} + \underbrace{4P}_{v_t \text{ fp32}} + \underbrace{4P}_{\theta^* \text{ fp32 master}}
```

FSDP FULL_SHARD divides this by N — but crucially, the *dtype* of m, v, θ* remains fp32 regardless of `param_dtype`. This is not a user option; it is wired into FSDP's flat-param-handle construction. When AllGather fires before a block's forward, it reads from the **bf16 compute copy** (a separate shard), not from the fp32 master.

**Why this split matters.** If the master were bf16:

- Every AdamW step `θ ← θ − lr · m̂ / (√v̂ + ε) − lr · λ · θ` performs an fp32→bf16 rounding per parameter per step.
- With `lr = 1e-5` and typical `||m̂ / √v̂|| ~ 1`, the update magnitude is `1e-5 · θ_scale ≈ 1e-5`.
- bf16's relative precision is ~1% = 1e-2. A 1e-5 update is 3 orders of magnitude below bf16's representable precision → the update is silently rounded to zero.

The fp32 master accumulates these sub-precision updates across steps; after enough steps the accumulated update crosses bf16's representable threshold and the compute copy picks it up. See [[excerpts/adam]] for the same argument applied to `v_t` underflow.

**Notice:** 8-bit Adam (bitsandbytes) departs from this contract by storing m, v in 8-bit blockwise with fp32 dequant-on-step. Works for SFT; has been reported to diverge at 70B+ pretraining scale exactly because of the master-copy argument above.

---

## Loss scaling — why bf16 eliminates it and fp16 + FSDP complicates it

Quoted:

> "bf16 (the modern default): same exponent range as fp32, so loss scaling is **not needed**; gradients never underflow."

Under fp16 DDP, the 2017 recipe is:

```
scaled_loss = S * loss
scaled_grads = backward(scaled_loss)     # grads scaled by S
# synchronize via AllReduce
grads_fp32 = cast(scaled_grads, fp32) / S
clip_grad_norm_(grads_fp32, max_norm)
adamw_step(w_fp32, grads_fp32)
```

Under fp16 FSDP, the ordering becomes subtle. The ReduceScatter reduces *scaled* fp16 gradients across ranks, which can overflow if `S` is large and the per-rank gradient norm is large. Two options:

1. **Unscale before ReduceScatter** — requires an extra pass over gradients. Costly.
2. **Accept the reduced-range ReduceScatter and handle overflow via dynamic loss scaling** — the `GradScaler` halves `S` on observed inf/NaN.

PyTorch's FSDP in fp16 mode (`ShardedGradScaler`) implements option 2 but adds an AllReduce-`SUM` on the per-rank "did any gradient overflow" bit. This is a synchronization point; at 64+ GPU scale, the tail latency is measurable.

**Why bf16 kills the complication.** bf16 has fp32's exponent range (~1e-38 to 3e38), so gradients never underflow and `S = 1` is sufficient. FSDP + bf16 drops the `ShardedGradScaler`, the overflow AllReduce, and the unscale-before-clip ordering requirement. This is the single biggest reason bf16 is the 2025 default.

---

## Gradients reduced in bf16 vs fp32 — the numerical question

The `reduce_dtype` choice is where FSDP-specific numerics show up. Consider a gradient tensor `g` with per-element magnitude 1e-4, reduced across N=64 ranks:

```math
g_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} g_i
```

In bf16 (7 mantissa bits ≈ 1% relative precision), each per-rank `g_i` has rounding error ~1e-6. Summing 64 of them, the accumulated rounding error is ~√64 · 1e-6 = 8e-6, still small relative to the reduced value of O(1e-4). Acceptable.

But if the gradient tensor contains a handful of outlier elements (|g| ~ 1), those dominate the rounding error at bf16. The paper's Figure 2 shows this distribution for SSD training — most gradients are in a narrow band, but a heavy tail crosses `|g| > 1`. Under bf16 ReduceScatter, the tail absorbs most of the rounding error; under fp32 ReduceScatter (23 mantissa bits), rounding error is negligible.

**Notice:** this is the justification for `reduce_dtype=fp32` at extreme scale. DeepSeek-V3 reports using fp32 for all-reduces on the hottest (largest-norm) tensors during late-stage pretraining; frontier runs tune this per-tensor via the grad-norm heavy-tail statistic.

---

## LayerNorm / softmax / loss in fp32 — the stability triangle

Quoted:

> "Keep LayerNorm/RMSNorm in fp32 (reduction-heavy; small numerical errors compound). Keep softmax computation in fp32 (exponentials). Keep cross-entropy loss in fp32."

Under FSDP, these ops sit **outside** the sharded parameter path. Their inputs/outputs flow through the bf16 compute stream, but the reduction itself is cast up to fp32 and back. PyTorch's autocast handles this automatically with `autocast(dtype=torch.bfloat16)`; FSDP's `MixedPrecision` doesn't need to know.

**A subtle FSDP interaction.** LayerNorm weights and biases are FSDP-sharded like any other parameter. During AllGather they arrive in bf16; the LayerNorm forward then casts activations to fp32 for the reduction, applies the bf16 weight cast-to-fp32, and casts back. This is a tiny overhead but an important correctness point — LayerNorm weights in pure bf16 all the way through would accumulate reduction error proportional to `seqlen`.

**Notice:** FSDP2 (2024+) introduces per-module dtype overrides via `fully_shard(..., mp_policy=...)`. A common pattern: shard LayerNorm separately with `param_dtype=torch.float32, reduce_dtype=torch.float32` while the rest of the model is bf16. Worth doing at 100B+.

---

## Common pitfalls — replayed under FSDP

Quoted:

> "Mixing fp16 and bf16 in the same run (e.g. fp16 forward, bf16 grads) → silent divergence.
> Forgetting to unscale before grad clipping → clipping threshold is off by S.
> Logging loss in fp16 → loss curves look quantized/jaggy; log in fp32.
> Using a tiny `eps` in AdamW under fp16 → division by zero."

Distributed translations:

- **Mixed fp16/bf16 across ranks.** If one rank has `param_dtype=bf16` and another `param_dtype=fp16` (a misconfigured world), the AllGather reconstructs a tensor that disagrees bitwise across ranks. NCCL does not type-check; the run silently diverges.
- **Unscale-before-clip under FSDP.** Becomes unscale-before-AllReduce-of-squared-norms. Getting the order wrong makes the global norm `S` times too large → clip fires on every step → effective LR is `1/S` of what you configured.
- **fp16 loss logging under DP.** Each rank's fp16 loss drops distinct mantissa bits; averaging 64 jaggy fp16 losses still looks jaggy. Log the scalar loss in fp32, always.
- **AdamW eps under fp16 FSDP.** Same as single-GPU; fp16's smallest representable positive normal is ~6e-8, so `eps=1e-8` underflows. Bump to 1e-5. bf16's range makes this a non-issue.

---

## Connections

- [[excerpts/fsdp-sft]] — `MixedPrecision(param_dtype, reduce_dtype)` is where this paper's recipe attaches to FSDP.
- [[excerpts/adam]] — the `12P` optimizer-state fp32 line is the master-weight argument.
- [[excerpts/gradient-clipping]] — unscale-before-clip ordering becomes unscale-before-AllReduce under FSDP.
- [[excerpts/sequence-packing]] — varlen attention runs in bf16; numerical reductions in softmax stay fp32.
- [[excerpts/loss-masking-prompt]] — loss is always fp32 regardless of param_dtype; log it in fp32 too.
- [[ch-05]] — synthesis and the bf16/bf16/fp32-master recipe.
