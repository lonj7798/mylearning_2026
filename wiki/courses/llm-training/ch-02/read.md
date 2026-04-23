<!-- chapter: ch-02
     track: foundations
     title: Numerical Precision and Stability
     sources: [[mixed-precision]], [[adam]], [[gradient-clipping]], [[batch-vs-layer-norm]], [[deepseek-v3]]
     figures: figures/precision-range.html
-->

# Chapter 2 — Numerical Precision and Stability

> **Core insight.** Modern LLM training is a multi-precision pipeline: bf16 for compute, fp32 for reductions, fp8 for the matmul tensor cores on H100/B100. The *rules about which op lives in which precision* are what keep the run from silently diverging.
>
> **Guideline.** Use bf16 everywhere you can in 2025 (no loss scaler needed); keep norm reductions, softmax, loss, and optimizer state in fp32; reserve fp8 for the matmul path with fp32 master weights and per-tensor or per-block scaling.

---

## Why this chapter exists

Precision bugs don't announce themselves. A run trained with the wrong `eps` on AdamW under fp16 produces a loss curve that looks fine until step 3000, when a single bad step produces `inf` and the training scaler helpfully skips it — for the next 500 steps. A run where the LayerNorm reduction accidentally happens in bf16 instead of fp32 converges, but the final perplexity is 0.5 points worse than you expected and you never figure out why. fp8 runs without per-tensor scaling drift for hundreds of steps before the amax history gets saturated and collapses.

The cost of getting precision right is small. The cost of getting it wrong is an unbudgeted restart at 80% of your compute. This chapter enumerates what every layer of the training stack needs and why.

All primary material is in [[mixed-precision]], with cross-references to [[adam]] and [[gradient-clipping]] for their interactions.

---

## 1. The four formats, with operational meaning

The layout of a floating-point number is `sign | exponent | mantissa`. Two axes matter: **range** (exponent bits) and **precision** (mantissa bits). Everything else follows.

| Format | Bits | Exp | Mant | Range | Precision | Typical use |
|---|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1e-38 to 3e38 | ~7 dec digits | master weights; norm reductions; softmax; loss |
| fp16 | 16 | 5 | 10 | ~6e-8 to 65504 | ~3 dec digits | Volta-era compute; **needs loss scaling** |
| bf16 | 16 | **8** | 7 | ~1e-38 to 3e38 | ~2 dec digits | 2025 default compute format; no loss scaling |
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 | coarse | H100+ forward matmul (activations × weights) |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 | coarser | H100+ backward matmul (gradients) |

See `figures/precision-range.html` for a side-by-side visualizer of where each format's representable values land on the real line.

The key trade: **fp16 has more precision than bf16 but a much narrower range**. For LLM gradients specifically, range dominates — gradients span many orders of magnitude, and anything below fp16's floor (~6e-8) underflows to zero silently. bf16 matches fp32 range, which is why 2023+ frontier runs switched.

**The bf16 / fp16 decision in one sentence.** If you have to support Volta (V100) hardware, use fp16 + dynamic loss scaling; otherwise use bf16 and delete the scaler code.

---

## 2. The fp16 recipe — and why bf16 mostly retires it

From [[mixed-precision]], Micikevicius et al. 2017's three-part fp16 recipe:

1. **fp32 master weights.** Optimizer state and a pristine parameter copy live in fp32. The bf16/fp16 weights used in forward are a *view*.
2. **Loss scaling.** Multiply the loss by `S` before `.backward()` so small gradients end up in fp16's representable range. Divide gradients by `S` before the optimizer step.
3. **fp32 matmul accumulation.** Tensor cores compute `C += A @ B` with fp16 inputs but an fp32 accumulator. Without this, long reductions saturate.

The full pipeline in code:

```python
# fp16 + dynamic loss scaler (PyTorch AMP)
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(batch).loss          # forward in fp16
    scaler.scale(loss).backward()         # scaled backward → fp16 grads ×S

    scaler.unscale_(optimizer)            # ← grads ÷ S back to real scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # NOW valid
    scaler.step(optimizer)                # step, or skip if any grad is inf/NaN
    scaler.update()                       # adjust S dynamically
```

Dynamic scaling starts `S = 2^15`; on any inf/NaN gradient it halves `S` and skips the step; every ~2000 successful steps it doubles `S`. The scaler state is a one-line training-resume bug waiting to happen if you forget to checkpoint it.

**bf16's simplification.** Switch to bf16 and the scaler code deletes itself:

```python
for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
    loss.backward()                       # grads live in bf16 range = fp32 range
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

Every open 2023+ frontier run (Llama 3, GPT-NeoX, Mistral, Qwen, DeepSeek V2/V3, OLMo, Tülu 3) uses this bf16 + fp32-master setup. If you're starting a training codebase today and don't need V100 compatibility, don't support fp16 at all.

---

## 3. Which ops must live in fp32 (the never-ending list)

Even under bf16 compute, **four kinds of ops must stay in fp32** regardless:

1. **Norm reductions** (LayerNorm / RMSNorm). The `mean(x²)` over 4096-dim vectors accumulates errors in bf16 that bias normalisation. Frameworks default to fp32 reductions here; never override it. See [[batch-vs-layer-norm]].
2. **Softmax.** Exponentials are range-sensitive; a `softmax` computed in bf16 on attention logits loses tokens at the right tail.
3. **Cross-entropy loss.** The final log-softmax + NLL must happen in fp32 for numerically stable gradients.
4. **Optimizer state + master weights.** AdamW's `m_t`, `v_t`, and the fp32 shadow copy. See [[adam]].

```python
# The universal pattern for a norm under mixed precision
class RMSNorm(nn.Module):
    def forward(self, x):                          # x: bf16 activations
        in_dtype = x.dtype
        x_fp32 = x.float()                         # promote for reduction
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(in_dtype) * self.weight
```

The `.float()` cast is not optional. Drop it and you'll see a slow, silent quality regression — exactly the hardest kind of bug to diagnose.

---

## 4. fp8 — the 2024+ frontier path

H100 and B100 tensor cores compute fp8 matmul at 2× the bf16 throughput. The deployment details, however, matter more than the flop count.

**Per-tensor scaling.** Before each matmul, compute `amax(x)` and pick a scale `s = fmax_E4M3 / amax(x)`, then cast `x_fp8 = (s · x).to(float8_e4m3)`. The matmul accumulates in bf16/fp32, and the product is unscaled on the way out. This is the "fp8 as accelerator feature, not storage format" pattern.

**E4M3 vs E5M2 split.** [[mixed-precision]] documents the forward/backward split: E4M3 (4 exponent bits) for forward activations and weights where precision matters more than range; E5M2 (5 exponent bits) for gradients where the range matters more. NVIDIA's Transformer Engine library automates this.

**DeepSeek-V3's block-wise variant.** DeepSeek-V3 uses 1×128 (weight rows) and 128×128 (activation tiles) block-wise scaling rather than per-tensor. Each block has its own scale factor. This tolerates heterogeneous distributions within a weight matrix at the cost of more scale-factor bookkeeping. The paper is in [[deepseek-v3]].

**Delayed scaling.** Computing `amax(x)` fresh on every call is expensive. Most implementations use *delayed* scaling: the scale for step t is computed from an amax history of steps t-16 to t-1, lagging by 1 step. Under stable training this is fine. In volatile phases (warmup, LR changes, RL rollouts) it can drift — which is why many frontier fp8 runs keep a bf16 fallback path and auto-switch.

**Universal fp8 rule.** Even in full fp8 training, residual streams, norms, and softmax stay in bf16 or fp32. fp8 is a matmul-specific feature. If anyone claims "we trained fully in fp8," they almost certainly mean "the matmuls are fp8 and everything else is bf16."

---

## 5. Stability pitfalls the precision stack creates

The bugs below are all precision-coupled. Know them by sight:

- **NaN from small `eps` under fp16.** AdamW's `1 / (√v̂ + ε)` can overflow when `v̂` is near zero and `ε = 1e-8` underflows in fp16. Fix: either move to bf16, or bump `eps` to `1e-5`.
- **Loss-scaled clipping is silently wrong.** Unscale gradients *before* calling `clip_grad_norm_`. See [[gradient-clipping]]. Ordering is `unscale → clip → step`.
- **Mixed fp16 and bf16 in one run.** This happens by accident — e.g. fp16 forward but bf16 gradient reduction under FSDP `reduce_dtype=bfloat16`. The two half-precision formats are not interchangeable; silent divergence.
- **Logging loss in bf16.** Loss curves look quantised/jaggy because bf16 only has ~2 decimal digits of precision. Cast `loss.float()` before `log()` or `.item()`.
- **Dropping fp32 reductions in norms.** See §3 — always promote.
- **fp8 amax collapse.** Under extreme gradient spikes the amax history saturates; subsequent scales are useless. Keep a bf16 fallback for the outlier-prone layers (embeddings, head).

---

## 6. The recommendation table

| Context | Compute dtype | Reductions | Grads | Opt state | Loss scaling |
|---|---|---|---|---|---|
| 2025 pretrain (default) | bf16 | fp32 | bf16 | fp32 | none |
| Pre-H100 hardware (V100) | fp16 | fp32 | fp16 (scaled) | fp32 | dynamic |
| H100+ frontier | bf16 + fp8 matmul | fp32 | bf16 (+ fp8 backward) | fp32 | per-tensor / block |
| SFT / DPO | bf16 | fp32 | bf16 | fp32 | none |
| RL (PPO/GRPO) | bf16 | fp32 | bf16 | fp32 | none; reward spikes tracked separately |
| Inference / eval | bf16 | bf16 OK | — | — | — |

---

## Connections and what's next

- **[[adam]] / ch-01** — optimizer state stays in fp32; bf16 `v̂` underflows on small gradients within ~100 steps.
- **[[gradient-clipping]] / ch-01** — the `unscale → clip → step` ordering is specifically a mixed-precision concern.
- **[[batch-vs-layer-norm]] / ch-03** — the norm-reduction precision rule comes back when we discuss RMSNorm / QK-norm placement.
- **ch-05 (FSDP)** — FSDP's `MixedPrecision(param_dtype, reduce_dtype, buffer_dtype)` is how you declare this policy across shards.
- **[[deepseek-v3]]** — canonical 2024 fp8 training recipe; the reference implementation for block-wise scaling.

## Further reading

- [[mixed-precision]] — full extract of Micikevicius 2017 plus bf16 / fp8 successor context.
- [[deepseek-v3]] — the per-block fp8 recipe in a production 671B MoE run.
- Karpathy's "recipe" (see [[karpathy-training-neural-net-recipe]]) — "start in fp32; add mixed precision only once training is stable".

## Companion visualization

**[figures/precision-range.html](figures/precision-range.html)** — side-by-side plot of where fp32 / fp16 / bf16 / fp8-E4M3 / fp8-E5M2 can represent values on the real line. Hover to see the boundary of each format's dynamic range and why bf16 matches fp32 in range even with only 7 mantissa bits.
