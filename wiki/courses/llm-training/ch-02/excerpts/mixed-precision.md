---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Micikevicius et al. — Mixed Precision Training (2017)"
source_url: https://arxiv.org/abs/1710.03740
created_at: "2026-04-23"
---

# Excerpt: Mixed Precision Training — the fp16 recipe that started it all

**Paper:** *Mixed Precision Training*
**Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA + Baidu)
**Venue:** ICLR 2018 (preprint Oct 2017)
**arXiv:** 1710.03740
**URL:** https://arxiv.org/abs/1710.03740

---

## Why this paper matters for the 2025 LLM trainer

This is the paper that defined "mixed precision." Every current training stack — PyTorch AMP, DeepSpeed, FSDP's `MixedPrecision(param_dtype, reduce_dtype)`, NVIDIA Transformer Engine — inherits its three-part structure. Even though bf16 has retired fp16-specific machinery for H100-era training, the framing survives: **storage precision is not compute precision is not accumulator precision is not optimizer precision**. Getting the four right is the whole game.

The paper is short (12 pages) and the algorithm fits on one page. What makes it foundational is the crisp accounting of *why* each trick is necessary; read §3 of the paper with a copy of [[batch-vs-layer-norm]] open for the norm-reduction rationale.

---

## 1. The three-part recipe (§3 of the paper)

The abstract crystallizes the approach:

> "The technique combines a) maintaining a single-precision (FP32) master copy of the weights that accumulates the gradients after each optimizer step, b) loss scaling to preserve small gradient values, and c) arithmetic with FP16 storage and FP32 accumulation."

Three independent fixes, each addressing a distinct fp16 failure mode. Remove any one of them and training diverges within tens of steps.

### 1a. fp32 master weights (§3.1)

```math
w_{fp32} \gets w_{fp32} - \alpha \cdot g_{fp32}
w_{fp16} \gets \text{cast}(w_{fp32}, \text{fp16})
```

Forward/backward use `w_fp16`, but the authoritative weight is `w_fp32`. Why: an SGD-style update `w_new = w - lr * g` can have `lr * g` be on the order of `1e-7`, which is a no-op in fp16 (the smallest positive normal is ~`6.10e-5`; subnormals reach ~`5.96e-8`). Whole-gradient information is lost to rounding.

Figure 2 of the paper quantifies this: training ResNet-50 on ImageNet purely in fp16 without a master copy loses 2+ percentage points of top-1 accuracy; restoring the fp32 master recovers the fp32 baseline exactly.

**For the 2025 trainer:** the fp32 master doubles parameter memory (bf16 weights + fp32 master = 6 bytes per param), and AdamW adds another 8 bytes (fp32 `m` and `v`), giving the familiar **14 bytes/param** that dominates optimizer memory. This is why ZeRO-1, ZeRO-2, and fp32-master sharding across ranks exist.

### 1b. Loss scaling (§3.2)

The single most-quoted figure in the paper is Figure 2 (gradient histogram for Multibox SSD):

> "Much of the FP16 representable range was left unused by the gradient values. [...] Note that activation gradient values below 2^-27 in magnitude were irrelevant to the training of this model, but values in the [2^-27, 2^-24) range were important to preserve."

Most gradient magnitude mass falls below fp16's representable floor. Multiply the loss by a scale `S` and by chain rule every gradient is multiplied by `S` too, shifting the histogram up into fp16's range. Divide gradients by `S` before the optimizer step to recover real-scale values.

```math
\tilde{L} = S \cdot L
\tilde{g} = \nabla \tilde{L} = S \cdot g  \qquad (\text{stored in fp16})
g_{fp32} = \text{cast}(\tilde{g}, \text{fp32}) / S
```

**Choosing `S`.** The paper recommends a constant `S` large enough that the smallest meaningful gradient lands above fp16's subnormal floor, but small enough that the largest gradient does not overflow `65504`. Dynamic loss scaling — which the paper discusses in §3.2 and which every framework now implements — starts `S = 2^15` and:

- on any inf/NaN gradient, **skip the step** and halve `S`
- after `N` (e.g. 2000) successful steps, double `S`

This auto-tunes `S` to the training gradient distribution over time.

**Notice:** the scaler has *state* (the current `S`, the success counter). Forgetting to checkpoint scaler state is a silent resume bug — on restart the scaler hunts for a good `S` for ~500 steps and those steps produce inflated (or skipped) gradients.

### 1c. fp32 accumulation in matmul (§3.3)

> "NVIDIA Volta GPUs introduce Tensor Cores, which multiply FP16 input matrices and accumulate the result into either FP16 or FP32 output. [...] In our experience, an FP32 accumulator is necessary to match the master-weights baseline."

Matrix multiply is `C[i,j] = Σ_k A[i,k] * B[k,j]`. The `Σ` is where fp16 fails: for `k = 4096`, adding 4096 fp16 products into a running fp16 sum loses LSBs each addition. With an fp32 accumulator, the partial sums stay fp32; only the final store is downcast.

**For the 2025 trainer:** this is automatic on Volta+ Tensor Cores, H100 Transformer Engine, and B100 — *but only if you use the framework's autocast / matmul wrappers*. A custom CUDA kernel written in pure fp16 (still found in some older codebases) silently drops the accumulator precision.

---

## 2. The precision landscape (table reproduced + extended)

The paper uses fp16 exclusively; bf16 and fp8 post-date it. But its framing — range vs. precision vs. accumulator — is the right vocabulary:

| Format | Bits | Exp | Mantissa | Min normal | Max | Notes |
|---|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~1.18e-38 | ~3.4e38 | Master weights, reductions, loss |
| fp16 | 16 | 5 | 10 | ~6.10e-5 (subnormal down to 5.96e-8) | 65504 | Needs loss scaling; underflow prone |
| bf16 | 16 | **8** | 7 | ~1.18e-38 | ~3.39e38 | Same range as fp32; no loss scaling |
| fp8-E4M3 | 8 | 4 | 3 | ~1.95e-3 (subnormal 1.95e-3 → bias detail varies) | 448 | Forward matmul |
| fp8-E5M2 | 8 | 5 | 2 | ~1.53e-5 | 57344 | Backward matmul |

The key derivation the paper forces you to do yourself:

```math
\text{fp16 dynamic range} = \log_2(65504 / 5.96\text{e-}8) \approx 40 \text{ binades}
\text{fp32 dynamic range} \approx 277 \text{ binades}
\text{bf16 dynamic range} \approx 277 \text{ binades (same exponent field as fp32)}
```

A *binade* is a factor-of-2 interval. fp16 covers 40 binades; a typical transformer's gradients span 30+ binades across layers and training phases. Loss scaling must position those 30 binades within fp16's 40 — which leaves little margin.

**bf16's quiet revolution.** bf16 sacrifices mantissa (7 bits = ~2 decimal digits) for an fp32-identical exponent field. For LLM gradients, **range dominates precision**: a bf16 gradient is less-precise but never underflows. This is the single reason every 2023+ frontier run (Llama 3, Qwen, DeepSeek V2/V3, OLMo, Mistral) uses bf16 + fp32-master and has no loss scaler code at all.

---

## 3. Why bf16 doesn't need loss scaling (derivation)

The fp16 trap: `fp16_min_normal ≈ 6.1e-5`. A gradient of `1e-6` rounds to `0` or a noisy subnormal. Loss scale `S = 2^15 = 32768` lifts it to `~3.3e-2`, safely in fp16 range.

bf16 has `min_normal ≈ 1.18e-38`. A gradient of `1e-6` sits 32 binades above the floor — no scaling needed, no overflow concern (max is `~3.4e38`, same as fp32).

The only cost: bf16's 7-mantissa-bit representation quantizes every gradient to ~`2^-7 ≈ 0.8%` relative precision. For stochastic gradients this is far below the batch noise floor and has no measurable effect on convergence. This is why the authors of §3.2 had to work so hard to defend loss scaling for fp16, and why the successor Brain Float design (Kalamkar et al. 2019) obsoleted the whole machinery once Ampere supported it.

---

## 4. The ordering rule (§3 composed)

The scaler, clipper, and optimizer compose in exactly one correct order under fp16:

```python
# 1. forward in fp16
with autocast(dtype=torch.float16):
    loss = model(batch).loss

# 2. scaled backward — gradients are fp16, multiplied by S
scaler.scale(loss).backward()

# 3. UNSCALE first — grads become fp32 at real scale
scaler.unscale_(optimizer)

# 4. clip on real-scale grads (see [[excerpts/gradient-clipping]])
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 5. step (or skip if any grad was inf/NaN)
scaler.step(optimizer)
scaler.update()
```

**Notice:** steps 3 and 4 cannot be swapped. If you clip before unscaling, the threshold `1.0` is actually `1.0 / S` in real terms — i.e. essentially zero, and the optimizer is starved. This is the single most common fp16 bug in hand-rolled training loops.

Under bf16 the recipe collapses to three lines: forward → `backward()` → `clip_grad_norm_(1.0)` → `step()`. No scaler, no ordering trap.

---

## 5. What the paper empirically demonstrated (Table 1)

The paper reports fp16-mixed-precision training matching fp32 baselines on:

| Task | Model | fp32 baseline | fp16 mixed |
|---|---|---|---|
| ImageNet classification | AlexNet | 56.77% | 56.93% |
| ImageNet classification | ResNet-50 | 73.85% | 73.87% |
| Detection (VOC 2007) | Multibox SSD | 76.9 mAP | 77.1 mAP |
| LM (1B Words) | Big LSTM | 39.09 (test ppl) | 39.99 |
| NMT EN→FR | GNMT | 34.96 BLEU | 34.87 BLEU |
| GAN | DCGAN | (visual parity) | (visual parity) |

Parity across vision, language, and generative tasks — the evidence that convinced the field fp16 training was not a curiosity. For the modern reader: the numerical properties that saved these models (master weights, loss scaling, fp32 accumulation) are exactly the ones that still save LLMs — only the format changed from fp16 to bf16/fp8.

---

## 6. Stability guidance the paper folds in implicitly (§3.4, §4)

Even in fp16 mixed precision, certain ops must stay fp32:

- **Softmax.** `exp(x)` on large `x` overflows fp16 at `x ≈ 11`.
- **LayerNorm reductions.** `mean(x²)` over long vectors accumulates bias in fp16.
- **Cross-entropy loss.** `log(softmax(x))` is a numerical-stability minefield.
- **Optimizer state.** Adam/AdamW `m`, `v`. Under fp16, `v̂` underflows to zero for any parameter with small gradient — see [[excerpts/adam]].

The paper discusses this as "certain operations benefit from FP32" but does not enumerate them exhaustively. The 2025 defaults in FSDP / Transformer Engine codify it:

```python
MixedPrecision(
    param_dtype   = torch.bfloat16,   # weights in bf16
    reduce_dtype  = torch.float32,    # gradient all-reduce in fp32
    buffer_dtype  = torch.float32,    # norms, running stats
)
```

---

## Connections

- [[ch-02]] — chapter home; §2 "The fp16 recipe" and §3 "which ops live in fp32" both trace here.
- [[excerpts/adam]] — why optimizer state must remain fp32 even under bf16.
- [[excerpts/gradient-clipping]] — the `unscale → clip → step` ordering rule.
- [[excerpts/batch-vs-layer-norm]] — norm reductions in fp32 are the single most common precision bug.
- [[excerpts/deepseek-v3]] — 2024's production fp8 recipe; extends this paper's framing to 8-bit.
