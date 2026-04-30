---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Kingma & Ba — Adam (2014) / Loshchilov & Hutter — AdamW (2017)"
source_url: https://arxiv.org/abs/1412.6980
created_at: "2026-04-23"
---

# Excerpt: Adam / AdamW — the numerical failure modes

**Papers:**
- *Adam: A Method for Stochastic Optimization* — Diederik P. Kingma, Jimmy Ba. ICLR 2015 (preprint Dec 2014). arXiv: [1412.6980](https://arxiv.org/abs/1412.6980)
- *Decoupled Weight Decay Regularization* — Ilya Loshchilov, Frank Hutter. ICLR 2019 (preprint Nov 2017). arXiv: [1711.05101](https://arxiv.org/abs/1711.05101)

This excerpt is the **precision/stability angle** on Adam: what breaks in fp16/bf16, why the `eps` placement matters, and why `v̂` must live in fp32. The optimization theory (bias correction, why AdamW decouples decay) is covered in ch-01. Here we are forensic numerical engineers.

---

## 1. The update rule, reproduced (Adam Algorithm 1)

```math
m_t \gets \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
v_t \gets \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
\hat{m}_t \gets m_t / (1 - \beta_1^t)
\hat{v}_t \gets v_t / (1 - \beta_2^t)
\theta_t \gets \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
```

Every one of the five lines has a precision trap under half-precision. We will walk each in turn.

---

## 2. `v_t` underflow under fp16 / bf16 (the single most common bug)

Consider a parameter whose gradient `g_t` hovers around `1e-4` over training (typical for mid-network LLM weights after warmup). The second-moment accumulator is:

```math
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
```

With `β_2 = 0.95` (modern LLM default, see [[excerpts/mixed-precision]] for the choice), `(1 - β_2) · g_t² = 0.05 · (1e-4)² = 5e-10`.

Now apply format floors:

| Format | Smallest representable positive | `v_t` representable? |
|---|---|---|
| fp32 | ~`1.18e-38` | Yes — 28 binades of headroom |
| fp16 | ~`5.96e-8` (subnormal) | **No** — underflows to zero |
| bf16 | ~`1.18e-38` | Yes — same range as fp32 |

**The fp16 failure:** `v_t` underflows to zero. `√v̂ = 0`, and the update becomes `α · m̂ / ε`. With `ε = 1e-8` and `α = 3e-4`, the step is `~3e4 · m̂` — 1e8× larger than intended. Every parameter with a small gradient gets blasted to infinity on the first step it underflows.

**The bf16 subtlety:** bf16 has fp32-range, so `v_t` does not underflow to zero. But bf16 has only 7 mantissa bits (~2 decimal digits). The update `β_2 · v_{t-1} + (1 - β_2) · g_t²` becomes a catastrophic-cancellation-class operation when `β_2 · v_{t-1}` is many orders of magnitude larger than the new increment. In bf16 this means the `v_t` update is *quietly dropped* for small-gradient parameters, which over ~100 steps drifts `v_t` toward staleness and the optimizer loses its adaptive scaling. [[excerpts/mixed-precision]] summarizes this: "bf16 `v_hat` underflows on small gradients within ~100 steps."

**The universal fix:** always store `m`, `v`, and the master weight copy in fp32. The 2025 FSDP idiom is:

```python
MixedPrecision(param_dtype=torch.bfloat16,   # compute in bf16
               reduce_dtype=torch.float32,   # gradient reduction in fp32
               buffer_dtype=torch.float32)   # norm buffers in fp32
# Optimizer state is always fp32 regardless of these flags.
```

The "14 bytes/param" figure (fp32 master = 4, fp32 `m` = 4, fp32 `v` = 4, bf16 weight = 2) comes directly from this rule. ZeRO-1 exists to shard exactly these 14 bytes across data-parallel ranks.

---

## 3. `ε` placement and the "divide by √v̂ + ε" trap

The paper places `ε` *inside* the square-root-plus:

```math
\theta_t \gets \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
```

An alternative placement (seen in some older TensorFlow code) is:

```math
\theta_t \gets \theta_{t-1} - \alpha \cdot \hat{m}_t / \sqrt{\hat{v}_t + \epsilon}
```

The two are nearly identical when `v̂ » ε²`, but diverge when `v̂ → 0`:

| Form | `v̂ = 0` behavior |
|---|---|
| `√v̂ + ε` | denominator = `ε`; update = `α m̂ / ε` (large but finite) |
| `√(v̂ + ε)` | denominator = `√ε`; update = `α m̂ / √ε` (also finite, *different magnitude*) |

PyTorch and JAX both use the first form (`√v̂ + ε`). Mixing optimizers across frameworks silently changes the effective step for warmup parameters.

**The fp16 NaN trap.** Default `ε = 1e-8` rounds to `1e-8` in fp32 but to `0` (subnormal) in fp16. The defense: bump `ε` to `1e-5` (or better, keep the optimizer in fp32 and never see the problem). The Micikevicius 2017 recipe ([[excerpts/mixed-precision]]) explicitly recommends this — see Table 1 of that paper for the failure curves.

---

## 4. Bias correction and the warmup interaction

From Algorithm 1:

```math
\hat{v}_t = v_t / (1 - \beta_2^t)
```

At `t = 1` with `β_2 = 0.95`, `(1 - β_2^1) = 0.05`, so `v̂_1 = v_1 / 0.05 = 20 · v_1`. The correction factor magnifies early-step estimates.

**The interaction with fp16 scaling.** If gradients are loss-scaled by `S = 2^15`, then `v_t` proportionally scales by `S² = 2^30`. The bias-correction amplifies this further. At `t = 1` the raw `v` magnitudes sit near fp32's comfortable middle range — nowhere close to overflow. Under fp16 you would already be dead. Under bf16 the bias correction is safe because bf16 shares fp32's exponent field.

**The interaction with `β_2 = 0.95` for LLMs.** Llama 1 and successors lowered `β_2` from the paper's `0.999` to `0.95`. This tightens `v̂`'s effective memory from ~1000 steps to ~20 steps, which is good for non-stationary LR schedules — but it also means `v̂` is noisier per step, and precision errors in `v` compound faster. Storing `v` in fp32 is doubly important at `β_2 = 0.95`.

---

## 5. AdamW: the weight-decay decoupling (Loshchilov & Hutter 2017)

The update rule changes only on the last line:

```math
\theta_t \gets \theta_{t-1} - \alpha \cdot \left( \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \cdot \theta_{t-1} \right)
```

Weight decay `λ · θ` is added **directly to the parameter update**, bypassing `m` and `v`. Contrast with L2 regularization applied to the loss:

```math
L_{\text{reg}} = L + \frac{\lambda}{2} \|\theta\|^2 \implies g_t^{\text{reg}} = g_t + \lambda \theta
```

In L2-Adam, `λ θ` enters `g_t`, so it gets exponentially averaged into `m` and squared into `v`. Parameters with large `v̂` receive *less* effective weight decay because the update divides by `√v̂`. This is why L2-Adam's optimal `(α, λ)` form a diagonal valley (AdamW Figure 1) — the two hyperparameters are coupled.

**Precision implication:** under AdamW, `λ θ` is computed in the *parameter dtype*, not the gradient dtype. If weights are bf16 and `λ = 0.1`, `θ = 1e-3`, then `λ θ = 1e-4` — well within bf16 range, safe. But if the optimizer step does `λ * θ_bf16` then casts to fp32 for accumulation, check the order-of-operations in your framework — PyTorch's `AdamW` promotes to the optimizer state dtype before multiplying.

---

## 6. The "no decay" group and numerical symmetry

Standard practice (GPT, Llama, Qwen): exclude LayerNorm `γ`, bias terms, and embeddings from weight decay. The rationale is partly statistical (these parameters don't benefit from shrinkage) and partly numerical: norm `γ` starts at `1.0` and must stay near `1.0` for the layer to preserve scale; decaying it toward `0` breaks the pre-trained normalization.

```python
decay_params = [p for n, p in model.named_parameters()
                if p.ndim >= 2 and "embed" not in n]
no_decay     = [p for n, p in model.named_parameters()
                if p.ndim < 2 or "embed" in n]
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay,     "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

**Notice:** the `eps=1e-8` default is fine only because the optimizer state is fp32. Under a badly-configured mixed-precision setup that leaks bf16 into the `sqrt(v̂) + eps` computation, you would need `eps=1e-5`.

---

## 7. Memory accounting (the 14 bytes/param number)

For a bf16 compute / fp32 optimizer setup:

| Component | Dtype | Bytes/param |
|---|---|---|
| Weights (compute) | bf16 | 2 |
| Master weights | fp32 | 4 |
| `m_t` (first moment) | fp32 | 4 |
| `v_t` (second moment) | fp32 | 4 |
| **Total optimizer-side** | | **14** |

Add gradient buffers (`grad` = bf16, 2 bytes) and activation memory, and a 70B model needs ~1.4 TB just for optimizer state — which is why ZeRO-1 shards these 14 bytes across DP ranks. The fp32-ness of `m`, `v`, and master is non-negotiable; the sharding is the only way to make them fit.

---

## 8. Pretraining / SFT / RL precision-defaults table

| Stage | `β_1` | `β_2` | `eps` | Optimizer state dtype | Weight dtype |
|---|---|---|---|---|---|
| Pretrain | 0.9 | 0.95 | 1e-8 | fp32 | bf16 |
| SFT | 0.9 | 0.95–0.999 | 1e-8 | fp32 | bf16 |
| RL (PPO/GRPO) | 0.9 | 0.95 | 1e-8 (or 1e-5 if fp16) | fp32 | bf16 |
| Any fp16 fallback (V100) | 0.9 | 0.95 | **1e-5** | fp32 | fp16 + master |

RL `eps` bumps exist because policy-gradient variance is higher, `v̂` can transiently drop low, and the `1/(√v̂+ε)` denominator must not explode. This is the same fp16 defense applied to a bf16 run whose `v̂` got unusually small from a rollout gap.

---

## Connections

- [[ch-02]] — §5 "stability pitfalls" enumerates `eps` traps and `v̂` underflow; both derive from §2 and §3 here.
- [[excerpts/mixed-precision]] — why master weights and optimizer state live in fp32; Micikevicius's §3.1 is the origin.
- [[excerpts/gradient-clipping]] — grad-clip happens between `g_t` and `m_t` / `v_t`; clip after unscaling.
- [[excerpts/batch-vs-layer-norm]] — LayerNorm `γ` is in the no-decay group; under bf16 the norm reduction is still fp32.
- [[excerpts/deepseek-v3]] — the AdamW state stays fp32 even in a full fp8 training recipe.
