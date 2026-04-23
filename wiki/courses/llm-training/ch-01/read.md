<!-- chapter: ch-01
     track: foundations
     title: Optimization Fundamentals for Transformers
     sources: [[adam]], [[gradient-clipping]], [[mixed-precision]], [[lr-schedules]], [[weight-init]]
     figures: figures/beta2-memory.html
-->

# Chapter 1 — Optimization Fundamentals for Transformers

> **Core insight.** AdamW with per-parameter adaptive step sizes + *decoupled* weight decay + global-norm gradient clipping is the entire optimizer story for 2025 LLM training. Every other piece of the training loop is built on top of this triad.
>
> **Guideline.** For pretraining, use `AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)` with `clip_grad_norm_=1.0`. Lower the LR (not the optimizer) for SFT and RL. Deviate from this only with evidence.

---

## Why this chapter exists

Most practitioners reach for `torch.optim.AdamW` without thinking. That works — until it doesn't. When it fails it fails silently: loss plateaus, norm explodes in step 3000, gradients NaN under fp16, `v_hat` underflows to zero on embedding rows, FSDP clips the wrong global norm. Every one of those bugs is an optimizer-level bug. This chapter builds the mental model you need to *recognize* and *fix* them, not just to pick hyperparameters.

Three things you should walk away with:

1. The exact algorithm box for Adam and AdamW, with the single line that differentiates them.
2. Why LLM defaults (`β₂=0.95`, `wd=0.1`) are different from the ImageNet defaults you may have learned, and what happens when you mix them up.
3. The `unscale → clip → step` ordering that gets silently wrong in half of production trainers.

All of these come from [[adam]] and [[gradient-clipping]] in the raw-data library; this chapter pulls them into a single coherent narrative.

---

## 1. The SGD → Adam → AdamW progression

Plain SGD with momentum takes a single scalar learning rate and applies it to every parameter equally:

```
theta_t = theta_{t-1} - alpha * grad_t                             # SGD
theta_t = theta_{t-1} - alpha * (rho * m_{t-1} + grad_t)           # SGD + momentum
```

This works well on convex-ish landscapes. It works badly on the loss surfaces of transformers: the per-parameter curvature varies by orders of magnitude between attention weights, embeddings, layer-norm scales, and FFN projections. One scalar LR cannot serve all of them.

**Adam** (Kingma & Ba 2014) solves the problem by maintaining, *per parameter*, a running estimate of the gradient's variance and rescaling each step by that variance. Parameters that see high-variance gradients get small steps; parameters with clean signal get full steps. The update is still gradient descent — just with per-parameter adaptive step sizes.

**AdamW** (Loshchilov & Hutter 2017) fixes a subtle but important bug. The original Adam paper recommended implementing weight decay by adding `λ * θ` to the loss (i.e. L2 regularization). But that term flows through the gradient, gets baked into the second-moment estimate `v_t`, and gets *adaptively rescaled* — meaning parameters with large `v_t` effectively see less weight decay. AdamW fixes this by applying the decay term directly to the parameter update, bypassing `m_t` and `v_t` entirely.

The fix is one line of code. The consequences are large enough that "Adam" in every 2024+ LLM report actually means AdamW.

---

## 2. Adam and AdamW, on paper and in code

From [[adam]], the update rule for a timestep `t` with gradient `g_t`:

```
m_t    = β₁ · m_{t-1} + (1 - β₁) · g_t                 # 1st moment (momentum)
v_t    = β₂ · v_{t-1} + (1 - β₂) · g_t²                # 2nd moment (variance)
m_hat  = m_t / (1 - β₁ᵗ)                               # bias correction
v_hat  = v_t / (1 - β₂ᵗ)
θ_t    = θ_{t-1} - α · m_hat / (√v_hat + ε)            # Adam update
θ_t    = θ_{t-1} - α · (m_hat / (√v_hat + ε) + λ·θ_{t-1})   # AdamW update  ← the whole change is λ·θ
```

Everything else is shared. The bias-correction (`m_hat`, `v_hat`) matters because `m_t` and `v_t` are zero-initialized — without correction, early steps underestimate the true gradient statistics by factors of `1 − βᵗ`.

Here's the actual PyTorch source for the core AdamW step (from `torch.optim.adamw._single_tensor_adamw`, simplified):

```python
# decoupled weight decay happens BEFORE the adaptive step
param.mul_(1 - lr * weight_decay)

# momentum + second-moment EMAs
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

# bias correction
bias_correction1 = 1 - beta1 ** step
bias_correction2 = 1 - beta2 ** step
step_size = lr / bias_correction1
bias_correction2_sqrt = bias_correction2 ** 0.5

# adaptive step
denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
param.addcdiv_(exp_avg, denom, value=-step_size)
```

The first line is the entire AdamW contribution. If you delete it, you have Adam. If you instead add `weight_decay * param` to `grad` before the EMA, you have L2-Adam — the historically "bad" optimizer that AdamW replaces.

**Practical pitfall.** In PyTorch, `torch.optim.Adam(..., weight_decay=0.1)` is *not* AdamW. It's L2-Adam, which is what Loshchilov & Hutter showed is broken. Always use `torch.optim.AdamW` explicitly. This is the single most common optimizer bug in hobby-scale training code.

---

## 3. LLM hyperparameter defaults — and why they differ

Here's the table from [[adam]], annotated with what each number means operationally:

| Hyperparameter | Pretrain | SFT | RL (PPO/GRPO) | What it does |
|---|---|---|---|---|
| `β₁` | 0.9 | 0.9 | 0.9 | momentum timescale (~10 step half-life) |
| `β₂` | **0.95** | 0.95 – 0.999 | 0.95 | variance-memory timescale (~20 steps vs ~1000) |
| `ε` | 1e-8 | 1e-8 | 1e-8 (or 1e-5 for fp16) | denominator floor |
| `weight_decay` | **0.1** | 0.0 – 0.01 | 0.0 | parameter-shrinkage rate |
| Peak `lr` | 1e-4 to 6e-4 | 1e-5 to 5e-5 | 1e-6 to 1e-5 | step-size envelope |

Two numbers need justification because they diverge from classical defaults:

**`β₂ = 0.95` (not 0.999).** The second moment `v_t` approximates per-parameter gradient variance. `β₂ = 0.999` gives it a ~1000-step memory; at LLM pretraining scale — where the loss landscape is non-stationary (curriculum, data shifts, LR warmup) — that memory is too long. `v_t` lags reality, and the effective step size is stale. `β₂ = 0.95` has a ~20-step memory and tracks the current gradient distribution. This was Llama 1's choice and is now standard across GPT, Qwen, DeepSeek, and OLMo.

See `figures/beta2-memory.html` for an interactive visualisation of how `v_hat` responds to a gradient spike under `β₂ ∈ {0.9, 0.95, 0.999}`.

**`weight_decay = 0.1` (not 1e-4).** Transformers over-parameterize far more than image classifiers. The usefulness of weight decay as a generalisation prior scales with the ratio of parameters to data tokens. `0.1` is empirically right for 100B-token pretraining runs; it's far too aggressive for a 10K-sample SFT where you typically drop it to zero or `0.01`.

**Exclusions.** Always exclude LayerNorm scales, biases, and embedding scale parameters from weight decay. The convention is a two-group optimizer:

```python
decay_params, no_decay_params = [], []
for n, p in model.named_parameters():
    (no_decay_params if any(k in n for k in ("bias", "norm.weight", "norm.bias"))
     else decay_params).append(p)
optim = AdamW([
    {"params": decay_params,    "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

This pattern costs 5 lines of code and changes end-of-pretraining perplexity by 0.1–0.5%. It's free.

---

## 4. Gradient clipping — the safety net

Adam's adaptive rescaling does *not* protect against exploding gradients. The intuition: `v_hat` is a *running* average, so a single step with a 100× gradient blows through the current scale before `v_t` catches up. The fix has existed since Pascanu 2013 ([[gradient-clipping]]) and is mandatory for LLM training.

**Global-norm clipping** — the only correct variant:

```
g_norm = sqrt(sum over all params of ||g_i||²)
if g_norm > c:
    g_i ← g_i · (c / g_norm)        # same rescale factor applied to every tensor
```

Direction is preserved; only magnitude is bounded. The two alternatives you'll see floating around are both worse:

- **Clip-by-value** (`g_i ← clip(g_i, -c, c)` elementwise) destroys the descent direction. Don't use it for language models.
- **Per-tensor norm clip** (loop `clip_grad_norm_(p, c)` over each parameter) biases the optimizer toward small tensors. Don't use this either.

**Modern defaults**: `max_grad_norm = 1.0` for pretraining and SFT; `0.5 – 1.0` for RL rollouts where reward spikes cause advantage outliers.

**The ordering that breaks silently**. Under mixed-precision and FSDP, the pipeline has to be exactly:

```
1. loss.backward()                    # accumulates scaled gradients
2. scaler.unscale_(optimizer)         # divide gradients by loss-scale S
3. clip_grad_norm_(params, 1.0)       # NOW the threshold is meaningful
4. scaler.step(optimizer)             # step + update scaler state
5. scaler.update()
```

Clip before unscale and the threshold is off by a factor of S (typically 2^15 or larger); the clip does nothing. Clip after step and, well, you didn't clip.

**FSDP/ZeRO-3 specifically**. The global norm must be computed *across all shards* before scaling. A naive `clip_grad_norm_(local_shards, 1.0)` under-counts the norm and produces inconsistent rescaling across ranks → silent divergence. Use `FullyShardedDataParallel.clip_grad_norm_` in PyTorch, or DeepSpeed's `engine.clip_grad_norm_`. Do not write your own.

**Instrumentation tip**. Track `pre_clip_grad_norm` as a training metric. A 100× spike in `pre_clip_grad_norm` usually precedes a loss spike by 1–5 steps. This is the single best early-warning signal for pretraining runs.

---

## 5. The "no momentum for transformers" myth

Every few years someone claims SGD+momentum with the right schedule can match AdamW on language modelling. The claim is technically true for small models and specific schedules. It's false at production scale.

Here's why. Transformer training updates three qualitatively different parameter groups:

1. **Embeddings** — highly sparse gradients (only rows for seen tokens get updated). Scalar LR wastes signal on rare tokens unless you compensate with adaptive scaling.
2. **Attention QKV projections** — heterogeneous curvature across heads; some heads specialize early and need fine updates, others are still exploring.
3. **FFN projections** — dense gradients, closer to classical MLP training.

AdamW handles all three with one knob. SGD+momentum requires per-layer-group LR tuning, gradient clipping per group, and carefully tuned warmup to avoid embedding blow-up. At 70B+ scale, nobody does this. AdamW is the default because it's *robust*, not because it's fastest on any single setting.

The modern subtlety: Lion (Chen 2023), Sophia (Liu 2023), Shampoo, and Muon are all attempts to beat AdamW on pretraining. As of late 2025 frontier reports (Llama 3, DeepSeek V3, Qwen 3, OLMo 3) still use AdamW. Muon has traction in some MoE contexts (Kimi K2 uses MuonClip), but for dense transformer pretraining AdamW remains the default.

---

## 6. The 2025 optimizer landscape — what actually ships

Quick field guide to the alternatives, ordered by frontier adoption:

- **AdamW** — default everywhere. Every open 2025 frontier recipe (Llama 3/4, Qwen 2.5/3/3.5, DeepSeek V3/R1, OLMo 2/3, Phi-3/4, Tülu 3, Nemotron-Ultra).
- **MuonClip / Muon** — Kimi K2 (Moonshot) uses MuonClip for the MoE pretraining stage; experimental but production-validated at 1T-parameter scale.
- **Lion** (Chen 2023) — sign-of-momentum optimizer. Faster per-step than AdamW, but variance of the final model quality is higher. Some community adoption for SFT; not frontier.
- **Sophia** (Liu 2023) — second-order-via-Gauss-Newton approximation. Impressive scaling-law paper; limited production evidence.
- **Shampoo** — distributed preconditioning. Works well at Google-scale; the sharding complexity has kept it out of most open stacks.
- **Online-Merging Optimizer** (Qwen 2.5) — variant that averages checkpoints during DPO. Task-specific, not a general-purpose AdamW replacement.

Rule of thumb for 2026: if your report does not say which optimizer it uses, it used AdamW.

---

## 7. Practitioner's cheat-sheet

```python
# The 90% case: pretraining, SFT, and RL, with the right defaults baked in.

import torch
from torch.optim import AdamW

def build_optimizer(model, lr, stage="pretrain"):
    betas = {"pretrain": (0.9, 0.95), "sft": (0.9, 0.95), "rl": (0.9, 0.95)}[stage]
    wd    = {"pretrain": 0.1,         "sft": 0.01,        "rl": 0.0}[stage]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if any(k in n for k in ("bias", "norm", "embed_tokens")) else decay).append(p)

    return AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8,
    )

# Training step — the canonical pipeline.
for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
    loss.backward()
    # bf16 does NOT need a scaler; fp16 does. See ch-02.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## Connections and what's next

- **[[mixed-precision]] / ch-02** — fp16 loss-scaling + `GradScaler.unscale_(optimizer)` before clip. bf16 removes most of this pain and is why 2025 runs use it by default.
- **[[lr-schedules]] / ch-03** — warmup is partly required *because* AdamW's bias-correction makes the first ~100 steps effectively run at a higher LR; warmup lets `v_hat` stabilize.
- **[[weight-init]] / ch-03** — μP width-transfer rules depend on AdamW's per-parameter update scale being `O(α)` regardless of width; do not mix μP with SGD.
- **[[ppo]] / ch-36** — PPO fine-tuning uses the same AdamW at LR ≈ 1e-6 because policy gradients are noisy. Reward spikes are the RL equivalent of embedding-blow-up; clip them for the same reason.
- **ch-05 / ch-07** — distributed training and failure-mode diagnosis revisit the FSDP clip-norm correctness issue and the silent-drift bugs from mis-ordered clip/step calls.

## Further reading

- [[adam]] — full extract of Kingma-Ba 2014 + Loshchilov-Hutter 2017.
- [[gradient-clipping]] — Pascanu 2013; canonical treatment.
- [[karpathy-training-neural-net-recipe]] — "monitor and clip the gradient norm" as a non-negotiable rule.
- [[mixed-precision]] — loss-scaling interaction that this chapter deliberately defers to ch-02.

## Companion visualization

**[figures/beta2-memory.html](figures/beta2-memory.html)** — interactive slider showing how the second-moment EMA `v_hat` responds to a single gradient spike under `β₂ ∈ {0.9, 0.95, 0.999}`. Use it to build intuition for why Llama picked 0.95.
