---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: adam (Kingma & Ba 2014 + Loshchilov & Hutter 2017) — schedule-interaction framing
source_url: https://arxiv.org/abs/1412.6980 ; https://arxiv.org/abs/1711.05101
created_at: "2026-04-23"
---

# Excerpt: Adam × Schedules — Why Warmup Exists Because of Bias Correction

**Sources:**
- Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015 — arxiv 1412.6980 — Algorithm 1
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019 — arxiv 1711.05101

---

## Framing for this chapter

Ch-01 introduced AdamW as *the* LLM optimizer and gave its full algorithm box. This excerpt does not repeat that mechanics tour. Instead, it zooms in on the *interaction* between AdamW and the LR schedule — specifically, on why AdamW's bias correction *forces* warmup into every pretraining recipe in a way that plain SGD + momentum does not.

This is the foundational answer to a question [[excerpts/lr-schedules]] raises and partially answers: "Why is warmup non-negotiable with AdamW?" The full answer requires walking through `m̂` and `v̂` step by step for the first ~50 iterations and seeing exactly what happens. That is what this excerpt does.

---

## 1. The bias correction in slow motion

From [[adam]] Algorithm 1, the AdamW update at timestep `t`:

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{aligned}
```

Initial state: `m_0 = 0`, `v_0 = 0`. Modern LLM settings: `β_1 = 0.9`, `β_2 = 0.95`.

Walk through step 1:
- `m_1 = 0.9 · 0 + 0.1 · g_1 = 0.1 g_1`
- `v_1 = 0.95 · 0 + 0.05 · g_1² = 0.05 g_1²`
- `m̂_1 = 0.1 g_1 / (1 - 0.9) = g_1`  ← bias correction exactly unbiases it
- `v̂_1 = 0.05 g_1² / (1 - 0.95) = g_1²`  ← same
- update: `θ_1 = θ_0 - α · (g_1 / √(g_1²) + ε) = θ_0 - α · sign(g_1)`

**This is the problem.** At step 1, `v̂_1 = g_1²` is an estimate of `E[g²]` based on exactly **one sample**. If `g_1` happens to be small (say, `0.01`) for some parameter, then `√v̂_1 = 0.01`, and the update magnitude is `α / 0.01 = 100 α`. For that parameter, the effective LR is 100× the nominal LR.

Step 2:
- `m_2 = 0.9 · 0.1 g_1 + 0.1 g_2 = 0.09 g_1 + 0.1 g_2`
- `v_2 = 0.95 · 0.05 g_1² + 0.05 g_2² = 0.0475 g_1² + 0.05 g_2²`
- `m̂_2 = m_2 / (1 - 0.9²) = m_2 / 0.19`
- `v̂_2 = v_2 / (1 - 0.95²) = v_2 / 0.0975`

Computing `v̂_2` numerically:
```
v̂_2 = (0.0475 g_1² + 0.05 g_2²) / 0.0975 ≈ 0.487 g_1² + 0.513 g_2²
```

Now `v̂_2` is an average of the two most recent squared gradients. Still only two samples. If both happen to be small, `√v̂_2` is small and the update overshoots.

### How long until `v̂` is reliable?

`v̂_t` is a weighted average of the last `~1/(1-β_2)` squared gradients. For `β_2 = 0.95`, that's `1/0.05 = 20` effective samples. Before step 20, the estimate has visible noise. Before step ~5, it is dominated by whichever `g²` happened first.

From [[adam]]:

> "At LR-warmup completion, `v_t` should reflect *recent* gradient variance to track non-stationary loss landscapes. `0.999` has effective memory of ~1000 steps; `0.95` has ~20 steps and reacts faster to phase changes."

And directly on the schedule interaction:

> "AdamW's bias correction makes the first ~100 steps effectively use a higher LR — this is *partly* why warmup is required; it lets `v̂` stabilize before full LR."

So the math is clear: for the first ~20 steps, `v̂` is a wild estimate. If you set `α = peak_lr` from step 1, some fraction of your parameters get 10–100× over-updates. With `peak_lr = 3e-4`, a 100× inflation means effective LR `3e-2` — which is enough to NaN a GPT-3-scale model in the first forward pass after the update.

---

## 2. Why warmup specifically solves this

Linear warmup (from [[excerpts/lr-schedules]]):

```math
\mathrm{lr}(t) = \mathrm{peak\_lr} \cdot \frac{t}{\mathrm{warmup\_steps}}
```

Combined with the AdamW update:

```math
\Delta\theta_t = -\frac{\mathrm{peak\_lr} \cdot t}{\mathrm{warmup\_steps}} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

At `t = 1`, `lr = peak_lr / warmup_steps`. For `warmup_steps = 2000` and `peak_lr = 3e-4`, `lr(1) = 1.5e-7`. Even with the 100× `v̂`-underestimate inflation, the effective step is `1.5e-5` — survivable.

By `t = 2000`, `lr = peak_lr` and `v̂` has effectively averaged 2000 samples — stable. Full LR is safe.

What does this mean for LLM training in 2025? **Warmup is not "polite optimization behaviour." It is a literal mathematical necessity given AdamW's initialisation of `v_0 = 0`.** If you could somehow initialise `v_0` to a reasonable prior (e.g. the expected squared-gradient magnitude from the previous run), you could skip warmup. This is exactly why RL fine-tuning ([[excerpts/lr-schedules]] §6) uses 0–50 steps of warmup — the base model provides a warm `v_0`.

---

## 3. The `β_2 = 0.95` choice and schedule shape

Ch-01 covered why `β_2 = 0.95` replaces the original `0.999`. For this chapter, the schedule implication is:

- `β_2 = 0.999` → effective memory ~1000 steps → warmup should be ≥ 1000 steps for `v̂` to stabilise.
- `β_2 = 0.95` → effective memory ~20 steps → warmup could in principle be ~20 steps.

Why do practitioners still use 2000–8000 warmup steps with `β_2 = 0.95`? Two reasons:
1. **Loss-landscape curvature** also changes in the first few hundred steps. Even with a stable `v̂`, the directions of `m̂` are still converging.
2. **Init interacts with early `v̂`**. If init is slightly off ([[excerpts/weight-init]]), the first hundred gradients are on a pre-equilibrium trajectory. Warmup lets that settle.

So the warmup-length choice is not purely from bias-correction math. It is a conservative blend of bias correction + init dynamics + caution.

---

## 4. The `ε` parameter and how it interacts with schedules

From [[adam]]:

> "Setting `eps` too small under fp16 → division-by-zero NaNs. Bump to `1e-5` if you see NaN in optimizer step."

Standard: `ε = 1e-8` in fp32. This is added inside `√(v̂) + ε` to prevent division by zero for parameters with zero gradient history.

Schedule interaction: during warmup, `v̂` is small for parameters that haven't been updated much. If `√v̂ ≪ ε`, the update becomes `α · m̂ / ε` — a constant magnitude regardless of gradient. This is rare (most parameters have some gradient) but can happen for dead embedding rows (tokens that haven't appeared yet) during early training. Not a problem in practice; just noting the edge case.

Under bf16/fp8, the issue flips: `ε = 1e-8` may round to zero in bf16 (bf16's smallest normal is ~`1.2e-38` but precision loss in addition can drop `1e-8` below the sum). This is why [[excerpts/mixed-precision]] recommends `ε = 1e-5` for bf16 Adam state — though most frameworks keep optimizer state in fp32 so the concern is moot.

---

## 5. Weight decay × schedule — the decoupling question

AdamW's innovation (Loshchilov & Hutter 2017):

```math
\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
```

Note the `α` *outside* the entire bracket. So weight decay is scaled by the current LR — meaning weight decay also *decays* on the cosine/WSD tail. This is intentional and widely used. But there's a wrinkle:

From [[adam]]:

> "AdamW's weight-decay coefficient `lambda` is **independent of learning rate**, which restores the principle that LR and regularization should be tunable independently."

That is true of `λ` as a *hyperparameter* — you tune it without re-tuning LR. But the *effective* decay per step is `α · λ`. As the schedule reduces `α`, effective decay also shrinks. For a cosine schedule ending at `min_lr = 0.1 × peak`, effective decay at end-of-training is 10% of peak decay.

Some recipes (a minority) apply weight decay with a separate constant coefficient, un-scaled by LR. This is sometimes called "fully-decoupled AdamW" or "AdamW-no-schedule-decay." It subtly changes the end-of-training behaviour — the model continues to shrink toward zero even as LR approaches zero. Not standard; mostly a research curiosity.

What does this mean for LLM training in 2025? Stick with the standard AdamW (α outside the bracket). Use `λ = 0.1` for pretraining, `λ ∈ [0, 0.01]` for SFT, `λ = 0` for RL. Don't overthink it.

---

## 6. The no-decay group — intersection with norms

From [[adam]]:

> "Not excluding LayerNorm/embedding bias parameters from weight decay (the 'no-decay group'). Standard practice; ~0.1–0.5% perplexity difference."

LayerNorm's `γ` and `β` parameters, embedding weights, and linear-layer biases are **excluded from weight decay**. The rationale: these parameters are not learning "features" — they are scale/shift parameters. Decaying them toward zero actively hurts:
- `γ` toward zero collapses the normalised activations.
- Embeddings toward zero destroy token representations.
- Biases toward zero shifts the model's prediction in an arbitrary direction.

The intersection with norms ([[excerpts/batch-vs-layer-norm]]): RMSNorm has no `β`, so the no-decay group for RMSNorm is just `γ` plus embeddings plus biases. LayerNorm includes `β` too.

Implementation in PyTorch:

```python
decay_params = [p for n, p in model.named_parameters()
                if p.dim() >= 2 and 'norm' not in n.lower() and 'embed' not in n.lower()]
no_decay_params = [p for n, p in model.named_parameters()
                   if p.dim() < 2 or 'norm' in n.lower() or 'embed' in n.lower()]
optim = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=peak_lr, betas=(0.9, 0.95))
```

---

## 7. Memory and state — optimizer cost vs schedule flexibility

From [[adam]]:

> "Adam stores `m_t` and `v_t` per parameter — **2x model parameters in optimizer state**. With fp32 state and bf16 weights, optimizer state is **4x** the model in bytes."

For a 70B model: 70B params × 4 bytes (fp32 m) + 70B × 4 bytes (fp32 v) = 560 GB of optimizer state. Vs 70B × 2 bytes (bf16 weights) = 140 GB. The optimizer state is 4× the model.

Schedule implication: if you want to **resume** training mid-schedule (e.g. WSD fork), you must checkpoint the optimizer state too. 560 GB per checkpoint. For WSD's "fork decay from any stable-phase checkpoint" story ([[excerpts/lr-schedules]]), you need disk budget for multiple 560-GB saves. This is a real operational constraint.

What does this mean for LLM training in 2025? Budget ~5× your model size in bytes for each full checkpoint. ZeRO-1/2/3 and FSDP shard the optimizer state across ranks — covered in [[ch-05]].

---

## 8. What warmup does *not* fix

Warmup prevents the `v̂`-underestimate issue at step 0. It does **not** fix:

- **Bad init**: if activation variance explodes at init, warmup slows the divergence but doesn't prevent it. See [[excerpts/weight-init]] — init audit must pass *before* training starts.
- **Post-norm depth**: post-norm's gradient-norm-grows-linearly-with-depth ([[excerpts/batch-vs-layer-norm]]) is orthogonal to warmup. Warmup helps, but pre-norm solves it.
- **Mixed-precision underflow**: if bf16 gradients underflow to zero, warmup doesn't help — the signal is literally gone. See [[excerpts/mixed-precision]].

Warmup is a targeted fix for one specific failure mode. It is load-bearing, but it is not a general-purpose safety net.

---

## Connections

- [[excerpts/lr-schedules]] — the companion excerpt; explains warmup shapes, cosine, WSD.
- [[excerpts/weight-init]] — init audit; bad init bypasses warmup's protection.
- [[excerpts/batch-vs-layer-norm]] — the no-decay group intersects with norm parameters; pre-norm vs post-norm is orthogonal to optimizer choice.
- [[excerpts/mixed-precision]] — optimizer state stays fp32 regardless of compute precision.
- [[ch-01]] — full Adam/AdamW mechanics (do not re-read; this excerpt assumes it).
- [[ch-03]] — synthesis with reference schedule code.
