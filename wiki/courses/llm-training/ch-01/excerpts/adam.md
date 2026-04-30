---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "Adam: A Method for Stochastic Optimization (Kingma & Ba 2014) + Decoupled Weight Decay Regularization (Loshchilov & Hutter 2017)"
source_url: https://arxiv.org/abs/1412.6980
created_at: "2026-04-23"
---

# Excerpt: Adam and AdamW — the LLM default optimizer

**Authors:** Diederik P. Kingma, Jimmy Ba (Adam, 2014); Ilya Loshchilov, Frank Hutter (AdamW, 2017)
**Year:** 2014 / 2017
**Venue:** ICLR 2015 (Adam); ICLR 2019 (AdamW, originally arXiv 2017)
**URLs:** https://arxiv.org/abs/1412.6980 (Adam), https://arxiv.org/abs/1711.05101 (AdamW)
**arXiv IDs:** 1412.6980, 1711.05101
**Raw-data source:** [[raw-data/adam]]

---

## The one-box algorithm (Algorithm 1, Kingma & Ba 2014)

```math
\begin{aligned}
m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t \\
v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^{\,2} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
```

This five-line block is what you will see, almost verbatim, in every modern framework's `AdamW.step()` — PyTorch, JAX/Optax, DeepSpeed, Megatron, TRL. Memorising it is non-optional for an LLM practitioner.

**What each line buys you.** Line 1 is momentum: an exponential moving average (EMA) of the gradient with half-life `ln(2)/ln(1/β₁) ≈ 6.6` steps for β₁=0.9. Line 2 is a per-parameter *variance* EMA, giving the optimizer an estimate of how noisy each coordinate's gradient is. Lines 3–4 are the bias correction (see derivation below). Line 5 is the update — note it is `m̂ / √v̂`, not `g / √v̂`. Dividing the first moment by the standard deviation of the second moment is what gives Adam its "almost learning-rate-free" reputation: the update magnitude is roughly `α` regardless of gradient scale, because `m̂ / √v̂` has order-1 magnitude whenever the gradient distribution is stationary.

Notice: `ε` sits *inside* the denominator to prevent `0/0`, not outside. A common bug — writing `m̂ / (√v̂ + ε)` as `m̂ / √(v̂ + ε)` — changes the update by orders of magnitude when `v̂` is small.

---

## Derivation: why `1 - β^t` is the right bias correction

The EMAs are initialised at `m_0 = v_0 = 0`. Ignoring higher-order terms, after `t` steps of a stationary gradient with true mean `E[g]`:

```math
\mathbb{E}[m_t] = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{\,t-i} \, \mathbb{E}[g_i] = (1 - \beta_1^{\,t}) \, \mathbb{E}[g]
```

So `m_t` is a biased estimator — it underestimates the true mean by the factor `(1 - β₁ᵗ)`. The first step has `m_1 = (1-β₁) g_1 = 0.1 g_1`, which is ten times smaller than the gradient! Without bias correction, the effective learning rate in step 1 would be 10× smaller than intended.

Dividing by `(1 - β₁ᵗ)` exactly undoes this. The same argument gives `v̂ = v / (1 - β₂ᵗ)`. As `t → ∞`, both correction factors → 1 and you can drop them. Most frameworks keep them for simplicity.

**Practical corollary — why warmup exists.** In the very first steps, `v̂` is extremely noisy (it is the debiased average of at most a handful of `g²` samples). When that noise is small in some coordinate, `m̂ / √v̂` produces a *huge* step on that coordinate. This is the real reason LLMs with AdamW need warmup: see [[excerpts/lr-schedules]] for the companion analysis of why 0-warmup causes first-100-step divergence at 7B+ scale.

---

## AdamW: the single-line fix that changed LLM training

Loshchilov & Hutter (2017, Algorithm 2) pointed out that the "weight decay" in every existing Adam implementation was actually L2 regularization added to the *loss*, i.e.

```math
g_t \leftarrow \nabla_\theta \mathcal{L}(\theta_{t-1}) + \lambda\, \theta_{t-1}
```

This looks innocent but is catastrophic under adaptive optimization: the `λθ` term flows through `m_t` and `v_t`, and coordinates with large `v̂` (the ones whose gradients are historically noisy) get their weight decay *divided by √v̂* — effectively *undoing* decay on exactly the parameters that most benefit from it.

AdamW's fix is to apply weight decay **directly on the parameter update**:

```math
\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \, \theta_{t-1} \right)
```

One consequence: in AdamW the optimal `λ` is **decoupled** from the optimal `α`. In the L2-Adam ancestor, the optimal `(α, λ)` sit on a diagonal ridge (Figure 1 of the AdamW paper), and changing one forces retuning the other. That single graph is the entire empirical contribution of the AdamW paper — and it is why every frontier-lab pretraining config has independent `learning_rate` and `weight_decay` fields.

Notice: calling `torch.optim.Adam(params, weight_decay=0.1)` does **not** give you AdamW — it gives you L2-regularized Adam, the thing the paper says to avoid. You must use `torch.optim.AdamW` explicitly.

---

## Why `β₂ = 0.95` (not `0.999`) for LLMs

The Kingma & Ba defaults are `β₁=0.9, β₂=0.999`. LLM pretraining almost universally uses `β₂=0.95` since Llama 1. The reason is effective memory.

For an EMA with decay `β`, the effective memory (time over which a sample influences the estimate) is `1 / (1 - β)` steps. Thus:

| `β₂` | Effective memory | Rough half-life |
|------|-----------------:|----------------:|
| 0.999 | 1000 steps | 693 steps |
| 0.99  | 100 steps  | 69 steps |
| 0.95  | 20 steps   | 14 steps |

LLM loss landscapes are profoundly non-stationary: data curriculum shifts, domain changes, and learning-rate decay all move the gradient distribution over hundreds of steps. `β₂ = 0.999` builds up `v̂` over ~1000 steps of stale variance estimates, *over-smoothing* the second moment and making the optimizer sluggish to adapt to new regimes. `β₂ = 0.95` cuts effective memory by ~50×, making `v̂` track *recent* variance and giving the optimizer better per-parameter scaling under curriculum changes.

The raw-data file states it plainly: "`0.999` has effective memory of ~1000 steps; `0.95` has ~20 steps and reacts faster to phase changes" — for LLMs this is the difference between a flat loss curve and a spiking one during domain mixing.

---

## Modern hyperparameters in practice

| Hyperparameter | Pretrain | SFT | RL (PPO / GRPO) |
|---|---|---|---|
| `β₁` | 0.9 | 0.9 | 0.9 |
| `β₂` | **0.95** | 0.95–0.999 | 0.95 |
| `ε` | 1e-8 | 1e-8 | 1e-8 (bump to 1e-5 for fp16) |
| `weight_decay` | **0.1** | 0.0–0.01 | 0.0 |
| Peak `α` | 1e-4 – 6e-4 (size-dependent) | 1e-5 – 5e-5 | 1e-6 – 1e-5 |

**Memory footprint of optimizer state** — this is the number every distributed-training engineer must internalise:

```math
\text{bytes}_{\text{optim}} = 2 \cdot P \cdot 4 = 8P \text{ bytes (fp32)}
```

for `P` parameters. With bf16 weights (2 bytes) and bf16 gradients (2 bytes), optimizer state is **4× the model's weight footprint** — a 70B model has `70 \cdot 10^9 \cdot 8 = 560 \, \mathrm{GB}` of optimizer state alone. This is why ZeRO-1 (sharding only the optimizer state) gives most of the wins of full ZeRO-3 and why 8-bit Adam ([Dettmers 2022](https://arxiv.org/abs/2110.02861)) was a major breakthrough for consumer fine-tuning.

---

## Common pitfalls

- **Using `weight_decay` on `torch.optim.Adam`**: you get L2-coupled Adam, not AdamW. Switch to `AdamW` explicitly. The bug is silent — training "works" but generalisation degrades 1–3% perplexity.
- **`ε` too small under fp16**: `√v̂ + ε` divides by ~0 when `v̂` underflows, producing NaN. Bump `ε` to `1e-5` or, better, switch to bf16 (see [[excerpts/mixed-precision]]).
- **Weight decay on norm / embedding parameters**: standard practice is the two-group optimizer pattern — LayerNorm weights, biases, and the embedding get `weight_decay=0`; everything else gets `weight_decay=0.1`. Skipping this is worth ~0.1–0.5% perplexity.
- **Forgetting bias correction under resume**: when resuming from a checkpoint at step `t`, `(1 - β₁ᵗ)` is essentially 1. If you accidentally reset the step counter you re-introduce bias. Always restore the optimizer state dict, not just `m`/`v`.

---

## Why AdamW beats alternatives in 2025

As of 2026, AdamW remains the default at frontier scale despite several challengers:

- **Lion** (Chen et al. 2023, arXiv 2302.06675): sign-based update, half the optimizer state. Competitive at small-to-mid scale; loses ~0.3 perplexity on 70B+ in most reported comparisons.
- **Sophia** (Liu et al. 2023, arXiv 2305.14342): Hessian-diagonal preconditioning. Faster convergence per step but 1.4× compute per step; wall-clock wins are mixed at scale.
- **Shampoo / SOAP** (Shi et al. 2023): full-matrix second-moment; strong on short runs but expensive.

All of these exist because AdamW's 4× optimizer-state overhead is painful. None have displaced it for 100B+ pretraining.

---

## Connections

- [[excerpts/gradient-clipping]] — AdamW's adaptive scaling does **not** prevent early-training exploding gradients; clipping is still mandatory. Ordering: `unscale → clip → AdamW.step()`.
- [[excerpts/mixed-precision]] — optimizer state must remain fp32. `v̂` underflows in fp16/bf16 for any parameter with small gradients, breaking the update.
- [[excerpts/lr-schedules]] — bias correction and warmup are two sides of the same coin; the companion excerpt explains why `lr=0 → peak` ramp is needed *because* of `(1-β₁ᵗ)`'s behaviour at small `t`.
- [[excerpts/weight-init]] — Adam's per-parameter scaling means the optimizer tolerates bad init *during training*, but the first forward pass still needs variance-preserving init; the two stabilisation mechanisms are complementary.
- [[ch-01]] — parent synthesis for the fundamentals of LLM training.
