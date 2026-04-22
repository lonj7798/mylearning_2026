<!-- scope: Adam optimizer + AdamW decoupled weight decay; default LLM optimizer
     deps: [[gradient-clipping]]
     see-also: [[lr-schedules]], [[mixed-precision]], [[weight-init]]
-->

# Adam: A Method for Stochastic Optimization (and AdamW)
- **Core Insight:** Per-parameter adaptive step sizes computed from running estimates of the first and second gradient moments give SGD-like convergence with almost no learning-rate tuning.
- **Guideline:** Use AdamW (decoupled weight decay) with `(beta1, beta2) = (0.9, 0.95)` and `weight_decay = 0.1` for LLM pretraining; never use plain Adam with L2 regularization for transformer training.
- **Authors:** Diederik P. Kingma, Jimmy Ba (Adam, 2014); Ilya Loshchilov, Frank Hutter (AdamW, 2017)
- **Year:** 2014 / 2017
- **URL:** https://arxiv.org/abs/1412.6980 ; https://arxiv.org/abs/1711.05101
- **Relevant topics:** optimization, weight decay, regularization, large-batch training

## Abstract
Adam combines the per-parameter scaling of RMSProp with the momentum of classical SGD. It maintains exponential moving averages of the gradient (first moment, `m_t`) and the squared gradient (second moment, `v_t`), bias-corrects them so early-iteration estimates are unbiased, and uses their ratio as the update direction. AdamW (Loshchilov & Hutter) observes that adding L2 regularization to Adam's *loss* couples weight decay strength to the per-parameter adaptive scaling — undoing weight decay on parameters with large `v_t`. The fix: apply weight decay directly to the parameter update, decoupled from the gradient. This single change makes Adam competitive with SGD-momentum on image classification and is the de-facto LLM optimizer.

## Key Contributions
- **Adam (2014)**: bias-corrected first/second moment estimates; nearly invariant to gradient rescaling; default hyperparameters that work across many tasks with no tuning.
- Theoretical convergence proof (later shown to have a flaw — fixed by AMSGrad — but Adam still dominates in practice).
- **AdamW (2017)**: decoupled weight decay; demonstrated that Adam's "poor generalization" was an artifact of L2-as-regularizer coupling.
- AdamW's weight-decay coefficient `lambda` is **independent of learning rate**, which restores the principle that LR and regularization should be tunable independently.
- Made it routine to set `beta_2 = 0.95` (instead of 0.999) for LLMs, where shorter second-moment memory tracks loss-landscape curvature more responsively.

## Key Figures/Tables to Study
- **Adam Algorithm 1**: the single algorithm box — memorize it; you'll see it in every framework's optimizer source.
- **AdamW Algorithm 2 vs L2-Adam**: side-by-side showing where the weight-decay term enters; this is the entire contribution.
- **AdamW Figure 1**: loss surfaces showing optimal `(lr, wd)` is decoupled in AdamW but coupled (diagonal valley) in L2-Adam — explains why AdamW is hyperparameter-stable.

## Technical Details
**Adam update** (timestep `t`, gradient `g_t`):
```
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t                  # 1st moment (momentum)
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2                # 2nd moment (variance)
m_hat = m_t / (1 - beta_1^t)                                 # bias correction
v_hat = v_t / (1 - beta_2^t)
theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
```

**AdamW update** (only the last line changes):
```
theta_t = theta_{t-1} - alpha * (m_hat / (sqrt(v_hat) + eps) + lambda * theta_{t-1})
```
i.e. weight decay `lambda * theta` is added directly to the update, **not** to the gradient `g_t` (which would feed into `m_t` and `v_t` and get adaptively scaled).

**Defaults from the original paper**: `alpha=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8`.

**Modern LLM defaults** (GPT-3 / Llama / Chinchilla / Qwen lineage):
| Hyperparameter | Pretrain | SFT | RL (PPO/GRPO) |
|---|---|---|---|
| `beta_1` | 0.9 | 0.9 | 0.9 |
| `beta_2` | **0.95** | 0.95–0.999 | 0.95 |
| `eps` | 1e-8 | 1e-8 | 1e-8 (sometimes 1e-5 for fp16) |
| `weight_decay` | **0.1** | 0.0–0.01 | 0.0 |
| Peak `lr` (AdamW) | 1e-4 to 6e-4 (size-dependent) | 1e-5 to 5e-5 | 1e-6 to 1e-5 |

**Why `beta_2 = 0.95` for LLMs**: at LR-warmup completion, `v_t` should reflect *recent* gradient variance to track non-stationary loss landscapes. `0.999` has effective memory of ~1000 steps; `0.95` has ~20 steps and reacts faster to phase changes (curriculum, data domain shift). This was Llama 1's choice and is now standard.

**Memory cost**: Adam stores `m_t` and `v_t` per parameter — **2x model parameters in optimizer state**. With fp32 state and bf16 weights, optimizer state is **4x** the model in bytes. This is why ZeRO-1/2/3 and 8-bit Adam exist.

**Common pitfalls**:
- Using `weight_decay` in `torch.optim.Adam` — that's L2-regularization-Adam, not AdamW. Use `torch.optim.AdamW` explicitly.
- Setting `eps` too small under fp16 → division-by-zero NaNs. Bump to `1e-5` if you see NaN in optimizer step.
- Not excluding LayerNorm/embedding bias parameters from weight decay (the "no-decay group"). Standard practice; ~0.1–0.5% perplexity difference.

## Connections
- **[[lr-schedules]]**: AdamW's bias correction makes the first ~100 steps effectively use a higher LR — this is *partly* why warmup is required; it lets `v_hat` stabilize before full LR.
- **[[mixed-precision]]**: optimizer state must stay in fp32 (master weights). Adam's `v_hat` underflows to zero in fp16 for any parameter with small gradients.
- **muP / [[weight-init]]**: AdamW's update scale is `O(alpha)` per parameter regardless of width, which is why μP requires AdamW LR to **not** scale with width (unlike SGD).
- **PPO / GRPO**: the same AdamW with much lower LR (1e-6) — RL fine-tuning is intentionally conservative because policy gradients are noisy.
- **Lion / Sophia / Shampoo**: 2023+ alternatives that try to beat AdamW on LLM pretraining; as of 2025 AdamW remains the default at frontier scale.
