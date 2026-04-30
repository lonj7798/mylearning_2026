<!-- scope: learning rate schedules — warmup, cosine, inverse-sqrt, WSD
     deps: [[adam]]
     see-also: [[weight-init]], [[gradient-clipping]]
-->

# Learning Rate Schedules: Warmup, Cosine, Inverse-Sqrt, WSD
- **Core Insight:** The optimal learning rate is non-stationary across training — it must be small at the start (so adaptive optimizer state can stabilize) and small at the end (so the model can settle into a sharp minimum), with a high-LR plateau in the middle.
- **Guideline:** Use **cosine** with linear warmup for fixed-budget pretraining; use **WSD (warmup-stable-decay)** when you want to extend or branch a run; never skip warmup with AdamW.
- **Authors:** Various — Vaswani 2017 (inverse-sqrt + warmup), Loshchilov 2017 (SGDR/cosine), Hu 2024 / DeepSeek (WSD)
- **Year:** 2017–2024
- **URL:** https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/1608.03983 ; https://arxiv.org/abs/2404.06395
- **Relevant topics:** optimization, training dynamics, scaling, continued pretraining

## Abstract (composite)
Learning rate scheduling is the practice of varying the optimizer step size over training. Four families dominate modern LLM training. **Warmup** linearly raises LR from 0 to peak over the first 0.1–2% of steps to avoid early divergence with adaptive optimizers. **Cosine annealing** (Loshchilov 2017) smoothly decays from peak to a small final LR (typically 10% of peak). **Inverse-square-root** (Vaswani 2017) decays as `1/sqrt(step)` after warmup; popular in the original Transformer and still used in some encoder pretrains. **WSD** (warmup-stable-decay; popularized by MiniCPM and DeepSeek) holds a constant LR for the bulk of training, then sharply decays at the end — enabling continued pretraining without re-tuning.

## Key Contributions
- **Warmup** (Goyal 2017 / Vaswani 2017): the linear ramp-up that prevents adaptive optimizers from taking destructive steps before `v_t` has accumulated.
- **Cosine annealing** (Loshchilov 2017, "SGDR"): smoother than step-decay, no hyperparameters beyond min/max, and demonstrably superior on image classification — adopted across LLM pretraining (GPT-3, Llama, Chinchilla).
- **Inverse-sqrt** (Vaswani 2017): the "Transformer schedule" — `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`. Tied to the model dimension; a self-tuning property.
- **WSD** (Hu 2024 / Hägele 2024): constant high-LR phase + short final decay matches cosine *at any chosen stop point*, without committing to a token budget upfront.
- Empirical finding (Chinchilla, Hoffmann 2022): cosine length should match training length — over- or under-shooting cosine costs ~0.5–1% on validation perplexity.

## Key Figures/Tables to Study
- **SGDR Figure 1** (Loshchilov): cosine with warm restarts; original motivation.
- **Chinchilla Figure A1**: cosine schedule mismatch costs are real but small.
- **MiniCPM/DeepSeek WSD figure**: the loss curve "step-down" at the start of decay phase; signature of WSD.
- **Vaswani Section 5.3**: the inverse-sqrt formula and its derivation from `d_model`.

## Technical Details

**Linear warmup** (always prepended):
```
lr(t) = peak_lr * t / warmup_steps          for t < warmup_steps
```
Typical `warmup_steps`: 2000 (small models), 2000–8000 (7B–70B), longer for very large LR.

**Cosine annealing** (most common LLM default):
```
lr(t) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * (t - warmup) / (T - warmup)))
```
`min_lr` typically `0.1 * peak_lr` (Llama, GPT-3) or `0.0` (some pretrains). `T` = total training steps.

**Inverse-square-root** (Transformer original):
```
lr(t) = d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))
```
Self-scales with model width; nice in theory but loses to cosine in practice for fixed budget.

**WSD — Warmup-Stable-Decay**:
```
phase 1 (warmup):     lr ramps 0 → peak_lr      [0, warmup]
phase 2 (stable):     lr = peak_lr               [warmup, T - decay]
phase 3 (decay):      lr → min_lr                [T - decay, T]
                      shape: linear, cosine, or 1-sqrt
```
Decay phase is typically **10–20% of total tokens**. Key advantage: the stable phase is *checkpoint-able* — you can fork off a 10%-decay run from any stable-phase checkpoint. This is how DeepSeek and MiniCPM produce many model variants from one pretraining trunk.

**Modern defaults**:
| Setting | Pretrain | SFT | RL |
|---|---|---|---|
| Warmup steps | 2000–8000 | 100–500 (~3% of total) | 0–50 |
| Schedule | cosine or WSD | cosine or constant | constant |
| Peak LR | 3e-4 (1B) → 1.2e-4 (70B) → 8e-5 (405B) | 2e-5 (Llama-3 SFT) | 1e-6 to 1e-5 |
| Min LR | 0.1 × peak | 0.1 × peak | — |

**Why warmup is essential with AdamW**: in the first few steps, `v_hat = (1-beta_2^t)^{-1} * v_t` is poorly estimated; the bias correction divides by a small number, inflating effective LR. Without warmup the first updates can NaN. Empirically: GPT-3 used 375M-token warmup; Llama-3 used 8000 steps; 0 warmup = divergence at 7B+.

**Common pitfalls**:
- Cosine sized for wrong horizon → significant final-loss penalty (Chinchilla showed +0.3–1% perplexity).
- Warmup too short with high LR (e.g. 100 steps for `lr=3e-4`) → loss spike at step ~150.
- Constant LR with no decay → final loss is 1–3% worse than cosine; OK for small experiments, bad for production.
- WSD decay phase too short (<5%) → underperforms cosine; too long (>30%) → wastes the stable phase.

## Connections
- **[[adam]]**: warmup is required *because* of Adam's bias correction; SGD with momentum tolerates zero warmup.
- **[[gradient-clipping]]**: high-LR-with-no-warmup failures present as a `grad_norm` spike in the first 50 steps; clipping alone doesn't save you.
- **[[weight-init]]** and μP: μP changes how peak LR scales with width, but the *schedule shape* is unchanged.
- **Continued pretraining / model souping**: WSD's stable-phase checkpoints are the natural inputs for [[early-stopping-and-checkpointing]] / model averaging.
- **RL fine-tuning**: usually constant LR after a tiny warmup — the policy is already good and you don't want LR-induced drift.
- **Karpathy** ([[karpathy-training-neural-net-recipe]]): "use a constant LR for sanity checking, then tune the schedule" — the schedule should be the *last* thing you tune.
