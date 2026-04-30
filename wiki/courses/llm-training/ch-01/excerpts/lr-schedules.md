---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "Learning Rate Schedules — Vaswani et al. (2017), Loshchilov & Hutter (2017, SGDR), Hu et al. (2024, MiniCPM / WSD)"
source_url: https://arxiv.org/abs/1706.03762
created_at: "2026-04-23"
---

# Excerpt: LR schedules — warmup, cosine, inverse-sqrt, WSD

**Authors (composite):** Ashish Vaswani et al. (inverse-sqrt + warmup, 2017); Ilya Loshchilov, Frank Hutter (SGDR/cosine, 2017); Shengding Hu et al. (MiniCPM/WSD, 2024); Alexander Hägele et al. (2024)
**URLs:**
- https://arxiv.org/abs/1706.03762 — Attention Is All You Need (Vaswani 2017)
- https://arxiv.org/abs/1608.03983 — SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov 2017)
- https://arxiv.org/abs/2404.06395 — MiniCPM (Hu et al. 2024)
**arXiv IDs:** 1706.03762, 1608.03983, 2404.06395
**Raw-data source:** [[raw-data/lr-schedules]]

---

## The meta-question schedules answer

The optimal learning rate is non-stationary across training. Early on, `v̂` in AdamW (see [[excerpts/adam]]) is a poor estimate and aggressive steps destroy the run. Late in training, the parameters sit near a sharp minimum and large steps overshoot it. In the middle — the bulk of training — a high, roughly-constant LR is what maximises token-efficiency. Every LR schedule in use today is an attempt to carve this three-phase shape out of wall-clock time.

The four dominant families are linear warmup + one of: cosine, inverse-sqrt, WSD, or constant. Each is studied below with its formula, origin, and when to use it.

---

## Linear warmup (universal prelude)

```math
\text{lr}(t) = \alpha_{\text{peak}} \cdot \frac{t}{T_w} \quad \text{for } t < T_w
```

Prepended to essentially every modern schedule. `T_w` (warmup steps) is typically 2000 for small models and 2000–8000 for 7B–70B. Pretraining runs with very large batch size or very high peak LR extend warmup to 10k+.

**Why warmup is required with AdamW — the precise reason.** Recall the bias-corrected `v̂` from [[excerpts/adam]]:

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^{\,t}}
```

At `t = 1` with `β₂ = 0.95`, the denominator is `1 - 0.95 = 0.05`. `v_1 = (1-β₂) g_1² = 0.05 g_1²`, so `v̂_1 = g_1²`. The update magnitude is:

```math
\frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = \frac{g_1}{|g_1| + \epsilon} \approx \text{sign}(g_1)
```

The first update is a **signed step of magnitude `α`**. For `α = 3e-4`, a single signed step of that size is typically small enough to survive, but at larger `α` (say `1e-3` for a small model) the signed step can push parameters far off the init manifold. A linear ramp from 0 to peak over `T_w` steps lets `v̂` accumulate enough samples to become a meaningful variance estimate before the effective LR reaches its peak.

Empirically: GPT-3 used 375M tokens of warmup. Llama-3 used 8000 steps. A 7B+ model with 0 warmup diverges within the first 100 steps — not slowly, not with a recoverable spike, but catastrophically.

SGD with momentum tolerates zero warmup; only adaptive optimizers need it. This alone is a strong argument that warmup is not about "finding good descent directions" but specifically about stabilising `v̂`.

---

## Cosine annealing (Loshchilov 2017, SGDR)

After warmup:

```math
\text{lr}(t) = \alpha_{\min} + \frac{1}{2}(\alpha_{\text{peak}} - \alpha_{\min}) \left( 1 + \cos\left( \pi \cdot \frac{t - T_w}{T - T_w} \right) \right)
```

for `T_w ≤ t ≤ T`, where `T` is total training steps. `α_min` is typically `0.1 · α_peak` (Llama, GPT-3) or `0` (some pretrains).

**Why cosine wins in practice.** Loshchilov's original motivation (SGDR) was warm restarts — cosine with periodic restarts to `α_peak`. The "restart" part was largely abandoned for LLMs; the smooth-decay part stuck. The smooth decay is empirically better than step decay (fewer transient loss bumps) and has no tuneable shape parameters beyond the min/max.

Notice the key property: **cosine is sized to the horizon**. The formula explicitly takes `T` as input. The Chinchilla paper (Hoffmann 2022, Figure A1) showed that a schedule sized for the wrong `T` costs ~0.3–1% perplexity: if you plan for `T` and stop early at `0.5T`, the LR is still at `~0.5 · α_peak` — you never got the benefit of decay. If you plan for `T` and continue past it, the LR has already decayed to `α_min` and further training is at a uselessly-small LR.

This is cosine's one weakness: you must commit to a token budget up front.

---

## Inverse-square-root (Vaswani 2017, the original Transformer schedule)

```math
\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\!\left(t^{-0.5}, \; t \cdot T_w^{-1.5}\right)
```

Section 5.3 of "Attention Is All You Need" — the schedule used to train the original Transformer. The `min` is a clever one-liner: for `t < T_w` it evaluates to `t · T_w^{-1.5}` (a linear warmup with slope `T_w^{-1.5}`); for `t > T_w` it evaluates to `t^{-0.5}` (inverse-sqrt decay). The two pieces are continuous at `t = T_w` since `T_w · T_w^{-1.5} = T_w^{-0.5}`.

The `d_model^{-0.5}` prefactor makes the schedule self-scale with width: a wider model gets a proportionally smaller LR. This was a (small) step toward the width-transfer properties that μP later formalised (see [[excerpts/weight-init]]).

**Why it lost to cosine for LLMs.** Inverse-sqrt decays too slowly for fixed-budget pretraining: after 1M steps, `t^{-0.5} ≈ 10^{-3}`, but cosine at `t = T` is already at `α_min = 0.1 · α_peak`. In a fixed-budget regime, cosine extracts more work out of the final decile of training. Inverse-sqrt still shows up in encoder pretrains and some machine-translation systems but is rare in modern autoregressive LLMs.

---

## WSD — Warmup-Stable-Decay (Hu 2024, MiniCPM / DeepSeek)

```
phase 1 (warmup):  α ramps 0 → α_peak     over [0, T_w]
phase 2 (stable):  α = α_peak              over [T_w, T - T_d]
phase 3 (decay):   α → α_min               over [T - T_d, T]
                   shape: linear, 1-sqrt, or cosine
```

`T_d` (decay length) is typically 10–20% of total training.

**Why WSD matters.** The stable phase has constant LR, so a checkpoint from *anywhere* in the stable phase is a valid "trunk" from which to fork a shorter decay. Hu et al. and Hägele et al. (2024) show empirically that a 10%-decay from a stable-phase checkpoint matches the loss curve of a full cosine sized to stop at the same point. This decouples schedule from token-budget commitment:

- Pretrain `70%` of tokens in the stable phase.
- Fork multiple variants from the trunk — each gets its own 20%-decay run targeting a different dataset mix, a different end-LR, or a different evaluation setting.

This is the schedule MiniCPM and DeepSeek use to produce multiple model variants from one trunk. It is strictly more flexible than cosine — and since the early-2024 publications, WSD has been displacing cosine in new pretraining recipes.

**The decay shape matters.** Linear decay is simplest. `1 - √(t/T_d)` decay is slightly better empirically (concave, more time near the peak). Cosine-shaped decay inside WSD (warmup-stable-cosine-decay) is another variant. The difference is <0.1 perplexity — minor — but the stable-phase advantage over cosine is 0.2–0.5 perplexity *when you don't know the horizon in advance*.

---

## Modern defaults at a glance

| Setting | Pretrain | SFT | RL |
|---|---|---|---|
| Warmup | 2000–8000 steps | 100–500 (~3% of total) | 0–50 |
| Schedule | cosine or **WSD** | cosine or constant | constant |
| Peak LR | 3e-4 (1B) → 1.2e-4 (70B) → 8e-5 (405B) | 2e-5 (Llama-3 SFT) | 1e-6 – 1e-5 |
| Min LR | 0.1 × peak | 0.1 × peak | — |

**Why RL uses constant LR with tiny or zero warmup.** The policy is already a strong LM by the time RL starts; the goal is conservative preference shaping, not large-scale feature learning. LR-schedule-induced drift is more likely to hurt than help. The Llama-3 post-training recipe uses `lr = 3e-7` constant for PPO and `lr = 1e-6` for DPO — flat, constant, no schedule.

---

## Chinchilla's schedule-sizing lesson

Figure A1 of Hoffmann et al. 2022 ("Chinchilla") plots final validation loss vs. the ratio of cosine length to actual training length. The curve has a clear minimum at `ratio = 1.0` — i.e. cosine must be sized to match training. Setting `T_cosine = 2T_train` (over-long cosine) costs ~0.3 perplexity; setting `T_cosine = 0.5T_train` (under-long cosine, then flat at `α_min`) costs ~1.0 perplexity. The penalty is real enough to justify recomputing the schedule whenever token-budget plans change.

This is arguably the strongest empirical argument for WSD: it makes the schedule-sizing problem go away.

---

## Common pitfalls

- **Warmup too short for large peak LR**: `T_w = 100` with `α_peak = 3e-4` spikes the loss at step ~150. Either lengthen warmup to 2000+ or lower peak LR.
- **Cosine sized for wrong horizon**: +0.3–1% perplexity; always resize when budget changes.
- **Constant LR without decay**: 1–3% worse than cosine; OK for ablation / sanity checking, not for production.
- **WSD decay too short (<5%)**: underperforms cosine; too long (>30%) wastes the stable phase.
- **Resetting the step counter on resume**: warmup re-triggers, loss spikes. Always restore scheduler state.

---

## The Karpathy heuristic

"Use a constant LR for sanity checking, then tune the schedule — the schedule should be the *last* thing you tune." The ordering of hyperparameter debugging goes: (1) init ([[excerpts/weight-init]]), (2) peak LR via a short LR-range test, (3) weight decay, (4) warmup length, (5) schedule shape. If the model diverges you almost never have a schedule problem — you have an init, LR, or clipping problem.

---

## Connections

- [[excerpts/adam]] — warmup is *required* because of Adam's bias correction; the first-step signed update analysis above is the precise reason.
- [[excerpts/gradient-clipping]] — high-LR-with-no-warmup failures manifest as `grad_norm` spikes in the first 50 steps; clipping masks the symptom but the underlying issue is unstable `v̂`.
- [[excerpts/weight-init]] — μP changes peak LR scaling with width but leaves schedule shape unchanged.
- [[excerpts/mixed-precision]] — under fp16, warmup is even more critical because noisy early gradients can push `v̂` into underflow or overflow.
- [[ch-01]] — parent chapter for training fundamentals.
