---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: lr-schedules (Vaswani 2017 + Loshchilov 2017 + Hu/MiniCPM 2024 + Hägele 2024)
source_url: https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/1608.03983 ; https://arxiv.org/abs/2404.06395 ; https://arxiv.org/abs/2405.18392
created_at: "2026-04-23"
---

# Excerpt: Learning-Rate Schedules — Warmup, Cosine, Inverse-Sqrt, WSD

**Sources (composite family):**
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017 — inverse-sqrt schedule, §5.3 — arxiv 1706.03762
- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017 — cosine annealing — arxiv 1608.03983
- Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies," 2024 — WSD — arxiv 2404.06395
- Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations," 2024 — constant-with-cooldown analysis — arxiv 2405.18392
- Goyal et al., "Accurate, Large Minibatch SGD," 2017 — linear warmup — arxiv 1706.02677
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), 2022 — cosine-horizon penalty, Figure A1

---

## Why one chapter covers four schedules

Four schedule families dominate modern LLM training. Each was proposed for a different problem, and each survives because its failure mode does not overlap with the others. The synthesis `[[ch-03]]` uses is: **warmup is always prepended; the mid/tail choice is cosine-for-fixed-budget or WSD-for-open-ended**. Inverse-sqrt is a legacy curiosity worth understanding because it is still in Vaswani's code and the `d_model` scaling is a conceptual hint at what μP later formalises.

The composite raw-data file [[lr-schedules]] opens with the single sentence that explains why this is a chapter at all:

> "The optimal learning rate is non-stationary across training — it must be small at the start (so adaptive optimizer state can stabilize) and small at the end (so the model can settle into a sharp minimum), with a high-LR plateau in the middle."

Two of those three claims have to be taught together with [[excerpts/adam]]: "small at the start" is a statement about `v̂` under AdamW's bias correction, not a general truth of gradient descent. Plain SGD with momentum does not need warmup. The schedule literature reads like a stand-alone topic, but half of it is a consequence of the optimizer choice.

---

## 1. Linear warmup — the ramp everybody prepends

```math
\mathrm{lr}(t) \;=\; \mathrm{peak\_lr} \cdot \frac{t}{\mathrm{warmup\_steps}} \qquad \text{for } t < \mathrm{warmup\_steps}
```

Goyal et al. (2017, §2.2) introduced this for large-batch ImageNet SGD — not for AdamW, not for Transformers. Their argument was batch-size-linear: if you scale the batch by `k`, you must scale LR by `k`, but you cannot start there because the loss surface around init does not tolerate a `k×` step. Warm up over the first 5 epochs.

The LLM adaptation flips the motivation. Warmup in AdamW is about **bias correction**, not batch size. Recall from [[excerpts/adam]] that the update is

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

At `t=1` and `β₂=0.999`, `1 - β₂¹ = 0.001`, so `v̂` is `v_t / 0.001 = 1000 × v_t`. But `v_t = 0.001 · g_1²`, so `v̂ = g_1²` — fine. The problem is subtler: the *direction* of `m̂/√v̂` depends on a single sample `g_1` for the first ~5 steps, and for parameters that happen to see tiny gradients early, `√v̂` is a wild underestimate and the update is correspondingly huge. The [[lr-schedules]] raw-data file puts this bluntly:

> "`v_hat = (1-beta_2^t)^{-1} * v_t` is poorly estimated; the bias correction divides by a small number, inflating effective LR. Without warmup the first updates can NaN. Empirically: GPT-3 used 375M-token warmup; Llama-3 used 8000 steps; 0 warmup = divergence at 7B+."

Notice: the 375M-token GPT-3 warmup and the 8000-step Llama-3 warmup are the same order of magnitude when you convert tokens-to-steps at their respective batch sizes. The heuristic has stayed remarkably stable across three model generations.

What does this mean for LLM training in 2025? **Never set `warmup_steps = 0` on a pretraining run.** The cheapest debug win in this entire chapter is to set warmup to 2000 and verify `grad_norm` is bounded over steps 1–100. See `[[ch-03]]` §1 for the default table.

---

## 2. Cosine annealing — Loshchilov & Hutter (SGDR, 2017)

The paper is titled "Stochastic Gradient Descent with Warm Restarts" and its original contribution is *restarts*. The LLM world took the base curve and threw away the restart part. The single-cycle formula is:

```math
\mathrm{lr}(t) \;=\; \mathrm{min\_lr} \;+\; \tfrac{1}{2}(\mathrm{peak\_lr} - \mathrm{min\_lr})\cdot\left(1 + \cos\left(\pi\cdot\frac{t - \mathrm{warmup}}{T - \mathrm{warmup}}\right)\right)
```

Derivation: you want a schedule that (a) starts at `peak_lr` when `t = warmup` (cosine argument = 0, cos = 1, expression collapses to `peak_lr`), (b) ends at `min_lr` when `t = T` (argument = π, cos = -1, expression collapses to `min_lr`), and (c) has zero first-derivative at both endpoints so the transitions are smooth. The cosine between 0 and π is the unique trig function with that property up to rescaling.

Two design conventions fix `min_lr`:

| Convention | `min_lr` | Used by |
|---|---|---|
| "10% floor" | `0.1 × peak_lr` | GPT-3, Llama-1/2/3, Chinchilla |
| "hard zero" | `0.0` | Some Qwen runs, T5-style |

The 10% floor is safer for continued fine-tuning because you never hit a literal zero-LR. Hard zero squeezes a marginal improvement in final loss but leaves you unable to resume.

### The Chinchilla horizon penalty

Hoffmann et al. (2022) Figure A1 is the single figure every pretraining practitioner memorises. They trained identical models with cosine schedules sized for `N`, `2N`, `4N`, and `0.5N` tokens, and measured val loss at each stopping point. The pattern:

- Cosine sized **exactly** for the token budget → best val loss.
- Cosine sized for **longer** than the budget (LR never reaches `min_lr`) → ~0.3–1% worse val perplexity.
- Cosine sized for **shorter** (you keep training past the nominal `T`, LR stuck at `min_lr`) → also worse.

What does this mean for LLM training in 2025? **Commit to a token budget before training starts.** If you don't know the budget, pick WSD (below). The [[lr-schedules]] commentary:

> "Cosine sized for wrong horizon → significant final-loss penalty (Chinchilla showed +0.3–1% perplexity)."

0.3–1% perplexity on a 70B model is the difference between "mid" and "frontier-class" on MMLU. The penalty is not decorative.

---

## 3. Inverse-square-root — Vaswani 2017

From Vaswani §5.3:

```math
\mathrm{lr}(t) \;=\; d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5},\; t \cdot \mathrm{warmup\_steps}^{-1.5}\right)
```

Let me derive how this becomes a warmup-then-decay curve. The two arguments of `min`:

- Argument A: `t · warmup^(-1.5)` — linear in `t`, rises from 0. Equivalent to a linear warmup with slope `warmup^(-1.5)`.
- Argument B: `t^(-0.5)` — decays as `1/√t`.

They are equal when `t · warmup^(-1.5) = t^(-0.5)`, i.e. `t^(1.5) = warmup^(1.5)`, i.e. `t = warmup`. So:

- For `t < warmup_steps`, argument A (linear) is smaller → LR grows linearly.
- For `t > warmup_steps`, argument B (`1/√t`) is smaller → LR decays as `1/√t`.

The peak LR at `t = warmup` is `d_model^(-0.5) · warmup^(-0.5)`. Notice: **the peak is set by `d_model` and `warmup_steps`, not by a user-chosen hyperparameter.** This is the self-scaling property. Double `d_model` → peak LR drops by √2; quadruple `warmup` → peak LR drops by 2.

What does this mean for LLM training in 2025? Inverse-sqrt is elegant theoretically — it tunes itself with model width — but empirically loses ~0.3% to cosine on fixed-budget pretraining. The self-scaling intuition survived, though, and resurfaced in [[excerpts/weight-init]] as the μP parametrisation. Inverse-sqrt is still the T5 default and appears in some encoder-only recipes (BERT-base uses it in the original 2018 code).

---

## 4. WSD — Warmup-Stable-Decay (Hu 2024 / MiniCPM)

Three phases, pick-your-shape decay:

```math
\mathrm{lr}(t) \;=\;
\begin{cases}
\mathrm{peak\_lr} \cdot t / \mathrm{warmup} & 0 \le t < \mathrm{warmup} \\
\mathrm{peak\_lr} & \mathrm{warmup} \le t < T - D \\
\mathrm{peak\_lr} \cdot \mathrm{decay\_shape}\!\left(\tfrac{t - (T-D)}{D}\right) & T - D \le t \le T
\end{cases}
```

`decay_shape(s)` is commonly `1 - s` (linear), `0.5(1 + \cos(\pi s))` (cosine half-cycle on just the tail), or `1 - \sqrt{s}` (MiniCPM's reported choice). `D` is typically 10–20% of `T`.

The key insight is the **checkpoint-ability of the stable phase**. You are training indefinitely at constant high LR. At any moment, you can fork the checkpoint, run the short decay for `D` steps, and get a "done" model. This turns one pretraining run into a family of models at different effective budgets without retraining.

The [[lr-schedules]] file calls out:

> "The stable phase is *checkpoint-able* — you can fork off a 10%-decay run from any stable-phase checkpoint. This is how DeepSeek and MiniCPM produce many model variants from one pretraining trunk."

Hu et al. (2024) observed the signature **step-down in loss at the start of the decay phase**. Loss is roughly flat during the stable phase (it was already near its stable-LR equilibrium), then drops sharply when decay kicks in — because the model can finally settle into a local minimum it was oscillating around. Cosine hides this behaviour because LR decays continuously.

### Hägele et al. (2024) — why WSD matches cosine

Hägele et al. analyse "constant + cooldown" schedules and prove (empirically and with a small analytic argument) that WSD with a sufficiently long decay approaches cosine's final loss within the noise floor, *for any training length*. This is the theoretical underpinning: WSD is not a compromise, it is a strict generalisation — cosine is approximately WSD with `D = T - warmup` and a specific decay shape.

What does this mean for LLM training in 2025? Newer open frontier recipes (DeepSeek-V2/V3, Qwen-2.5, MiniCPM) are all WSD. If you're starting a fresh pretrain and you don't know your final token budget, WSD is the strict winner. If you do know the budget, cosine and WSD are tied. See `[[ch-03]]` §1 for the defaults table.

---

## 5. Constant + cooldown — Hägele 2024 framing

A special case of WSD where there is *no* warmup (or only 50–100 steps) and the stable LR is held constant, with cooldown as above. This is the standard for RL fine-tuning (PPO/GRPO per [[excerpts/adam]]) and for small continued-pretraining runs. The logic: the base model is already a good initialisation, `v̂` is already calibrated from pretraining, so the "warmup because of bias correction" argument vanishes.

Hägele showed that even for *pretraining*, constant + cooldown with no warmup can match cosine if you're willing to accept a 3–5% higher early-step loss (which is quickly amortised). Most frontier teams still prepend warmup out of caution.

---

## 6. The practical defaults — why the numbers are what they are

From [[lr-schedules]]:

| Setting | Pretrain | SFT | RL |
|---|---|---|---|
| Warmup steps | 2000–8000 | 100–500 (~3% of total) | 0–50 |
| Schedule | cosine or WSD | cosine or constant | constant |
| Peak LR | 3e-4 (1B) → 1.2e-4 (70B) → 8e-5 (405B) | 2e-5 (Llama-3 SFT) | 1e-6 to 1e-5 |
| Min LR | 0.1 × peak | 0.1 × peak | — |

The **peak LR decreases with model scale**. This is not μP (μP would *width-invariantly* transfer LR; see [[excerpts/weight-init]] §μP). It is an empirical observation from sweeps: larger models are more sensitive to high LR, partly because residual-stream variance grows with depth (see [[excerpts/batch-vs-layer-norm]] §pre-norm) and partly because `v̂` has more entries to underestimate.

The **SFT LR is 10–20× smaller** than pretrain LR. Reason: the pretrained model is already near a minimum; large SFT steps push the weights off the ridge of "generally capable" onto the ridge of "good at this specific SFT dataset" — overfitting. The 2e-5 Llama-3 SFT LR is deliberately conservative.

The **RL LR is another 10–100× smaller** than SFT. Policy gradients are much noisier than SL gradients, and every off-policy step risks destroying the model's behaviour. See [[ch-36]] (PPO) for the full story.

---

## 7. Common pitfalls (from the source)

Quoting [[lr-schedules]]:

> "Cosine sized for wrong horizon → significant final-loss penalty (Chinchilla showed +0.3–1% perplexity). Warmup too short with high LR (e.g. 100 steps for `lr=3e-4`) → loss spike at step ~150. Constant LR with no decay → final loss is 1–3% worse than cosine; OK for small experiments, bad for production. WSD decay phase too short (<5%) → underperforms cosine; too long (>30%) → wastes the stable phase."

The first pitfall is the expensive one at frontier scale. The second is the one every new engineer hits. The third is why "just run with constant LR" is such a common blog-post failure mode — it looks cheap, it looks fine in the loss curve, but final evals take a 2% hit.

---

## 8. Interaction with AdamW (the reason schedules *matter*)

The line from [[lr-schedules]]:

> "AdamW's bias correction makes the first ~100 steps effectively use a higher LR — this is *partly* why warmup is required; it lets `v_hat` stabilize before full LR."

This is why Chapter 3 is specifically about LR, init, and norms *as a bundle*. The optimizer is ch-01, but its dynamics — particularly the first few hundred steps — dictate the schedule shape you need. Non-adaptive optimizers (SGD+momentum) can start at peak LR from step 1 and use a pure `1/t` or cosine decay with no warmup. Every AdamW recipe needs warmup. Every μP-parametrised AdamW recipe still needs warmup (μP fixes the *peak value* across widths; it does not obviate the bias-correction issue).

---

## Connections

- [[excerpts/adam]] — why warmup exists; `β₂=0.95` and its effect on the first 100 steps.
- [[excerpts/weight-init]] — μP LR transfer rules, init scale's interaction with initial `grad_norm`.
- [[excerpts/batch-vs-layer-norm]] — pre-norm's `O(1)` gradient norm means schedules can be more aggressive than with post-norm.
- [[excerpts/mixed-precision]] — loss-scale and schedule interact: unscale → clip → step ordering, then LR is applied.
- [[ch-03]] — synthesis.
- [[ch-01]] — AdamW mechanics.
- [[ch-36]] — PPO/GRPO LR conventions.
