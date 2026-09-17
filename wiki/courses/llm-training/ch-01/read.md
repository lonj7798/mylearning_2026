<!-- chapter: ch-01
     track: foundations
     kind: content
     title: Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability
     deps: [ch-00]
     sources: [[adam]], [[gradient-clipping]], [[pytorch-adamw-clip-amp]], [[heavy-tailed-class-imbalance-adam]], [[why-weight-decay-modern-dl]], [[small-scale-proxies-instabilities]], [[gpt-3-few-shot]], [[rlhf-instructgpt]], [[rlhf-instructgpt-optimizer-settings]], [[llama-2-recipe]], [[llama-3]], [[llama-3-recipe]], [[olmo-2]], [[olmo-2-optimizer-settings]], [[deepseek-v3-recipe]], [[kimi-k2]], [[kimi-k2-recipe]], [[glm-4-5-recipe]], [[qwen-2.5-recipe]], [[lora-learns-less-forgets-less]], [[lora-without-regret]], [[rls-razor]], [[online-merging-optimizer]], [[fantastic-pretraining-optimizers]], [[muon-scalable-moonlight]], [[openrlhf-ppo-recipe]]
     figures: figures/beta2-memory.html, figures/clip-unscale-order.html
     revised: 2026-09 (generality revision)
-->

# Chapter 01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability

> **Core insight.** In the common case Adam moves each coordinate by approximately at most the learning rate per step, independent of the gradient's scale ([[adam]] §2.1), so the learning-rate schedule and the number of steps set how far a training stage moves a model. In pretraining, this per-coordinate normalization lets the loss on rare tokens fall at a rate that gradient descent does not reach ([[heavy-tailed-class-imbalance-adam]] Fig. 1), and weight decay acts through the effective learning rate and bf16 stability rather than as a regularizer ([[why-weight-decay-modern-dl]] §3). In fine-tuning, the same update budget governs forgetting: full fine-tuning of Llama-2-7B on code instructions lowered a three-benchmark retention average from 0.595 after 1 epoch to 0.414 after 16 epochs ([[lora-learns-less-forgets-less]] Table S6). Alternative optimizers that lower pretraining loss at small scale showed about a 1.1× speedup and no downstream gain at 1.2B parameters once each optimizer was tuned ([[fantastic-pretraining-optimizers]] §4.1, Table 5).
>
> **Guideline.** When pretraining a dense Transformer without the budget for an optimizer sweep, use AdamW with β₁ = 0.9, β₂ = 0.95, ε = 1e-8, weight decay 0.1 in the PyTorch form, no decay on token embeddings, and global-norm clipping at 1.0, because this combination is the released OLMo 2 7B configuration ([[olmo-2-optimizer-settings]]) and OLMo 2's ablations found lower gradient norms with ε = 1e-8 than with 1e-5 and fewer spikes without embedding decay (arXiv:2501.00656v3 §3.4.1-§3.4.2). No report in this chapter ablates β₂ at scale, so treat β₂ as a value to tune when a small-scale sweep is possible ([[fantastic-pretraining-optimizers]] Table 36). When fine-tuning a general checkpoint, choose the peak learning rate and the number of steps or epochs from a measured target-gain versus held-out-retention curve, because forgetting grew with epochs ([[lora-learns-less-forgets-less]] Table S6) and weight-space distance predicted forgetting worse than KL divergence on new-task inputs in a toy setting ([[rls-razor]] Table 1). When an RLHF stage regresses held-out tasks, add an anchor to the pretraining distribution or to the SFT delta rather than only a larger KL coefficient, because in InstructGPT a 100× larger KL coefficient did not remove the regressions that a pretraining-gradient mix removed ([[rlhf-instructgpt-optimizer-settings]] App. E.6). When fp16 loss scaling is used, unscale gradients before clipping, because clipping scaled gradients makes the threshold invalid according to the PyTorch documentation ([[pytorch-adamw-clip-amp]]); with bf16 and no loss scaling the order does not apply. When an alternative optimizer is proposed, compare it at the end of training against a tuned AdamW baseline on a breadth suite and test SFT with both optimizers; otherwise keep AdamW.

## Why this chapter matters for a general-purpose model

An optimizer converts gradients into parameter changes. For a general-purpose model this conversion matters at three points of the pipeline (pre-training → mid-training → SFT → preference optimization → RL → evaluation).

1. **Pre-training coverage.** Next-token prediction is a classification problem over the vocabulary (128,000 tokens for Llama 3, [[llama-3-recipe]]) with heavy-tailed token frequencies ([[heavy-tailed-class-imbalance-adam]] §1.1). Which tokens the optimizer learns within a fixed token budget determines how much of the long tail (rare words, identifiers, low-resource languages) the base model covers (§2).
2. **Stability of long runs.** Loss spikes and divergence waste compute and can leave a worse checkpoint. Weight decay, ε, β₂, and gradient clipping are the optimizer settings that open reports change to control them (§3-§5).
3. **Retention during post-training.** Every post-training stage starts from a checkpoint that already has broad ability. The learning rate, the number of steps, the form of the update (full, low-rank, merged), and the anchors used during optimization determine how much of that ability survives (§6). ch-30a measures forgetting in detail; this chapter covers the optimizer side.

The measurable problems are: slower loss decrease on rare-token classes; loss and gradient-norm spikes; and held-out benchmark drops after fine-tuning. Terms used below: a *parameter* θ is one trainable number (a coordinate) or the vector of all of them; the *gradient* g is the derivative of the training loss with respect to θ; the *learning rate* α (also written η or lr) scales each update; a *step* is one optimizer update; *global norm* is the L2 norm of the gradient over all parameters.

## §1 AdamW: the update rule, symbol by symbol

**Definition.** Adam is a stochastic optimizer that keeps two exponential moving averages per parameter, one of the gradient and one of the squared gradient, and divides the first by the square root of the second ([[adam]] Algorithm 1). AdamW is Adam with *decoupled* weight decay: the decay term shrinks the parameter directly and does not pass through the moment estimates ([[adam]] AdamW Algorithm 2).

**The problem it addresses.** Gradient magnitudes differ by orders of magnitude between coordinates. With one learning rate, gradient descent moves coordinates with large gradients far and coordinates with small gradients little. Adam's per-coordinate division makes the step size close to α for every coordinate.

**Mechanism.**
1. Compute the stochastic gradient g_t on a minibatch.
2. Update the first moment m_t (a running mean of g) and the second moment v_t (a running mean of g²).
3. Divide each by (1 − β^t) to remove the bias from initializing both at zero.
4. Step each coordinate by α·m̂_t/(√v̂_t + ε).
5. For AdamW, also shrink each decayed parameter by a factor that depends on λ.

**Formula** ([[adam]] Algorithm 1; AdamW Algorithm 2, line 12):

```
m_t = β₁·m_{t−1} + (1 − β₁)·g_t
v_t = β₂·v_{t−1} + (1 − β₂)·g_t²
m̂_t = m_t / (1 − β₁^t)          v̂_t = v_t / (1 − β₂^t)
θ_t = θ_{t−1} − η_t·( α·m̂_t / (√v̂_t + ε) + λ·θ_{t−1} )      # AdamW, paper form
```

θ_t: parameters after step t. g_t: gradient at step t. m_t, v_t: first and second moment estimates (zero at t = 0). β₁, β₂ ∈ [0, 1): decay rates of the two averages. m̂_t, v̂_t: bias-corrected estimates. α: base step size. ε: a small constant that keeps the denominator positive. η_t: schedule multiplier. λ: weight-decay coefficient. All operations are element-wise.

Adam's step has an approximate bound: with ε = 0, |Δθ_t| ≲ α for each coordinate, and the step is invariant to multiplying all gradients by a constant c, because m̂ scales by c and √v̂ by c ([[adam]] §2.1). The paper gives the looser bound α·(1 − β₁)/√(1 − β₂) for the case (1 − β₁) > √(1 − β₂) and says it is reached only in the most severe sparsity, when a coordinate's gradient was zero at every earlier step. The bound α is approximate: with β₂ = 0.999 the condition holds (0.1 > 0.032), and §4, worked example 2, shows steps of 1.81α after a one-step spike and 5.30α during a steady gradient rise.

**The implementation most runs use.** PyTorch v2.5.0 `AdamW` multiplies the parameter by (1 − lr·weight_decay) before the Adam step ([[pytorch-adamw-clip-amp]], `torch/optim/adamw.py` L368-L429):

```python
# Perform stepweight decay
param.mul_(1 - lr * weight_decay)
# Decay the first and second moment running average coefficient
exp_avg.lerp_(grad, 1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
bias_correction1 = 1 - beta1**step
bias_correction2 = 1 - beta2**step
step_size = lr / bias_correction1
bias_correction2_sqrt = bias_correction2**0.5
denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
param.addcdiv_(exp_avg, denom, value=-step_size)
```

Here `lr` already contains the schedule, so the per-step decay is lr·λ. In the paper form the decay is η_t·λ, which does not scale with the base α. [[small-scale-proxies-instabilities]] calls the paper form *independent* weight decay (§3.2.2). PyTorch `Adam` with `weight_decay` implements L2 regularization instead: `grad = grad.add(param, alpha=weight_decay)` (`torch/optim/adam.py` L366-L367), so the decay term is divided by √v̂ together with the loss gradient.

**Worked example 1 (two AdamW steps, one coordinate).** Take θ₀ = 1.0, α = 1e-3, β₁ = 0.9, β₂ = 0.95, ε = 1e-8, λ = 0.1, PyTorch form, gradients g₁ = 0.2 and g₂ = −0.1.
- Step 1: decay θ ← 1.0 × (1 − 1e-3 × 0.1) = 0.9999. m₁ = 0.1 × 0.2 = 0.02; v₁ = 0.05 × 0.04 = 0.002. m̂₁ = 0.02/0.1 = 0.2; v̂₁ = 0.002/0.05 = 0.04; √v̂₁ = 0.2. Ratio 1.0, so θ₁ = 0.9999 − 0.001 = 0.9989. For any g₁, bias correction gives m̂₁ = g₁ and v̂₁ = g₁², so the first step always has size α.
- Step 2: decay θ ← 0.9989 × 0.9999 = 0.998800. m₂ = 0.9 × 0.02 + 0.1 × (−0.1) = 0.008; v₂ = 0.95 × 0.002 + 0.05 × 0.01 = 0.0024. m̂₂ = 0.008/0.19 = 0.0421; v̂₂ = 0.0024/0.0975 = 0.0246; √v̂₂ = 0.1569. Ratio 0.268, so θ₂ = 0.998800 − 0.000268 = 0.998532. The sign change of the gradient reduced the step to 27% of α.

**Worked example 2 (L2 versus decoupled decay).** Two coordinates both have θ = 1, α = 1e-3. Coordinate A has typical |g| = 10, so √v̂ ≈ 10; coordinate B has |g| = 0.01, so √v̂ ≈ 0.01. With an L2 coefficient 1e-4 inside Adam (and the L2 term small relative to g), the decay part of the step is α·λθ/√v̂: 1e-3 × 1e-4 / 10 = 1e-8 for A and 1e-3 × 1e-4 / 0.01 = 1e-5 for B, a 1,000× difference. With PyTorch AdamW and λ = 0.1, both coordinates decay by α·λ·θ = 1e-4 per step. This is the inequivalence stated in [[adam]] (AdamW §2, Proposition 2): under L2, "weights with large typical gradient magnitude s are regularized by a smaller relative amount".

**Evidence.** The AdamW paper's reported gain is a 15% relative test-error improvement for Adam on CIFAR-10 and ImageNet32x32 image classification (AdamW §1, Figs. 2-3); it has no language-model experiment. For one-epoch GPT-2-124M training, an L2 penalty added to the loss reproduced the training-loss benefit of weight decay (Fig. 21), and SGD with momentum showed a similar benefit (Fig. 22) ([[why-weight-decay-modern-dl]]) (Result, single study).

**Conditions and limits.** The evidence that decoupling itself matters is from image classification. For LLM pretraining, the effect that has been measured is the effect of weight decay in general (§3), not decoupled versus L2.

**Implication for a general-purpose model.** A λ taken from a report transfers only when the training code implements the same form of decay: the same λ = 0.1 gives different per-step decay under the paper form and the PyTorch form (worked in §3).

## §2 Why Adam trains language models faster than gradient descent: heavy-tailed token frequencies

**Definition.** *Heavy-tailed class imbalance* means that class frequencies follow a power law, so rare classes together account for a large share of the samples. In next-token prediction each vocabulary item is a class ([[heavy-tailed-class-imbalance-adam]] §1.1).

**The problem, stated as a measurement.** GPT2-Small trained on WikiText-103 was evaluated by training loss per group of classes, where each group holds about 10% of the samples sorted by frequency. "SGD makes little to no progress on low-frequency classes while Adam makes progress on all groups" (Fig. 1). Because most samples come from infrequent tokens, the average loss also falls slowly under SGD (Abstract).

**Mechanism** (weighted quadratic model, §3.1).
1. Give each class k a loss f_k(w_k) = ½‖w_k‖² weighted by its frequency π_k, with Σπ_k = 1.
2. The gradient with respect to w_k is π_k·w_k, so gradient descent shrinks w_k by the factor (1 − απ_k) per step.
3. The step size cannot exceed about 1/π₁ without destabilizing the most frequent class, so rare classes receive steps of about π_k/π₁ of the maximum.
4. Sign descent, a simplified form of Adam, divides the gradient by its magnitude, so each class moves by α per step independent of π_k.

**Formula** (§3.1): gradient descent gives w_k^(t) = (1 − απ_k)^t · w_k^(0); sign descent gives w_k^(t) = w_k^(t−1) − α·sign(f_k′(w_k^(t−1))). w_k^(t): parameters of class k after t steps. π_k: frequency of class k. α: step size.

**Worked example.** Let π₁ = 0.5 and a rare class π_k = 0.001, and use α = 1/π₁ = 2, the step-size limit stated in §3.1. The frequent class reaches zero in one step: (1 − 2 × 0.5) = 0. The rare class keeps (1 − 2 × 0.001)^100 = 0.998^100 = 0.819 of its initial error after 100 steps and 0.998^1000 = 0.135 after 1,000 steps. Sign descent with step 0.01 moves both classes from w = 1 to 0 in 100 steps.

**Evidence.** The frequency gap was reproduced with a CNN on MNIST plus about 10,000 added classes of 5 samples each (Fig. 2), with ResNet18 on an ImageNet subset with π_k ∝ 1/k (Fig. 3), and with a softmax linear model on random inputs (Fig. 4). It appears with full-batch gradient descent, so minibatch noise is not required (§2.3). Sign descent recovers Adam's behavior (Fig. 5), and upweighting the loss of rare classes improves SGD (Appendix F) ([[heavy-tailed-class-imbalance-adam]]). Status: Result (single study, several architectures).

**Conditions and limits.** Every measurement is training loss; no held-out capability was evaluated. The largest language model is GPT2-Small. The authors "do not claim that class imbalance is the only reason Adam outperforms SGD" (§1.1).

**Implication for a general-purpose model (Interpretation).** The long tail of a pretraining corpus (specialized vocabulary, code identifiers, text in low-resource languages) is made of rare tokens. An optimizer that normalizes per coordinate fits these classes within the same token budget in which gradient descent fits mainly the frequent classes. Whether this changes downstream coverage has not been measured; ch-12a (knowledge acquisition) and ch-13a (multilingual coverage) treat the data side. Adam's cost is two state tensors per parameter; the training-memory course covers optimizer-state memory.

## §3 Weight decay: effective learning rate and stability in one-pass pretraining, overfitting control only in multi-epoch stages

**Definition.** Weight decay multiplies decayed parameters by a factor slightly below 1 at every step. Three forms appear in practice:

| Form | Per-step effect | Where it appears |
|---|---|---|
| L2 in the loss | g ← g + λ_L2·θ, then divided by √v̂ | PyTorch `Adam(weight_decay=…)` ([[pytorch-adamw-clip-amp]]) |
| Independent (paper AdamW) | θ ← θ − η_t·λ·θ | [[adam]] Algorithm 2 line 12; [[small-scale-proxies-instabilities]] §3.2.2 |
| Coupled to LR (PyTorch AdamW) | θ ← θ·(1 − lr_t·λ) | PyTorch `AdamW`; OLMo 2: "every parameter is multiplied by 1 − (0.1 · lr) at every step" ([[olmo-2-optimizer-settings]], §3.4.2); Llama 3 scaling-law runs: "The weight decay at each step is set to 0.1 times the learning rate at that step" ([[llama-3-recipe]], §3.2.1) |

**The problem it addresses.** In classical learning theory weight decay limits model capacity to reduce overfitting. LLM pretraining is close to one pass over the data, and [[why-weight-decay-modern-dl]] reports that the generalization gap of GPT-2-124M is close to zero even without weight decay (Fig. 19). The measurable effects of weight decay in this regime are final training loss and divergence.

**Mechanism** ([[why-weight-decay-modern-dl]] §3, Interpretation by the authors).
1. For sign-like updates, decay keeps the parameter norm ‖w_t‖₂ smaller.
2. The direction w/‖w‖₂ then changes with an *effective learning rate* η_t/‖w_t‖₂, which is larger than without decay.
3. Early in training the larger effective LR raises the loss (the gradient-noise term grows with the step size); late in training, when the LR has decayed, the run with weight decay reaches lower loss because it made faster progress on the initial-condition term.
4. Weight decay also prevents late-training divergence under bfloat16 weights.

**Formula.** Effective LR η_eff ∝ η_t/‖w_t‖₂. Per-step decay under PyTorch AdamW: d_t = lr_t·λ; decay timescale 1/(lr·λ) steps at constant lr. η_t, lr_t: learning rate at step t. ‖w_t‖₂: L2 norm of the weights. λ: decay coefficient.

**Worked example.** OLMo 2 7B uses peak LR 3e-4 and λ = 0.1 in the coupled form ([[olmo-2-optimizer-settings]]). The per-step decay at peak LR is 3e-4 × 0.1 = 3e-5; the timescale is 1/3e-5 = 33,333 steps; after 10,000 steps at that LR the decay alone leaves (1 − 3e-5)^10,000 = 0.741 of a weight. One OLMo 2 7B step holds 1,024 × 4,096 = 4,194,304 tokens, so the 4T-token stage 1 is 953,674 steps (derived from Table 3), about 29 timescales at peak LR, fewer after the cosine decay. Under the independent form with λ = 1e-4 ([[small-scale-proxies-instabilities]] §2.1), the decay at schedule multiplier 1 is 1e-4 per step at every LR; under the coupled form with λ = 0.1 it is 3e-5 at LR 3e-4 and 1e-2 at LR 0.1. Raising the LR in the coupled form also raises the decay, which is why the two forms have different LR sensitivity.

**Evidence.**
- GPT-2-124M on OpenWebText, 50,000 iterations, batch 256, context 256, AdamW LR 6e-4, β₂ = 0.95, clipping 1.0: final training loss is lower with λ = 0.1 and 0.3 than with λ = 0 (Fig. 6). A run without decay whose LR schedule matches the effective LR of the λ = 0.1 or 0.3 run reproduces its whole loss curve in float32; the same runs diverge in bfloat16 (Fig. 7). At context 1,024 and LR 1e-3 without decay, all three bfloat16 seeds diverge late in training and do not recover, while λ > 0 prevents it (Fig. 8) ([[why-weight-decay-modern-dl]]). Status: Result (single study, 124M).
- Independent decay (λ = 1e-4) gave lower LR sensitivity than the coupled default (λ = 0.1) for 2.4M-1.2B models on C4; weight decay also mitigated output-logit divergence for the larger models tested ([[small-scale-proxies-instabilities]] §3.1.2, §3.2.2, Fig. 3, Fig. 6). Status: Result (single study).
- Decaying token embeddings in an OLMo 2 run shrank the embedding norm, raised the gradient norm, and gave gradient spike score 0.16 versus 0.092 without embedding decay ([[olmo-2-optimizer-settings]], §3.4.2, Fig. 10). The released OLMo 2 7B config sets `decay_embeddings: false` and `decay_norm_and_bias: true`, so it decays normalization weights; no ablation of norm or bias decay is reported. GLM-4.5 applies Muon to all parameters except word embeddings, biases, and RMSNorm weights, and does not print the optimizer or decay used for those ([[glm-4-5-recipe]], §2.4).
- Muon without weight decay let "both the weight and the layer output's RMS keep growing to a large scale, exceeding the high-precision range of bf16"; an 800M model trained on 100B tokens reached lower validation loss with decay than without ([[muon-scalable-moonlight]] §2.2, Fig. 2). Status: Replicated for the stability role (two independent groups, [[why-weight-decay-modern-dl]] and [[muon-scalable-moonlight]]).

**Contrast with multi-epoch stages.** Overfitting is measurable when data repeat. InstructGPT SFT models "overfit on validation loss after 1 epoch", yet 16 epochs improved RM score and human preference ratings; the 6B reward model overfit within a few epochs, with "obvious deterioration in the validation loss" ([[rlhf-instructgpt-optimizer-settings]] §3.5, App. C.1-C.2). In code instruction tuning of Llama-2-7B, weight decay of 5e-5 or 1e-4 learned and forgot about as much as full fine-tuning without decay, and its target performance deteriorated at epochs 8 and 16 ([[lora-learns-less-forgets-less]] §4.5, Fig. 4). Decoupled decay pulls weights toward zero, not toward the starting checkpoint θ₀ (from the formula θ ← θ(1 − lr·λ)), so it does not anchor a fine-tuned model to its base. With the Llama 2-Chat SFT values lr = 2e-5 and λ = 0.1 ([[llama-2-recipe]] §3.1), a hypothetical 10,000 steps at that LR shrink every decayed weight by (1 − 2e-6)^10,000 = 0.980 independent of the data.

**Conditions and limits.** The effective-LR explanation was tested at 124M parameters with 50,000 iterations ([[why-weight-decay-modern-dl]] §4 names the lack of large-scale experiments). Which parameter groups to exclude from decay is supported by an ablation only for embeddings.

**Implication for a general-purpose model.** In pretraining, weight decay is a stability and final-loss setting that interacts with the LR schedule and precision, so it is tuned together with them. In fine-tuning, weight decay is not a retention control in the cited evidence; §6 lists controls with measured retention effects.

## §4 β₂ and ε: what open reports chose and what has been measured

**Definition.** β₂ sets the memory of the second-moment average; ε sets a floor under the denominator √v̂. The *averaging timescale* of an exponential moving average with rate β is 1/(1 − β) steps; its *half-life* is ln 0.5 / ln β steps.

**Worked example 1 (timescales).** β = 0.9: timescale 10 steps, half-life 6.6 steps. β = 0.95: 20 and 13.5. β = 0.98: 50 and 34.3. β = 0.999: 1,000 and 692.8.

**Worked example 2 (one coordinate after a gradient change).** Start in steady state with g = 1, so m = v = 1 and the update equals α. At step 0 the gradient is 10 for one step, then returns to 1; β₁ = 0.9.
- β₂ = 0.999: v = 0.999 + 0.001 × 100 = 1.099, √v = 1.048; m = 0.9 + 1.0 = 1.9. The spike step is 1.9/1.048 = 1.81α. Updates stay above α through step +28, then fall slightly below α (0.96α at +50 and +100) while v remains elevated for hundreds of steps.
- β₂ = 0.95: v = 0.95 + 0.05 × 100 = 5.95, √v = 2.44. The spike step is 1.9/2.44 = 0.78α. Updates stay below α (minimum 0.65α at +14) and return to within 1% of α at step +108.
- Gradual rise instead of a spike (gradient grows from 1 to 10 over 20 steps and stays): the peak update is 5.30α for β₂ = 0.999 and 1.15α for β₂ = 0.95.

So a long β₂ memory produces steps larger than α while gradients *grow*; after a one-step spike, the elevated v makes subsequent steps *smaller* than α. The interactive figure [figures/beta2-memory.html](figures/beta2-memory.html) lets the reader vary the spike size, switch between a spike and a gradual rise, and read the update size for β₂ = 0.95, 0.98, and 0.999.

**Worked example 3 (ε).** In steady state the per-coordinate update is about α·r/(r + ε), where r is the coordinate's gradient RMS. For r = 1e-6: ε = 1e-8 gives 0.990α, ε = 1e-5 gives 0.091α. For r = 1e-8 and ε = 1e-8 the update is 0.5α.

**Evidence: values in reports.**
- GPT-3 (all versions, arXiv 2020-05): Adam β₁ = 0.9, β₂ = 0.95, ε = 1e-8, global-norm clipping 1.0, weight decay 0.1 ([[gpt-3-few-shot]] App. B). The value β₂ = 0.95 is therefore in the published record from 2020, before the later reports below.
- InstructGPT (arXiv 2022-03): all models Adam with β₁ = 0.9, β₂ = 0.95, fp16 weights with fp32 master copies ([[rlhf-instructgpt-optimizer-settings]] App. C).
- Llama 2 (2023-07): AdamW β₂ = 0.95, eps = 1e-5, weight decay 0.1, clipping 1.0 in pretraining and in PPO ([[llama-2-recipe]] §2.2, §3.2.3).
- DeepSeek-V3: AdamW β₂ = 0.95, weight decay 0.1, clipping 1.0 ([[deepseek-v3-recipe]] §4.2). Llama 3.1 405B: AdamW, betas and weight decay not printed ([[llama-3-recipe]] §3.4.1).
- OLMo 2 7B: betas (0.9, 0.95), ε 1e-8 ([[olmo-2-optimizer-settings]], config L44-L53). None of these reports gives an ablation of β₂.

**Evidence: measurements.**
- OLMo 2 lowered ε from 1e-5 to 1e-8: "The lower value allows for larger updates early in training ... the gradient norm settles much more quickly and remains permanently lower" (§3.4.1, Fig. 9, about 8,000 steps; model size of the ablation not stated) ([[olmo-2-optimizer-settings]]). Status: Result (single study).
- Gradient RMS of the first MLP layer falls with model size and LR and is around the default ε = 1e-8 at the largest scale and LR tested; for a 4.8B model at LR 0.3, ε = 1e-15 improved loss and ε = 1e-6 diverged ([[small-scale-proxies-instabilities]] §3.4, Figs. 11-12; one run per value). Status: Result (single study). OLMo 2 and this paper agree in direction (smaller ε helped); they measure different settings.
- In a one-at-a-time sweep of tuned AdamW at 130M parameters, 1× Chinchilla data, C4-EN loss: β₂ = 0.98 gave 3.529, 0.95 gave 3.535, 0.9 gave 3.545; β₁ = 0.95 gave 3.539 and β₁ = 0.98 gave 3.882; ε from 1e-25 to 1e-10 moved loss by at most 0.002 ([[fantastic-pretraining-optimizers]] Table 36). The paper accepts a hyperparameter change when validation loss improves by more than 3e-3 (§3.2); the 0.006 difference between β₂ = 0.95 and 0.98 is twice that threshold, and seeds are not reported. Status: Result (single study, small scale).

**Conditions and limits.** Which β₂ is best at frontier scale is an Open question in this chapter's sources. The small-scale sweep is at 130M parameters. [[small-scale-proxies-instabilities]] mentions β₂ only in related work (footnote 1, §4).

**Implication for a general-purpose model.** β₂ = 0.95 is a documented starting value, not a measured optimum at scale. In large runs, per-layer gradient RMS logged against ε shows whether ε limits the update. A steady rise of gradient magnitude, not a single spike, is the case in which a long β₂ memory gives large steps.

## §5 Gradient clipping by global norm

**Definition.** Global-norm clipping rescales the whole gradient vector when its L2 norm exceeds a threshold c, keeping its direction ([[gradient-clipping]] Algorithm 1).

**The problem it addresses.** Gradient norms spike during training: OLMo-0424 had spikes "in the loss, and more frequently, in the gradient norm" ([[olmo-2-optimizer-settings]] §3). Adam's first step after such a gradient can exceed α (§4, worked example 2), and for SGD the step grows in proportion to the gradient.

**Formula** ([[gradient-clipping]] §3.2):

```
ĝ ← ∂E/∂θ
if ‖ĝ‖ ≥ threshold:  ĝ ← (threshold / ‖ĝ‖) · ĝ
```

ĝ: gradient over all parameters. ‖ĝ‖: its L2 norm, the square root of the sum of squared entries of every tensor. threshold: c. PyTorch computes the same coefficient without a branch ([[pytorch-adamw-clip-amp]], `clip_grad.py` L94-L109):

```python
total_norm = torch.linalg.vector_norm(
    torch.stack([norm.to(first_device) for norm in norms]), norm_type
)
clip_coef = max_norm / (total_norm + 1e-6)
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
```

The function returns the pre-clip `total_norm` (L124), which is the value to log.

**Worked example 1 (global versus per-tensor).** Two tensors have gradient norms 3 and 4, so the global norm is √(9 + 16) = 5. With c = 1 the coefficient is 0.2 and the norms become 0.6 and 0.8; the ratio 3:4 is preserved. Clipping each tensor separately to 1 gives norms 1 and 1, which changes the ratio to 1:1 and so changes the update direction. Pascanu et al. report that their norm-based variant and Mikolov's element-wise variant "behave similarly" in their RNN experiments (§3.2); for Transformers, this chapter's sources give no comparison.

**Where clipping sits relative to loss unscaling.** fp16 training multiplies the loss by a loss scale S so that small gradients stay representable (ch-02). The PyTorch documentation states that clipping without unscaling makes the threshold "invalid", and gives the order `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()`; `unscale_` must be called once per step after all accumulated gradients are present ([[pytorch-adamw-clip-amp]], `amp_examples.rst` L67-L117). bf16 training does not use a GradScaler, so S = 1 and the order does not matter.

**Worked example 2 (clip before unscale).** GradScaler's initial scale is S = 2^16 = 65,536 (`grad_scaler.py` L122). Let the true gradient norm be 0.8 and c = 1.
1. Documented order: the norm after unscaling is 0.8 < 1, so the coefficient is 1 and the optimizer receives norm 0.8.
2. Clip before unscale: the clip sees 65,536 × 0.8 = 52,428.8, the coefficient is 1/52,428.8 = 1.91e-5, and after `scaler.step` unscales, the optimizer receives norm 0.8 × 1.91e-5 = 1.53e-5. Clipping triggers at every step; it is not a no-op.
3. Effect on AdamW (simplified: equal magnitude in every coordinate, N = 1e9 parameters): the per-coordinate gradient is 1.53e-5/√1e9 = 4.83e-10, below ε = 1e-8, so the update is about 4.83e-10/(4.83e-10 + 1e-8) = 0.046α. Adam's scale invariance (§1) would cancel a constant factor, but ε does not scale, so the run trains at about 5% of its intended step size.

The calculator [figures/clip-unscale-order.html](figures/clip-unscale-order.html) lets the reader change S, the gradient norm, c, N, and ε and compare what the optimizer receives under the two orders.

**Threshold choice and values in reports.** Pascanu et al. suggest setting the threshold from "statistics on the average norm over a sufficiently large number of updates" and found training "not very sensitive" to it in their RNN experiments ([[gradient-clipping]] §3.2). GPT-3, Llama 2, DeepSeek-V3, and OLMo 2 7B clip at 1.0 (§4 citations). In the 130M AdamW sweep, no clipping and a threshold of 2.0 both gave 3.534 versus 3.529 at 1.0 ([[fantastic-pretraining-optimizers]] Table 36). Kimi K2 and GLM-4.5 do not print a clipping value ([[kimi-k2-recipe]]; [[glm-4-5-recipe]]).

**Monitoring.** OLMo 2 reports that OLMo-0424 had spikes "in the loss, and more frequently, in the gradient norm", that "more dramatic spikes in gradient norm often preceded training loss spikes", and defines a *spike score*: "the percentage of values in a time series that are at least seven standard deviations away from a rolling average of the last 1,000 values"; a new initialization changed the gradient-norm spike score from 0.40 to 0.03 ([[olmo-2-optimizer-settings]], §3, §3.2). No source in this chapter gives a fixed lead time between a gradient-norm spike and a loss spike.

**Interaction with RL reward outliers.** The REINFORCE-style gradient is (1/N)·Σ_i A_i·∇log π(y_i | x), where A_i is the advantage of sample y_i, π the policy, and N the number of samples (ch-37). The gradient norm grows with the magnitude of the advantages, and an outlier reward also changes the *signs* of other samples' advantages. Worked example: five samples for one prompt with rewards [1, 0, 1, 0, 20].
1. Mean baseline: advantages [−3.4, −4.4, −3.4, −4.4, 15.6]. The two correct samples (reward 1) receive negative advantages, which is a negative gradient in the sense of ch-43a (their likelihood is pushed down).
2. Group normalization (divide by the population standard deviation 7.81; the sample standard deviation is 8.74; GRPO, ch-40): [−0.435, −0.563, −0.435, −0.563, 1.997]. The scale shrinks; the signs do not change.
3. Global-norm clipping rescales the summed gradient; it also leaves these signs unchanged.
4. Reward clipping at 10 (OpenRLHF's default range is (−10, 10), [[openrlhf-ppo-recipe]] `train_ppo_ray.py` L437): rewards [1, 0, 1, 0, 10], advantages [−1.4, −2.4, −1.4, −2.4, 7.6]; the signs are still flipped. Clipping at 2 gives [0.2, −0.8, 0.2, −0.8, 1.2], and the correct samples become positive again.

Gradient clipping bounds the step size; reward and advantage design determine the step direction. OpenRLHF's default gradient clip is 1.0 (L474), Llama 2-Chat PPO uses 1.0 ([[llama-2-recipe]] §3.2.3), and the RL's Razor LLM experiments use max grad norm 1 with the note that weight decay and max gradient norm were "manually ablated" with "no significant effect on results" ([[rls-razor]] App. B.1 Table 2).

**Implication for a general-purpose model.** The pre-clip global norm and the clip coefficient are the two values to log at every step. A clip coefficient below 1 on most steps is a signal to check the threshold, the loss-scaling order, and the reward scale, because each of these changes how far every update moves the model.

## §6 Update size in fine-tuning and retention of prior ability

**Definition.** *Update size* is how far a training stage moves the model from its starting checkpoint θ₀. It can be measured in weight space (for example ‖θ_T − θ₀‖) or in output space (for example E_{x∼τ}[KL(π₀ ‖ π)], the KL divergence between the base policy π₀ and the fine-tuned policy π on inputs x from the new task τ, [[rls-razor]] §1). *Forgetting* is a drop on held-out tasks the stage did not target. *Alignment tax* is forgetting caused by an alignment stage ([[rlhf-instructgpt-optimizer-settings]] §1). An *anchor* is a training term that pulls the model toward a reference distribution (for example pretraining text) or toward reference weights (for example the SFT checkpoint).

**The problem.** A fine-tuning stage improves a target and can lower held-out scores. Full fine-tuning of Llama-2-7B on Magicoder-Evol-Instruct-110K raised HumanEval pass@1 from 0.302 after 1 epoch to 0.497 after 8 epochs, after which it fell to 0.416 after 16, while the forgetting average (mean of HellaSwag, ARC-Challenge, WinoGrande; higher means less forgetting) fell from 0.595 to 0.446 to 0.414 ([[lora-learns-less-forgets-less]] Tables S5-S6).

**Mechanism.**
1. Adam bounds each coordinate's step by approximately α_t in the common case (§1; exceptions up to 5.30α with β₂ = 0.999 in §4), so the total movement of a coordinate after T steps is at most about Σ_{t=1..T} α_t.
2. The realized movement is smaller when gradient signs alternate (§1, worked example 1).
3. Weight decay shrinks weights toward zero, not toward θ₀ (§3).
4. The optimizer can restrict the form of the update (low rank in LoRA) or mix in an anchor direction (pretraining gradients, the SFT delta).

**Formula (upper bound).** |θ_{T,i} − θ_{0,i}| ≲ Σ_{t=1..T} α_t. θ_{T,i}: coordinate i after T steps. α_t: learning rate at step t. For a cosine schedule from α_max to 0 over T steps, Σα_t ≈ α_max·T/2.

**Worked example.** Constant α = 1e-5: 1,000 steps give a per-coordinate bound of 0.01; 16,000 steps give 0.16. A cosine schedule from 2e-5 to 0 over 10,000 steps gives about 2e-5 × 10,000 / 2 = 0.1. Doubling the epochs doubles T and doubles the bound at the same schedule shape.

**Evidence.**
- *Steps and epochs.* Forgetting "tends to worsen with training duration", instruction fine-tuning forgets more than continued pretraining, and code forgets more than math ([[lora-learns-less-forgets-less]] §4.2). Llama 3 observed that DPO on short-context data did not hurt long-context performance of a long-context SFT model and suspected this "is due to the fact that our DPO recipe has fewer optimizer steps than SFT" ([[llama-3]] §4.3.4; Interpretation by the authors).
- *Learning rate.* For InstructGPT PPO at 1.3B and 6B, a scan from 2.55e-6 to 2.55e-5 found that "All runs with learning rate greater than 8.05e-6 diverged" without the pretraining mix, and PPO with the mix was "less sensitive to change of the learning rate" ([[rlhf-instructgpt-optimizer-settings]] App. E.9).
- *Form of the update.* In code instruction tuning, LoRA rank 256 reached HumanEval 0.498 at epoch 4 with forgetting average 0.631, while full fine-tuning reached 0.497 at epoch 8 with 0.446; rank 16 learned less (0.358 at epoch 4) and kept 0.652 ([[lora-learns-less-forgets-less]] Tables S5-S6). The best LoRA learning rates were about 10× those of full fine-tuning (App. B). [[lora-without-regret]] (practitioner-evidence) reports that LoRA on all layers matches full fine-tuning loss on Tulu3 and OpenThoughts3 subsets when not capacity-limited, matches full fine-tuning in policy-gradient RL "even with ranks as low as 1", and has an optimal LR about 10× that of full fine-tuning (fit multiplier 9.8); it does not measure forgetting. Status for the 10× LR ratio: Replicated (two sources). Status for LoRA reducing forgetting at matched target accuracy: Result (single study, code).
- *Weight distance versus output distance.* On an MNIST-based toy task, a quadratic fit of forgetting on forward KL gave R² 0.96, on reverse KL 0.93, on weight change L1 0.34, and on Fisher-weighted L2 and spectral-norm weight change 0.58; for Qwen 2.5 3B-Instruct the KL fit gave R² 0.71 ([[rls-razor]] §4, §6 Table 1). The authors also found that apparently sparse RL weight updates came from bfloat16 weights, where "small parameter updates ... can fail to cross the representational threshold", and disappeared in float32 (§6). ch-30a records a study in which the KL-forgetting link "does not always hold". Status: Open question for LLMs.
- *Anchors inside optimization.* InstructGPT PPO-ptx adds γ·E_{x∼D_pretrain}[log π(x)] to the RL objective with γ = 27.8 and 8× as many pretraining examples as RL episodes; with γ ≥ 20 the regressions on public NLP datasets were recovered at 1.3B, while with γ = 0 raising the KL coefficient to 2.0, "100 times of the default value", did not fix them ([[rlhf-instructgpt-optimizer-settings]] §3.5, App. C.4, App. E.6). PPO-ptx still lagged GPT-3 on DROP, SQuADv2, and translation (§4.2).
- *Online Merging Optimizer.* At each step the update is combined with the SFT delta τ_r = θ_SFT − θ_base: θ^(t) = θ^(t−1) + (1 − α)·F_R(Δθ) + α·F_R(τ_r), where Δθ is the Adam update, F_R a random sparsification that keeps each entry with probability p, and α the merging weight ([[online-merging-optimizer]] Eq. 3-4). In off-policy DPO on UltraFeedback, OnDARE changed the benchmark average (Table 1 "Benchmark Avg.") relative to AdamW by +0.5 (Qwen1.5-1.8B, AdamW 41.8), +1.1 (Qwen1.5-7B, 58.4), and +1.3 (LLaMa-3-8B, 58.0), and MT-Bench by +0.24, +0.12, +0.19, where the authors estimate MT-Bench standard deviation at about 0.05 (Table 1, App. B). A KL-penalty baseline changed the averages by +0.4, +0.1, +1.0. Seeds per configuration are not reported. Qwen2.5 Instruct used this optimizer for 1 epoch of DPO at LR 7e-7 without printing its hyperparameters ([[qwen-2.5-recipe]] §4.2). Status: Result (single group).
- *Optimizer mismatch.* Moonlight checkpoints at 1.2T tokens fine-tuned for 2 epochs on tulu-3-sft-mixture scored MMLU 55.7 and GSM8K 68.0 with Muon pretraining and Muon SFT, and 50.2 and 64.9 with Muon pretraining and AdamW SFT; AdamW-pretrained checkpoints scored 52.0 and 64.6 with AdamW SFT ([[muon-scalable-moonlight]] Table 6). On Qwen2.5-7B (AdamW-pretrained), Muon SFT scored GSM8K 85.8 versus 89.8 and MMLU 70.8 versus 71.4 for Adam SFT; the authors describe these as on par (Table 7, §3.5.2). Kimi K2 used Muon for SFT and RL after Muon pretraining ([[kimi-k2-recipe]] §3.1, §3.2.3). Status: Result (single group).

**Conditions and limits.** The epoch and LoRA evidence uses one base model (Llama-2-7B), decoupled LionW instead of AdamW, and three multiple-choice retention benchmarks ([[lora-learns-less-forgets-less]] App. A). The online-merging gains are 0.5-1.3 points on averages without reported seeds. The Σα_t bound is loose and says nothing about which directions move.

**Implication for a general-purpose model.** Peak LR × steps is a quantity to choose from measurements: when compute allows a sweep, each candidate value is evaluated for target gain and for held-out retention on the suites fixed in ch-00. KL to the starting policy on target prompts is a monitor that needs no held-out data; the stop decision rests on held-out evaluations, because the KL-forgetting relation is an Open question for LLMs. When a stage is expected to regress broad ability, prefer a mechanism with measured retention (pretraining mix, low-rank updates, online merging, fewer steps) to weight decay.

## §7 Alternatives to AdamW, judged by downstream breadth

**Definitions** ([[fantastic-pretraining-optimizers]] §3; [[muon-scalable-moonlight]] §2).
- *Lion*: w_{t+1} = w_t − η·sign(β₂·m_t + (1 − β₂)·g_t); it keeps only a first moment.
- *Muon*: M_t = μM_{t−1} + ∇L_t; O_t = Newton-Schulz(M_t), which approximates UVᵀ for M_t = UΣVᵀ; W_t = W_{t−1} − η_t(0.2·O_t·√max(A, B) + λW_{t−1}) for an A × B matrix. The factor 0.2·√max(A, B) matches AdamW's typical update RMS of 0.2-0.4, so the AdamW learning rate and weight decay can be reused; AdamW still updates embeddings, the LM head, and normalization weights.
- *Matrix-based preconditioners* (Shampoo family, Soap, Kron) multiply gradients by matrices instead of per-entry scalars.

**The problem.** Papers proposing new optimizers reported 1.4-2× pretraining speedups over AdamW, measured as tokens to a target loss, often against baselines with fixed or untuned hyperparameters ([[fantastic-pretraining-optimizers]] §1).

**Evidence.**
- Tuning only the learning rate of the GPT-3 recipe for a 100M model gave AdamW itself up to a 2× speedup; Lion's optimal weight decay is about 0.6 versus about 0.1 for AdamW. Against a tuned AdamW, the best estimated speedup was 1.4×; matrix-based methods gave about 1.3× below 520M parameters and about 1.1× at 1.2B and 8× Chinchilla data, where they "no longer" gave downstream improvements. The 10-task average at 1.2B and 193B tokens was AdamW 67.15, NAdamW 66.70, Muon 66.98. A scaling-law fit predicts slightly higher loss for Muon than AdamW at 7B and 1× Chinchilla. Rankings between optimizers can flip during learning-rate decay ([[fantastic-pretraining-optimizers]] Abstract, §4.1, Table 5, Fig. 1).
- Moonshot AI reports that Muon needs about 52% of AdamW's training FLOPs under compute-optimal training ([[muon-scalable-moonlight]] §3.2, Fig. 1a). At 1.2T tokens, before cooldown, Muon versus AdamW on the same 2.24B-activated MoE: MMLU 60.4 vs 60.2, BBH 43.2 vs 45.3, HumanEval 37.2 vs 29.3, MATH 19.8 vs 16.1, CMMLU 58.8 vs 58.2 (Table 4). Gains are concentrated in code and math; BBH is lower.
- Production use: Kimi K2 (1.04T total, 32B activated) was pre-trained on 15.5T tokens with MuonClip, which rescales query and key weights when a head's maximum attention logit exceeds τ = 100; a mid-scale vanilla Muon run exceeded logits of 1,000 ([[kimi-k2]] §2.1). GLM-4.5 (355B total, 32B activated) used Muon with update RMS 0.2 ([[glm-4-5-recipe]] §2.4). Neither report trains an AdamW counterpart, so neither measures a downstream difference caused by the optimizer.

Status: the size of Muon's advantage at frontier scale is an Open question; the two controlled studies disagree (about 2× compute efficiency in [[muon-scalable-moonlight]]; 1.1-1.4× after per-optimizer tuning up to 1.2B in [[fantastic-pretraining-optimizers]]), and they differ in tuning protocol, data-to-model ratio, and scale.

**Conditions and limits.** The controlled comparison stops at 1.2B parameters; its downstream suite is base-model multiple-choice and cloze tasks. Moonlight's Table 4 compares one run per optimizer at an intermediate checkpoint.

**Implication for a general-purpose model.** An optimizer change is a pipeline change: it alters hyperparameter optima, the per-domain profile of downstream scores, and the fine-tuning optimizer that works best afterward (§6). Evaluate it on the full breadth suite at the end of training and after SFT, not on pretraining loss alone.

## Recipe

All rows are optimizer-related settings as printed. Status dates: 2026-09-14 rows were verified in the linked library card; 2026-09-15 rows were read at the locus for this chapter.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GPT-3 (all versions) | all sizes in the report | pretrain-stable | Adam β₁, β₂, ε; clipping; weight decay | 0.9, 0.95, 1e-8; global norm 1.0; 0.1 | arXiv:2005.14165v4 App. B ([[gpt-3-few-shot]]) | verified 2026-09-15 | no ablation reported |
| InstructGPT SFT | 1.3B, 6B / 175B | SFT | optimizer; LR; batch; epochs; schedule | Adam (0.9, 0.95); 9.65e-6 / 5.03e-6; 32 / 8; 16 epochs; cosine to 10%, no warmup | arXiv:2203.02155v1 App. C, C.1 ([[rlhf-instructgpt-optimizer-settings]]) | verified 2026-09-15 | geometric search over 7 LRs (1.3B, 6B) and 5 LRs (175B), epochs by geometric search; final model chosen by RM score (§3.5) |
| InstructGPT PPO / PPO-ptx | 1.3B, 6B | RL | LR scan; divergence | 2.55e-6 to 2.55e-5; PPO without ptx diverged above 8.05e-6 | arXiv:2203.02155v1 App. E.9 | verified 2026-09-15 | Fig. 38 human evaluations; final checkpoints by Likert score; single final LR not printed |
| InstructGPT PPO-ptx | 1.3B-175B | RL | KL β; ptx γ; pretraining examples | 0.02; 27.8; 8× the RL episodes | arXiv:2203.02155v1 App. C.4 | verified 2026-09-15 | App. E.6: γ ≥ 20 recovers regressions at 1.3B; β up to 2.0 with γ = 0 does not |
| Llama 2 | 7B-70B | pretrain-stable | AdamW β₁, β₂, eps; weight decay; clipping | 0.9, 0.95, 1e-5; 0.1; 1.0 | arXiv:2307.09288v2 §2.2 ([[llama-2-recipe]]) | verified 2026-09-14 | no ablation reported |
| Llama 2-Chat | all | SFT | LR; weight decay; batch; epochs | 2e-5 cosine; 0.1; 64; 2 | arXiv:2307.09288v2 §3.1 ([[llama-2-recipe]]) | verified 2026-09-14 | no ablation reported |
| Llama 2-Chat | all | RL | optimizer; LR | AdamW (0.9, 0.95), eps 1e-5, weight decay 0.1, clipping 1.0; constant 1e-6 | arXiv:2307.09288v2 §3.2.3 ([[llama-2-recipe]]) | verified 2026-09-14 | no ablation reported |
| Llama 3 scaling-law models | 40M-16B | pretrain-stable | weight decay | "0.1 times the learning rate at that step" | arXiv:2407.21783v3 §3.2.1 ([[llama-3-recipe]]) | verified 2026-09-14; wording re-read 2026-09-15 | n/a |
| Llama 3.1 405B | 405B | pretrain-stable | optimizer; betas, weight decay, clipping | AdamW; not printed | arXiv:2407.21783v3 §3.4.1 ([[llama-3-recipe]]) | verified 2026-09-14 (optimizer); not reported (betas, decay, clipping; body checked by card) | n/a |
| Llama 3.1 ("largest models") | 405B | SFT; preference | LR; steps | SFT 1e-5 over 8.5K-9K steps; DPO 1e-5 | arXiv:2407.21783v3 §4.1.3, §4.1.4 ([[llama-3-recipe]]) | verified 2026-09-14 | SFT values "work well across different rounds and data mixes"; no numbers |
| OLMo 2 7B | 7B | pretrain-stable | AdamW betas; ε; weight decay; decay groups; clipping; peak LR | (0.9, 0.95); 1e-8; 0.1; `decay_norm_and_bias: true`, `decay_embeddings: false`; 1.0; 3.0e-4 | github.com/allenai/OLMo@600a8d0 configs/official-1124/OLMo2-7B-stage1.yaml L44-L53, L90 ([[olmo-2-optimizer-settings]]) | verified 2026-09-15 | ε: arXiv:2501.00656v3 §3.4.1 Fig. 9 (1e-8 vs 1e-5); embeddings: §3.4.2 Fig. 10 (spike score 0.092 vs 0.16) |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | AdamW β₁, β₂; weight decay; clipping | 0.9, 0.95; 0.1; 1.0 | arXiv:2412.19437v2 §4.2 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | no ablation reported |
| Kimi-K2-Base | 1.04T / 32B act. | pretrain-stable | optimizer; weight decay; QK-Clip τ; gradient clipping | MuonClip; 0.1; 100; not printed | arXiv:2507.20534v2 §2.1, §2.5 ([[kimi-k2-recipe]]) | verified 2026-09-14; clipping not reported | App. D: τ = 30 on a 0.5B-activated / 3B-total MoE, negligible loss change; Muon vs AdamW not re-ablated |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT; RL | optimizer | Muon | arXiv:2507.20534v2 §3.1, §3.2.3 ([[kimi-k2-recipe]]) | verified 2026-09-14 | cites Moonlight for fine-tuning Muon-pretrained checkpoints with Muon |
| GLM-4.5 | 355B / 32B act. | pretrain | optimizer; weight decay | Muon (N = 5, μ = 0.95, update RMS 0.2) except word embedding, bias, RMSNorm weights; 0.1 | arXiv:2508.06471v1 §2.4 ([[glm-4-5-recipe]]) | verified 2026-09-14 | Muon "can accelerate convergence and tolerate larger batch sizes"; no numbers |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | preference | optimizer; LR; epochs | Online Merging Optimizer; 7e-7; 1; its hyperparameters not given | arXiv:2412.15115v2 §4.2 ([[qwen-2.5-recipe]]) | verified 2026-09-14 | no ablation reported |
| Moonlight | 2.24B act. / 15.29B total | pretrain-stable | optimizer; update scale; weight decay; peak LR | Muon with 0.2·√max(A, B) scaling; 0.1; 4.2e-4 | arXiv:2502.16982v1 §2.2 Eq. 4, §3.3 ([[muon-scalable-moonlight]]) | verified 2026-09-15 | Table 4: vs Moonlight-A (AdamW) at 1.2T tokens, one run each |
| Small-scale proxy default (Wortsman et al.) | 2.4M-1.2B | pretrain-stable | AdamW β₁, β₂, ε; clipping; weight decay | 0.9, 0.95, 1e-8; global norm 1; independent 1e-4 | arXiv:2309.14322v2 §2.1 ([[small-scale-proxies-instabilities]]) | verified 2026-09-14 | Fig. 6 (independent decay: lower LR sensitivity); ε = 1e-15 only for a 4.8B run at LR 0.3 (Fig. 12) |
| Tuned AdamW baseline (Wen et al.) | 130M, 1× Chinchilla | pretrain-stable | β₁, β₂, ε, LR, clipping, weight decay, batch, warmup | 0.9, 0.98, 1e-20, 0.008, 1, 0.1, 128 sequences of 4,096 tokens, 2,000 steps | arXiv:2509.02046v2 Table 36 ([[fantastic-pretraining-optimizers]]) | verified 2026-09-15 | Table 36 one-at-a-time sweep on C4-EN loss; seeds not reported |
| Llama-2-7B code IFT (Biderman et al.) | 7B | SFT | optimizer; LoRA LR; weight decay; clipping; batch | decoupled LionW (0.9, 0.95); 2e-4 (r = 16, 64), 1e-4 (r = 256); 0; norm 1; 192 | arXiv:2405.09673v2 App. A ([[lora-learns-less-forgets-less]]) | verified 2026-09-15 | App. B Fig. S1: best full fine-tuning LR 5e-5 (code), best LoRA 5e-4, 2 epochs |
| RL's Razor LLM runs | Qwen 2.5 3B-Instruct | SFT; RL | AdamW; clipping; weight decay; LR sweeps | max grad norm 1; 0; SFT {1e-5 … 9e-5}; GRPO {1e-5 … 5e-5}, KL 0 | arXiv:2509.04259v1 App. B.1 Table 2 ([[rls-razor]]) | verified 2026-09-15 | weight decay and clipping "manually ablated", no significant effect |
| OpenRLHF default (no model) | any | RL | gradient clipping; reward clip | 1.0; (−10, 10) | OpenRLHF@64c1cc4 `openrlhf/cli/train_ppo_ray.py` L474, L437 ([[openrlhf-ppo-recipe]]) | verified 2026-09-14 | no ablation reported (framework default) |

Starting point for a small general-purpose run. For a dense decoder-only pretraining run in the OLMo 2 7B regime (7B parameters, batches of 1,024 × 4,096 tokens, a 5T-token cosine schedule truncated at 4T), the verified OLMo 2 7B row gives AdamW with betas (0.9, 0.95), ε 1e-8, weight decay 0.1 in the coupled form with embeddings excluded, global-norm clipping 1.0, and peak LR 3.0e-4. For models near 130M parameters trained on 1× Chinchilla data, the verified tuned row differs (β₂ 0.98, LR 0.008, batch 128), so values should be re-tuned at that scale rather than copied from the 7B row. For fine-tuning, the verified rows are size-scoped: Llama 2-Chat SFT used LR 2e-5 with cosine decay, weight decay 0.1, batch 64, and 2 epochs, and PPO used a constant LR 1e-6 with clipping 1.0 (7B-70B); Qwen2.5 DPO used 1 epoch at LR 7e-7 with the Online Merging Optimizer (0.5B-72B). No verified row states a fine-tuning LR that was chosen by a held-out retention measurement, so the budget should be swept as described in §6.

## Generalization lens

**(a) What increases breadth.**
- Per-coordinate normalized updates learn rare-token classes that gradient descent leaves at high loss within the same number of steps ([[heavy-tailed-class-imbalance-adam]] Fig. 1; training loss, not downstream breadth).
- Weight decay lowers final pretraining loss through the effective LR and prevents late bf16 divergence ([[why-weight-decay-modern-dl]] Figs. 6-8); lower C4 loss tracked HellaSwag accuracy in the optimizer benchmark ([[fantastic-pretraining-optimizers]] §4.1).
- Muon at 1.2T tokens raised code and math scores relative to AdamW on the same model (HumanEval 37.2 vs 29.3, MATH 19.8 vs 16.1) with smaller changes on MMLU and a drop on BBH ([[muon-scalable-moonlight]] Table 4).

**(b) What causes narrowing or forgetting.**
- More epochs of full fine-tuning: forgetting average 0.595 → 0.414 from 1 to 16 epochs ([[lora-learns-less-forgets-less]] Table S6); fewer unique HumanEval generations after full fine-tuning (§4.5).
- High RL learning rates: InstructGPT PPO without ptx diverged above 8.05e-6 ([[rlhf-instructgpt-optimizer-settings]] App. E.9).
- RLHF without a pretraining anchor: regressions on SQuAD, DROP, HellaSwag, and WMT French-English ([[rlhf-instructgpt-optimizer-settings]] §1).
- A fine-tuning optimizer that differs from the pretraining optimizer: Muon-pretrained Moonlight scored MMLU 50.2 after AdamW SFT versus 55.7 after Muon SFT ([[muon-scalable-moonlight]] Table 6).
- Mechanisms that do not help retention in the cited settings: weight decay and dropout in fine-tuning ([[lora-learns-less-forgets-less]] §4.5); a larger KL coefficient alone in InstructGPT ([[rlhf-instructgpt-optimizer-settings]] App. E.6).

**(c) How to measure it for this stage.**
- Pretraining: loss by token-frequency bucket (the grouping of [[heavy-tailed-class-imbalance-adam]] Fig. 1), final-checkpoint downstream suites with base-model formats (ch-00), gradient-norm spike score ([[olmo-2-optimizer-settings]] §3.2), and per-layer gradient RMS against ε ([[small-scale-proxies-instabilities]] §3.4).
- Optimizer comparisons: end-of-training comparisons after per-optimizer tuning, because rankings flip during LR decay ([[fantastic-pretraining-optimizers]] Abstract).
- Fine-tuning: a target-gain versus held-out-retention curve over LR and epochs ([[lora-learns-less-forgets-less]] Fig. 3), KL(π₀ ‖ π) on target prompts as a monitor ([[rls-razor]] §4), and output diversity (unique generations per prompt, [[lora-learns-less-forgets-less]] §4.5). Report paired uncertainty as defined in ch-00; the online-merging gains of 0.5-1.3 points are of the size that needs it ([[online-merging-optimizer]] Table 1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Using `torch.optim.Adam(weight_decay=…)` while intending AdamW | No error; parameters with small gradients shrink faster than parameters with large gradients (1,000× in §1, worked example 2) | Confirm the optimizer class; `adam.py` L366-L367 adds the decay to the gradient |
| Copying λ from a report without matching the decay form | Per-step decay differs by a factor of the LR (1e-4 independent vs 3e-5 coupled at LR 3e-4) | Compute lr_t·λ or η_t·λ from the training code; log parameter norms |
| Decaying token embeddings with λ = 0.1 | Embedding norm falls, gradient norm rises, more spikes (spike score 0.16 vs 0.092) | Log embedding L2 norm and gradient-norm spike score ([[olmo-2-optimizer-settings]] §3.4.2) |
| ε = 1e-5 in a large pretraining run | Slower early loss decrease and a higher gradient norm ([[olmo-2-optimizer-settings]] Fig. 9) | Log per-layer gradient RMS and compare it with ε |
| Clipping before `scaler.unscale_` under fp16 | Pre-clip norm logged near S·‖g‖ (for example 5e4); clip coefficient near 1/(S·‖g‖), for example 1.9e-5, on every step; slow loss decrease | Assert the order `backward → unscale_ → clip → step → update`; log the clip coefficient ([[pytorch-adamw-clip-amp]]) |
| Calling `unscale_` before gradient accumulation finishes | `RuntimeError` on a second call, or clipping applied to a partial gradient | Clip once per optimizer step on the accumulated gradient (`amp_examples.rst` L114-L117) |
| Reading β₂ = 0.999 as "steps too large after a spike" | Wrong fix chosen for a divergence that follows a gradual gradient rise | Simulate or log m̂/√v̂ per layer; post-spike steps are smaller, rising gradients give large steps (§4) |
| Treating global clipping as protection against reward outliers | Correct RL samples receive negative advantages; clip coefficient normal | Log advantage signs and reward range per group; clip or normalize rewards (§5) |
| Adding epochs because the target metric still improves | Held-out retention falls (0.595 → 0.414) while HumanEval peaks and declines | Evaluate a retention suite at every epoch ([[lora-learns-less-forgets-less]] Tables S5-S6) |
| Using weight-space distance as the only forgetting monitor | Large weight changes without forgetting and small ones with it | Log KL(π₀ ‖ π) on target prompts and run held-out evaluations ([[rls-razor]] Table 1) |
| Declaring a new optimizer better from mid-run loss or an untuned baseline | The ranking reverses after LR decay or after tuning AdamW's LR | Compare at the end of training with per-optimizer sweeps and a breadth suite ([[fantastic-pretraining-optimizers]] Fig. 1, §4.2) |
| Fine-tuning a checkpoint with a different optimizer than it was pretrained with, without a check | Lower MMLU and GSM8K than with the matching optimizer | Run a short SFT with both optimizers ([[muon-scalable-moonlight]] Tables 6-7) |
| Pure bf16 weights in low-LR RL | A share of parameters never changes (sparse updates) | Compare update sparsity with a float32 or master-weight run ([[rls-razor]] §6; ch-02) |

## Check your understanding

1. Adam's step is invariant to multiplying every gradient by a constant, yet clipping before unscaling under fp16 reduces AdamW's update to about 5% of its intended size in the §5 example. Explain which term in the update breaks the invariance and why the effect grows with the number of parameters.
2. Using the weighted quadratic model of §2, explain why raising the learning rate of gradient descent cannot close the gap on rare classes, and what property of sign descent removes the dependence on π_k.
3. [[why-weight-decay-modern-dl]] reproduces a weight-decay run with a no-decay run whose LR schedule follows the effective learning rate, but only in float32. What does this imply about which of weight decay's two effects is replaceable by a schedule, and why does it matter for bf16 training?
4. In the §4 example, β₂ = 0.999 gives a larger step than β₂ = 0.95 at the spike but smaller steps 100 steps later. Trace m and v through both phases and state which gradient pattern makes a long β₂ memory risky.
5. InstructGPT found that a 100× larger KL coefficient did not remove regressions that a pretraining-gradient mix removed. Explain the difference between the two anchors in terms of which distribution each term constrains, and relate it to the Online Merging Optimizer's use of the SFT delta.
6. [[rls-razor]] finds weight-change L1 a weak predictor of forgetting (R² 0.34) while the Σα_t bound in §6 is stated in weight space. Explain why both statements can hold, and what you would log during a fine-tuning run to decide when to stop.
7. Moonlight reports about 2× compute efficiency for Muon, while [[fantastic-pretraining-optimizers]] reports about 1.1× at 1.2B parameters. List the protocol differences that could produce both results, and design an evaluation that would decide whether to switch a general-purpose pretraining run to Muon, including the SFT stage.

## Connections

- Previous: **ch-00 — What General Capability Means and How It Is Measured.** Supplies the held-out suites, retention reporting, and paired-uncertainty format used in §6 and the Generalization lens.
- Next: **ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch.** Covers fp16 loss scaling, bf16, and fp8, which determine whether the clipping order of §5 applies and whether small updates are representable (§6).
- **ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization.** Schedules set α_t in the §6 budget; μP and initialization are treated there, not here.
- **ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen.** Computing the global gradient norm across shards.
- **ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining** and **ch-13a — Multilingual Coverage and Vocabulary as Capability Axes.** The data side of rare-token learning (§2).
- **ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures.** Full pretraining recipe comparison.
- **ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate** and **ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control.** Measurement and control of the forgetting evidence summarized in §6.
- **ch-30c — Weight Averaging and Model Merging for Generalist Models.** Offline merging, the post-hoc counterpart of the Online Merging Optimizer.
- **ch-37 — Policy-Gradient Foundations for Language Models**, **ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax**, **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity**, **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO**, **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** The RL objective, PPO-ptx, the KL-forgetting question, group normalization, and negative advantages used in §5-§6.

## Sources

- [[adam]] — chapter excerpt checked against Adam arXiv:1412.6980v9 (Algorithm 1, §2.1 step bound and scale invariance, §3 bias correction) and AdamW arXiv:1711.05101v3 (Algorithm 2, Propositions 1-2, image-classification evidence). The library card `classics/adam.md` had not been re-verified on 2026-09-15.
- [[gradient-clipping]] — chapter excerpt of Pascanu, Mikolov, Bengio arXiv:1211.5063v2: Algorithm 1 and threshold heuristic (§3.2). The library card had not been re-verified.
- [[pytorch-adamw-clip-amp]] — PyTorch v2.5.0 AdamW and Adam steps, `clip_grad_norm_`, GradScaler initial scale, and the documented unscale-then-clip order.
- [[heavy-tailed-class-imbalance-adam]] — frequency-group training loss for SGD vs Adam, weighted quadratic model, sign descent, reweighting.
- [[why-weight-decay-modern-dl]] — weight decay in one-pass GPT-2-124M training: effective LR, final loss, bf16 divergence, L2 equivalence for this effect.
- [[small-scale-proxies-instabilities]] — independent versus coupled weight decay, gradient RMS near ε, ε = 1e-15 at 4.8B, default AdamW and clipping settings.
- [[gpt-3-few-shot]] — GPT-3 App. B optimizer, clipping, and weight-decay values.
- [[rlhf-instructgpt]] — library card for InstructGPT (unverified on 2026-09-15; its hyperparameter table is not used).
- [[rlhf-instructgpt-optimizer-settings]] — InstructGPT Adam betas, SFT epochs and overfitting, RM epochs, PPO LR divergence, PPO-ptx coefficients and ablations.
- [[llama-2-recipe]] — Llama 2 pretraining, SFT, and PPO optimizer rows.
- [[llama-3]] — Llama 3 observation that short-context DPO did not hurt long-context ability, attributed by the authors to fewer optimizer steps (§4.3.4).
- [[llama-3-recipe]] — Llama 3 scaling-law weight decay wording, 405B optimizer disclosure, SFT and DPO learning rates.
- [[olmo-2]] — library card for the OLMo 2 report (unverified on 2026-09-15); linked for the report itself.
- [[olmo-2-optimizer-settings]] — OLMo 2 7B released optimizer config, ε and embedding-decay ablations, spike score, LR crossover.
- [[deepseek-v3-recipe]] — DeepSeek-V3 AdamW betas, weight decay, and clipping.
- [[kimi-k2]] — MuonClip mechanism and the vanilla Muon logit growth observation.
- [[kimi-k2-recipe]] — Kimi K2 optimizer, weight decay, τ, and Muon in SFT and RL.
- [[glm-4-5-recipe]] — GLM-4.5 Muon settings and parameter groups.
- [[qwen-2.5-recipe]] — Qwen2.5 DPO with the Online Merging Optimizer at LR 7e-7.
- [[lora-learns-less-forgets-less]] — epoch-by-epoch learning and forgetting for LoRA and full fine-tuning, weight decay as a forgetting control, LR sweeps.
- [[lora-without-regret]] — LoRA versus full fine-tuning in SFT and RL, 10× learning-rate ratio (practitioner-evidence).
- [[rls-razor]] — KL on new-task inputs versus weight-space measures as forgetting predictors; bf16 update sparsity; LLM optimizer settings.
- [[online-merging-optimizer]] — OnDARE/OnTIES update rule, DPO results against AdamW and regularization baselines, hyperparameter effects.
- [[fantastic-pretraining-optimizers]] — tuned comparison of eleven optimizers up to 1.2B, downstream Table 5, AdamW sweep Table 36.
- [[muon-scalable-moonlight]] — Muon update with weight decay and RMS matching, compute-efficiency claim, Moonlight vs Moonlight-A downstream scores, SFT optimizer mismatch.
- [[openrlhf-ppo-recipe]] — OpenRLHF default gradient clipping and reward clipping.
