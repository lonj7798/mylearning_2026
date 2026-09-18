<!-- scope: Adam optimizer (primary artifact) and its AdamW decoupled-weight-decay variant
     deps: [[gradient-clipping]]
     see-also: [[lr-schedules]], [[mixed-precision]], [[weight-init]]
-->

# Adam: A Method for Stochastic Optimization
- **Core Insight:** Adam rescales each parameter's step by bias-corrected running estimates of the first and second gradient moments, which makes the step approximately bounded by the stepsize hyperparameter α and invariant to a diagonal rescaling of the gradients (Abstract; §2.1).
- **Guideline:** When using an adaptive optimizer with weight-decay regularization, use the decoupled form (AdamW, arXiv:1711.05101 Algorithm 2 line 12), because L2 regularization added to the gradient is scaled down by the adaptive preconditioner for the weights with large second moments (§2, Proposition 2). Otherwise the optimal λ and the optimal α stay coupled (Figure 2).
- **Authors:** Diederik P. Kingma, Jimmy Lei Ba (Adam). Secondary artifact covered here: Ilya Loshchilov, Frank Hutter (AdamW).
- **Year:** 2014 (arXiv v1 2014-12; ICLR 2015). Secondary: AdamW 2017 (arXiv v1 2017-11; ICLR 2019).
- **URL:** https://arxiv.org/abs/1412.6980 (primary). Secondary: https://arxiv.org/abs/1711.05101
- **Source type:** paper
- **Relevant topics:** optimization, adaptive methods, weight decay, regularization

## Abstract
Adam is a first-order method for stochastic objectives that keeps exponential moving averages of the
gradient (first moment `m_t`) and of the elementwise squared gradient (second raw moment `v_t`),
bias-corrects both, and takes the ratio as the update direction. The paper states it is computationally
efficient, has small memory requirements, is invariant to diagonal rescaling of the gradients, handles
non-stationary objectives and sparse or noisy gradients, gives a regret bound under the online convex
optimization framework, and introduces AdaMax, an infinity-norm variant (Abstract). The AdamW paper
shows L2 regularization and weight decay are equivalent for plain SGD up to a rescaling by the learning
rate but not for adaptive methods, and applies the decay directly in the parameter update (§2).

## Key Contributions
- Bias correction of both moment estimates (Algorithm 1; §3 derives the `1/(1 − β₂^t)` factor).
- Effective-step analysis: with ε = 0, `|Δ_t| ≤ α·(1−β₁)/√(1−β₂)` when `(1−β₁) > √(1−β₂)`, else
  `|Δ_t| ≤ α` (§2.1).
- Defaults reported for the tested problems: `α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8` (Alg. 1 caption).
- AdaMax, with `v_t` replaced by an exponentially weighted infinity norm, giving `|Δ_t| ≤ α` (§7.1).
- (AdamW) Propositions 1-3 separating weight decay from L2 regularization for SGD, for a general
  adaptive optimizer, and for an adaptive optimizer with a fixed preconditioner (§2).

## Key Figures/Tables to Study
- **Adam Algorithm 1** (p. 2): the full update, including the two bias-correction lines.
- **Adam Figures 1-3** (§6): training cost on logistic regression (MNIST, IMDB), multilayer nets and a
  CIFAR-10 convnet, vs AdaGrad, RMSProp, SGDNesterov, AdaDelta.
- **AdamW Algorithm 2** (arXiv:1711.05101 p. 3): L2-Adam and AdamW in one box; only line 6 and line 12
  differ. **AdamW Figure 2** (§4.2): CIFAR-10 test error of a 26 2x64d ResNet over an (α, λ) grid.

## Technical Details
**Adam update** (Algorithm 1, timestep `t`, gradient `g_t`; all operations elementwise):
```
m_t   = β₁·m_{t-1} + (1 − β₁)·g_t
v_t   = β₂·v_{t-1} + (1 − β₂)·g_t²
m̂_t   = m_t / (1 − β₁^t)
v̂_t   = v_t / (1 − β₂^t)
θ_t   = θ_{t-1} − α · m̂_t / (√v̂_t + ε)
```
`α` is the stepsize, `β₁, β₂ ∈ [0,1)` the moment decay rates, `ε` a numerical floor (Algorithm 1).

**AdamW update** (arXiv:1711.05101 Algorithm 2 line 12):
```
θ_t = θ_{t-1} − η_t · ( α·m̂_t/(√v̂_t + ε) + λ·θ_{t-1} )
```
`η_t` is the schedule multiplier returned by `SetScheduleMultiplier(t)` (line 11), `λ` the weight-decay
factor (line 1). The decay term sits outside the adaptive ratio but still inside `η_t`, so it follows the
learning-rate schedule; it is not fed into `m_t` or `v_t` as it is in L2-Adam (line 6).

**Reported numbers.** AdamW reaches 15% relative improvement in test error over L2-Adam, across
CIFAR-10 and ImageNet32x32, budgets of 100-1800 epochs, and fixed / step-drop / cosine schedules
(arXiv:1711.05101 §1, Figures 1-3). **Result (single study), image classification only.** Those grids
used 26 2x64d and 26 2x96d ResNets (11.6M and 25.6M parameters) at batch size 128 (§4). Adam's own
experiments are logistic regression, a 2-layer fully connected net, and a CIFAR-10 convnet (§6.1-§6.3);
no language model is trained in either paper.

**β₂ = 0.95 in LLM pretraining.** This is a later practice, not a claim of either paper. GPT-3 reports
"Adam with β₁ = 0.9, β₂ = 0.95, and ε = 1e-8", global gradient-norm clipping at 1.0, and weight decay
0.1 for all model sizes (arXiv:2005.14165 App. B). The Adam paper's own defaults keep β₂ = 0.999
(Algorithm 1 caption). **Replicated** in later reports; the earliest locus in this library is GPT-3
(2020-05), not Llama 1.

**Optimizer-state memory.** Adam keeps `m_t` and `v_t` per parameter (Algorithm 1): 8 bytes per
parameter in fp32, 4x the parameter bytes when parameters are bf16. **Derived**, not stated.

## Recipe ledger
Moved to [[adam-recipe]]: paper defaults, plus the GPT-3 pretraining settings.

## Findings relevant to generality
The AdamW paper's generality claim is about held-out test error on image classification: decoupled
weight decay "yields substantially better generalization performance" than L2 regularization in Adam
(§5), measured as CIFAR-10 and ImageNet32x32 test error (Figures 2-3). Neither paper measures retention
of prior capability, forgetting, output diversity, or any language-model evaluation. Claims about
optimizer settings and breadth of capability in SFT or RL must come from reports that run those stages.

## Connections
- **[[lr-schedules]]**: AdamW's `η_t` multiplies both the adaptive step and the decay term (Alg. 2
  lines 11-12), so the schedule changes the effective decay per step.
- **[[gradient-clipping]]**: GPT-3 pairs these settings with global-norm clipping at 1.0 (App. B).
- **[[mixed-precision]]**: the moment estimates and master parameters are the tensors whose precision
  choice is discussed there; neither paper studies reduced precision.
- **[[weight-init]]**: width-dependent learning-rate rules belong to the parameterization, not to Adam.

## Verification
- Checked on 2026-09-18 against https://arxiv.org/abs/1412.6980 (v9, 2017-01-30),
  https://arxiv.org/abs/1711.05101 (v3, 2019-01-04), and https://arxiv.org/abs/2005.14165 App. B.
- Corrections to the previous card version:
  - "β₂ = 0.95 … was Llama 1's choice" → GPT-3 already reports β₁ = 0.9, β₂ = 0.95 in 2020-05
    (arXiv:2005.14165 App. B), three years before Llama 1.
  - "λ is **independent of learning rate**" → the update multiplies the decay term by the schedule
    multiplier `η_t` (Algorithm 2 line 12); the paper claims the *optimal settings* become "much more
    independent" / "largely decoupled" (§1, §4.2). The card's formula also omitted `η_t`; restored.
  - "AdamW Algorithm 2 vs L2-Adam" implied two algorithm boxes → both variants are printed in the
    single Algorithm 2 box (lines 6 and 12).
  - "makes Adam competitive with SGD-momentum on image classification", no number → replaced with the
    reported 15% relative test-error improvement over L2-Adam (§1).
  - "AdamW Figure 1: loss surfaces" → Figure 1 shows final test error under three LR schedules; the
    (α, λ) grid comparison is Figure 2 (§4.2).
- Removed as unsupported by the source:
  - "μP requires AdamW LR to **not** scale with width (unlike SGD)" — not in either paper.
  - Convergence-proof flaw "fixed by AMSGrad" — a claim of Reddi et al. 2018, not of this paper.
  - "~0.1–0.5% perplexity difference" for the no-decay group; "bump ε to 1e-5 on NaN" and the fp16
    division-by-zero mechanism — no source for any of these.
  - The "Modern LLM defaults (GPT-3 / Llama / Chinchilla / Qwen lineage)" table — it merged model
    families into single cells and gave unsourced SFT and RL ranges. The GPT-3 rows are kept in
    [[adam-recipe]] with their locus; other per-model values belong in each model's own card.
  - "Lion / Sophia / Shampoo … as of 2025 AdamW remains the default at frontier scale" — no source.
  - "β₂ = 0.95 tracks loss-landscape curvature and reacts faster to phase changes" — not reported.
- Not reported by the source: any language-model experiment; behaviour at LLM batch sizes or parameter
  counts; interaction with mixed precision; effect on task breadth or forgetting.
