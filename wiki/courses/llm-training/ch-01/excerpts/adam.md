---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "Adam: A Method for Stochastic Optimization (Kingma & Ba) + Decoupled Weight Decay Regularization (Loshchilov & Hutter)"
source_url: https://arxiv.org/abs/1412.6980
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Adam and AdamW, checked against the primary papers

**Adam:** Diederik P. Kingma, Jimmy Ba. arXiv:1412.6980 (v1 2014-12; read v9, 2017-01-30). ICLR 2015.
**AdamW:** Ilya Loshchilov, Frank Hutter. arXiv:1711.05101 (v1 2017-11; read v3, 2019-01-04). ICLR 2019.
**Status:** rewritten on 2026-09-15 from the two PDFs. The library card `classics/adam.md` had no Verification section on that date; its β₂ history, μP connection, and "bias correction makes early steps larger" statements are not used by ch-01.

## Adam, Algorithm 1 (Kingma & Ba)
```
m_0 ← 0, v_0 ← 0, t ← 0
while θ_t not converged:
    t ← t + 1
    g_t ← ∇_θ f_t(θ_{t−1})
    m_t ← β1 · m_{t−1} + (1 − β1) · g_t
    v_t ← β2 · v_{t−1} + (1 − β2) · g_t²
    m̂_t ← m_t / (1 − β1^t)
    v̂_t ← v_t / (1 − β2^t)
    θ_t ← θ_{t−1} − α · m̂_t / (√v̂_t + ε)
```
"Good default settings for the tested machine learning problems are α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10⁻⁸" (Algorithm 1 caption).

## Step size bound and scale invariance (§2.1)
- With ε = 0 the effective step is Δ_t = α · m̂_t / √v̂_t. "The effective magnitude of the steps taken in parameter space at each timestep are approximately bounded by the stepsize setting α, i.e., |Δ_t| ⪅ α."
- The bound |Δ_t| ≤ α·(1 − β1)/√(1 − β2) applies only "in the most severe case of sparsity".
- "The effective stepsize Δ_t is also invariant to the scale of the gradients; rescaling the gradients g with factor c will scale m̂_t with a factor c and v̂_t with a factor c², which cancel out."

## Bias correction (§3)
- v_t = (1 − β2) Σ_{i=1..t} β2^{t−i} g_i² (Eq. 1); E[v_t] = E[g_t²]·(1 − β2^t) + ζ (Eq. 4). Dividing by (1 − β2^t) removes the bias from the zero initialization.
- "In case of sparse gradients ... it is exactly this case of small β2 where a lack of initialisation bias correction would lead to initial steps that are much larger." (Here "small β2" refers to 1 − β2 being small, the paper's sparse-gradient case.)

## AdamW (Loshchilov & Hutter)
- Weight decay as defined by Hanson & Pratt: θ_{t+1} = (1 − λ)θ_t − α∇f_t(θ_t) (§2, Eq. 1).
- Proposition 1: for plain SGD, weight decay λ equals L2 regularization with λ′ = λ/α.
- Proposition 2: for an optimizer with a preconditioner M_t ≠ kI (Adam), "there exists no L2 coefficient λ′" that reproduces decoupled weight decay.
- Algorithm 2, line 6 (L2 variant, the term highlighted in the paper): g_t ← ∇f_t(θ_{t−1}) + λθ_{t−1}.
- Algorithm 2, line 12 (AdamW, the decoupled term): θ_t ← θ_{t−1} − η_t ( α·m̂_t/(√v̂_t + ε) + λθ_{t−1} ). η_t is the schedule multiplier; in this line λ is multiplied by η_t but not by α.
- "with L2 regularization ... weights x with large typical gradient magnitude s are regularized by a smaller relative amount than other weights" (§2).
- Stated observations (§1): optimal weight decay depends on the total number of batch passes (the longer the run, the smaller the optimal decay); Adam benefits from a scheduled LR multiplier such as cosine annealing.
- Evidence scope: decoupled weight decay gave a "15% relative improvement in test error" for Adam on CIFAR-10 and ImageNet32x32 image classification (§1, Figs. 2-3). No language-model experiments.

## Implementation difference used in ch-01
PyTorch `AdamW` multiplies the parameter by (1 − lr · weight_decay) (see [[pytorch-adamw-clip-amp]]), so its decay term scales with the learning rate, while Algorithm 2 line 12 scales λ only by the schedule multiplier η_t. [[small-scale-proxies-instabilities]] calls the paper form "independent" weight decay (§3.2.2).

## Verification
- Checked on 2026-09-15 against arXiv:1412.6980v9 (Algorithm 1, §2, §2.1, §3) and arXiv:1711.05101v3 (Abstract, §1, §2, Algorithms 1-2).
- Removed from the previous excerpt version: LLM default tables (not in either paper), "Llama picked 0.95" attribution, μP statements.
