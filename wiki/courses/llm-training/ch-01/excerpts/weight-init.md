---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "Weight initialization: Glorot & Bengio (2010), He et al. (2015), GPT-2 (Radford 2019), μP (Yang & Hu 2022)"
source_url: https://arxiv.org/abs/2203.03466
created_at: "2026-04-23"
---

# Excerpt: Weight initialization — Xavier, He, GPT-2, residual scaling, μP

**Authors (composite):** Xavier Glorot, Yoshua Bengio (Glorot/Xavier, 2010); Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (He init, 2015); Alec Radford et al. (GPT-2, 2019); Greg Yang, Edward Hu et al. (μP, 2022)
**URLs:**
- https://proceedings.mlr.press/v9/glorot10a.html — Understanding the difficulty of training deep feedforward neural networks (Glorot 2010)
- https://arxiv.org/abs/1502.01852 — Delving Deep into Rectifiers (He 2015)
- https://arxiv.org/abs/2203.03466 — Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (Yang 2022)
**arXiv IDs:** 1502.01852, 2203.03466
**Raw-data source:** [[raw-data/weight-init]]

---

## The unifying principle — variance preservation

Consider a linear layer `y = Wx` with `x ∈ ℝ^{fan_in}`, `W_{ij}` i.i.d. with mean 0 and variance `σ²`, and `x_i` i.i.d. with variance `Var(x)`. The output variance is:

```math
\text{Var}(y_j) = \sum_{i=1}^{\text{fan\_in}} \text{Var}(W_{ji} x_i) = \text{fan\_in} \cdot \sigma^2 \cdot \text{Var}(x)
```

For activations to stay at roughly constant scale across depth, we need `Var(y) = Var(x)`, which gives:

```math
\sigma^2 = \frac{1}{\text{fan\_in}} \quad \text{(forward preservation)}
```

A symmetric argument on the backward pass — gradients propagate through `W^T` — gives `σ² = 1/fan_out`. Glorot's compromise averages these:

```math
\boxed{\sigma^2_{\text{Xavier}} = \frac{2}{\text{fan\_in} + \text{fan\_out}}}
```

This is the entire derivation behind Xavier/Glorot init. It assumes (a) zero-mean weights, (b) i.i.d. samples, and (c) a **linear or odd-symmetric** activation (tanh). The forward pass can be loose; the backward condition is what usually breaks when init is wrong.

---

## He / Kaiming: the ReLU correction

ReLU zeroes out the negative half of its input. If `z = Wx` has a symmetric distribution around 0 (which it does if `W` and `x` have zero mean), then `E[ReLU(z)²] = ½ E[z²]`. The forward-preservation condition must compensate for this factor of 2:

```math
\text{Var}(\text{ReLU}(Wx)) = \frac{1}{2} \cdot \text{fan\_in} \cdot \sigma^2 \cdot \text{Var}(x)
```

Setting this equal to `Var(x)` gives:

```math
\boxed{\sigma^2_{\text{He}} = \frac{2}{\text{fan\_in}}}
```

Figure 1 of He et al. 2015 is the canonical "init matters" plot: a 30-layer model converges with He init (the activation norm stays roughly constant across depth) and fails entirely with Xavier init (activations vanish after ~10 layers because the ½ factor compounds geometrically). For a 30-layer stack with Xavier init, the final activation magnitude is `(1/2)^{15} ≈ 3 × 10^{-5}` of the input — the network's output is numerically zero regardless of input.

LeCun init (`σ² = 1/fan_in`) is a middle option — correct for linear networks or SELU, but loses a factor of 2 against He for ReLU. Rarely used in 2026.

---

## GPT-2 / Megatron: the residual scaling trick

Modern Transformers use Pre-LN with residual connections: `x ← x + block(LN(x))`. The problem: the residual stream variance **grows linearly with depth** if each block output has order-1 variance.

Concretely, if `block_l(LN(x))` has variance `σ²_block` (roughly constant over `l`), then after `L` layers:

```math
\text{Var}(x_L) \approx \text{Var}(x_0) + L \cdot \sigma^2_{\text{block}}
```

For `L = 96` (GPT-3 scale) this is a ~100× variance inflation by the final layer. The final LayerNorm divides by this inflated std, but the gradient path through all the earlier layers sees variance that grows with depth — numerical instability and loss spikes at 70B+ scale.

**GPT-2's fix** (Radford 2019, Section 2.3): scale the output projections by `1/√(2L)`:

```
W_attn_output.data *= 1.0 / sqrt(2 * L)    # the W_O projection in attention
W_mlp_output.data  *= 1.0 / sqrt(2 * L)    # the W_2 projection in MLP
```

where `L` is the number of residual blocks. The `2L` counts two residual adds per block (attention + MLP). After this scaling, each block contributes `1/(2L)` to the residual stream variance, making total variance `Var(x_0) + L · 1/(2L) = Var(x_0) + 1/2` — bounded regardless of depth.

Megatron, OPT, Llama, and OLMo-2 all adopt this rule. OLMo-2's tech report explicitly credits residual scaling with eliminating several reproducible mid-training loss spikes at 7B scale.

The full GPT-2/Megatron init recipe in practice:

```
every Linear layer:   N(0, 0.02)
embedding layer:      N(0, 0.02)
W_O in attention:     N(0, 0.02); then scale by 1/sqrt(2L)
W_2 in MLP:           N(0, 0.02); then scale by 1/sqrt(2L)
LayerNorm weight:     1.0
LayerNorm bias:       0.0
```

The `0.02` constant is a tuned shortcut for `√(2/d_model)` at `d_model ≈ 1024–4096`. For very wide models you may want to recompute it, though `0.02` is surprisingly robust across the 1B–100B range.

---

## μP — hyperparameter transfer across width

μP (Yang & Hu 2022, "Tensor Programs V") is a re-parametrization that makes the *optimal* learning rate, init scale, and other HPs **width-invariant**. The practical promise: sweep on a 40M-param proxy model, transfer the best HPs to 6B+ without re-tuning.

**The abc-parametrization.** μP assigns each parameter group three exponents `(a, b, c)` that govern how init scale and LR scale with width `d`. The recipe (for Adam):

| Layer | Init scale | LR multiplier | Forward multiplier |
|---|---|---|---|
| Input (embed) | `O(1)` | `O(1)` | — |
| Hidden (middle linears) | `O(1/√d)` | `O(1)` for Adam; `O(1/d)` for SGD | — |
| Output (LM head) | `O(1/d)` | `O(1/d)` | multiply logits by `1/d` |

The derivation comes from Tensor Programs theory: under these exponents, every feature-learning quantity in the network (pre-activations, gradients, updates) has a well-defined non-trivial `d → ∞` limit. Under standard (non-μP) init, those quantities either blow up or vanish as `d` grows, so the optimal HP drifts with scale.

**The empirical proof (μP Figure 1).** Plot loss vs. peak LR for many widths. Under standard init, the curves do not overlap — the optimum shifts left (smaller LR) as width increases. Under μP, the curves overlap: the optimum LR is the same at `d = 256` as at `d = 8192`.

**What transfers under μP:** peak LR, optimizer betas, init scale, LR schedule shape.
**What does NOT transfer:** depth-dependent quantities (residual scaling still needs `1/√(2L)`), batch size (compute-optimal scaling laws still apply, per Chinchilla), data mix.

Cerebras-GPT publicly demonstrated μP transfer; reports suggest GPT-4's HP sweep used μP-style tuning on a small proxy. The compute savings at 100B+ scale are enormous — a full sweep at 100B scale costs tens of millions of dollars; the same sweep at 40M costs hundreds.

---

## Embedding init — the tied-weights trap

Embeddings are usually tied to the LM head (shared weights). The embedding layer is indexed by discrete tokens, so `fan_in = 1` in a strict sense (one active row per token). Standard init for the embedding would use `Var = 2/d_model`, but because the same tensor is also used as the LM head weights, the *output* projection sees `fan_in = d_model`. A single scale can't be optimal for both roles.

Common pragmatic choices:

- **GPT-2 / Llama**: embedding initialised at `N(0, 0.02)`, same as linears. The LM head inherits this. Works empirically but embeddings can drift to large norm during training — monitored as a failure signal.
- **μP**: embedding is the "input layer" with `O(1)` init. Because μP also multiplies output logits by `1/d`, the effective LM head is `O(1/d)` — the mismatch is absorbed.
- **T5**: initializes embeddings at `N(0, 1.0)` but scales the output by `1/√d_model` before the softmax.
- **Some modern recipes**: very small embedding init (`N(0, 1e-5)` or even zeros) because the forward pass only reads one row per token — a small embedding still produces a non-trivial activation after the first attention block.

The takeaway: embedding init is the most recipe-specific part of the init spec. Follow your base recipe (GPT-2, T5, or μP) end-to-end; don't mix.

---

## The quick init audit (before hitting "go")

Four cheap checks that catch ~95% of init bugs:

1. **Forward variance preservation.** After init, run one forward pass on random input; check that the activation variance is within 2× of its value at each block. If it drifts by 10×+ across depth, the residual scaling is wrong.
2. **Backward gradient magnitude.** Similar check on gradient norm at each layer — should be within an order of magnitude.
3. **Initial loss.** For an LM with vocab `V`, uniform prediction has loss `ln(V)`. GPT-2's `V = 50257` gives initial loss `≈ 10.82`. If your initial loss is much higher (say 20), the output projection init is way off.
4. **First 100 steps monotonic.** Loss should decrease smoothly from step 0. If there is a spike in steps 1–50, either warmup is too short (see [[excerpts/lr-schedules]]) or init is too hot.

Karpathy's recipe crystallises this: "init well — at minimum, ensure the initial loss equals the loss of a uniform-output baseline."

---

## Common pitfalls

- **Forgetting residual scaling on 70+-layer models**: guaranteed loss spikes mid-training.
- **Using PyTorch default (`uniform(-√(1/fan_in), √(1/fan_in))`) for embeddings**: the uniform's std is `√(1/(3·fan_in))`, much smaller than `0.02`. Tied weights then amplify; embedding norm grows unboundedly.
- **Mixing μP and non-μP layers**: forgetting to scale the LM head defeats HP transfer entirely.
- **`init_std = 0.05`** (too hot): first-layer gradients saturate softmax; training stalls at initial loss.
- **Init audit skipped**: bug class where the model "trains" but 1–2% off baseline — the cheapest regression to catch and the one most likely to be missed.

---

## Why Pre-LN is init-forgiving (and Post-LN isn't)

Pre-LN Transformers apply LayerNorm *inside* the residual block: `x + block(LN(x))`. The LN normalises away the magnitude of `x` before the block sees it, so even a badly-scaled input produces a well-scaled block input. Post-LN (original Transformer, 2017) applies LN *after* the residual add: `LN(x + block(x))` — so the block input is un-normalised and init errors compound across depth.

This is why GPT-2 and every subsequent frontier LLM use Pre-LN. The init robustness is real: a Pre-LN model with Xavier init (wrong recipe) still trains; the equivalent Post-LN model diverges.

---

## Connections

- [[excerpts/adam]] — Adam's per-parameter scaling partially compensates for init errors at the optimization level, but the first forward pass (before any step) still needs variance preservation.
- [[excerpts/gradient-clipping]] — bad init shows up as abnormally large initial gradient norms; clipping is a symptom-damper, not a fix.
- [[excerpts/lr-schedules]] — μP changes the *value* of peak LR but not the *shape* of the schedule; warmup is still required.
- [[excerpts/mixed-precision]] — under fp16, badly-initialised activations are more likely to overflow/underflow; init sensitivity rises with narrower numerical formats.
- [[ch-01]] — parent chapter for training fundamentals.
