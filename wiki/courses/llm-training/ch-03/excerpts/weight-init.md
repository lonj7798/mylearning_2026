---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: weight-init (Glorot 2010 + He 2015 + GPT-2 Radford 2019 + Tensor Programs V / Yang-Hu 2022)
source_url: https://proceedings.mlr.press/v9/glorot10a.html ; https://arxiv.org/abs/1502.01852 ; https://arxiv.org/abs/2203.03466
created_at: "2026-04-23"
---

# Excerpt: Weight Initialization — Glorot, He, GPT-2 / Residual Scaling, μP

**Sources (composite family):**
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS 2010 — Xavier init
- He, Zhang, Ren, Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification," ICCV 2015 — He/Kaiming init — arxiv 1502.01852
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019 — `N(0, 0.02)` + `1/√(2L)` residual trick, §2.3
- Kaplan et al., "Scaling Laws for Neural Language Models," 2020 — residual scaling derivation — arxiv 2001.08361
- Yang, Hu, et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (μP / muTransfer), 2022 — arxiv 2203.03466

---

## Why init is a chapter-3 topic and not a chapter-1 one

Init sits between the optimizer and the architecture. An optimizer doesn't know what scale its parameters start at; an architecture doesn't know how to sample its own weights. The practitioner has to. Get it wrong and the first 50 gradient steps are spent fixing the init rather than doing anything useful — worse, a bad init interacts with AdamW's `v̂` bias correction ([[excerpts/adam]]) to produce loss NaNs in the warmup phase.

The composite raw-data file [[weight-init]] starts with:

> "Initial weights must be scaled so that activation and gradient variances are preserved across layers — otherwise training diverges or vanishes before the optimizer can correct it."

This is a statement about **signal propagation**, a concept from Glorot's 2010 thesis that precedes modern deep learning. Forward activations should have bounded variance at every layer; backward gradients should too. If either grows or shrinks geometrically in depth, the optimizer is chasing a moving target.

---

## 1. Variance-preservation derivation (Xavier, 2010)

Consider a linear layer `y = Wx` with no nonlinearity. Assume:
- `x ∈ ℝ^{fan_in}` has i.i.d. entries with zero mean and variance `Var(x)`.
- `W ∈ ℝ^{fan_out × fan_in}` has i.i.d. entries from `N(0, σ²)`, independent of `x`.
- Bias is zero.

Compute `Var(y_j)` for any fixed output index `j`:

```math
y_j = \sum_{i=1}^{\mathrm{fan\_in}} W_{ji} x_i
```

Because `W_{ji}` and `x_i` are independent with zero mean:

```math
\mathrm{Var}(y_j) = \sum_{i=1}^{\mathrm{fan\_in}} \mathrm{Var}(W_{ji} x_i) = \sum_{i=1}^{\mathrm{fan\_in}} \mathrm{Var}(W_{ji}) \cdot \mathrm{Var}(x_i) = \mathrm{fan\_in} \cdot \sigma^2 \cdot \mathrm{Var}(x)
```

For `Var(y) = Var(x)` (variance preservation in the forward direction), we need

```math
\sigma^2 = \frac{1}{\mathrm{fan\_in}}
```

Now run the same argument on the backward pass. The gradient `∂L/∂x = W^T (∂L/∂y)`. By the symmetric calculation:

```math
\mathrm{Var}(\partial L/\partial x) = \mathrm{fan\_out} \cdot \sigma^2 \cdot \mathrm{Var}(\partial L / \partial y)
```

For backward preservation: `σ² = 1 / fan_out`.

Glorot's compromise — average the two requirements harmonically:

```math
\boxed{\sigma^2 = \mathrm{Var}(W) = \frac{2}{\mathrm{fan\_in} + \mathrm{fan\_out}}}
```

This is the Xavier/Glorot rule. It is exact for linear or symmetric-tanh networks; for anything else it is a heuristic.

### He/Kaiming correction for ReLU (2015)

ReLU zeroes out (on expectation) half its inputs. So the effective variance *after* ReLU is half the input variance. To preserve forward variance through `ReLU(Wx)`:

```math
\mathrm{Var}(y) = \tfrac{1}{2} \cdot \mathrm{fan\_in} \cdot \sigma^2 \cdot \mathrm{Var}(x)
```

Setting this equal to `Var(x)`:

```math
\boxed{\sigma^2 = \frac{2}{\mathrm{fan\_in}}}
```

He's key empirical result (Figure 1 of the paper): a 30-layer model converges with He init but fails with Xavier init. The factor of 2 is decisive at depth.

What does this mean for LLM training in 2025? Transformers don't use ReLU — they use GELU, SwiGLU, SiLU, all of which are between linear and ReLU. You could derive a custom factor for each. In practice modern code skips the derivation entirely and uses a hand-tuned constant: `N(0, 0.02)`. See §3.

---

## 2. The GPT-2 / Megatron rule: `N(0, 0.02)` everywhere

Radford et al. (GPT-2, 2019) §2.3 specifies:

> "We scale the weights of residual layers at initialization by a factor of `1/√N` where `N` is the number of residual layers."

And the default init for everything else is `N(0, 0.02)`. This rule has **propagated essentially unchanged** through GPT-2 → GPT-3 → Llama-1/2/3 → Qwen → Megatron-Turing 530B → DeepSeek-V3 → OLMo-2. The [[weight-init]] source:

> "GPT-2 / Llama: all linear layers `N(0, 0.02)`. Embedding layer same. The two "residual projections" per block (output of attention `W_O`, output of MLP `W_2`) are additionally scaled by `1/√(2L)`. Reason: residual stream variance grows linearly in depth without it; loss-spike risk at 100+ layers."

Why `0.02`? It's `√(1/2500)`, which is close to `√(2/d_model)` for `d_model = 100`. The standard is frozen from the GPT-2 era when models were 768–1600 wide. For modern 4096–16384 widths, a principled He-style init would give `σ = √(2/d_model) ≈ √(2/4096) ≈ 0.022` — **numerically almost identical to the 0.02 constant**. The industry keeps the constant because it's easy to communicate and well-tested.

---

## 3. Residual-projection scaling: the `1/√(2L)` derivation

This is the rule that every modern frontier LLM applies, and most tutorials gloss over it. Here is the full derivation, following Kaplan et al. (2020, App B) and the GPT-2 paper.

### Setup

A pre-norm Transformer block is:

```math
x_{\ell+1} = x_\ell + \mathrm{Sublayer}(\mathrm{LN}(x_\ell))
```

where `Sublayer` is either attention or MLP. In a block, there are **two** residual additions — one for attention, one for MLP — so `2L` total residual additions across the network.

Assume at init each `Sublayer(LN(x_ℓ))` output is i.i.d. Gaussian with variance `σ²_sub`, independent of `x_ℓ`. Then:

```math
\mathrm{Var}(x_{\ell+1}) = \mathrm{Var}(x_\ell) + \sigma^2_{\text{sub}}
```

Iterating from `x_0` (embedding, variance `σ²_emb`):

```math
\mathrm{Var}(x_L) = \sigma^2_{\text{emb}} + 2L \cdot \sigma^2_{\text{sub}}
```

So **residual-stream variance grows linearly in depth**. For `L = 32` blocks, that's a 64× variance blowup between the embedding and the LM head.

### The fix

We want `Var(x_L) ≈ Var(x_0)`. That requires `σ²_sub ≪ σ²_emb / (2L)`. If we scale the Sublayer's *output-projection* weights by `α`, its output variance scales by `α²`. Choose:

```math
\alpha = \frac{1}{\sqrt{2L}} \;\Longrightarrow\; \alpha^2 = \frac{1}{2L} \;\Longrightarrow\; 2L \cdot \alpha^2 \cdot \sigma^2_{\text{sub,raw}} = \sigma^2_{\text{sub,raw}}
```

So `Var(x_L) ≈ σ²_emb + σ²_sub_raw` — bounded in depth. This is the GPT-2 trick.

### Which weights get scaled

From [[weight-init]]:

> "The two "residual projections" per block (output of attention `W_O`, output of MLP `W_2`) are additionally scaled by `1/√(2L)`."

Specifically, **only** the *last* linear of each sub-layer (the one whose output writes into the residual). The inner linears (`W_Q`, `W_K`, `W_V`, `W_1` of MLP) keep `N(0, 0.02)`. Scaling those would break the sub-layer's signal-to-noise ratio at init.

### Empirical evidence

Llama-2, Llama-3, Qwen, DeepSeek, OLMo-2, Megatron-Turing all apply this scaling. The [[weight-init]] source notes:

> "OLMo-2: also adopts the residual-scale; reports it eliminated several loss spikes."

What does this mean for LLM training in 2025? If you're building a Transformer from scratch, **you must implement this**. The GPT-2 paper's 2019 line remains the single most important init detail in modern LLM code. The reference code in `[[ch-03]]` §5 shows the pattern.

---

## 4. Embedding init — a separate rule

Embeddings are special. From [[weight-init]]:

> "Embedding init: typically smaller (e.g. `N(0, 1e-5)`) or zero — the LM head's tying with the embedding causes large gradients otherwise."

When the embedding table `E` is tied with the LM head (same matrix used for `x @ E^T` to produce logits), the embedding participates in *every* gradient step. If `E` is initialised at `N(0, 0.02)` and the LM head output goes through softmax, the initial logits have magnitude `O(0.02 · √d_model · √vocab_size)` which can saturate softmax. Scaling the embedding down to `N(0, 1e-5)` (or even zero) makes initial logits nearly uniform — initial loss equals `ln(vocab_size)`, the uniform-prediction baseline.

Modern practice splits:

- **Untied** (Llama-3 405B, DeepSeek-V3): embedding at `N(0, 0.02)`, separate `N(0, 0.02)` head.
- **Tied** (many smaller models, GPT-2): embedding at `N(0, 0.02)` or even `N(0, 1e-5)` for safety.

---

## 5. μP / muTransfer — Yang & Hu 2022

μP solves a different problem: **hyperparameter transfer across model widths**. Normally, the optimal LR at width 256 is not optimal at width 8192 — you have to re-sweep every time you scale. μP re-parametrises the network so the same LR works at every width.

### The abc-parametrization (simplified)

Classify each weight matrix by its input/output role:

| Role | Examples | Init scale | AdamW LR multiplier | Forward multiplier |
|---|---|---|---|---|
| Input layer | embedding | `O(1)` | `O(1)` | — |
| Hidden layer | `W_Q`, `W_V`, `W_O`, `W_1`, `W_2` | `O(1/√d)` | `O(1)` | — |
| Output layer | LM head | `O(1/d)` | `O(1)` | `1/d` applied to logits |

"Hidden init `O(1/√d)`" is just He init for the new width. "Output init `O(1/d)`" is strictly smaller than He — this is the μP twist. "Output multiplier `1/d` on logits" rescales after the matmul so the logit magnitude is width-invariant.

### Why this works

The intuition (formalised in Tensor Programs V): under this scaling, the *statistics* of every hidden activation, every gradient, every Adam moment, are width-invariant at initialization. So the "best LR" — which is a function of those statistics — is also width-invariant.

The [[weight-init]] source captures the empirical result:

> "μP Figure 1 (Yang/Hu): loss-vs-LR curves for many widths *overlap* under μP and *diverge* under standard init — the empirical proof of HP transfer."

What does this mean for LLM training in 2025? Frontier models reportedly use μP-style transfer (GPT-4's HP sweep is rumoured to; Cerebras-GPT publicly demonstrates it). At 100B+ scale, sweeping on a 40M proxy saves enormous compute. The catch: you must be disciplined. Mix one non-μP layer (forget to rescale the LM head) and the transfer property collapses.

### What transfers, what doesn't

From [[weight-init]]:

> "Hyperparameters that DO transfer under μP: peak LR, optimizer betas, init scale, LR schedule shape.
> That do NOT transfer: depth-dependent quantities, batch size (compute-optimal scaling laws still apply), data mix."

Depth is the awkward case. μP is a *width* theory. Transferring across depth requires additional scaling rules (sometimes called "depth-μP"), which are less settled. In practice, people sweep depth and width together and pick a point on the Chinchilla curve.

---

## 6. The init audit — verify before you train

From [[weight-init]]:

> 1. After init (no training), forward-pass variance should be roughly preserved across all blocks (within 2x).
> 2. Backward gradient norm at every layer should be within an order of magnitude.
> 3. Initial loss should be near `ln(vocab_size)` for an LM (uniform-prediction loss).
> 4. First 100 training steps' loss should monotonically decrease without spikes.

Audit step 3 is the cheapest and most diagnostic. For `vocab_size = 32000`, `ln(32000) ≈ 10.37`. If your initial loss is `15`, your embedding or LM head is mis-initialised. If it's `8`, some bias in your init is already predicting something (wrong — a random init should predict uniformly).

What does this mean for LLM training in 2025? Add these four checks to your training harness. They catch more bugs per line-of-code than any other class of validation.

---

## 7. Common pitfalls

From [[weight-init]]:

> - "Forgetting the residual-projection scale on a 70+-layer model → loss spikes mid-training."
> - "Using PyTorch default init (`uniform(-√(1/fan_in), √(1/fan_in))`) for embeddings → embedding norm explodes; tie-weights amplifies it."
> - "Mixing μP and non-μP layers (e.g. forgot to scale the LM head) → defeats the entire transfer property."
> - "Setting `init_std` too high (e.g. `0.05`) → first-layer gradients saturate softmax → training stalls."

The PyTorch default pitfall is nasty. `nn.Linear(in, out)` uses `kaiming_uniform_(a=√5)`, which is *not* He init — it's a legacy uniform variant whose variance is roughly `1/(3 · fan_in)`. Close enough for many image models, wrong for Transformers. Always explicitly call `nn.init.normal_(mean=0, std=0.02)` on your linears.

---

## Connections

- [[excerpts/lr-schedules]] — peak LR is tied to init scale through AdamW's bias correction dynamics.
- [[excerpts/adam]] — AdamW's update is per-parameter, so bad init shows up as huge `v̂` for some parameters → small effective LR there and normal LR elsewhere → uneven training.
- [[excerpts/batch-vs-layer-norm]] — pre-norm absorbs minor init errors; post-norm cannot, which is partly why post-norm is extinct.
- [[excerpts/mixed-precision]] — residual-stream variance that grows with depth is also the source of norm-reduction precision issues under bf16/fp8.
- [[ch-03]] — synthesis with reference code.
- [[ch-01]] — AdamW basics.
