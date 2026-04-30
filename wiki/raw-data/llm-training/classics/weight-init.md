<!-- scope: weight initialization — Xavier, He/Kaiming, μP for width transfer
     deps: [[batch-vs-layer-norm]]
     see-also: [[adam]], [[lr-schedules]]
-->

# Weight Initialization: Xavier/Glorot, He/Kaiming, and μP
- **Core Insight:** Initial weights must be scaled so that activation and gradient variances are preserved across layers — otherwise training diverges or vanishes before the optimizer can correct it.
- **Guideline:** For a Transformer, initialize all linear layers `N(0, 0.02)` or `N(0, sqrt(2/d_model))`; scale residual-projection weights by `1/sqrt(2L)` (GPT-2 trick); use **μP** when you need hyperparameter transfer from small to large models.
- **Authors:** Xavier Glorot & Yoshua Bengio (2010); Kaiming He et al. (2015); Greg Yang & Edward Hu et al. (μP, 2022)
- **Year:** 2010 / 2015 / 2022
- **URL:** https://proceedings.mlr.press/v9/glorot10a.html ; https://arxiv.org/abs/1502.01852 ; https://arxiv.org/abs/2203.03466
- **Relevant topics:** signal propagation, training stability, hyperparameter transfer, scaling laws

## Abstract (composite)
**Glorot/Xavier (2010)**: derives the variance condition for keeping forward activations and backward gradients bounded across layers, assuming linear (or symmetric tanh) activations. Result: `Var(W) = 2 / (fan_in + fan_out)`.
**He/Kaiming (2015)**: extends Glorot to ReLU, which zeroes half the activations and thus halves the forward variance. Result: `Var(W) = 2 / fan_in`. This single fix enabled training of 30-layer-plus CNNs (VGG, ResNet) for the first time.
**μP / muTransfer (Yang 2022, Hu et al.)**: re-parametrizes a network so the *optimal* learning rate, init scale, and other hyperparameters are width-invariant. Tune on a 40M-param proxy and transfer to 6B+ without re-tuning — the technique behind multiple frontier-model HP sweeps (notably Cerebras-GPT and parts of GPT-4's HP search).

## Key Contributions
- **Xavier**: variance-preservation condition; the first principled init.
- **He/Kaiming**: ReLU-aware variance; unblocked deep CNN training.
- **GPT-2 / Megatron init**: `N(0, 0.02)` for all linear layers, with residual-projection scaling `1/sqrt(2L)` (where `L` = number of residual blocks). Empirically beats Xavier/He for autoregressive Transformers.
- **μP**: introduces "abc-parametrization" (init scale, multiplier, LR scale, all width-dependent). The result: the loss curve at small width is a faithful surrogate for large width.
- **Embedding init**: typically smaller (e.g. `N(0, 1e-5)`) or zero — the LM head's tying with the embedding causes large gradients otherwise.

## Key Figures/Tables to Study
- **He 2015 Figure 1**: 30-layer model converges with He init, fails with Xavier — the canonical "init matters" plot.
- **μP Figure 1** (Yang/Hu): loss-vs-LR curves for many widths *overlap* under μP and *diverge* under standard init — the empirical proof of HP transfer.
- **GPT-2 paper Section 2.3**: the residual scaling rule `weight *= 1/sqrt(N)` for the second linear in each MLP and the output projection in attention.

## Technical Details

**Variance-preservation derivation (Xavier)**: for a linear layer `y = Wx`, if `x_i` are i.i.d. with variance `Var(x)` and `W_ij ~ N(0, sigma^2)`, then `Var(y_j) = fan_in * sigma^2 * Var(x)`. To preserve variance: `sigma^2 = 1 / fan_in`. Symmetric reasoning on the backward pass gives `1 / fan_out`. Average:
```
Xavier:    Var(W) = 2 / (fan_in + fan_out)         # tanh / linear
He:        Var(W) = 2 / fan_in                     # ReLU (forward only)
LeCun:     Var(W) = 1 / fan_in                     # SELU / linear
```

**Transformer init recipes (in practice)**:
- **GPT-2 / Llama**: all linear layers `N(0, 0.02)`. Embedding layer same. The two "residual projections" per block (output of attention `W_O`, output of MLP `W_2`) are additionally scaled by `1/sqrt(2L)`. Reason: residual stream variance grows linearly in depth without it; loss-spike risk at 100+ layers.
- **T5**: `N(0, sqrt(1/d_in))` (Glorot-flavored), and pre-LN means init is less critical.
- **Megatron**: same `0.02` rule; explicitly scales `W_O` and `W_2` for stability up to 530B params.
- **OLMo-2**: also adopts the residual-scale; reports it eliminated several loss spikes.

**μP — the key idea**:
- Init the "input layer" `O(1)`, "hidden layers" `O(1/sqrt(d))`, "output layer" `O(1/d)`.
- LR multipliers: input `O(1)`, hidden `O(1/d)` for SGD or `O(1)` for Adam, output `O(1/d)`.
- Multiply output logits by `1/d` (the "output multiplier").
After this re-parametrization, the *optimal LR found at width 256* is also optimal at width 8192. You sweep on a tiny model and transfer.

**Hyperparameters that DO transfer under μP**: peak LR, optimizer betas, init scale, LR schedule shape.
**That do NOT transfer**: depth-dependent quantities, batch size (compute-optimal scaling laws still apply), data mix.

**Common pitfalls**:
- Forgetting the residual-projection scale on a 70+-layer model → loss spikes mid-training.
- Using PyTorch default init (`uniform(-sqrt(1/fan_in), sqrt(1/fan_in))`) for embeddings → embedding norm explodes; tie-weights amplifies it.
- Mixing μP and non-μP layers (e.g. forgot to scale the LM head) → defeats the entire transfer property.
- Setting `init_std` too high (e.g. `0.05`) → first-layer gradients saturate softmax → training stalls.

**Quick init audit checklist for a new architecture**:
1. After init (no training), forward-pass variance should be roughly preserved across all blocks (within 2x).
2. Backward gradient norm at every layer should be within an order of magnitude.
3. Initial loss should be near `ln(vocab_size)` for an LM (uniform-prediction loss).
4. First 100 training steps' loss should monotonically decrease without spikes.

## Connections
- **[[batch-vs-layer-norm]]**: pre-LN Transformers are much more init-tolerant than post-LN — LayerNorm absorbs init mistakes. This is one of the key reasons pre-LN won.
- **[[adam]]**: Adam's per-parameter scaling makes raw init *less* critical for first-order optimization, but residual-stream growth still matters at depth.
- **[[lr-schedules]]**: μP changes the *value* of peak LR but not the *shape* of the schedule.
- **[[gradient-clipping]]**: bad init shows up as enormous initial grad norms — clipping masks the symptom; fix the init.
- **Frontier HP sweeps**: GPT-4 reportedly used μP-style transfer; Cerebras-GPT publicly demonstrated it; the saved compute is enormous at 100B+ scale.
- **Karpathy** ([[karpathy-training-neural-net-recipe]]): "init well — at minimum, ensure the initial loss equals the loss of a uniform-output baseline."
