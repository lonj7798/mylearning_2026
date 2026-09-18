<!-- scope: weight initialization — Glorot 2010, He 2015, GPT-2 residual scaling, and μP (Tensor Programs V)
     deps: [[batch-vs-layer-norm]]
     see-also: [[adam]], [[lr-schedules]], [[olmo-2]], [[small-scale-proxies-instabilities]]
-->

# Weight Initialization: Glorot/Xavier, He/Kaiming, and μP
- **Core Insight:** Initialization variance is set by the fan-in of a layer and by the activation function — `Var(W) = 2/(n_in + n_out)` for symmetric activations (Glorot 2010 Eq. 16) and `Var(W) = 2/n_l` for ReLU (He 2015 Eq. 10) — while μP additionally makes the learning rate width-dependent per layer type, so that the optimal learning rate found at one width also applies at larger widths (Tensor Programs V Table 3, Fig. 1).
- **Guideline:** When the network is deep and uses ReLU, use `std = sqrt(2/n_l)`, because on a 30-layer model (27 conv + 3 fc) He initialization converged while Glorot initialization stalled with diminishing gradients (He 2015 Fig. 3). When transferring hyperparameters across width, put the model in μP and copy the tuned learning rate; the results are validated for width, and empirically for depth, batch size, sequence length and training time on Transformers (Tensor Programs V Table 1, §6.1).
- **Authors:** Xavier Glorot, Yoshua Bengio (2010); Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015); Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, et al. (2022)
- **Year:** 2010 (AISTATS) / 2015 (arXiv v1 2015-02; ICCV) / 2022 (arXiv v1 2022-03, v2 2022-03-28)
- **URL:** https://proceedings.mlr.press/v9/glorot10a.html ; https://arxiv.org/abs/1502.01852 ; https://arxiv.org/abs/2203.03466 ; https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf (GPT-2 §2.3)
- **Source type:** paper (composite classics card: Glorot 2010, He 2015, GPT-2 2019 §2.3, Tensor Programs V 2022)
- **Relevant topics:** signal propagation, training stability, hyperparameter transfer, width scaling

## Summary
Glorot and Bengio (2010) derive an initialization variance that keeps forward activations and backward gradients
from shrinking or growing across layers, under a linear or symmetric-activation assumption. He et al. (2015) redo
the derivation for rectifier nonlinearities and obtain a variance larger by a factor of two. Yang, Hu et al. (2022)
introduce μTransfer: parametrize the target model in the Maximal Update Parametrization (μP), tune on a smaller
proxy, and transfer the hyperparameters zero-shot.

## Key Contributions
- Glorot and Bengio (2010, Eq. 16): `Var(W) = 2/(n_in + n_out)`, averaging the forward and backward conditions.
- He et al. (2015, §2.2, Eq. 10): a sufficient condition `½ n_l Var[w_l] = 1` for ReLU layers, giving a zero-mean
  Gaussian with `std = sqrt(2/n_l)` and `b = 0`; the backward condition Eq. (14) uses `n̂_l` instead of `n_l`, and
  the paper states it is sufficient to use either Eq. (10) or Eq. (14) alone.
- Tensor Programs V (2022): Table 3 gives the μP scaling of initialization variance, SGD learning rate and Adam
  learning rate for three parameter groups; Table 1 lists which hyperparameters transfer; §7.3 and §7.4 tune
  BERT-large through a 13M proxy and GPT-3 6.7B through a 40M proxy.

## Key Figures/Tables to Study
- **He 2015 Fig. 3** (30-layer: He converges, Glorot stalls) and **Fig. 2** (22-layer: both converge).
- **Tensor Programs V Fig. 1:** training loss against learning rate for widths 128–8192 — under μP the optimum is stable across width, under standard parametrization it shifts. **Table 3** scaling rules, **Table 1** transfer.

## Technical Details

### Variance-preservation derivation
For a linear layer `y = Wx` with i.i.d. inputs and `W_ij ~ N(0, σ²)`, `Var(y_j) = n_in σ² Var(x)`, so preserving
forward variance requires `σ² = 1/n_in` and the backward condition gives `1/n_out`. He et al. restate the Glorot
forward result as `n_l Var[w_l] = 1`, "which can be implemented as a zero-mean Gaussian distribution whose std is
sqrt(1/n_l)", and note that over `L` layers this std is `1/√2^L` of theirs (He 2015, "Comparisons with Xavier").
`Glorot (2010, Eq. 16): Var(W) = 2/(n_in + n_out)` for symmetric activations in the linear regime;
`He (2015, Eq. 10): Var(W) = 2/n_l` for ReLU in the forward case, where `n_l` is the fan-in.

### What He et al. actually measured
- 22-layer model: both converge, He starts reducing error earlier (Fig. 2). On the 14-layer model of Table 2,
  Glorot gives 33.90/13.44 top-1/top-5 error and He 33.82/13.34; the paper states "We have not observed clear
  superiority of one to the other on accuracy."
- 30-layer model (27 conv + 3 fc): He converges, Glorot "completely stalls the learning, and the gradients are
  diminishing as monitored in the experiments" (Fig. 3). That model reached 38.56/16.59, "clearly worse than the
  error of the 14-layer model"; the paper reports no benefit from the extra depth in its ImageNet experiments.

### GPT-2 residual scaling (Radford et al. 2019, §2.3)
"We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual
layers." The same section moves layer normalization to the input of each sub-block and adds a final layer norm.

### μP scaling rules (Tensor Programs V Table 3)
Values in parentheses are standard parametrization where it differs. `fan_in` of a bias vector is 1.

| | Input weights and all biases | Output weights | Hidden weights |
|---|---|---|---|
| Init. variance | 1/fan_in | 1/fan_in² (1/fan_in) | 1/fan_in |
| SGD LR | fan_out (1) | 1/fan_in (1) | 1 |
| Adam LR | 1 | 1/fan_in (1) | 1/fan_in (1) |

Table 8 gives an equivalent, easier-to-implement formulation in which output weights have init variance 1 and a multiplier `1/fan_in`, and hidden-weight Adam LR is again `1/fan_in`.

### What transfers (Tensor Programs V Table 1)
- μTransferable: "optimization related, init, parameter multipliers, etc". Not μTransferable: "regularization
  (dropout, weight decay, etc)", because the required amount depends on data size as well as model size (§6).
- μTransferred across: "width, depth*, batch size*, training time*, seq length*", where `*` marks dimensions
  validated empirically on Transformers only.
- Limits stated in §6.1: transfer held for width ≥ about 256, depth ≥ about 4, batch size ≥ 32, sequence length
  ≥ 128 and training steps ≥ 5000 within the ranges tested; the best initialization standard deviation did not
  transfer across depth, and depth transfer worked only for pre-layernorm Transformers.

### μTransfer results
BERT-large test loss 1.683 with a 13M-parameter proxy vs 1.731 for the Megatron default (§7.3, Table 6); GPT-3 6.7B
validation loss 1.98 with a 40M-parameter proxy vs 2.03 for a re-run with the original hyperparameters, at a tuning
cost of "only 7% of total pretraining cost" (§7.4, Table 7), with two confounds stated in §7.4 (an absolute-attention
re-run baseline, and FP32 vs FP16 precision).

## Recipe ledger
Proxy sizes, tuning budgets, per-run values and the stated confounds are in [[weight-init-recipe]].

## Connections
- [[batch-vs-layer-norm]] — Tensor Programs V footnote 9: Glorot initialization scales like μP asymptotically but
  causes logit blowup once layernorm or batchnorm is added, while μP does not.
- [[adam]] — μP's width-dependent learning-rate rule belongs to the parametrization, not the optimizer;
  [[lr-schedules]] — μP changes the peak learning rate value, not the schedule shape.
- [[olmo-2]] — initialization ablation with measured spike scores; MiniCPM (arXiv:2404.06395 App. A.1) — an applied
  μP variant searched on a 0.009B proxy; [[small-scale-proxies-instabilities]] — a simple μParam stabilized the optimal learning rate without
  improving loss or removing the need for qk-layernorm.
- [[gradient-clipping]] — large initial gradient norms are a symptom clipping hides; [[karpathy-training-neural-net-recipe]] — the initial loss should match a uniform-output baseline.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1502.01852 (arXiv v1) and https://arxiv.org/abs/2203.03466
  (arXiv v2, 28 Mar 2022), full texts. Glorot Eq. 16 checked through the He 2015 restatement and the ch-03 excerpt.
- Corrections to the previous card version:
  - "LR multipliers: input O(1), hidden O(1/d) for SGD or O(1) for Adam, output O(1/d)" → reversed. Table 3: hidden
    weights take SGD LR 1 and Adam LR 1/fan_in; input weights SGD LR fan_out and Adam LR 1; output weights 1/fan_in for both.
  - "Init the input layer O(1), hidden layers O(1/sqrt(d)), output layer O(1/d)" → Table 3 gives initialization
    *variance* 1/fan_in for input and hidden weights and 1/fan_in² for output weights.
  - "Multiply output logits by 1/d" → the `1/fan_in` output multiplier belongs to the Table 8 formulation, not Table 3.
  - "Tune on a 40M-param proxy and transfer to 6B+" → the 40M proxy is the GPT-3 6.7B case (§7.4); BERT used 13M (§7.3).
  - "This single fix enabled training of 30-layer-plus CNNs (VGG, ResNet) for the first time" → He 2015 reports the
    30-layer model converging but performing worse than the 14-layer model (38.56/16.59 vs 33.82/13.34) and states
    no benefit from the extra depth. ResNet is a later paper.
  - "He 2015 Figure 1: 30-layer converges with He, fails with Xavier" → that is Figure 3; Figure 2 is the 22-layer case.
  - "DO transfer: peak LR, betas, init scale, schedule shape / do NOT transfer: depth-dependent quantities, batch
    size, data mix" → Table 1 lists regularization as the non-transferable class and lists depth, batch size,
    training time and sequence length as dimensions transferred *across*.
- Removed as unsupported by the source:
  - "OLMo-2 also adopts the residual-scale; reports it eliminated several loss spikes" — OLMo 2 §3.2 initializes
    every parameter from `N(0, 0.02)` and reports removing the earlier scaled initialization; see [[olmo-2]].
  - "Embedding init typically smaller (e.g. N(0, 1e-5)) or zero" — no source for this value.
  - "GPT-4 reportedly used μP-style transfer" and "Cerebras-GPT publicly demonstrated it" — not in these papers.
  - "T5: N(0, sqrt(1/d_in))"; "Megatron ... for stability up to 530B params"; the four "common pitfalls"; the
    four-step "init audit checklist"; and the claim that pre-LN "won" because LayerNorm absorbs init mistakes.
- Not reported: a base standard deviation for the GPT-2 residual rule or whether its `N` counts blocks or sub-layers (GPT-2 §2.3); a μP rule for depth scaling.
