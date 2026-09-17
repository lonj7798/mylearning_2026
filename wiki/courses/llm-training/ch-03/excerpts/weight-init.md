---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: weight-init (composite) — Glorot & Bengio 2010; He et al. 2015; Radford et al. 2019 (GPT-2); Yang et al. 2022 (Tensor Programs V, μP); OLMo et al. 2025 (OLMo 2 §3.2); Hu et al. 2024 (MiniCPM App. A.1); Wortsman et al. 2023 (§3.2.4)
source_url: https://proceedings.mlr.press/v9/glorot10a.html ; https://arxiv.org/abs/1502.01852 ; https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf ; https://arxiv.org/abs/2203.03466 ; https://arxiv.org/abs/2501.00656 ; https://arxiv.org/abs/2404.06395 ; https://arxiv.org/abs/2309.14322
created_at: "2026-04-23"
revised_at: "2026-09-15"
note: "Rewritten for the 2026-09 revision. The library card classics/weight-init.md was not yet re-verified on 2026-09-15 and contains errors: a reversed μP learning-rate table (hidden-weight Adam LR written as O(1)), a claim that OLMo 2 uses depth-scaled residual init, an embedding init of N(0, 1e-5), and a claim that GPT-4 used μP. This excerpt keeps only statements read in the primary sources."
---

# Excerpt: initialization and μP, as printed

## Variance-preserving initialization

- Glorot & Bengio (2010, normalized initialization, Eq. 16): Var(W) = 2 / (n_in + n_out), derived for symmetric
  activations near the linear regime.
- He et al. (2015, arXiv:1502.01852 §2.2, Eq. 10): Var(W) = 2 / n_l for ReLU layers, where n_l is the fan-in.
- Tensor Programs V footnote 6 (arXiv:2203.03466v2): in standard parametrization "the init. variance ∝ 1/fan_in,
  so the same insights here apply with e.g. He initialization."

## GPT-2 residual scaling (Radford et al. 2019, §2.3)

"A modified initialization which accounts for the accumulation on the residual path with model depth is used. We
scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual
layers." The same section: "Layer normalization (Ba et al., 2016) was moved to the input of each sub-block ... and an
additional layer normalization was added after the final self-attention block." The paper does not print a base standard deviation and does not state whether N counts blocks or
sub-layers.

## OLMo 2 (arXiv:2501.00656 §3.2, Fig. 4; official config)

- OLMo 2: "we initialize every parameter from a normal distribution with a mean of 0 and a standard deviation of
  0.02." OLMo-0424 (scaled initialization, Zhang et al. 2019) "scaled input projections by 1/√d_model, and output
  projections by 1/√(2·d_model·layer_idx) at every layer."
- Ablation: a baseline "that reproduces spikes quickly" by "mainly reducing the warmup period"; "the new
  initialization had no loss spikes, and the spike score for the L2 norm of the gradient went from 0.40 to 0.03."
  Spike score = percentage of values at least seven standard deviations from a rolling average of the last 1,000
  values. "The new initialization converges slightly slower."
- `OLMo2-7B-stage1.yaml`: `init_fn: normal`, `init_std: 0.02`, `init_cutoff_factor: 3` (truncation at 3 standard
  deviations); §2.3 describes "a truncated normal distribution with a mean of 0 and a standard deviation of 0.02".

## μP (Tensor Programs V, arXiv:2203.03466v2)

Table 3 (μP, with standard parametrization in parentheses where it differs):

| | Input weights and all biases | Output weights | Hidden weights |
|---|---|---|---|
| Init. variance | 1/fan_in | 1/fan_in² (1/fan_in) | 1/fan_in |
| SGD LR | fan_out (1) | 1/fan_in (1) | 1 |
| Adam LR | 1 | 1/fan_in (1) | 1/fan_in (1) |

- Transformer μP also uses "1/d attention instead of 1/√d" (Table 3 caption, Definition 4.1).
- Table 1: μTransferable = "optimization related, init, parameter multipliers, etc"; not μTransferable =
  "regularization (dropout, weight decay, etc)"; μTransferred across = "width, depth*, batch size*, training
  time*, seq length*", where * means "empirically validated only on Transformers".
- §7.4: GPT-3 6.7B (32 blocks, width 4096) tuned by random search on a width-256 proxy of "roughly 40 million
  trainable parameters"; "The total tuning cost was only 7% of total pretraining cost." Table 7 validation loss:
  6.7B+μP 1.98, 6.7B re-run 2.03. Stated confounds: the re-run "mistakenly used absolute attention", and the μP model
  was trained in FP32 "to avoid" frequent divergences while the original and re-run used FP16.

## μP in MiniCPM (arXiv:2404.06395v3 §3.1, §3.3, App. A.1, Table 7)

- Operations: embedding output × scale_emb; block output scaled by scale_depth/√num_layers before the residual add;
  2-D tensor init std = init_std/√(d_m/d_base); 2-D tensor LR = 1/(d_m/d_base) of the base LR; output logits ×
  1/(d_m/d_base). Attention softmax scaling not applied.
- Search on a 0.009B model gave scale_depth = 1.4, scale_emb = 12, init_std = 0.1, lr = 0.01 (App. A.1). Optimal
  base LR "remains around 0.01" from 0.04B to 0.5B and was confirmed at 2.1B (§3.3, Fig. 3).

## μP in Wortsman et al. (arXiv:2309.14322v2 §3.2.4)

muParam (simple), which scales the LR of linear layers by base-fan-in/fan-in with base width 256, stabilized the
optimal LR but did not improve loss or reduce LR sensitivity, and did not remove the need for qk-layernorm at high
LR (Fig. 8, Fig. E.4; paraphrase from the verified card [[small-scale-proxies-instabilities]]).
