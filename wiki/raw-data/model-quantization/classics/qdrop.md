<!-- scope: QDrop — random quantization dropout during PTQ calibration
     deps: brecq, adaround
     see-also: integer-only-inference, omniquant
-->

# QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization
- **Core Insight:** Calibration distributions and inference distributions differ; randomly disabling per-layer activation quantization during PTQ optimisation (with probability p ≈ 0.5) regularises the rounding search and closes the train-test gap, especially below 4-bit.
- **Guideline:** Wrap BRECQ/AdaRound's block-wise optimisation with a Bernoulli-p mask on activation quantization — each forward pass, each layer's activation quantizer is bypassed independently with probability p; set p ≈ 0.5 for 2-bit, decay to 0 over the last 25% of iterations.
- **Authors:** Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu, Fengwei Yu
- **Year:** 2022 (ICLR)
- **URL:** https://arxiv.org/abs/2203.05740
- **Relevant topics:** PTQ regularisation, quantization dropout, sub-4-bit, calibration-test gap

## Abstract
QDrop observes that PTQ methods like AdaRound and BRECQ over-fit the rounding search to the small calibration set, hurting test accuracy at extreme bit-widths. The fix is a quantization-level dropout: during each PTQ forward pass, each layer's activation quantization is independently disabled with probability p. This injects controlled stochasticity into the calibration-time loss landscape and forces the learned rounding to be robust to whether an upstream activation is quantized or not. On CIFAR and ImageNet at 2-bit, QDrop strictly improves every BRECQ/AdaRound baseline.

## Key Contributions
- Identifies calibration-vs-test distribution gap as the dominant 2-bit PTQ failure mode.
- Introduces a Bernoulli-p mask on per-layer activation quantization during optimisation.
- Empirically derives p ≈ 0.5 as near-optimal; theoretical justification via expected-loss bound.
- Compatible drop-in with AdaRound, BRECQ, OmniQuant — adds a single line.
- New SOTA for ≤4-bit PTQ at the time of publication.

## Key Figures/Tables to Study
- **Figure 3** — accuracy vs drop probability p; flat plateau around p=0.5.
- **Table 2** — 2/3/4-bit ResNet-18 PTQ: BRECQ vs BRECQ+QDrop.

## Technical Details

### The dropout rule
For each forward pass during PTQ optimisation, and for each layer ℓ:
`activation_ℓ = Q(activation_ℓ)  with prob p`
`activation_ℓ = activation_ℓ      with prob 1−p`
Weight quantization is always applied (the target deployment configuration).

### Block-wise loss under dropout
For block B_k:
`L_QDrop = E_{m ~ Bernoulli(p)^L} ‖f_k(X) − f̂_k(X; W_k, m)‖²`
where m is the per-layer dropout mask sampled fresh each iteration. Estimated by Monte Carlo with one sample per step.

### Why p = 0.5
- p = 0: pure quantized forward; over-fits to compounded error path.
- p = 1: pure FP forward during optimisation; ignores quantization noise entirely.
- p = 0.5: each layer sees both regimes equally; learned rounding is robust to either.

The paper's theoretical lemma shows the expected loss interpolates linearly between the two extremes, with the minimum-variance estimator at p = 0.5.

### Annealing
Linear decay p_t = 0.5 · (1 − t/T_decay) over the last 25% of iterations so the final rounding decisions are evaluated under the actual deployment forward (all quantized).

### Integration with AdaRound / BRECQ
A single addition to the existing PTQ loop:
```
for step in range(N):
    mask = bernoulli(p, size=L)         # NEW
    y_fp  = f_block_fp(X)
    y_q   = f_block_q(X, dropout=mask)  # NEW arg
    loss  = ‖y_fp − y_q‖² + λ·reg(V)
    loss.backward(); optimizer.step()
```

### Empirical effect
- AdaRound 2-bit ResNet-18: 52.84% → BRECQ 51.93% → BRECQ+QDrop **54.27%**.
- Effect grows monotonically as bit-width drops.

## Connections
- [[brecq]] — primary host; QDrop is a regulariser around BRECQ's objective.
- [[adaround]] — also benefits, smaller delta because layer-wise loss is less compounded.
- [[integer-only-inference]] — same target deployment, different optimisation step.
- [[omniquant]] — adopts QDrop-style stochasticity in modern LLM PTQ recipes.
- [[gptq]] — does not use QDrop (sequential exact update precludes random dropout) but the failure mode QDrop fixes also motivates GPTQ's per-row exact correction.
