<!-- scope: analysis paper on coupling between weight quantization and KV-cache quantization
     deps: [[wkvquant]], [[kivi]]
     see-also: [[gptq]], [[awq]], [[kvquant]]
-->

# Coupling between Weight Quantization and KV-Cache Quantization
- **Core Insight:** Weight quant and KV-cache quant are *not* independent — quantization errors in W_k, W_v propagate into the KV cache, where they are then re-quantized, producing compound error that can be larger than the sum of individual contributions if not jointly calibrated.
- **Guideline:** When deploying W4 + KV4, calibrate them *jointly* (single block-reconstruction loss covering both) rather than sequentially; coupling-aware calibration recovers ~0.3 PPL on LLaMA-2-7B at W4-KV4 vs naive sequential pipelines.
- **Authors:** (placeholder — covered in joint W+KV papers including the [[wkvquant]] cross-block analysis and the [[qserve]] SmoothAttention section; consolidated as a topic-level entry)
- **Year:** 2024
- **URL:** consolidated; see [[wkvquant]] (https://arxiv.org/abs/2402.12065) and [[qserve]] (https://arxiv.org/abs/2405.04532)
- **Relevant topics:** coupled quantization, error propagation, joint calibration, W+KV

## Abstract
This entry consolidates the analytical observation — surfaced by WKVQuant, QServe (SmoothAttention) and KVQuant ablations — that errors introduced by W4 quantization in the K/V projection matrices propagate forward into the cache and are *then* re-quantized to KV4, producing compound errors larger than the sum of independent contributions. Joint calibration (a single objective for W and KV scales) addresses this; sequential calibration leaves PPL on the table.

## Key Contributions (consolidated)
- Formalises the compound-error problem: `ε_total = ε_KV(W + ε_W)` is not linear in ε_W.
- Demonstrates that sequential calibration (W first, then KV given quantized W) under-counts the interaction.
- Joint optimisation objective: cross-block reconstruction loss covering both quant decisions.
- SmoothAttention-style equivalent-transformation insertion as a mitigation that decouples without joint training.

## Key Figures/Tables to Study
- WKVQuant ablation table: sequential W4 → KV4 vs joint W4+KV4 — joint wins by ~0.3 PPL.
- QServe SmoothAttention diagram (Figure 5) — the per-head scaling that re-couples without retraining.
- KVQuant ablation: order-of-operations matters.

## Technical Details

### Error propagation analysis
Let W_k_hat = W_k + ε_W. Then K_t_hat = W_k_hat · x_t = K_t + ε_W · x_t. After KV4 quant:
`K_t_final = Quant(K_t + ε_W · x_t)`
Quantization error is non-linear in its input (rounding boundary depends on scale that depends on inputs), so the combined error is *not* simply ε_W · x_t + ε_KV. Empirically the combined PPL hit is 1.5–2× the sum.

### Joint calibration objective
Replace the standard per-linear `min_{s_W} ||y_FP − y_W4||²` and per-layer `min_{s_KV} ||y_FP − y_KV4||²` with a single objective:
`min_{s_W, s_KV} || y_FP − y_{W4, KV4} ||²`
covering a block. Requires the activation pass to be computed with both quantizations active during calibration; doubles the compute per calibration step but no algorithmic complexity.

### Decoupling via SmoothAttention (QServe)
Introduce per-head invertible scaling s such that `Q' = Q s`, `K' = K / s`. Distributes magnitude between Q and K so the KV4 path sees a flatter K distribution — reduces ε_KV without retraining W.

### Decoupling via cross-block regularization (WKVQuant)
`L = Σ_b ||h_b^FP − h_b^quant||² + λ Σ_{b<b'} ||h_{b'}^FP − h_{b'}^quant||²`
The cross-block term explicitly accounts for compounding across multiple blocks of (W4, KV4).

### Practical recipe
- If you have a calibration corpus: do joint calibration.
- If not: insert SmoothAttention with per-head scales derived from activation statistics (one calibration pass).
- Always validate at long context, where coupling errors compound across more decode steps.

## Connections
- Joint W+KV paper: [[wkvquant]].
- SmoothAttention solution: [[qserve]].
- Quant ordering analysis in: [[kvquant]] ablations, [[gear]] ablations.
- General error propagation: [[quantization-error-propagation]] (classical).
- KV-only siblings: [[kivi]], [[skvq]], [[qaq]].
