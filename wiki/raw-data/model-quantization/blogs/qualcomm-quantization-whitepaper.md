<!-- scope: Qualcomm AI Research PTQ + QAT production whitepaper
     deps: [[int8]], [[adaround]]
     see-also: [[qualcomm-ai-research]], [[brecq]], [[data-free-quantization]]
-->

# Qualcomm AI Research — Neural Network Quantization Whitepaper
- **Core Insight:** Qualcomm's whitepaper is the single most-cited consolidation of production PTQ + QAT practice — taxonomy, design choices, and a decision tree for when to escalate from naive PTQ to advanced PTQ to QAT.
- **Guideline:** Use the whitepaper's escalation ladder — start with PTQ + per-channel weight quant + simple percentile clipping; only move to AdaRound, BRECQ, or QAT when accuracy drop exceeds tolerance.
- **Authors:** Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort (Qualcomm AI Research)
- **Year:** 2021 (updated 2022)
- **URL:** https://arxiv.org/abs/2106.08295
- **Relevant topics:** PTQ, QAT, per-channel quantization, AdaRound, calibration, deployment

## Summary
The Qualcomm whitepaper "A White Paper on Neural Network Quantization" is the canonical practitioner reference for production neural-network quantization. It cleanly separates the design space into three orthogonal axes — quantization granularity (per-tensor / per-channel / per-group), quantization scheme (symmetric / asymmetric / power-of-two), and quantization method (PTQ naive / PTQ advanced / QAT). It introduces a decision tree: try naive PTQ first; if accuracy degrades, add cross-layer equalization and bias correction (data-free); if still inadequate, run AdaRound (a few hundred calibration samples); if that fails, go to full QAT with STE. The paper has become the de-facto reference for hardware-friendly quantization recipes and is the foundation of Qualcomm's AIMET (AI Model Efficiency Toolkit) software.

## Key Points
- Defines the **PTQ → advanced PTQ → QAT escalation ladder** still used in 2026.
- Recommends per-channel weight quant + per-tensor activation quant as the production default.
- Cross-Layer Equalization (CLE) folds outlier channels into adjacent layers without data.
- Bias correction compensates for the mean shift introduced by weight rounding.
- AdaRound learns per-weight rounding direction via the layer-wise reconstruction loss.
- QAT uses STE through the quantizer and learns the quantization range.

## Technical Details

### Quantization formulation (uniform affine)
```
x_int = clamp(round(x / s) + z, q_min, q_max)
x_q   = s · (x_int − z)
```
- `s` (scale) and `z` (zero-point) per tensor / channel / group.
- Symmetric: `z = 0` always (typical for weights with INT8 signed).
- Asymmetric: `z ≠ 0` (typical for activations with ReLU).

### Granularity decision rule
| Tensor | Recommended granularity | Why |
|--------|------------------------|-----|
| weights (Conv/Linear) | per-output-channel | per-channel is free (broadcast over batch) |
| activations | per-tensor | per-channel adds memory + complex broadcasting |
| KV cache | per-token or per-head | distribution shifts across tokens |

### Escalation ladder
1. **Naive PTQ** — uniform symmetric weight quant per-channel, percentile activation clipping.
2. **+ CLE** — equalize the per-channel weight ranges across consecutive layers.
3. **+ Bias correction** — subtract the bias shift from `E[Wx] − E[Q(W)x]`.
4. **+ AdaRound** — learn rounding direction with `min ||Wx − Q(W)x||²` over ~1024 calibration samples.
5. **QAT** — fake-quant nodes in forward; STE in backward; train for ~5-10% of original schedule.

### Cross-Layer Equalization (CLE)
For a ReLU-bounded pair `Wₙ → Wₙ₊₁`, find a diagonal scaling `s` such that scaled ranges are equalized:
```
W̃ₙ = Wₙ · diag(s)⁻¹
W̃ₙ₊₁ = diag(s) · Wₙ₊₁
```
Network function is preserved exactly (homogeneity of ReLU); the quantization error is reduced.

### Bias correction
For weight quantization with rounding error `ΔW = Q(W) − W`:
```
b_corrected = b − E[ΔW · x]
```
Computed from a small calibration set (or from BatchNorm statistics in data-free mode).

## Connections
- [[adaround]] — same authors; the advanced-PTQ step in the ladder.
- [[brecq]] — block-wise reconstruction extension by the same group.
- [[data-free-quantization]] — DFQ method (Nagel 2019) that becomes CLE + bias correction in this paper.
- [[qualcomm-ai-research]] — lab that produced this whitepaper and the AIMET toolkit.
- [[intel-quantization]] — adjacent industry effort (Intel Neural Compressor) with similar API surface.
