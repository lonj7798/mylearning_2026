<!-- scope: PACT — learn the activation clipping threshold inside QAT
     deps: straight-through-estimator, dorefa-net
     see-also: lsq, lsq-plus, quantization-mapping
-->

# PACT: Parameterized Clipping Activation for Quantized Neural Networks
- **Core Insight:** Activations after ReLU have unbounded range; quantizing to k bits requires a clip threshold α, and treating α as a learnable parameter trained jointly with weights (via STE) recovers most of the FP accuracy at low bit-widths.
- **Guideline:** Replace ReLU with PACT(x; α) = clip(x, 0, α); init α ≈ 10 (or per-layer max); add a small L2 penalty on α to discourage degenerate growth; apply uniform k-bit quantization on the clipped range.
- **Authors:** Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, Kailash Gopalakrishnan
- **Year:** 2018
- **URL:** https://arxiv.org/abs/1805.06085
- **Relevant topics:** activation quantization, learnable clip, k-bit QAT, ReLU replacement

## Abstract
PACT (PArameterized Clipping acTivation) addresses the asymmetry of QAT: weights have bounded distributions but post-ReLU activations are heavy-tailed and unbounded, so a fixed clip threshold either wastes resolution or clips too aggressively. PACT replaces ReLU with a clip-to-α function where α itself is a learned parameter, optimised by SGD via the straight-through estimator. With 4-bit weights and 4-bit activations, PACT matches FP ResNet-50 accuracy on ImageNet — the first method to do so at this precision.

## Key Contributions
- Learnable per-layer clip threshold α optimised alongside weights.
- Gradient of α through STE allowing joint SGD optimisation.
- L2 regularisation on α to prevent unbounded growth.
- Empirical breakthrough: 4-bit ImageNet ResNet-50 within ~1% of FP.
- Composable with any uniform k-bit weight quantizer (DoReFa, learned, etc.).

## Key Figures/Tables to Study
- **Figure 2** — α dynamics across training: learns layer-specific clip.
- **Table 3** — ImageNet ResNet ablation: PACT vs ReLU+fixed clip across bit-widths.

## Technical Details

### PACT activation
`y = PACT(x; α) = 0.5·(|x| − |x − α| + α) = clip(x, 0, α)`

### Quantize the clipped output
`y_q = round(y · (2^k − 1) / α) · α / (2^k − 1)`
i.e. uniform k-bit quantization on [0, α].

### Backward (STE w.r.t. x)
`∂y/∂x = 1[0 ≤ x ≤ α], 0 otherwise`
Same as clipped STE — gradient passes through the active region.

### Backward (gradient w.r.t. learned α) — the key formula
`∂L/∂α = ∂L/∂y · ∂y/∂α = ∂L/∂y · 1[x ≥ α]`
i.e. α only receives gradient from saturated (clipped) samples. Intuition: if too many samples are clipped, ∂L/∂α pushes α up; if α is so high that quantization noise dominates, the loss landscape pushes α down via the indirect path through y_q.

### Regularisation
`L_total = L_task + λ·α²`
Prevents α from growing unbounded when training noise drives it upward. Typical λ = 10⁻⁴.

### Initialisation
α₀ = max activation observed in a calibration pass, or a fixed constant like 6 (matches ReLU6).

## Connections
- [[straight-through-estimator]] — both the activation backward and the α gradient route through STE.
- [[dorefa-net]] — supplies the uniform k-bit quantizer applied after PACT clip.
- [[lsq]] — generalises PACT by learning the step size Δ (which equals α/(2^k−1)) rather than α directly; mathematically equivalent for symmetric quantizers.
- [[lsq-plus]] — extends LSQ to asymmetric signed clip [α_neg, α_pos], generalising PACT.
- [[quantization-mapping]] — the calibration-vs-learning taxonomy PACT sits inside.
