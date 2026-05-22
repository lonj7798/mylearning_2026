<!-- scope: DFQ — weight equalization + bias correction without calibration data
     deps: quantization-mapping
     see-also: zeroq, smoothquant, integer-only-inference
-->

# Data-Free Quantization Through Weight Equalization and Bias Correction (DFQ)
- **Core Insight:** Per-channel weight magnitude imbalance is the dominant source of per-tensor-quant error; you can rescale consecutive layers to equalize ranges using only the weights (no data), then correct the residual mean error analytically — getting INT8 accuracy without a single calibration sample.
- **Guideline:** For any pair of layers L_i → L_{i+1} sharing a positive-homogeneous activation (ReLU, ReLU6), compute per-channel rescale factors `s_c = √(r_i^c / r_{i+1}^c)`, divide W_i by s and multiply W_{i+1} by s; then estimate per-channel quantization bias from W stats and subtract from b.
- **Authors:** Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling
- **Year:** 2019 (ICCV)
- **URL:** https://arxiv.org/abs/1906.04721
- **Relevant topics:** data-free PTQ, weight equalization, bias correction, per-tensor → per-channel

## Abstract
DFQ tackles the practical pain point that calibration data is sometimes unavailable (privacy, deployment-time-only quantization, on-device). Two purely-weight-based tricks suffice: (1) cross-layer weight equalization exploits the positive scaling invariance of ReLU, balancing per-channel weight magnitudes across consecutive layers; (2) analytical bias correction estimates the systematic offset that quantization introduces in each output channel and subtracts it from the bias. On MobileNetV2 — the QAT worst-case — DFQ recovers near-FP32 INT8 accuracy without touching a single image.

## Key Contributions
- Cross-layer equalization (CLE): a closed-form per-channel rescale that minimises range mismatch.
- Bias correction (BC): analytical estimate of mean quantization error per channel, absorbed into bias.
- Demonstrates per-tensor INT8 quantization for MobileNetV2 within ~0.5% of FP32 — without data.
- Sets up the equivalent-transformation paradigm later resurrected by SmoothQuant for LLMs.

## Key Figures/Tables to Study
- **Figure 4** — per-channel weight range histograms before/after equalization (visualises the bug + the fix).
- **Table 5** — MobileNetV2 INT8 accuracy: DFQ vs naive vs QAT.

## Technical Details

### ReLU positive-scaling invariance
For W_i, W_{i+1} connected by ReLU and any per-output-channel S > 0:
`ReLU(W_i x + b_i) → W_{i+1} · ReLU(W_i x + b_i) + b_{i+1}`
`= W_{i+1}S^{-1} · ReLU(SW_i x + Sb_i) + b_{i+1}`
i.e. dividing W_i by S and multiplying W_{i+1} by S leaves outputs unchanged.

### Cross-layer equalization (CLE)
Let r_i^c = per-channel range of W_i (output channel c) and r_{i+1}^c = per-input-channel range of W_{i+1}. Choose:
`S^c = (1/r_{i+1}^c) · √(r_i^c · r_{i+1}^c) = √(r_i^c / r_{i+1}^c)`
After applying S, both layers have geometric-mean ranges, minimising max-per-channel divergence from per-tensor scale.

Iterate over all adjacent (L_i, L_{i+1}) pairs to convergence (~5 sweeps).

### Bias correction (BC)
Let ε_c = E[(W_c − Q(W_c)) x] = (W_c − Q(W_c)) · E[x] per output channel c.
E[x] is approximated from the previous layer's BN running mean. Then:
`b_c ← b_c − ε_c`
absorbing the mean quantization error into the bias.

### Why it works without data
- CLE uses only weight statistics (ranges).
- BC uses BN's running mean as a free, training-time-recorded surrogate for E[x].
- Both fixes are deterministic, closed-form, and require zero forward passes through data.

### Limits
- CLE requires positive-homogeneous activations (ReLU/ReLU6); fails on Swish/h-swish.
- BC requires BN; fails on LayerNorm-only networks (transformers).

## Connections
- [[quantization-mapping]] — DFQ is the data-free corner of that taxonomy.
- [[zeroq]] — different no-data approach: synthesise calibration data from BN stats.
- [[smoothquant]] — LLM-era reincarnation of the equivalent-transformation idea (migrate activation outliers into weights).
- [[awq]] — also uses per-channel scaling absorbed into adjacent weights.
- [[integer-only-inference]] — DFQ targets the same INT8 deployment pipeline.
