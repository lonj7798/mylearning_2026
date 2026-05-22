<!-- scope: OCS — duplicate-and-halve outlier channels to fit a quantizer grid
     deps: percentile-clipping, oscar
     see-also: smoothquant, awq, llm-int8
-->

# Improving Neural Network Quantization without Retraining using Outlier Channel Splitting (OCS)
- **Core Insight:** Instead of clipping outlier channels (which loses information) or assigning them more bits (which breaks GEMM regularity), duplicate each outlier channel and halve the weights — the linear layer's output is exact (since two copies of W/2 add back to W) but the per-channel maximum is now half the original, fitting the quantizer's clip range.
- **Guideline:** For any input channel c with |W_:,c|_max > θ (e.g. θ = quantile_99), split c into two output replicas: replace channel c in the next layer with two copies, each at W_:,c/2; this is exact in FP and reduces the quantizer's required range by 2× per split; iterate splits until all channels fit the target range; no retraining required.
- **Authors:** Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Christopher De Sa, Zhiru Zhang
- **Year:** 2019 (ICML)
- **URL:** https://arxiv.org/abs/1901.09504
- **Relevant topics:** outlier handling, channel splitting, equivalent transformation, data-free PTQ

## Abstract
OCS targets the same outlier-activation pain point as percentile clipping but with a different fix: instead of saturating the outlier channel (information loss), it duplicates the channel and halves the weight. The mathematical insight is that an input channel x_c with weight column W_:,c is equivalent to two copies of x_c with weight columns W_:,c/2 — summing back to the original output. This redistributes outlier magnitude across two duplicated channels, each now half the size, so the quantizer's effective clip range can be tightened. OCS is data-free, retraining-free, and complementary to any subsequent calibration; on ImageNet ResNet and VGG, it recovers 1–2% over plain percentile clipping at 6-bit, and was a structural inspiration for SmoothQuant.

## Key Contributions
- Channel splitting as an exact (no information loss) equivalent transformation for outlier handling.
- Greedy algorithm: split channels with the largest |W_max| / quantile gap, iterate.
- Data-free, training-free: applies in seconds.
- Empirical improvement of 0.5–2% over baselines at 6/4-bit PTQ on ImageNet CNNs.
- Conceptual ancestor of SmoothQuant's per-channel rescale (SmoothQuant is the smooth generalisation: continuous per-channel scale instead of discrete split).

## Key Figures/Tables to Study
- **Figure 1** — diagram showing channel splitting on a single conv layer.
- **Figure 3** — per-channel weight magnitude before/after splitting iterations.
- **Table 3** — ResNet-50 6-bit PTQ: OCS vs percentile-clipping baseline.

## Technical Details

### The splitting transformation
Linear layer: `y = W x = Σ_c W_:,c · x_c`.
Pick an input channel c. Replace the c-th column W_:,c with two columns W_:,c/2 and W_:,c/2, and replace the c-th input feature x_c with two identical copies x_c, x_c. Then:
`y' = ... + (W_:,c/2) · x_c + (W_:,c/2) · x_c + ... = ... + W_:,c · x_c + ...`
Exactly equal to y. But the per-column maximum is now half: |W_:,c/2|_max = |W_:,c|_max / 2.

### Activation duplication
The duplicated input is sourced by duplicating the previous layer's output channel. For convolution, this means the previous conv has two output channels with identical weights (which can be merged at deploy time into a single channel with 2× output replication, or kept as two separate channels with shared weights). For BN+ReLU, the duplication is straightforward.

### Selection rule (greedy)
For each input channel c, compute its outlier score:
`s_c = |W_:,c|_max / quantile_99(|W|)`
Sort channels by s_c. Split the top-K channels (or split iteratively until all s_c ≤ threshold).

### Splitting iteration
- Start with original W.
- For each iteration: pick the channel with the largest s_c, split.
- Re-evaluate s_c (a split channel's halved magnitude may now be below threshold; other channels may rise to the top).
- Stop when no channel exceeds the threshold OR a max-split budget is hit.

### Cost
- Each split increases the layer's input dimension by 1.
- Typical: ~5–10% input dimension growth for clean 6-bit PTQ.
- Negligible inference overhead given GEMM amortisation.

### Why it's data-free
Splitting decision depends only on weight statistics. Activation statistics matter only if you want to split by activation outliers (a variant), but the basic algorithm is purely weight-driven.

### Limits
- Per-channel weights only — doesn't fix per-token activation outliers.
- Doesn't address shared activation outliers that appear across many channels (the LLM regime — see [[llm-int8]]).
- 4-bit regime needs combination with QAT or AdaRound.

## Connections
- [[percentile-clipping]] — alternative outlier handling (lossy clip vs lossless split).
- [[oscar]] — same problem in BERT, addressed by LN γ migration instead of splitting.
- [[data-free-quantization]] — same data-free, training-free philosophy (DFQ uses ReLU rescaling; OCS uses duplication).
- [[smoothquant]] — LLM-era generalisation: continuous per-channel scaling factor instead of discrete duplication; the conceptual heir.
- [[awq]] — per-channel scaling driven by activation magnitude during weight-only PTQ.
- [[llm-int8]] — extreme outlier regime where neither splitting nor smoothing alone suffices; mixed-precision INT8/FP16 path needed.
