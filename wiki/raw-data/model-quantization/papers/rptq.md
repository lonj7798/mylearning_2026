<!-- scope: RPTQ — reorder channels into similar-range clusters before per-cluster activation quant
     deps: [[zeroquant]], [[smoothquant]]
     see-also: [[awq]], [[atom]]
-->

# RPTQ: Reorder-based Post-training Quantization for Large Language Models
- **Core Insight:** Activation quantization fails not because of single-token outliers but because of *inter-channel range variance* — different hidden dimensions have systematically different scales; reordering the channels into clusters of similar-range and assigning one scale per cluster (instead of per-tensor) makes 3-bit activation PTQ viable.
- **Guideline:** When pushing activations below INT8, cluster the C_in channels (k-means on per-channel max-abs from calibration) into K=32–64 groups, permute the channels in-place, and quantize per-cluster; fold the reorder permutation into the previous LayerNorm and next Linear so inference is free.
- **Authors:** Zhihang Yuan, Lin Niu, Jiawei Liu, Wenyu Liu, Xinggang Wang, Yuzhang Shang, Guangyu Sun, Qiang Wu, Jiaxiang Wu, Bingzhe Wu
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2304.01089
- **Relevant topics:** activation channel reordering, cluster-wise quant, sub-INT8 activation, OPT-175B

## Abstract
RPTQ targets activation quantization in LLMs and identifies that the binding constraint is the variance of per-channel ranges, not isolated outliers. The fix is structural: cluster the input channels by their typical magnitude (k-means on per-channel max-abs over calibration data), permute the activation tensor so each cluster's channels are contiguous, and assign a separate INT-k scale per cluster. The permutation is folded into the preceding LayerNorm (γ, β reordered) and the next Linear (weight rows reordered), so runtime adds nothing. With 32–64 clusters, RPTQ achieves 3-bit activations on OPT-175B with up to 80% memory reduction and limited accuracy loss.

## Key Contributions
- Reframes the activation-quant problem as one of *channel-range variance* rather than outlier presence.
- Cluster-wise (instead of per-tensor or per-token) activation quantization with K = 32–64 clusters.
- Permutation absorbed into adjacent ops → zero runtime overhead.
- First viable A3 (3-bit activation) PTQ on 175B-class models.

## Key Figures/Tables to Study
- **Figure 3:** per-channel activation range histogram before/after clustering — the spread shrinks dramatically inside each cluster.
- **Table 4:** OPT-175B W4A3 vs W4A8 — RPTQ keeps within 2 ppl at A3.

## Technical Details

### Channel clustering
For input activation `X ∈ R^{T × C_in}`, compute per-channel statistic
```
r_j = max_t |X_{t, j}|     (collected over calibration data)
```
Run k-means on `{r_j}` with K clusters. Result: partition `{1, …, C_in} = ⊔_k S_k`, each cluster containing channels of similar magnitude.

### Permutation π
Define a permutation `π` that orders channels so cluster S_1 occupies indices `[0, |S_1|)`, S_2 occupies `[|S_1|, |S_1|+|S_2|)`, etc.

### Per-cluster quantization
For each cluster k:
```
s_k = max_{j ∈ S_k} r_j / (2^{b-1} − 1)
X̂_{·, S_k} = round(X_{·, S_k} / s_k) · s_k
```
One scale per cluster → far fewer scales than per-channel, but range-matched within cluster.

### Folding the permutation
For the canonical pattern `LayerNorm(γ, β) → Linear(W) → … → Linear(V)`:
- Reorder LayerNorm affine: `γ ← γ[π]`, `β ← β[π]`.
- Reorder Linear input dim: `W ← W[π, :]`.
- For the *output* side (Linear → next-Linear → activation quant), apply π on the output channels of the producing Linear: `W ← W[:, π_out]`.
- No runtime gather/scatter is needed.

### Quantization grid
Standard symmetric INT-k. b = 3 for activations is the target; weights typically GPTQ-INT4. Combined as W4A3.

### Hyperparameters
| Knob | Value |
|------|-------|
| Clusters K | 32–64 |
| Activation bits | 3 (target), 4 / 8 (easier) |
| Weight bits | 4 (GPTQ companion) |
| Calibration | 512 sequences × 512 tokens |
| Cluster algorithm | k-means on max-abs |
| Symmetric / asymmetric | symmetric |

## Connections
- The activation-outlier-as-channel-variance framing extends: [[llm-int8]], [[smoothquant]].
- Per-cluster scale lineage carried forward: [[atom]] (W4A4 + KV4 with sub-channel reorder).
- Activation-aware weight quant cousin: [[awq]].
- Rotation-based successors that obviate clustering by Gaussianising activations: [[quarot]], [[spinquant]], [[duquant]].
