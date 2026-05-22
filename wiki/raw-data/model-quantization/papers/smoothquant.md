<!-- scope: SmoothQuant — equivalent-transformation W8A8 PTQ that migrates outlier difficulty from X into W
     deps: [[llm-int8]], [[int8]]
     see-also: [[awq]], [[omniquant]], [[outlier-channel-splitting]], [[oscar]]
-->

# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- **Core Insight:** Activation outliers and weight magnitudes can be rebalanced by a per-channel diagonal `diag(s)` that is mathematically invisible to the layer (`(X·diag(s)⁻¹)·(diag(s)·W) = X·W`); pushing some difficulty from X into W makes both INT8-quantizable, enabling true W8A8 LLM PTQ with no training.
- **Guideline:** For W8A8 deployment, run SmoothQuant offline with migration strength `α = 0.5` (try `α ∈ {0.4, 0.5, 0.8}` per model family); fuse `diag(s)⁻¹` into the preceding LayerNorm and `diag(s)` into the next Linear weight, then standard per-tensor/per-token INT8 GEMM works.
- **Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **Year:** 2022 (ICML 2023)
- **URL:** https://arxiv.org/abs/2211.10438
- **Relevant topics:** equivalent transformation, activation-outlier migration, W8A8 PTQ, fused scaling into LayerNorm

## Abstract
LLMs above ~6.7B contain per-channel activation outliers that defeat naive INT8 activation quantization (cf. [[llm-int8]]). SmoothQuant observes that those outliers are restricted to a small fraction of channels and that the same channels have *small* weight magnitudes — so one can introduce a per-channel diagonal `s` that scales activations *down* (smoothing) and weights *up*, leaving the matmul output unchanged. The optimal `s` is a power-law of the per-channel activation and weight maxes, controlled by a migration strength α (default 0.5). With this preprocessing, standard per-token activation × per-channel weight INT8 GEMM achieves W8A8 PTQ on OPT/BLOOM/LLaMA up to 530B with ≤0.1 ppl loss and 1.5–2× speedup over FP16.

## Key Contributions
- The first practical W8A8 LLM PTQ — both weights *and* activations to INT8 — at 175B+ scale.
- A closed-form equivalent transformation that requires no calibration objective optimization, no gradients, no extra parameters at inference.
- The migration-strength knob α giving a smooth tradeoff between "activations easy, weights hard" (α→1) and "weights easy, activations hard" (α→0).
- Fuses cleanly into the architecture: `diag(s)⁻¹` is absorbed into the preceding LayerNorm's affine; `diag(s)` is absorbed into the next Linear's weight. Zero runtime overhead.
- Enables single-node serving of OPT-175B and BLOOM-176B with INT8 throughput.

## Key Figures/Tables to Study
- **Figure 2:** the per-channel activation magnitude heatmap before/after smoothing — the visual proof of the migration.
- **Figure 3:** the equivalent transformation diagram with `diag(s)⁻¹` folded into the LayerNorm.
- **Table 4/5:** OPT/BLOOM W8A8 results vs LLM.int8() and ZeroQuant — SmoothQuant is the only one that's actually faster than FP16 at 175B.

## Technical Details

### Equivalent transformation
For input `X ∈ R^{T × C_in}` and weight `W ∈ R^{C_in × C_out}`, introduce diagonal `s ∈ R^{C_in}`:
```
Y = X · W
  = (X · diag(s)⁻¹) · (diag(s) · W)
  = X̂ · Ŵ
```
Mathematically identical; the per-channel scale `s_j` shrinks the j-th activation column and grows the j-th weight row.

### Choice of `s` (migration strength)
Per input channel j:
```
s_j = max(|X_{·,j}|)^α  /  max(|W_{j,·}|)^{1−α}
```
- `α` is the **migration strength** controlling how much of the difficulty moves from X into W.
- α = 0 → trivial s = 1 / max|W| → all difficulty stays in activations (no smoothing).
- α = 1 → s = max|X| → entire activation max moved into weights.
- α = 0.5 (default) → balanced; the activation max and weight max after transformation are equal.

After transformation: `max(|X̂_{·,j}|) = max(|X_{·,j}|)^{1−α} · max(|W_{j,·}|)^α` and `max(|Ŵ_{j,·}|)` is the same — symmetric.

### Calibration
A small calibration set (≈512 sequences, 512 tokens) is used only to compute per-channel `max(|X|)`. No optimization, one forward pass.

### Architectural fusion
For a typical transformer block `LayerNorm → Linear`, SmoothQuant absorbs:
- `diag(s)⁻¹` into the LayerNorm's affine `γ_j ← γ_j / s_j`, `β_j ← β_j / s_j`.
- `diag(s)` into the subsequent Linear's weight `W_j ← s_j · W_j`.

So `s` exists only at calibration time; inference graph is unchanged.

### Per-model α
| Model family | α |
|--------------|---|
| OPT | 0.5 |
| BLOOM | 0.5 |
| LLaMA | 0.85 (stronger outliers) |
| GLM | 0.75 |
| Falcon | 0.6 |

### Final inference quant
After smoothing: standard W8A8 with per-token activation absmax + per-channel weight absmax, dispatched to INT8 tensor cores. Achieves 1.51× speedup vs FP16 on OPT-175B with no accuracy loss.

### Hyperparameters
| Knob | Value |
|------|-------|
| α (migration strength) | 0.5 (LLaMA: 0.85) |
| Calibration sequences | 512 × 512 tokens |
| Activation quant | per-token, INT8 absmax |
| Weight quant | per-channel, INT8 absmax |
| Symmetric / asymmetric | symmetric |

## Connections
- The problem this solves: [[llm-int8]] (isolates outliers in FP16) and Dettmers' blog [[dettmers-llm-int8-blog]].
- Pre-LLM lineage of "move outlier difficulty around" idea: [[oscar]] (Outlier Suppression with Equalization), [[outlier-channel-splitting]].
- Activation-aware *weight-only* cousin (only scales W, no equivalent move): [[awq]].
- Learnable extension (gradient-trained transformation): [[omniquant]], [[affinequant]].
- Rotation-based descendants that go further (full unitary instead of diagonal): [[quip]], [[quarot]], [[spinquant]], [[duquant]], [[flatquant]].
- Framework references: integrated into TensorRT-LLM [[tensorrt-llm-quant]] and vLLM [[vllm-quant]].
