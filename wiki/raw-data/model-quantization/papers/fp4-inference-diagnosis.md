<!-- scope: Layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4 inference
     deps: [[nvfp4]], [[mx-formats]], [[mxfp4-native-hardware-2026]]
     see-also: [[awq]], [[smoothquant]], [[quarot]], [[spinquant]]
-->

# Diagnosing FP4 Inference: Layer-wise and Block-wise Sensitivity Analysis of NVFP4 and MXFP4
- **Core Insight:** FP4 sensitivity is not uniform across transformer components: MLP up/down projections dominate degradation, attention projections are much less sensitive, and early blocks can be fragile, especially under MXFP4.
- **Guideline:** Do component-wise and block-wise ablations before committing to full FP4 inference; protect MLP up/down projections and do not blindly leave only final layers in higher precision.
- **Authors:** Musa Cim, Burak Topcu, Mahmut Taylan Kandemir
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2603.08747
- **Relevant topics:** FP4 inference, NVFP4, MXFP4, sensitivity analysis, Qwen2.5, mixed precision

## Abstract
This paper studies how FP4 inference behaves across transformer components and depth. It evaluates MXFP4 and NVFP4 on Qwen2.5 models at 0.5B, 7B, and 14B scales using controlled component-wise and block-wise isolation. The result is a diagnostic map: MLP up- and down-projection layers are consistently the most sensitive, gate projections are moderately sensitive, attention projections are much less sensitive, and sensitivity does not always concentrate in final blocks.

## Key Contributions
- Provides a controlled FP4 inference sensitivity study across both MXFP4 and NVFP4.
- Tests three Qwen2.5 model sizes, giving a limited but useful scale axis.
- Identifies MLP up/down projections as the main components to protect in mixed-precision FP4 deployment.
- Shows that early transformer blocks can be highly sensitive, particularly with MXFP4.
- Supplies evidence that "keep the last N layers high precision" is an incomplete heuristic for FP4.

## Key Figures/Tables to Study
- Component-wise sensitivity heatmap: the best visual summary of where FP4 hurts.
- Depth-wise block ablation: shows early-block fragility.
- NVFP4 vs MXFP4 comparison: illustrates how block-scale design changes error profile.
- Cross-scale table over Qwen2.5 0.5B/7B/14B: useful for whether small-model diagnostics transfer.

## Technical Details

### Component ranking
The headline qualitative ranking:
```
MLP up/down projections  >  gate projection  >  attention projections
```
This connects directly to [[awq]] and [[smoothquant]], which already treat activation/weight outliers in MLP paths as the most important low-bit obstacle.

### Why this matters for mixed precision
FP4 deployments often need exceptions. This paper gives a more concrete exception policy:
- Raise precision for MLP up/down projections first.
- Check early blocks, not only final blocks.
- Treat NVFP4 and MXFP4 as separate deployment targets, not interchangeable "FP4".

### Format contrast
NVFP4 uses a 16-element block with an FP8 scale and per-tensor scaling in the NVIDIA format. MXFP4 uses a 32-element block with an E8M0 shared exponent. The wider MXFP4 block can amplify sensitivity when a block contains a few high-energy channels.

## Connections
- [[nvfp4]] and [[mx-formats]] — the two FP4 families compared.
- [[mxfp4-native-hardware-2026]] — same author lineage; training-side diagnosis for MXFP4.
- [[awq]] — activation-aware scaling often protects the same MLP-heavy paths.
- [[quarot]] / [[spinquant]] — rotations can reduce component outliers before FP4 quantization.
- [[nvfp4-qad]] — QAD can recover accuracy after FP4 insertion; this paper tells you where recovery is most needed.

## Notes
This is a diagnostic paper, not a new quantization algorithm. It is valuable in the course because it teaches students how to evaluate FP4 failures instead of treating FP4 as a single global switch.
