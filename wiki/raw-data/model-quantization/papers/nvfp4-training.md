<!-- scope: NVFP4 native LLM pretraining on Blackwell (NVIDIA, 2025/2026)
     deps: [[nvfp4]], [[mx-formats]]
     see-also: [[microscaling-formats]], [[deepseek-v3-fp8]], [[mxfp4-pretraining]], [[quartet-ii]], [[nvfp4-qad]]
-->

# Pretraining Large Language Models with NVFP4
- **Core Insight:** With a 16-element block-scaled FP4 format (E4M3 block scale + FP32 per-tensor scale), 2-D consistent quantization across forward and backward, random Hadamard transforms on selected tensors, and stochastic rounding on gradients, a 12B model can be pretrained in 4-bit precision on 10T tokens and match an FP8 baseline.
- **Guideline:** Treat NVFP4 pretraining as FP8 + extra precautions: keep the per-tensor FP32 scale outside the block to absorb global range, use random Hadamard on the two GEMMs that see outliers (Wgate/Wup and Wo), apply stochastic rounding only on the backward GEMM, and leave the final ~few % of layers (head, last block, embeddings) in higher precision.
- **Authors:** Micikevicius, Mishra et al. (NVIDIA, ~89 co-authors)
- **Year:** 2025 (submitted 2025-09-29; rev. 2026-03-04)
- **URL:** https://arxiv.org/abs/2509.25149
- **Relevant topics:** FP4 native pretraining, Blackwell tensor cores, microscaling, two-level scaling, stochastic rounding, random Hadamard

## Abstract
NVIDIA's NVFP4 pretraining paper documents the first publicly reported 4-bit pretraining run at frontier scale: a 12B-parameter dense transformer trained for 10T tokens entirely in NVFP4 (with a small set of high-precision exceptions), matching the loss curve and downstream-eval averages of an FP8 baseline. The recipe rests on four ingredients: (1) the NVFP4 format itself — 16-element FP4 (E2M1) blocks with an FP8 (E4M3) block scale and an FP32 per-tensor scale — which Blackwell's 5th-gen Tensor Cores execute natively; (2) a *2-D consistent* scaling scheme that uses the same block layout on the forward GEMM and its corresponding backward GEMMs (input-grad and weight-grad); (3) random Hadamard transforms applied to the activations entering the FFN's gate/up and the attention output projection, to spread outliers across the 16-element block; (4) stochastic rounding on the backward GEMMs to keep the gradient estimator unbiased.

## Key Contributions
- Defines the production NVFP4 block scheme: **16 FP4 elements + 1 E4M3 block scale + 1 FP32 per-tensor scale** (two-level scaling). The FP32 tensor scale fixes the global range; the per-block FP8 scale fixes local dispersion; the 4-bit element carries the actual data.
- Shows a 12B / 10T-token NVFP4 pretrain matches FP8 in loss and average downstream score — first 4-bit run of this size.
- **2-D consistent quantization:** forward computes the GEMM on activation blocks of size [B, 16]; the backward weight-grad GEMM reuses the *same* block layout (transposed) so that the quantization noise cancels rather than compounds across the forward/backward pair.
- **Selective random Hadamard transform (RHT):** applied only to the two layers where outliers are concentrated (FFN-gate input, attention-output input); RHT spreads the outlier energy across the 16 elements of a block so a single rogue channel no longer dominates the block scale.
- **Stochastic rounding** on the backward GEMM only; deterministic round-to-nearest on forward. Backward SR preserves the unbiased-gradient property; forward RNE keeps inference-time match.
- **Selective high-precision layers:** the embedding, the final RMSNorm + LM-head, and a handful of attention LayerNorms stay in BF16 — only ~3 % of FLOPs.

## Key Figures/Tables to Study
- **Figure 1 (paper):** the four-component recipe diagram (NVFP4 + 2-D consistent + RHT + SR) — the canonical summary slide.
- **Figure 4:** the loss curve overlay vs FP8 baseline across all 10T tokens — flat gap throughout, no divergence at the tail.
- **Table 2:** ablation that removes each of the four ingredients in turn and shows the loss / eval drop — the cleanest evidence that all four are load-bearing.
- **Section 3 layout figure:** the block-scaled tensor diagram showing the FP32 scale wrapping the FP8 scales which wrap the 16-element FP4 blocks.

## Technical Details

### NVFP4 format (recap, see [[nvfp4]] for the spec page)
- Element: FP4 E2M1 (1 sign, 2 exponent, 1 mantissa) — representable values are {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
- Block: **16 elements** sharing one **E4M3** scale (note: shorter blocks than MX's 32-element block, and a richer FP8 scale instead of MX's E8M0).
- Per-tensor: one **FP32** scalar, applied multiplicatively on top of the block scale (the "two-level" structure).
- Hardware: native on Blackwell 5th-gen Tensor Cores (SM 10.x) — the FP4 GEMM consumes the FP4 elements, FP8 block scale, and FP32 tensor scale as separate operands.

### 2-D consistent quantization
For a linear layer Y = X · W^T (X: [B·S, C], W: [O, C]), three GEMMs are needed across the step:
1. Forward: X[1,16] blocks × W[16,1] blocks → Y.
2. Input-grad: dY × W → dX, requires W blocked along O.
3. Weight-grad: dY^T × X → dW, requires X blocked along B·S.

If each of the three GEMMs picks its own block layout independently, the quantization noise on the same tensor (X or W) seen by forward and backward is *uncorrelated* and the gradient becomes biased relative to the forward computation. The 2-D scheme stores the tensor with *both* block layouts (it's the same data, just two scale tables), so forward and weight-grad see exactly the same NVFP4 values for X, and forward and input-grad see exactly the same NVFP4 values for W. The cost is one extra small scale table per tensor (≈ 0.4 % memory).

### Random Hadamard transform (RHT)
- Applied to the **input** of FFN-gate (W_gate · X) and to the **input** of the attention output projection (W_o · X) — the two activations consistently identified across QuaRot/SpinQuant work as carrying the worst per-channel outliers.
- Hadamard matrix H is a fixed random sign matrix of size 128 × 128 (so it composes neatly with the 128-wide hidden tiles that Blackwell loads).
- The transform is folded into the corresponding weight (W' = W · H^T), so at inference time there is no extra kernel cost — just a one-shot offline reshape of W.
- Result: per-block amax drops by ~2-3× on those layers, which is what keeps the 16-element block scale tight enough for FP4 to work.

### Stochastic rounding
- Forward: round-to-nearest-even (RNE). Inference uses the same code path; we want deterministic outputs.
- Backward: SR — for each element, with probability (x - ⌊x⌋) round up, else down. This keeps E[round_SR(x)] = x, so the gradient estimator stays unbiased.
- Empirically, SR on forward had no benefit but hurt inference parity; SR on backward was load-bearing for matching FP8 loss.

### Selective high precision
- Embedding lookup: BF16 (the embedding table itself stays BF16, ~0.6 % of params).
- Final RMSNorm + LM-head: BF16.
- A handful of attention LayerNorms identified by sensitivity analysis: BF16.
- Everything else (all linear layers in all 40 blocks): NVFP4 weight + NVFP4 activation + FP32 partial-sum accumulation.

### Run specifics
| Knob | Value |
|------|-------|
| Model | 12B dense transformer |
| Tokens | 10T |
| Block size | 16 |
| Block scale | FP8 E4M3 |
| Tensor scale | FP32 |
| Forward rounding | RNE |
| Backward rounding | Stochastic |
| RHT layers | FFN-gate input, attn-out input |
| BF16 layers | embed, head, final RMSNorm |
| Hardware | Blackwell GB200 |
| Loss gap vs FP8 | matches within run-to-run noise |

## Connections
- [[nvfp4]] — the format spec (separate file under `formats/`); this paper is the training procedure that uses it.
- [[mx-formats]] / [[microscaling-formats]] — the OCP MX family; NVFP4 is a more aggressive variant (smaller block, richer scale) of the same shared-exponent idea.
- [[deepseek-v3-fp8]] — the FP8 ancestor recipe; NVFP4 essentially generalizes DSV3's per-block scaling from 128-wide FP8 blocks to 16-wide FP4 blocks with a richer two-level scale.
- [[mxfp4-pretraining]] — the academic MXFP4 pretrain study; uses 32-element MX blocks with E8M0 scale instead of 16-element NVFP4.
- [[quartet-ii]] — 2026 follow-up that replaces ordinary stochastic rounding with MS-EDEN for lower-error unbiased NVFP4 gradient estimation.
- [[nvfp4-qad]] — 2026 production recovery recipe for NVFP4 inference checkpoints via distillation.
- [[quarot]] / [[spinquant]] — the rotation lineage that motivated the selective Hadamard transform used here.
- [[blackwell-quantization]] — model-report page on the hardware that natively executes NVFP4.
