---
chapter: ch-17
course: model-quantization
phase: read
excerpt_of: "Pretraining Large Language Models with NVFP4 (Micikevicius et al., NVIDIA 2025)"
source_url: https://arxiv.org/abs/2509.25149
created_at: "2026-05-21"
---

# Excerpt: NVFP4 native pretraining — four ingredients

**Authors:** Micikevicius, Mishra et al. (NVIDIA, ~89 co-authors)
**Year:** 2025 (submitted 2025-09-29; rev. 2026-03-04)
**URL:** https://arxiv.org/abs/2509.25149
**Raw-data source:** [[raw-data/nvfp4-training]]

---

## What was done

A **12B-parameter dense transformer** trained for **10T tokens** entirely in NVFP4 (with ~3 % of FLOPs in BF16), matching the loss curve and downstream-eval averages of an FP8 baseline. First publicly reported 4-bit pretraining run at frontier scale.

---

## NVFP4 format recap

Three operand levels per GEMM:

- **Element:** FP4 E2M1. Representable: `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
- **Block scale:** 16 elements share one E4M3 (FP8) scale. *Fractional*, not power-of-two.
- **Tensor scale:** one FP32 scalar per tensor.

The hardware MMA (Blackwell 5th-gen Tensor Core) consumes all three operands and applies the scales during FP32 accumulation. No software dequant inside the GEMM loop.

Vs MXFP4: 32-element block with E8M0 (power-of-two) scale, no per-tensor scale. NVFP4's smaller block + richer scale halves the per-block MSE.

---

## The four ingredients (Table 2 ablation)

Each one is load-bearing — removing any one degrades loss / eval.

### (1) NVFP4 format itself

Two-level scaling structure:
- FP32 tensor scale absorbs *global* range.
- FP8 (E4M3) block scale absorbs *local* dispersion within 16 elements.
- 4-bit element carries the residual.

Without the per-tensor scale, the 16-element block scale has to absorb both global and local — fails at scale.

### (2) 2-D consistent quantization

For `Y = X · Wᵀ`, three GEMMs:

1. Forward: `X[1,16] × W[16,1] → Y`.
2. Input-grad: `dY × W → dX` (W blocked along O).
3. Weight-grad: `dYᵀ × X → dW` (X blocked along B·S).

If each GEMM picks its block layout independently, the quant noise on the same tensor X (or W) seen by forward and backward is *uncorrelated* → biased gradient relative to forward.

**Fix:** store each tensor with *both* block layouts (same data, two scale tables, ~0.4 % extra memory). Forward and weight-grad now see *identical* NVFP4 X; forward and input-grad see identical NVFP4 W.

### (3) Selective Random Hadamard Transform (RHT)

Applied to the input of:
- **FFN-gate** (`W_gate · X`)
- **Attention output projection** (`W_o · X`)

These are the two activations consistently identified across [[quarot]] / [[spinquant]] as carrying the worst per-channel outliers.

H is a **fixed random ±1 Hadamard of size 128×128** (composes with Blackwell's 128-wide hidden tiles). Folded into the weight offline: `W' = W · Hᵀ`. Zero inference cost — just a one-shot offline reshape.

Result: per-block amax drops by **~2–3×** on these layers, which is what keeps the 16-element block scale tight enough for FP4 to resolve.

### (4) Stochastic rounding on backward only

- **Forward:** round-to-nearest-even (RNE). Inference uses the same code path; deterministic outputs.
- **Backward:** for each FP4 element, round up with probability `x − ⌊x⌋`, else down. Preserves `E[round_SR(x)] = x` → unbiased gradient estimator.

Empirically: SR on forward had *no benefit* and *hurt inference parity*. SR on backward was load-bearing.

---

## Selective high precision

| Component | Precision | Why |
|-----------|-----------|-----|
| Embedding table | BF16 | precision-sensitive, ~0.6 % of params |
| Final RMSNorm | BF16 | tiny param count, large quality cost if quantized |
| LM-head | BF16 | output head, perplexity bottleneck |
| Attention LayerNorms (a handful, sensitivity-flagged) | BF16 | identified empirically |
| All 40 transformer blocks' Linear layers | NVFP4 | bulk of FLOPs |

Total: ~3 % of FLOPs stay BF16.

---

## Run specifics

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
| BF16 layers | embed, head, final RMSNorm, selected norms |
| Hardware | Blackwell GB200 |
| Loss gap vs FP8 | matches within run-to-run noise |

---

## Why the FP8 → FP4 step worked

Reading [[deepseek-v3-fp8]] and this paper together: the trajectory is *more structure in the scale*, not *fewer bits in the element*.

| Step | What changed |
|------|--------------|
| BF16 → FP8 (TE) | element bits ÷2; per-tensor scale |
| FP8 per-tensor → FP8 per-block (DSV3) | scale becomes per-128-block |
| FP8 per-block → NVFP4 | element bits ÷2; scale becomes per-16-block + per-tensor; RHT + SR + 2-D consistency added |

Each step needed the previous step's scale design to be *more structured* — block scale alone didn't work for FP4 because the 16-element block has too little range absorption. Adding the FP32 tensor scale (and the offline RHT to flatten outliers) is what closes the gap.

---

## Connections

- [[nvfp4]] — format spec page (separate file under `formats/`).
- [[deepseek-v3-fp8]] / [[excerpts/deepseek-v3-fp8]] — the FP8 ancestor recipe.
- [[mxfp4-pretraining]] / [[excerpts/mxfp4-pretraining]] — academic OCP-MXFP4 counterpart.
- [[mxfp4-native-hardware-2026]] / [[excerpts/mxfp4-native-hardware-2026]] — 2026 plot twist on Wgrad failure.
- [[quartet-ii]] — 2026 NVFP4 successor with MS-EDEN gradient estimator.
- [[nvfp4-qad]] — inference-time recovery via distillation.
- [[blackwell-quantization]] — the hardware substrate.
- [[quarot]] / [[spinquant]] / ch-14 — the rotation lineage motivating RHT.
- [[ch-17]] — parent synthesis.
