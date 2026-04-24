---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "DeepSeek-AI — DeepSeek-V3 Technical Report (2024)"
source_url: https://arxiv.org/abs/2412.19437
created_at: "2026-04-23"
---

# Excerpt: DeepSeek-V3 — the production fp8 training recipe

**Paper:** *DeepSeek-V3 Technical Report*
**Authors:** DeepSeek-AI et al.
**Year:** 2024
**arXiv:** [2412.19437](https://arxiv.org/abs/2412.19437)
**URL:** https://arxiv.org/abs/2412.19437

DeepSeek-V3 is the **canonical 2024 fp8 reference implementation**: a 671B-parameter MoE with 37B active params/token, trained on 14.8T tokens in 2.788M H800 GPU-hours (~$5.58M at $2/GPU-hr). This excerpt focuses on the fp8 mixed-precision stack from §3 of the paper — per-block scaling, fp32 accumulator reconstruction, and the absence of a traditional loss scaler.

The architecture story (Multi-head Latent Attention, DeepSeekMoE, auxiliary-loss-free balancing, Multi-Token Prediction) is covered in the MoE and attention chapters; here we stay strictly on the precision recipe.

---

## 1. The high-level fp8 framework (§3.3 of the paper)

From the paper:

> "We propose a fine-grained mixed-precision framework utilizing the FP8 data format for training DeepSeek-V3. [...] Most compute-density operators, e.g. GEMMs, are implemented in FP8, while a few operations that are precision-sensitive (embedding module, output head, MoE gating modules, normalization operators, and attention operators) are kept in their original precisions (BF16 or FP32)."

The structure, decoded:

| Operator | Precision | Rationale |
|---|---|---|
| Attention / MLP matmuls (the GEMMs) | **fp8** | Where the FLOPs live; 2× speedup over bf16 on H100/H800 |
| Embedding lookup | bf16 | Sparse, precision-sensitive, fp8 quantization noise dominant |
| Output head (LM head) | bf16 | Softmax + cross-entropy downstream; fp8 logits would distort |
| MoE gating (router) | bf16 | Discrete routing decisions amplify quantization error |
| LayerNorm / RMSNorm | fp32 reductions | As [[excerpts/batch-vs-layer-norm]] §2 — never compromise |
| Attention softmax | fp32 | Exponentials are range-sensitive |
| Residual stream | bf16 | Carries outlier magnitudes across layers |
| Optimizer state (m, v, master weight) | fp32 | As [[excerpts/adam]] §2 |

**Notice:** fp8 is *only* the matmul path. The marketing line "we trained in fp8" is really "our GEMMs are fp8 and everything else is bf16/fp32." The infrastructure choice to carve out precision-sensitive ops from fp8 is the main engineering content of §3.3.

---

## 2. Per-block scaling (§3.3.2, Figure 6 of the paper)

Rather than per-tensor scaling (one scale per activation/weight tensor, as in NVIDIA Transformer Engine's default), DeepSeek-V3 uses **per-block** scaling:

- **Activations:** `1 × 128` tile-wise scaling — one scale factor per row of 128 elements along the feature dimension.
- **Weights:** `128 × 128` block-wise scaling — one scale factor per 128×128 weight tile.

The scaling formula, for a block `B` of activations:

```math
s_B = \frac{\text{fmax}_{\text{E4M3}}}{\text{amax}(B)} = \frac{448}{\max_{x \in B} |x|}
B_{\text{fp8}} = \text{cast}\big(s_B \cdot B, \text{float8\_e4m3}\big)
```

And on the matmul output, the scale is divided back:

```math
C_{\text{fp32}} += \frac{1}{s_A \cdot s_B} \cdot \text{fp8\_matmul}(A_{\text{fp8}}, B_{\text{fp8}})
```

**Why per-block beats per-tensor.** A weight matrix's elements are not drawn from a single distribution; different rows (different attention heads, different MoE experts) have radically different magnitude profiles. A single per-tensor scale compresses the common case (small-magnitude weights) into a narrow E4M3 range, wasting representational bandwidth. Per-block scaling gives each `128 × 128` region its own dynamic range, tolerating heterogeneous distributions at the cost of `O(D/128 · D/128)` scale factors per weight matrix (negligible — a few KB for a 4096×4096 weight).

**Contrast with per-tensor E4M3 (H100 Transformer Engine default):** One scale per entire tensor. Works well for activations where outliers are rare; fails for weight matrices whose rows span 3+ orders of magnitude (common in MoE experts where some experts are under-utilized and have near-zero rows).

---

## 3. fp32 accumulator reconstruction (§3.3.2)

The paper is explicit about the accumulator:

> "In order to achieve accurate FP8 GEMM computations, we adopt two techniques: (1) fine-grained quantization, and (2) increasing accumulation precision."

H100 Tensor Cores natively support fp8-input with fp22 (!) accumulator, but the paper observes fp22 is insufficient over long reductions:

> "During the FP8 accumulation on Tensor Cores, the intermediate results are accumulated using the limited bit width. The accumulation precision of this process is critical to the accuracy of FP8 training."

Their fix: **two-level accumulation.** The innermost Tensor Core matmul accumulates `K_inner = 128` into a local fp22 partial sum; after `K_inner` elements the partial sum is promoted to **fp32** and accumulated into a CUDA-core-resident fp32 register. The outer loop over `K/128` block-chunks is fp32.

Pseudocode:

```python
# Conceptual: fp8 GEMM with fp32 outer accumulator
C_fp32 = torch.zeros(M, N, dtype=torch.float32)
for k_block in range(0, K, 128):
    A_block_fp8 = A[:, k_block:k_block+128]  # fp8-E4M3, scale s_A
    B_block_fp8 = B[k_block:k_block+128, :]  # fp8-E4M3, scale s_B
    # Tensor Core: fp8 × fp8 → fp22 accumulator over K=128
    partial_fp22 = tc_fp8_matmul(A_block_fp8, B_block_fp8)
    # Promote and accumulate in fp32
    C_fp32 += partial_fp22.to(torch.float32) / (s_A[k_block:k_block+128]
                                                * s_B[k_block:k_block+128])
```

**Why this matters numerically.** An inner fp22 accumulation over 128 elements has mantissa precision `~2^-10 ≈ 0.1%`, fine for a sum of 128 terms each `~O(1)`. But a full `K = 4096` reduction in pure fp22 would accumulate `4096 / 128 = 32` of these partial sums, and their individual rounding errors sum. Promoting each partial sum to fp32 before outer-accumulation contains the error.

**For the 2025 LLM trainer.** This is an H100-specific trick that NVIDIA's `cuBLAS` / Transformer Engine expose via `FP8_E4M3_FP32_GEMM` kernels. Writing a custom fp8 matmul without two-level accumulation is a subtle correctness bug that only shows up at 7B+ scale after 100B tokens.

---

## 4. No explicit loss scaler (§3.3, contrast with fp16)

The paper notes what is *absent* from their stack:

- No dynamic loss scaler.
- No `GradScaler.unscale_()` call.
- No inf/NaN skip logic tied to a scale factor.

Why: per-tensor scaling (the `s_A`, `s_B` factors) **is** the loss-scaling mechanism, but applied at the operator granularity rather than the loss granularity. Each fp8 GEMM has its own forward-and-backward scales. The scales are tracked by an **amax history**:

```math
s_t^{(A)} = \frac{\text{fmax}_{\text{E4M3}}}{\max\big(\text{amax}(A_{t-1}), \text{amax}(A_{t-2}), \ldots, \text{amax}(A_{t-16})\big)}
```

— "delayed scaling" with a 16-step window. Computing a fresh `amax` every step is expensive, so the prior 16 steps' `amax` values are cached and the scale for step `t` uses their max. Under stable training this is fine; in volatile phases (warmup, LR jumps, curriculum transitions) it can drift, which is why the paper keeps bf16 fallback paths for embedding, head, and gating.

**Contrast with fp16 loss scaling.** fp16's single global `S` rides on the whole loss; fp8's per-tensor scales ride on each matmul independently. fp8 is finer-grained, more expensive to track, and more robust — but only because the `amax` machinery is implemented correctly.

---

## 5. E4M3 forward / E5M2 backward choice

Standard NVIDIA recipe, adopted by DeepSeek-V3:

| Path | Format | Mantissa | Max | Rationale |
|---|---|---|---|---|
| Forward matmul (weights × activations) | E4M3 | 3 bits | 448 | Activations/weights have narrow range; prioritize precision |
| Backward matmul (gradients) | E5M2 | 2 bits | 57344 | Gradients span wider range; prioritize range |

The table from [[excerpts/mixed-precision]] lists these ranges:

| Format | Bits | Exp | Mantissa | Range |
|---|---|---|---|---|
| fp8-E4M3 | 8 | 4 | 3 | ~2e-7 to 448 |
| fp8-E5M2 | 8 | 5 | 2 | ~6e-8 to 57344 |

E5M2's extra exponent bit buys `57344 / 448 ≈ 128×` more range at the cost of half the mantissa precision. Gradients benefit from range (they span many orders of magnitude across layers, especially late in training); activations benefit from precision (they are roughly `O(1)` across the network once normalization is applied).

**Notice:** the paper does not use a single fp8 flavor throughout. The forward/backward split is hardware- and empirically-justified, not ideological.

---

## 6. Precision-sensitive operator carve-outs (§3.3.3)

The paper lists operators kept in bf16/fp32, each with a one-line reason:

1. **Embedding module** (bf16) — sparse lookup amplifies quantization error; the same token always maps to the same row, so per-row fp8 quantization noise is deterministic and accumulates through training.
2. **Output head** (bf16) — the LM head's logits feed cross-entropy; fp8 quantization noise on logits translates to biased token probabilities.
3. **MoE gating** (bf16) — router `softmax` over 256 experts; fp8 quantization of gate logits would flip routing decisions stochastically.
4. **Normalization** (fp32 reductions) — as [[excerpts/batch-vs-layer-norm]] §2; the `mean(x²)` reduction is the single most precision-sensitive op.
5. **Attention operators** (bf16 / fp32 softmax) — softmax exponentials, see [[excerpts/mixed-precision]] §6.

The "fp8 matmul, bf16 everything else" rule is not a compromise; it is the **design**. Each carve-out is justified by a specific failure mode.

---

## 7. Cost reporting (§1, Table 1)

| Phase | H800 GPU-hours | USD @ $2/GPU-hr |
|---|---|---|
| Pretraining (14.8T tokens) | 2,664,000 | $5,328,000 |
| Context extension (32K → 128K) | 119,000 | $238,000 |
| Post-training (SFT + RL) | 5,000 | $10,000 |
| **Total** | **2,788,000** | **$5,576,000** |

The cost number — ~$5.58M — is what made DeepSeek-V3 famous. The fp8 recipe contributes a ~2× speedup on the GEMM path versus a bf16 baseline; without it, pretraining alone would cost ~$10M+. Precision engineering is cost engineering at frontier scale.

**What the paper explicitly claims:** no irrecoverable loss spikes, no rollbacks across the 14.8T-token pretraining. The stability comes from (a) correct fp32 carve-outs, (b) per-block scaling tolerating weight-distribution heterogeneity, (c) fp32 master weights and optimizer state absorbing the fp8 quantization noise.

---

## 8. What a 2025 practitioner should steal

If you are not training a 671B MoE (most readers), the transferable lessons:

- **For a bf16-only pretrain:** ignore fp8 entirely; the DeepSeek-V3 paper's §3.1 and §3.2 (DualPipe, cross-node all-to-all) matter more than §3.3.
- **For a 70B-class run with fp8 ambition:** use Transformer Engine's per-tensor `E4M3/E5M2` first; graduate to per-block only if you see scale-factor starvation on specific weight matrices (common in MoE, rare in dense).
- **Universal:** keep norms in fp32, optimizer state in fp32, softmax in fp32, embeddings and head in bf16, and test `amax` history serialization across checkpoint resumes.
- **The carve-out list (§3.3.3)** is the 2025 best-practice canonical list. Deviations require specific empirical justification.

---

## Connections

- [[ch-02]] — §4 "fp8 — the 2024+ frontier path" is sourced directly from this paper's §3.3.
- [[excerpts/mixed-precision]] — the fp16 precursor to fp8's per-tensor scaling; the paper's E4M3/E5M2 split extends Micikevicius's framing.
- [[excerpts/adam]] — optimizer state remains fp32 in DeepSeek-V3 despite fp8 GEMM.
- [[excerpts/gradient-clipping]] — DeepSeek-V3 uses `max_grad_norm = 1.0` throughout; global-norm computed in fp32 as always.
- [[excerpts/batch-vs-layer-norm]] — the "LayerNorm in fp32" rule is enforced in DeepSeek-V3's normalization carve-out.
