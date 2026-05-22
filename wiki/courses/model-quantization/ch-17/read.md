<!-- chapter: ch-17
     track: frontier-2025-2026
     title: Low-Precision Pretraining: FP8 DeepSeek V3 + NVFP4/MXFP4 + Blackwell
     sources: [[deepseek-v3-fp8]], [[fp8-formats-paper]], [[fp8-lm]], [[transformer-engine]], [[transformer-engine-fp8]], [[nvfp4-training]], [[mxfp4-pretraining]], [[mxfp4-native-hardware-2026]], [[quartet-ii]], [[nvfp4-qad]], [[blackwell-quantization]]
     figures: figures/fp8-fp4-block-scaling.html
-->

# Chapter 17 — Low-Precision Pretraining: FP8 DeepSeek V3 + NVFP4/MXFP4 + Blackwell

> **Core insight.** Once you push training precision below BF16 the *per-tensor scale* (the only knob FP16/BF16 mixed-precision had) is no longer expressive enough — outliers crush it. The 2024–2026 frontier replaced it with a *block scale* (every 16–128 elements get their own scale) and, for FP4, a *two-level scale* (per-block FP8 + per-tensor FP32). With this one structural change, FP8 became the production default in 2024 (DeepSeek-V3 14.8T-token run, < 0.25 % loss gap vs BF16) and FP4 became the production default in 2025–2026 (NVFP4 12B / 10T-token run matches FP8).
>
> **Guideline.** When training in FP8: use **per-block scaling** (1×128 activation tile + 128×128 weight block), keep **BF16 master weights**, and **promote partial sums to FP32 every 4 WGMMA inside the GEMM** to defeat the Hopper FP22 accumulator. When training in NVFP4 on Blackwell: same skeleton + **16-element FP4 blocks with FP8 (E4M3) block scale + FP32 tensor scale**, **stochastic rounding on the backward GEMM only**, and a **random Hadamard transform** on the two layers (FFN-gate input, attn-out input) that carry the worst per-channel outliers. Anything that worked at BF16 — exclusion lists for embed/head/RMSNorm, gradient clipping, AdamW defaults — still applies.

---

## Why this chapter exists

The FP8 papers of 2022 ([[fp8-formats-paper]], [[fp8-lm]]) showed FP8 *could* train an LLM. They didn't show that FP8 could train a *frontier* LLM with no loss spikes for 14.8T tokens across 671B-parameter MoE — that was DeepSeek-V3's December 2024 contribution. NVIDIA's 2025 NVFP4 paper then did the same one bit-width down: 12B dense, 10T tokens, FP4 weights *and* activations, matching FP8 baseline. Quartet-II (2026) sharpened the gradient estimator. The 2026 MXFP4-on-native-hardware study (Cim et al.) added a diagnostic plot twist — weight-gradient quantization is the convergence-killer, not forward-pass quantization.

What you need to walk away with:

1. The exact difference between **per-tensor delayed scaling** ([[transformer-engine]]), **per-block online scaling** ([[deepseek-v3-fp8]]), and **two-level block + tensor scaling** ([[nvfp4-training]]).
2. The four-ingredient NVFP4 recipe: format + 2-D consistent quantization + random Hadamard + stochastic rounding — and *why each one is load-bearing* (the ablation in [[nvfp4-training]] Table 2 removes each in turn).
3. The 2026 trajectory: FP8 → FP4 in *training* is now the production default on Blackwell, with QAD ([[nvfp4-qad]]) as the inference-recovery path for post-trained checkpoints that didn't see FP4 during pretraining.

This chapter is dense — go slowly. It's the bridge from "quantization as an inference afterthought" to "quantization as a first-class pretraining design decision."

---

## 1. The FP8 → FP4 progression at a glance

| Recipe | Year | Element | Scale | Block | Master weights | Tested at |
|--------|------|---------|-------|-------|----------------|-----------|
| TE DelayedScaling ([[transformer-engine]]) | 2022 | E4M3 fwd / E5M2 bwd | per-tensor FP32 from amax history | none | BF16 | up to 175B |
| FP8-LM L3 ([[fp8-lm]]) | 2023 | E4M3 fwd / E5M2 bwd | per-tensor (predicted) | none | BF16 | 175B (Megatron) |
| **DSV3 FP8** ([[deepseek-v3-fp8]]) | 2024 | **E4M3 everywhere** | **per-block online** | **1×128 act / 128×128 wt** | BF16 | **671B MoE / 14.8T tokens** |
| MXFP4 ([[mxfp4-pretraining]]) | 2025 | FP4 E2M1 | E8M0 (power-of-two) | 32 elements | BF16 | up to 6.7B |
| **NVFP4** ([[nvfp4-training]]) | 2025 | FP4 E2M1 | **FP8 (E4M3) + FP32 tensor** | **16 elements** | BF16 | **12B dense / 10T tokens** |
| Quartet-II ([[quartet-ii]]) | 2026 | NVFP4 + MS-EDEN | same | same | BF16 | 1.9B / 38B tokens (so far) |
| Native MXFP4 ([[mxfp4-native-hardware-2026]]) | 2026 | FP4 E2M1 | E8M0 | 32 elements | BF16 | Llama-3.1-8B on AMD MI355X |

The arrow of progress is *more structure in the scale*, not *fewer bits in the element*. Each row in this table takes the previous row's scale design and refines it.

---

## 2. FP8 mixed precision — the 2022–2023 baseline

### 2.1 The two-format split

[[fp8-formats-paper]] (Micikevicius et al., NVIDIA/Arm/Intel 2022) is the joint industry spec. Two encodings:

| Format | Sign | Exp | Mantissa | Bias | Max | Smallest normal | Role |
|--------|------|-----|----------|------|-----|-----------------|------|
| **E4M3** | 1 | 4 | 3 | 7 | 448 | 2⁻⁶ | forward activations + weights (precision) |
| **E5M2** | 1 | 5 | 2 | 15 | 57344 | 2⁻¹⁴ | backward gradients (range) |

Neither format alone can serve both forward and backward. Weights live in O(1); gradients live in O(10⁻⁶)–O(10⁻²). E4M3's 4-bit exponent gives it precision (12.5 % relative) at the cost of half-width range; E5M2's 5-bit exponent gives it FP16-scale range at the cost of 25 % relative precision.

### 2.2 TE DelayedScaling — the per-tensor recipe

[[transformer-engine]] / [[transformer-engine-fp8]] is NVIDIA's reference library. The core idea is *delayed* per-tensor scaling:

```python
# Per training step, per tensor:
amax_history[t % H] = current_amax        # H = amax_history_len (default 1024 in TE 1.x)
new_amax = max(amax_history)
scale = fp8_max / (new_amax * (1 + margin))
scale_inv = 1 / scale
# scale_inv is used in the NEXT step's forward — no sync needed with the GEMM
```

The "delayed" trick is what lets FP8 hide its scale computation behind the GEMM: the scale is already ready before the next forward starts. The cost is staleness — if activations drift faster than the amax history can track, the scale lags and the FP8 tensor clips.

At 175B scale ([[fp8-lm]] Table 3), TE-style FP8 reached **39 % memory reduction and 75 % wall-clock speedup vs BF16 Megatron**.

### 2.3 FP8-LM — extending FP8 beyond the GEMM

[[fp8-lm]] (Peng et al., Microsoft 2023) pushed FP8 into the *optimizer and communication* layer with three opt-in levels:

- **L1** — FP8 GEMM only (same as TE).
- **L2** — FP8 gradients + FP8 all-reduce (per-bucket scale on the wire).
- **L3** — FP8 Adam states: `m` in E4M3 (modest range, precision matters), `v` in E5M2 (very wide range from g², precision matters less).

The L3 update reads back to FP32 transiently:

```
m_fp32 = s_m · m_fp8
v_fp32 = s_v · v_fp8
m_fp32 = β1 · m_fp32 + (1 − β1) · g
v_fp32 = β2 · v_fp32 + (1 − β2) · g²
m_fp8, s_m = quant_E4M3(m_fp32)
v_fp8, s_v = quant_E5M2(v_fp32)
```

Net Adam state memory: `2 bytes/param + scales` ≈ **4× smaller than FP32** Adam. Open-sourced as `MS-AMP`.

---

## 3. DeepSeek-V3 FP8 — the per-block recipe that scaled

Per-tensor FP8 hits a wall at frontier scale. A single outlier channel can crush a per-tensor scale: the max governs the whole tensor's quantization grid, so most of the dynamic range is wasted on the outlier. The SmoothQuant observation ([[smoothquant]], ch-09) showed activation outliers are heavily channel-localized — DSV3 ([[deepseek-v3-fp8]]) generalized SmoothQuant from *inference PTQ* into a *training* recipe.

### 3.1 Per-block scaling

Three numbers to memorize:

- **Activations:** **1 × 128** tile (per-token, per-128-channel). One E4M3 scale per tile, picked **online** from the tile's amax — no history, no delay.
- **Weights:** **128 × 128** block. One scale per block, picked online during the forward.
- **Element format:** **E4M3 throughout** — forward, weight-grad, activation-grad. (Note: this *deviates* from the TE convention of E4M3-fwd / E5M2-bwd. DSV3 can use E4M3 for gradients because the per-block scale already absorbs the wide dynamic range that E5M2 was needed for.)

The 1×128 tile keeps the scale tight enough that one outlier channel doesn't crush the rest. Per-tensor FP8 lost > 3 % loss at the same training budget; per-block lost < 0.25 %.

### 3.2 FP32 partial-sum promotion inside the GEMM

This is the load-bearing kernel-level trick. The H800 Tensor Core's internal FP8→FPx accumulator has roughly **14-bit mantissa precision** (FP22-like). After ~32 successive multiply-adds, the partial sum starts losing the smaller contributions — silent drift, no overflow warning.

DSV3 promotes the partial sum to FP32 in CUDA-core registers **every 4 WGMMA instructions** (each WGMMA does a fixed tile of FP8 matmul). The cost is a few extra FP32 adds per group; the gain is bit-exact accumulation across the full K dimension. Combined with per-block scales, the dequantized output is essentially identical to a BF16 GEMM.

### 3.3 What stays in higher precision

DSV3 keeps the precision-sensitive ops out of FP8:

- **Embedding lookup** — BF16.
- **RMSNorm scales** — BF16.
- **Routing gate logits** (MoE) — BF16.
- **Attention softmax** — FP32 inside, BF16 outside.
- **Cross-entropy loss head** — BF16/FP32.
- **MoE combine step** (all-to-all on the way back) — BF16 (the *dispatch* leg is FP8, but the gradient-carrying combine is not).
- **Master weights** — BF16.
- **Optimizer:** FP32 first moment + BF16 second moment.

The pattern: anything that's a *small parameter count but high precision-sensitivity* op stays in BF16. The win is on the giant Linear layers in the 256 expert FFNs.

### 3.4 The result

| Knob | Value |
|------|-------|
| Element format | E4M3 (all forward + backward + grad) |
| Activation tile | 1 × 128 |
| Weight block | 128 × 128 |
| Master-weight dtype | BF16 |
| Optimizer m / v dtype | FP32 / BF16 |
| WGMMA promote interval | 4 |
| FP8 comm | dispatch only; combine in BF16 |
| **Loss gap vs BF16** | **< 0.25 % relative** |
| Wall-clock vs BF16 | ~2× |

This is the recipe that trained DSV3 (671B MoE, 14.8T tokens) on 2.664M H800-hours with no loss spikes — the first publicly documented frontier-scale FP8 run. TE 2.x has since absorbed the per-block scaling idea as a first-class recipe.

---

## 4. NVFP4 — FP4 native pretraining on Blackwell

### 4.1 The format ([[nvfp4]], [[blackwell-quantization]])

NVFP4 is the format Blackwell's 5th-generation Tensor Cores consume natively. Three operand levels per GEMM:

- **Element:** FP4 E2M1 (1 sign, 2 exponent, 1 mantissa). Representable values: `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` — eight magnitudes.
- **Block scale:** **16 elements share one E4M3 (FP8) scale.** E4M3 is fractional, not power-of-two like MX's E8M0. This is what halves the per-block MSE vs MXFP4 on identical data.
- **Tensor scale:** **One FP32 scalar per tensor**, applied multiplicatively on top of the block scale.

The hardware MMA instruction takes all three operands at once and applies the scales during FP32 accumulation. There is no software dequant step inside the GEMM loop.

Compared to MXFP4 ([[microscaling-formats]]): MXFP4 uses 32-element blocks and an E8M0 (pure power-of-two) scale. NVFP4's smaller block + richer scale gives it ~2× lower per-block MSE — the difference between "matches FP8 with extra effort" and "needs heroic effort to match BF16."

### 4.2 The four-ingredient recipe

[[nvfp4-training]] (Micikevicius et al., NVIDIA 2025) trained a 12B dense transformer for 10T tokens in NVFP4 and matched the FP8 loss curve. Four ingredients, all load-bearing per the Table 2 ablation:

**(a) The NVFP4 format itself.** Two-level scaling lets the FP32 tensor scale absorb global range while the FP8 block scale absorbs local dispersion; the 4-bit element only has to carry the residual.

**(b) 2-D consistent quantization.** A linear `Y = X · Wᵀ` has three GEMMs across a training step:

1. Forward: X[1,16] blocks × W[16,1] blocks → Y.
2. Input-grad: dY × W → dX, requires W blocked along O.
3. Weight-grad: dYᵀ × X → dW, requires X blocked along B·S.

If each GEMM picks its own block layout, the quant noise on the same tensor X (or W) is *uncorrelated* across forward and backward → biased gradient. The 2-D scheme stores each tensor with *both* block layouts (same data, two scale tables, ~0.4 % extra memory) so forward and weight-grad see identical NVFP4 X.

**(c) Selective random Hadamard transform (RHT).** Applied to the input of FFN-gate (`W_gate · X`) and the input of the attention output projection (`W_o · X`) — the two activations consistently identified as carrying the worst per-channel outliers across the QuaRot ([[quarot]]) / SpinQuant ([[spinquant]]) lineage (ch-14). H is a fixed random ±1 Hadamard of size 128×128 (composes with the 128-wide hidden tiles Blackwell loads). The transform is folded into the weight offline (`W' = W · Hᵀ`) — *zero inference cost*. Per-block amax drops by ~2–3× on these layers, which is what keeps the 16-element block scale tight enough for FP4.

**(d) Stochastic rounding on the backward GEMM only.** Forward uses RNE (so inference is deterministic — inference uses the same code path). Backward uses SR: for each FP4 element, round up with probability `x − ⌊x⌋`, else down. This preserves `E[round_SR(x)] = x` so the gradient estimator stays unbiased. Empirically, SR on forward had no benefit but hurt inference parity; SR on backward was load-bearing.

### 4.3 Selective high precision

Same pattern as DSV3 — keep ~3 % of FLOPs in BF16:

- Embedding table.
- Final RMSNorm + LM-head.
- A handful of attention LayerNorms flagged by sensitivity analysis.
- Everything else in all 40 transformer blocks: NVFP4 weight + NVFP4 activation + FP32 accumulation.

### 4.4 The result

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
| Loss gap vs FP8 baseline | matches within run-to-run noise |

NVIDIA's framing in [[nvfp4-training]]: this is the first publicly reported 4-bit pretraining run at frontier scale. The trajectory through DSV3 (per-block FP8) → NVFP4 (per-block FP4 + tensor scale) is a single thread of "more structure in the scale, less bits in the element."

---

## 5. MXFP4 — the OCP cousin and a 2026 plot twist

### 5.1 MXFP4 with RHT + SR (AISTATS 2025)

[[mxfp4-pretraining]] (Tseng et al.) was the first academic study of OCP MXFP4 — 32-element blocks of FP4 E2M1 with an E8M0 shared exponent (8-bit unsigned exponent only, no mantissa, no sign — a pure power-of-two scale). No per-tensor outer scale (unlike NVFP4).

Naïve MXFP4 pretraining diverges. Two fixes make it work:

- **RHT on the GEMM input** (folded into weight). Theoretical bound: per-block max drops to `O(√(log d / d))` of the tensor's L2 norm — what keeps the E8M0 scale tight enough for FP4 to resolve.
- **Stochastic rounding on backward.** Same purpose as in NVFP4.

With both, GPT-1.3B / 2.7B / 6.7B trained in MXFP4 match the BF16 baseline on loss + downstream, with ~1.7× backward speedup vs BF16 and ~1.3× vs FP8 on supported hardware.

### 5.2 The 2026 native-hardware reality check

[[mxfp4-native-hardware-2026]] (Cim et al., AMD, May 2026) revisited MXFP4 pretraining on AMD Instinct MI355X — *native* FP4 hardware, not emulation. They enabled FP4 progressively across Fprop, Dgrad, and Wgrad on Llama-3.1-8B pretraining on C4. The result was diagnostic:

| Path | Meaning | Diagnostic finding |
|------|---------|---------------------|
| Fprop | forward activation × weight GEMM | relatively stable under MXFP4 |
| Dgrad | activation-gradient GEMM | modest added token cost |
| **Wgrad** | **weight-gradient GEMM** | **main convergence degradation driver** |

And — contrary to the 2025 recipe — *stochastic rounding and randomized Hadamard rotations were insufficient once Wgrad was quantized.* **Deterministic Hadamard rotations** restored stable optimization.

The lesson: forward-pass stability does not imply Wgrad stability. The 2025 results stand at their tested scope (GPT-6.7B with selective FP4); full-pipeline MXFP4 training at the 8B / native-hardware boundary needs deterministic rotations and more careful Wgrad handling. This is the current frontier as of mid-2026.

### 5.3 Quartet-II — better gradient estimator for NVFP4

[[quartet-ii]] (Panferov et al., 2026) targets the residual NVFP4 accuracy gap by replacing stochastic rounding with **MS-EDEN**, an unbiased *microscaling* quantizer that uses the block structure itself to produce an unbiased estimate with > 2× lower variance than scalar SR. Tested at 1.9B / 38B tokens; Blackwell kernel reports up to 4.2× over BF16 linear layers. This is the strongest direct follow-up to [[nvfp4-training]] as of 2026 — the trajectory is *better gradient estimators*, not bigger models.

---

## 6. NVFP4-QAD — recovering post-trained checkpoints for NVFP4 inference

[[nvfp4-qad]] (NVIDIA, 2026) addresses the deployment-side problem. Frontier models go through SFT, RL, distillation, safety data, and model merging. Replaying that stack with NVFP4 quantization inserted is expensive and unstable. QAD reduces the recovery problem to matching a frozen BF16 teacher:

```
L_QAD = KL(softmax(z_T / T) || softmax(z_S / T))
```

where `z_T` is the BF16 teacher's logits and `z_S` is the NVFP4 student's logits. The student model has NVFP4 quantization inserted (weights *and* activations — W4A4 deployment); training data is recovery data (not the original SFT/RL corpus), and the only loss is the soft-target KL.

Reported on AceReason Nemotron, Nemotron 3 Nano, Nemotron Nano V2, Nemotron Nano V2 VL, and Llama Nemotron Super v1. The relevance: this is the *production* path for shipping NVFP4 inference on Blackwell when the original model wasn't NVFP4-pretrained. For a NVFP4-pretrained model, [[nvfp4-training]] is sufficient; for everything else, run QAD.

---

## 7. Blackwell — the hardware that made FP4 production

[[blackwell-quantization]] is the model-report page on the hardware. The quant-relevant facts:

- **5th-gen Tensor Core**: native NVFP4 MMA instruction; consumes FP4 elements + E4M3 block scale + FP32 tensor scale as separate operands. The per-block scale dispatch is done in silicon, not software.
- **Coexisting precision tiers**: FP4 / FP6 / FP8 / BF16 / FP16 / TF32 / INT8 / INT4, all native. Mix per layer.
- **B200 per chip**: ~10 PFLOPS FP4 dense / ~20 PFLOPS FP4 sparse; ~5 PFLOPS FP8; ~192 GB HBM3e; ~8 TB/s bandwidth.
- **GB200 NVL72**: 72 B200 + 36 Grace in a single rack; ~13.4 EFLOPS FP4 sparse aggregate.
- **B300 / GB300** (2025–2026): ~2× FP4 throughput vs B200, ~288 GB HBM3e, attention-specific acceleration.

Memory math: NVFP4 is **3.5× less memory than FP16** and **1.8× less than FP8**. Per NVIDIA's NVFP4 inference blog, < 1 % quality drop with proper calibration on representative LLMs (the specific framing in [[blackwell-quantization]]: "as of the 2025 product announcements").

---

## 8. The 2026 production default

Putting the trajectory together:

- **Pretraining (frontier labs, 2026):** FP8 with per-block scaling is the safe default (DSV3 recipe). NVFP4 is moving from "first successful run" (12B / 10T) toward "production default on Blackwell."
- **Inference (Blackwell):** NVFP4 is the default. FP8 is the conservative tier. BF16 only for layers sensitivity-analysis flags. For post-trained models that didn't see FP4 in training: run QAD ([[nvfp4-qad]]) to recover.
- **Inference (Hopper, non-Blackwell):** FP8 is the deployment target; W4A16 (Marlin/Machete, see ch-19) remains the open-source serving baseline.

The arrow: every two years the production training precision halves. BF16 (2020–2022) → FP8 (2023–2024) → FP4 (2025–2026). The technique that made each step possible was *more structure in the scale*. The technique that will let us go further (FP3? FP2 training?) is not yet established; Quartet-II's MS-EDEN and the deterministic-Hadamard MXFP4 result are pointers.

---

## Practitioner's cheat-sheet

```python
# FP8 training (TE 1.x, Hopper / H100) — the DSV3-style block-scaled path
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# 2025-era default: still DelayedScaling for stability margin
recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID,
                        amax_history_len=1024, amax_compute_algo="max")

# DSV3-style per-block scaling (TE 2.x)
# from transformer_engine.common.recipe import MXFP8BlockScaling
# recipe = MXFP8BlockScaling()

model = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, ...)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    loss = model(inputs).loss
loss.backward()
optimizer.step()

# NVFP4 training (TE 2.x, Blackwell only)
# from transformer_engine.common.recipe import NVFP4
# recipe = NVFP4(block_size=16, block_scale_dtype="E4M3",
#                tensor_scale_dtype="FP32", backward_rounding="stochastic",
#                forward_rounding="rne", rht_layers=["ffn_gate_in", "attn_out_in"])

# Exclusion list — same as BF16 mixed-precision: embed, lm_head, final norm stay BF16
```

---

## Common pitfalls

- **Per-tensor FP8 at 70B+:** lost > 3 % loss at frontier scale ([[deepseek-v3-fp8]]). Use per-block from the start; don't even prototype with per-tensor.
- **Forgetting FP32 partial-sum promotion on H800/H100:** the FP22 accumulator silently loses precision after ~32 multiply-adds. Promote every 4 WGMMA.
- **SR on forward for NVFP4:** no quality benefit *and* breaks inference parity. SR is backward-only.
- **Skipping RHT on FFN-gate / attn-out:** the 16-element NVFP4 block scale is too tight for outliers in these specific layers. RHT bound proven in [[mxfp4-pretraining]]; selectively applied in [[nvfp4-training]].
- **Quantizing embed / RMSNorm / LM-head:** they're a tiny fraction of FLOPs and the most precision-sensitive. Always exclude.
- **Assuming forward-FP4-stable = full-FP4-stable:** the [[mxfp4-native-hardware-2026]] result — Wgrad quantization dominates instability. Test each GEMM separately.
- **Using MXFP4 calibration on NVFP4 (or vice versa):** the formats are *not* interchangeable. MXFP4 = 32-element block, E8M0 scale, no tensor scale. NVFP4 = 16-element block, E4M3 scale, FP32 tensor scale. The block size + scale-format choices are load-bearing.

---

## Connections and what's next

- **[[fp8-formats-paper]] / ch-02** — the E4M3 / E5M2 spec; you should have memorized the bit layout before reading this chapter.
- **[[smoothquant]] / ch-09** — the activation-outlier observation that DSV3's per-block scaling fundamentally exploits. Same root insight, generalized from PTQ into training.
- **[[quarot]] / [[spinquant]] / ch-14** — the rotation lineage; [[nvfp4-training]]'s selective Hadamard transform is a direct descendant.
- **[[nvfp4-qad]] / ch-19** — inference-time recovery for post-trained models; the deployment-side complement to NVFP4 pretraining.
- **[[microscaling-formats]] / ch-16** — the OCP MX family that MXFP4 instantiates; NVFP4 is the NVIDIA variant.
- **[[blackwell-quantization]]** — the hardware that natively executes NVFP4; required reading for understanding *why* the format choices in [[nvfp4-training]] were what they were.
- **ch-18** — KV-cache quantization in 2026; the data-oblivious turn that mirrors this chapter's "more structure in the scale" theme on the inference side.
- **ch-19** — production kernels (Marlin / Machete / TRT-LLM / vLLM / llama.cpp); how the algorithms in this chapter actually run.

## Further reading

- [[deepseek-v3-fp8]] — primary source for the per-block FP8 recipe; §3.3 of the DSV3 tech report.
- [[nvfp4-training]] — primary source for the 4-ingredient NVFP4 recipe.
- [[fp8-lm]] — the FP8-beyond-the-GEMM (gradient + optimizer + comm) path.
- [[mxfp4-native-hardware-2026]] — the 2026 plot twist; deterministic Hadamard, Wgrad as the failure mode.
- [[quartet-ii]] — the 2026 NVFP4 gradient-estimator successor.

## Companion visualization

**[figures/fp8-fp4-block-scaling.html](figures/fp8-fp4-block-scaling.html)** — interactive visualization of how per-tensor → per-block → two-level (per-block + per-tensor) scaling progressively tightens the dynamic range each element has to cover. Slider over outlier-channel magnitude shows the per-tensor scale collapsing while per-block / two-level absorb the spike.
