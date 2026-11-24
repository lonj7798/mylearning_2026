<!-- chapter: ch-06
     track: rl-pipeline
     kind: content
     title: Low-Precision Rollouts: MXFP8 Weights and KV Caches
     deps: [ch-02]
     sources: [[mai-rl-infrastructure]], [[verl-rollout-correction-and-lowprec]], [[fp16-train-infer-mismatch]], [[deepseek-train-infer-consistency]], [[mai-thinking-1]], [[truncated-importance-sampling]], [[slime]], [[scalerl]]
     figures: figures/precision-map.html
-->

# Chapter 6 — Low-Precision Rollouts: MXFP8 Weights and KV Caches

> **Core insight.** Rocket lowers sampler precision only where the memory saving is large and the numerical cost to sampler–learner agreement is measured to be small, and it makes the learner reproduce every quantization the sampler applies. MoE expert weights are "over 95% of the model's parameters", so storing them in MXFP8 "nearly halves the sampler's weight memory footprint" and allows a smaller expert-parallel degree ([[mai-rl-infrastructure]] p. 8). Compute stays in BF16: the experts are dequantized before the matmul, because MXFP8 matmuls "produced larger sampler–learner log-probability discrepancies" than BF16 matmuls on dequantized weights (p. 8). The learner then "quantiz[es] MoE weights to MXFP8 and immediately dequantiz[es] them to BF16", so both sides multiply by the same weight values (p. 8). The KV cache gets the same storage/compute split, but only for global-attention layers: with local:global attention at 5:1, quantizing both kinds "introduced substantially larger" discrepancies than quantizing global KV alone, so local KV and "the final three global attention layers" stay BF16, and the learner fake-quantizes K/V to match. The result is "approximately 67% more global-context tokens" at roughly the same local-attention capacity (p. 9). The predecessor, MAI-Thinking-1, kept BF16 weights and KV throughout the sampler (p. 7; [[mai-thinking-1]] Sec. 3.6.4); the change was driven by MAI Base 2's size.
>
> **Guideline.** When the model no longer fits the sampler fleet at the target concurrency in BF16, quantize the largest parameter group first (MoE experts) as storage only and keep BF16 compute, because Rocket measured larger log-prob gaps with MXFP8 matmuls than with dequantize-then-BF16 (p. 8), and verl's end-to-end MXFP8 GEMM runs measured `rollout_corr/kl` of 0.0045 against 0.0018 for BF16 on Qwen3-30B-A3B (verl@8357f1e `docs/low_precision/mxfp8.md` L116–121). When the sampler stores a tensor in a quantized format, apply the same quantizer in the learner's forward pass, because otherwise every expert matmul differs between the two engines at θ_old, and the TIS experiments show that INT8 rollouts without correction degrade PPO significantly ([[truncated-importance-sampling]] Experiments). When you quantize the KV cache, start with the layers whose cache grows with context (global attention), keep the layers whose cache is bounded by a window in BF16, and measure the sampler–learner gap after each step, because Rocket's ablation found that quantizing local KV as well gave "substantially larger" gaps (p. 9) and verl measured that an FP8 KV cache raised the mismatch of every arm, BF16 included (`mxfp8.md` L140–141).

---

## Why this chapter matters

Low-precision rollout is the most common reason a "synchronous, on-policy" RL run is in fact off-policy at the same weights. ch-03 separated two parts of the sampler–learner gap: parameter staleness and same-version engine mismatch. This chapter is about a deliberate source of the second part. The design question is not whether to use low precision, since a 3T-parameter MoE needs it. The question is which tensors to quantize, in what role (storage or compute), and how the learner reproduces it. Rocket's answer is one design point. verl, slime and DeepSeek-V4 choose others, and their measurements show what each choice costs.

**Interactive figure.** [figures/precision-map.html](figures/precision-map.html) redraws Fig. 5 as a clickable map of sampler and learner tensors, coloured by storage and compute precision, with a layer-type selector (local, quantized global, final-three global). It includes two calculators: weight memory as a function of the expert fraction, and global-KV token capacity as a function of the number of global layers. Use it with §2–§5.

---

## §1 Why the sampler moves to lower precision

### 1.1 Rocket's two generations

"For MAI Thinking 1, we prioritized numerical fidelity by retaining BF16 weights and KV caches throughout the sampler. Since MAI Base 2 is a much larger model, we selectively use MXFP8 for MoE weights and KV caches to accelerate rollouts while maintaining training stability" (p. 7). MAI-Thinking-1's report states the reason for its choice: "we use bf16 for both the learner and the inference engine in our RL runs; this yields substantially smaller numerics gaps and more stable training than the lower-precision alternatives we evaluated" ([[mai-thinking-1]] Sec. 3.6.4). Which alternatives were evaluated, and the size of the gap, are not stated.

The challenge statement in the Base 2 post gives the pressure: "Fitting a larger model and longer contexts into the sampler requires lower-precision storage" (p. 1). MAI-Thinking-1 is 35B active / 1T total parameters; MAI Base 2 is described as three trillion parameters (p. 1).

### 1.2 What low precision buys at rollout

verl's low-precision doc lists three benefits of low-precision rollout for large MoE models, whose rollout "is bound by memory capacity and bandwidth" (verl@8357f1e `docs/low_precision/qat.md` L5–9):

1. it "shrinks weight memory, leaving more room for KV cache and higher concurrency";
2. it "reduces memory traffic during decode, improving throughput";
3. it "uses FP8/FP4 Tensor Cores on Hopper/Blackwell to speed up prefill".

Rocket's storage-only design keeps the first two and gives up the third, because the matmul runs in BF16. The post says the design "retains most of the speed benefits while keeping the sampler closer to the learner" (p. 8). No speed number is given.

The same doc states the cost: "a numerical gap between the high-precision trainer and the low-precision rollout (training-inference mismatch). It makes importance sampling and off-policy correction less accurate and, in bad cases, destabilizes training" (L11).

### 1.3 The format

MXFP8 is "the OCP microscaling FP8 format: E4M3 elements with one E8M0 scale per 32-element block, decoded natively by Blackwell tensor cores" (verl@8357f1e `docs/low_precision/mxfp8.md` L5–6). Per element, storage is 8 bits plus 8/32 bits of shared scale, so 8.25 bits against 16 for BF16, a ratio of 0.516. E4M3 has 3 mantissa bits. For comparison, BF16 has 7 and FP16 has 10 ([[fp16-train-infer-mismatch]] Table 1).

---

## §2 MoE expert weights: MXFP8 storage, BF16 compute

### 2.1 Memory

"MoE expert weights account for over 95% of the model's parameters. Storing these weights in MXFP8 rather than BF16 nearly halves the sampler's weight memory footprint, enabling a smaller expert-parallel (EP) degree and fewer GPUs per sampler replica" (p. 8).

A consistency check with the format from §1.3. If a fraction f of parameters is in MXFP8 at 8.25 bits and the rest stays BF16, total weight memory relative to all-BF16 is f · 0.516 + (1 − f). At f = 0.95 this is 0.540, a 46% reduction. That matches "nearly halves". The check assumes the scale overhead of a 32-element block and ignores padding and metadata. The post does not give the exact fraction or the measured footprint.

**Why a smaller EP degree matters.** If one replica's weights take half the memory, the same expert set fits on fewer GPUs. Fewer GPUs per replica means more replicas from a fixed fleet, and the MoE all-to-all spans fewer devices. The post states the first consequence ("fewer GPUs per sampler replica"). It does not quantify the communication effect.

### 2.2 Why compute stays in BF16

"In our experiments, performing the expert matrix multiplications in MXFP8 produced larger sampler–learner log-probability discrepancies than dequantizing the MXFP8 weights to BF16 and multiplying them with BF16 activations. We therefore store MoE weights in MXFP8 but dequantize them to BF16 for computation" (p. 8).

This is an ablation result reported without numbers. The post does not give a mechanism. One consequence of the two designs can be read from what each side must reproduce:

- **Storage-only.** The sampler computes x · deq(q(W)) in BF16. The learner can compute exactly the same product if it applies the same q and deq to W. The activations x are BF16 in both.
- **MXFP8 matmul.** An MXFP8 GEMM also quantizes the activation operand, and its accumulation is done inside a low-precision tensor-core kernel. To match, the learner would need the same activation quantizer and the same kernel accumulation order. The sampler's decode kernel and the learner's training kernel usually differ.

So storage-only reduces the quantization difference to a deterministic function of the weights, which the learner can reproduce. This reading is an inference from the description. It is not stated in the post.

verl's MXFP8 recipe makes the other choice. It trains with Transformer Engine MXFP8 GEMMs and serves the rollout with MXFP8 GEMMs (DeepGEMM or FlashInfer kernels), with "only the GEMM inputs … cast to MXFP8", weights kept in BF16 for training, and `lm_head`, token embedding and MoE routers kept in BF16 on the rollout (`mxfp8.md` L20–46, L48–51). Its Qwen3-30B-A3B measurements on one 4×B200 machine, mean of steps 1–10 (`mxfp8.md` L116–130):

| Training / rollout precision | Step time (s) | Generation (s) | `update_actor` (s) | `rollout_corr/kl` | Score |
|---|---|---|---|---|---|
| BF16 / BF16 | 506 | 134 | 244 | 0.0018 | −0.130 |
| FP8-TW / BF16 | 487 | 133 | 225 | 0.0058 | −0.150 |
| FP8-BW / FP8-BW | 489 | 115 | 241 | 0.0090 | −0.169 |
| MXFP8 / MXFP8 | 452 | 102 | 221 | 0.0045 | −0.160 |

The doc's reading: "MXFP8 cuts the step time by 11% against BF16 (generation −24%, `update_actor` −9%) … with the lowest rollout–trainer mismatch of the three low-precision arms", and "Ten steps are enough for speed and mismatch, not for the score" (L129–131). These are single short runs. The mismatch with MXFP8 compute on both sides is 2.5× the BF16 value, which is consistent in direction with Rocket's finding that MXFP8 matmuls widen the gap.

### 2.3 Learner fake quantization

"To improve training stability, we reproduce the sampler's weight quantization in the learner by quantizing MoE weights to MXFP8 and immediately dequantizing them to BF16 for computation" (p. 8).

Step by step, what this does:

1. The learner holds high-precision master weights W.
2. In each forward pass, for every MoE expert matrix, it computes W̃ = deq(q(W)), where q is the MXFP8 quantizer.
3. It multiplies BF16 activations by W̃ in BF16.
4. At θ_old, the sampler multiplies by deq(q(W)) from its stored MXFP8 copy. If q is the same function on both sides, W̃ is bit-identical to the sampler's dequantized weight, and the expert matmul inputs agree.

**Not stated in the post:** how gradients pass through q (verl and DeepSeek use a straight-through estimator, below), and which side produces the sampler's MXFP8 copy during weight transfer (ch-05).

**Same format is not the same bits.** verl's doc shows why the quantizer must be literally the same function. Comparing its trainer-side exporter with Transformer Engine on the same BF16 weight: the two "pick the same E8M0 scale for every block except those whose amax is exactly `448 * 2**k`, where compressed-tensors picks the next one up; in such a block only elements below `amax / 14336` dequantize differently, by at most about `4.4e-6 * amax`" (`mxfp8.md` L42–46). Two correct MXFP8 implementations disagree on a measure-zero set of blocks. For bitwise parity (ch-07), measure-zero is not zero. verl's QAT path removes the problem by construction: "parameter export invokes the same kernels without dequantization, making their quantized values and scales bit-exact by construction" (`qat.md` L181–185), and its trainer forward "uses fake-quantized weights (quantize then dequantize) and gradients flow through a straight-through estimator" (L36).

### 2.4 How other stacks handle the learner side

| Stack | Sampler weights | Learner forward | Learner reproduces quantization? | Source |
|---|---|---|---|---|
| Rocket (Base 2) | MXFP8 storage for experts, BF16 compute | fake-quantized experts (q → deq), BF16 compute | yes | p. 8 |
| Rocket v1 | BF16 | BF16 | not needed | [[mai-thinking-1]] Sec. 3.6.4 |
| verl MXFP8 recipe | MXFP8 GEMMs (linear + MoE), BF16 `lm_head`, embeddings, routers | TE MXFP8 GEMMs | partly: same format, different kernels | `mxfp8.md` L20–46 |
| verl QAT | FP8_BLOCK / MXFP8 / MXFP4 / NVFP4 | fake-quantized, STE; export uses the same kernels | yes, bit-exact quantized values | `qat.md` L36, L181–185 |
| slime production path | FP8 (128×128 blockwise) checkpoint | BF16 Megatron | no | slime@79aa41a `docs/en/advanced/low-precision.md` L3–7 |
| slime INT4 QAT (beta) | INT4 | fake QAT via env flags | yes | `low-precision.md` L97–124 |
| DeepSeek-V4 | native FP4 expert weights | FP4 QAT, dequantized to FP8 for compute, STE | yes, and rollout uses the real deployed kernels | [[deepseek-train-infer-consistency]] Sec. 5.2.1 |

slime's recommended path ("BF16 training in Megatron + FP8 rollout/inference in SGLang") does not make the learner reproduce the rollout quantization. slime quantizes the updated BF16 weights "before sending them to SGLang" using the checkpoint's quantization config (`low-precision.md` L5–7). The resulting gap is left to the loss-side corrections that slime offers (TIS / MIS weights, ch-07). Its docs also note that "FP8 training plus FP8 inference can improve inference throughput and reduce training/inference mismatch in some settings" (L58), without numbers in-tree.

DeepSeek-V4 states the strongest version of the rule: "we directly use native FP4 quantized weights instead of simulated quantization. This ensures that model behavior during sampling is fully consistent with online deployment" ([[deepseek-train-infer-consistency]] Sec. 5.2.1). Its FP4 → FP8 dequantization for compute is lossless when the scale ratio within each 128×128 block is below a threshold, which the authors verify empirically (same section). That is the same storage/compute split as Rocket's, one format lower.

---

## §3 KV cache: global layers only

### 3.1 The storage/compute split for KV

"We apply the same separation of storage and compute precision to the sampler's global-attention KV cache. Within the attention kernel, quantized KV entries are dequantized to BF16 for computation. The learner applies corresponding KV fake quantization to reproduce the sampler's quantization effects" (p. 8). In Fig. 5 the sampler path is "Quantize + append" into the KV cache and "Dequantize" into attention. The learner path is "Quantize → dequantize" on K/V before attention.

The benefit: "The smaller KV cache footprint supports longer sequences and higher rollout concurrency, while reducing the need to offload KV entries or retract requests under memory pressure" (p. 9). MAI-Thinking-1 had described KV memory as "the main bottleneck" for single-turn generations of up to 128k tokens ([[mai-thinking-1]] Sec. 3.6.4).

### 3.2 Why only global layers

"MAI Base 2 mixes local and global attention layers at a ratio of 5:1, and our ablations showed that quantizing both local- and global-attention KV caches introduced substantially larger sampler–learner log-probability discrepancies than quantizing global-attention KV caches alone. We therefore retain BF16 KV caches for local-attention, and, as a numerical precaution, for the final three global attention layers. All other global-attention layers use MXFP8 storage" (p. 9).

Two separate claims, with different evidence:

| Choice | Evidence | Location |
|---|---|---|
| Keep local-attention KV in BF16 | ablation: quantizing local + global gave "substantially larger" gaps than global alone; no numbers | p. 9 |
| Keep the final three global layers in BF16 | "as a numerical precaution"; no ablation reported | p. 9 |

The memory side of the trade can be reasoned from the attention types, although the post does not spell it out. A local (sliding-window) layer stores K/V for at most the window, whatever the sequence length. A global layer stores K/V for every token. For long agentic contexts, global KV is the part that grows. Quantizing it gives the capacity gain for long sequences. Quantizing local KV would touch five times as many layers for a saving that does not grow with context. The ablation shows that the numerical cost of that saving was large.

### 3.3 The 67% figure

"The resulting KV cache configuration accommodates approximately 67% more global-context tokens than the BF16 KV cache configuration, while maintaining roughly the same local-attention cache capacity" (p. 9).

The figure depends on the number of global layers and on the scale granularity of the KV format, and the post gives neither. Under explicit assumptions (every global layer has the same per-token KV size, MXFP8 KV costs 8.25 bits per element as for weights, the global-KV memory budget is fixed, and n_g global layers with the last three in BF16), the capacity multiplier is

```
capacity ratio = 16 · n_g / (16 · 3 + 8.25 · (n_g − 3))
```

which tends to 16 / 8.25 ≈ 1.94 as n_g grows and is lower for small n_g. The precision-map figure lets you evaluate it. Because the post does not state n_g or the KV scale layout, the 67% cannot be reproduced from the post alone. Treat any n_g you infer from it as a guess.

### 3.4 KV precision in open stacks

- **slime** exposes FP8 KV as a rollout-only flag: `--sglang-kv-cache-dtype fp8_e4m3`. "This is a rollout-side setting. It does not change Megatron training precision" (slime@79aa41a `docs/en/advanced/low-precision.md` L44–52). The learner does not fake-quantize K/V.
- **verl** ran its Qwen3-8B DAPO arms with `kv_cache_dtype=fp8` in every arm: "It raises the mismatch of every arm, so these KL values do not compare with the other runs on this page" (`mxfp8.md` L140–141). In that setting, with FP8 KV everywhere, BF16 and MXFP8 arms both measured `rollout_corr/kl` 0.0088, against 0.0297 for FP8-TW training with a BF16 rollout, whose entropy fell from 0.307 to 0.062 over 32 steps (L144–161).
- **MAI-Thinking-1** controlled KV pressure differently: it disabled prefix caching for single-turn workloads "so that sliding-window-attention tokens can be fully evicted during long generations" ([[mai-thinking-1]] Sec. 3.6.4).

---

## §4 What stays in BF16 or FP32

Fig. 5's legend has three classes: "MXFP8 storage", "BF16 storage/compute", "FP32 accumulation" (p. 8). From the figure and text:

| Tensor or operation | Sampler | Learner | Source |
|---|---|---|---|
| MoE expert weights | MXFP8 storage → dequantize → BF16 matmul | quantize → dequantize → BF16 matmul | p. 8, Fig. 5 |
| Global-attention KV (except the final three layers) | MXFP8 storage ("Quantize + append") → dequantize in attention | K/V quantize → dequantize | p. 8–9, Fig. 5 |
| Local-attention KV; final three global layers' KV | BF16, "no conversion" | BF16 | Fig. 5 footnote, p. 9 |
| Q/K/V projections, output projection | drawn without a quantize step (BF16) | same | Fig. 5 |
| Attention compute | BF16 | BF16 | p. 8 |
| MoE expert combine (routing-weighted sum) | FP32, computed locally after all-to-all | same | p. 10 (ch-07) |
| Embeddings, `lm_head`, router | not stated | not stated | — |

The text layer of Fig. 5 does not show which box carries the FP32-accumulation colour. The FP32 MoE combine is stated in the parity section (p. 10). Rocket says nothing about `lm_head` precision. Two open sources show why the output head matters: ScaleRL measured that FP32 LM-head logits raised the fitted asymptote from 0.52 to 0.61 ([[scalerl]] Fig. 5b), and the TIS authors tried an FP32 `lm_head` patch in vLLM and found that "the mismatch problem still exists after applying both patches" ([[truncated-importance-sampling]] Summary). verl keeps `lm_head`, token embedding and MoE routers in BF16 on its MXFP8 rollout "as in training" (`mxfp8.md` L40–41).

---

## §5 Evidence: what is measured and what is not

| Claim | Type | Source |
|---|---|---|
| MXFP8 matmul gives larger log-prob gaps than dequantize + BF16 matmul | ablation, no numbers | [[mai-rl-infrastructure]] p. 8 |
| Quantizing local + global KV gives "substantially larger" gaps than global only | ablation, no numbers | p. 9 |
| Final three global layers kept BF16 | precaution, no ablation | p. 9 |
| "Nearly halves" sampler weight memory | consistent with 0.95 · 0.516 + 0.05 = 0.54 (§2.1 check) | p. 8 |
| ≈67% more global-context tokens | capacity statement; inputs not given | p. 9 |
| Learner fake quantization improves training stability | stated as the purpose; no ablation shown | p. 8 |
| MXFP8 GEMMs both sides: kl 0.0045 vs BF16 0.0018; −11% step time | 10-step single runs, Qwen3-30B-A3B | verl `mxfp8.md` L116–130 |
| MXFP8 rollout on Qwen3-8B GSM8K: kl 0.0063–0.0071 vs 0.0006–0.0008; accuracy tracks BF16 over 10 steps | single runs | verl `mxfp8.md` L82–86 |
| INT8 rollouts degrade PPO significantly; INT8 + TIS recovers to BF16-like performance | controlled, Qwen2.5-0.5B GSM8K | [[truncated-importance-sampling]] Experiments |
| BF16 rounding is the main mismatch source; FP16 on both sides cuts sequence-level mismatch ≈24× | controlled, offline + training | [[fp16-train-infer-mismatch]] Fig. 2, Sec. 3.5 |
| FP32 inference + BF16 training stable but "nearly three times slower" | controlled | [[fp16-train-infer-mismatch]] Sec. 4.4 |

The fp16 paper points the opposite way from the efficiency pressure in this chapter: its fix for mismatch is more mantissa bits on both sides, and its caveats note that "efficiency pressure pushes toward FP8" and that very large models may face FP16 overflow ([[fp16-train-infer-mismatch]] Sec. 6). Rocket's design and the fp16 paper agree on one principle: the two engines must compute with the same numbers, whatever the format. Rocket enforces it by reproducing the sampler's quantization in the learner. The fp16 paper enforces it by choosing a format whose rounding differences are small.

There is no reported training-curve comparison, for Rocket, between an MXFP8 sampler and a BF16 sampler at Base 2 scale. The closest Rocket evidence is Fig. 6 (bitwise-matched vs baseline on L42), which concerns parity, not precision (ch-07).

---

## Recipe

1. **Size the problem.** Compute sampler weight bytes and KV bytes per token per layer type at BF16 for your target concurrency and context. Quantize only if BF16 does not fit or the replica count is too low.
2. **Rank tensors by memory share.** For MoE models, experts dominate (Rocket: over 95% of parameters). For long agentic contexts, global-attention KV dominates the KV budget.
3. **Quantize storage, not compute, first.** Store the chosen tensors in MXFP8 (or FP8 blockwise on Hopper) and dequantize to BF16 inside the kernel.
4. **Reproduce the quantization in the learner** with the same quantizer function (same scale rounding, same block layout). Prefer one kernel used for both learner fake quantization and export (verl QAT), and pass gradients by straight-through estimation.
5. **Verify bit equality** of the dequantized weights between the learner's fake-quantized tensor and the sampler's loaded tensor on a few parameters after every weight transfer during bring-up.
6. **Quantize KV layer type by layer type.** Start with global layers, keep windowed layers and the last few global layers in BF16, and fake-quantize K/V in the learner.
7. **Measure the gap after each change** with identical weights and inputs (mean |Δ log-prob| or `rollout_corr/kl`), and keep a BF16 reference arm for at least the first hundred steps, longer than the 10-step runs in verl's doc.
8. **Keep the output head and router in high precision** unless you have measured otherwise (verl BF16; ScaleRL FP32 head).
9. **If you cannot reproduce the quantization in the learner,** enable a loss-side correction (TIS / MIS, ch-07) and log the correction's clip or mask fraction.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Quantizing the rollout but training on unquantized weights | Non-zero sampler–learner gap at identical weights; entropy collapse in bad cases (verl FP8-TW: 0.307 → 0.062) | Log-prob gap at staleness 0; entropy trend vs a BF16 arm |
| Using two different MXFP8 quantizers on the two sides | Rare blocks dequantize differently; parity tests fail intermittently | Compare scales block by block (verl's amax = 448·2^k case) |
| Turning on FP8 KV as a "free" capacity win | Gap rises for every precision arm | verl `mxfp8.md` L140–141; measure with and without KV quantization at fixed weights |
| Quantizing sliding-window KV with global KV | Larger gap for little capacity gain | Rocket's ablation (p. 9); compute capacity gain per layer type |
| Running MXFP8 matmuls in the sampler only | Gap larger than storage-only quantization | Compare storage-only vs MXFP8 GEMM at fixed weights (Rocket p. 8) |
| Judging a precision change by 10-step reward | Score differences within noise; late collapse missed | verl's own caveat (L84, L131); run longer and track the gap over training |
| Assuming the gap is constant | Gap grows during RL (Rocket p. 9) | Re-measure the gap at later checkpoints on fixed inputs |

---

## Check your understanding

1. Rocket stores experts in MXFP8 but multiplies in BF16. List the three benefits of low-precision rollout from verl's doc and state which ones Rocket keeps and which it gives up. Why might the one it gives up matter less for decode-heavy agentic rollouts than for prefill-heavy ones?
2. Explain why storage-only quantization can be reproduced exactly in the learner while an MXFP8 GEMM is harder to reproduce. Which part of your explanation is an inference rather than a stated result?
3. Using f = 0.95 and 8.25 bits per MXFP8 element, compute the sampler weight memory relative to BF16. What would the figure be if the post's "over 95%" were 98%?
4. Two correct MXFP8 quantizers can disagree on blocks whose amax is exactly 448 · 2^k. Why does such a rare disagreement matter for bitwise parity but hardly at all for accuracy?
5. Give the memory argument for quantizing global-attention KV before local-attention KV in a 5:1 local:global model, and say what Rocket's ablation adds that the memory argument cannot.
6. Using the capacity formula in §3.3, find how many global layers would make the ratio equal 1.67 under the stated assumptions. List the assumptions that could make this inference wrong.
7. slime's production path quantizes the rollout to FP8 and trains in BF16 without fake quantization; DeepSeek-V4 trains with FP4 QAT and rolls out with native FP4. Explain which part of the sampler–learner gap each leaves, and which correction each therefore needs.

---

## Connections

- **Previous: ch-02** — where the sampler and learner sit in the loop.
- **Next: ch-07** — bitwise parity: the remaining kernel, parallelism and reduction-order differences after precision is matched; the FP32 MoE combine.
- **ch-03** — the split between parameter staleness and same-version engine mismatch; this chapter is about the second.
- **ch-05** — weight transfer must deliver the format the sampler stores.
- **ch-08** — long agentic contexts, the reason global-KV capacity matters.
- **llm-training ch-43** — entropy collapse, the failure signature of an uncorrected precision gap.
- **llm-training ch-44a** — long-context RL and the KV budget.
- **llm-training ch-40** — the GRPO ratio that a precision gap perturbs at θ_old.

---

## Sources

- [[mai-rl-infrastructure]] — MXFP8 expert storage, BF16 compute, learner fake quantization, global-only KV quantization, 67% capacity, Fig. 5 (p. 7–9).
- [[mai-thinking-1]] — Rocket v1 BF16 sampler and learner and the reason given; KV as the single-turn bottleneck.
- [[verl-rollout-correction-and-lowprec]] — MXFP8 recipe, quantizer comparison, QAT with bit-exact export, FP8 KV effect, 30B-A3B and 8B tables (verl@8357f1e docs).
- [[slime]] — BF16 training + FP8 rollout production path, rollout-only FP8 KV, INT4 QAT (slime@79aa41a docs).
- [[deepseek-train-infer-consistency]] — FP4 QAT with native FP4 rollout and lossless FP4 → FP8 dequantization.
- [[fp16-train-infer-mismatch]] — mantissa bits as the mismatch driver; FP16 on both sides; precision ablation.
- [[truncated-importance-sampling]] — INT8/FP8 rollout experiments and the FP32 `lm_head` patch result.
- [[scalerl]] — FP32 LM head raises the fitted asymptote (0.52 → 0.61).
