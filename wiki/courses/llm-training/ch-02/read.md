<!-- chapter: ch-02
     track: foundations
     kind: content
     title: Numerical Precision, Determinism, and Train–Inference Mismatch
     deps: [ch-01]
     sources: [[mixed-precision]], [[fp8-formats-for-deep-learning]], [[pytorch-amp-autocast-gradscaler]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[scaling-laws-for-precision]], [[numerical-nondeterminism-llm-inference]], [[thinkingmachines-defeating-nondeterminism]], [[rollout-training-mismatch-tis]], [[minimax-m1]], [[small-scale-proxies-instabilities]], [[olmo-2]], [[olmo-2-official-configs]]
     figures: figures/precision-range.html
     revised: 2026-09 (generality revision)
-->

# Chapter 02 — Numerical Precision, Determinism, and Train–Inference Mismatch

> **Core insight.** A floating-point format fixes which magnitudes survive (range, set by exponent bits) and how coarsely they are rounded (precision, set by mantissa bits). Mixed-precision training matches full-precision loss by keeping master weights, gradient reduction, and selected operations in higher precision, and the published FP8 evidence is a loss comparison: DeepSeek-V3's FP8 runs stay within 0.25% relative loss of BF16 at about 16B parameters on 1.33T tokens and about 230B on 0.9T tokens, with no downstream benchmark comparison reported ([[deepseek-v3]] App. B.1). The same rounding changes measurements after training: under BF16 greedy decoding, DeepSeek-R1-Distill-Qwen-7B's AIME'24 accuracy has a 9.15-point standard deviation across 12 GPU-type, GPU-count, and batch-size configurations, and 0 in FP32 ([[numerical-nondeterminism-llm-inference]] Table 3). In RL, the sampler and the learner assign different probabilities to the same token under the same weights (maximum difference 1.0 on DAPO with Qwen2.5-32B), so nominally on-policy training is off-policy ([[rollout-training-mismatch-tis]] Fig. 1).
>
> **Guideline.** When the hardware supports bf16, train under bf16 autocast with fp32 master weights and fp32 gradient reduction, because bf16 has the fp32 exponent range (no gradient underflow, no loss scaling) but a relative spacing of 2^-7, which erases small weight updates (§2, §3; [[olmo-2-official-configs]]). Otherwise, with fp16, use dynamic loss scaling and unscale before gradient clipping ([[mixed-precision]] §3.2). When a lower-precision format (FP8, or a quantized deployment) replaces a higher one, compare the two on the development and unseen suites of ch-00, because the cited parity evidence is loss or perplexity only ([[deepseek-v3]] App. B.1; [[fp8-formats-for-deep-learning]] Table 4; [[scaling-laws-for-precision]] §6). When two checkpoints are compared with greedy decoding, run both under one recorded configuration and in FP32 or with FP32 computation (LayerCast), or sample multiple runs (the study used 16 or 64 on AIME'24) and report the mean, because BF16 configuration changes alone gave a 9.15-point standard deviation on AIME'24 ([[numerical-nondeterminism-llm-inference]] §3). When RL rollouts come from an inference engine and gradients from a training backend, record sampler log-probabilities and log sampler–learner mismatch every step; correct for it (ch-54) or remove its numerical sources, because uncorrected mismatch was followed by reward collapse in [[thinkingmachines-defeating-nondeterminism]] and by no reward growth in [[minimax-m1]] §3.2.

## Why this chapter matters for a general-purpose model

Numerical format is chosen once and then affects every stage of the pipeline (pre-training → mid-training → SFT → preference optimization → RL → evaluation). Four measurable problems follow.

1. **Training outcome.** A value that underflows to zero or overflows to infinity changes the update. In the original mixed-precision study, a Multibox SSD detector trained in FP16 without loss scaling diverged, and a Mandarin speech model whose FP16 weights were updated directly lost 80% relative accuracy ([[mixed-precision]] Table 2, §3.1).
2. **Deployment outcome.** A checkpoint is often quantized after training. Loss degradation from post-training quantization grows with tokens per parameter in the models of [[scaling-laws-for-precision]] (Finding 1), so additional training tokens can lower the loss of the full-precision model while raising the loss of its quantized version (§3.1 of that paper).
3. **Measurement.** A claimed gain in general capability is a score difference. If runtime configuration moves the score by more than the claimed difference, the difference is not measured (§6; ch-00 §6).
4. **RL correctness.** Policy-gradient methods assume the samples come from the policy being updated. Numerical differences between the inference engine and the trainer break that assumption without raising an error (§7).

Terms used throughout. A *dtype* is a numeric storage format (fp32, bf16, fp16, fp8). *Mixed precision* stores and computes different tensors in different dtypes in one training step. *Autocast* is the framework mechanism that picks a dtype per operation. *Master weights* are the full-precision parameter copy the optimizer updates. A *reduction* is an operation that combines many elements into fewer (a sum, a mean, a softmax normalizer). *Rollout* is a response sampled from the policy during RL; the *sampler* generates rollouts and the *learner* computes gradients.

## §1 Floating-point formats: range, precision, and rounding

**Definition.** A binary floating-point number has a sign bit, e exponent bits, and m mantissa bits. Range is the interval of magnitudes the format can hold; precision is the spacing between neighboring representable values.

**Problem.** A gradient, update, or activation outside the range becomes 0 or infinity; a value inside the range is rounded to the nearest representable value, and the rounding error depends on the spacing at that magnitude.

**Formula.** For a normal number with biased exponent field E (1 ≤ E ≤ 2^e − 2 in IEEE-style formats; E4M3 also uses E = 15 for finite values) and mantissa field f:

`x = (−1)^s · (1 + f / 2^m) · 2^(E − bias)`,  `bias = 2^(e−1) − 1`

For a subnormal number (E = 0): `x = (−1)^s · (f / 2^m) · 2^(1 − bias)`.

- s is the sign bit; E the stored exponent field; f the integer value of the m mantissa bits; bias the offset that centers the exponent range.
- Neighboring normal numbers with the same exponent are 2^(E − bias − m) apart, which is 2^-m at 1.0. Rounding is to the nearest value, ties to even.

**Formats used in LLM training.** Values for E4M3 and E5M2 are from [[fp8-formats-for-deep-learning]] Table 1. Values for fp32, bf16, and fp16 follow from the formula (status: derived) and match `torch.finfo` in PyTorch 2.12.0.

| Format | sign/exp/mantissa bits | smallest subnormal | smallest normal | largest | spacing at 1.0 |
|---|---|---|---|---|---|
| fp32 | 1/8/23 | 2^-149 ≈ 1.40e-45 | 2^-126 ≈ 1.18e-38 | ≈ 3.40e38 | 2^-23 ≈ 1.19e-7 |
| bf16 | 1/8/7 | 2^-133 ≈ 9.18e-41 | 2^-126 ≈ 1.18e-38 | ≈ 3.39e38 | 2^-7 ≈ 7.81e-3 |
| fp16 | 1/5/10 | 2^-24 ≈ 5.96e-8 | 2^-14 ≈ 6.10e-5 | 65,504 | 2^-10 ≈ 9.77e-4 |
| fp8 E5M2 | 1/5/2 | 2^-16 ≈ 1.53e-5 | 2^-14 ≈ 6.10e-5 | 57,344 | 0.25 |
| fp8 E4M3 | 1/4/3 | 2^-9 ≈ 1.95e-3 | 2^-6 ≈ 1.56e-2 | 448 (no infinity) | 0.125 |

E4M3 reuses most special-value bit patterns for finite numbers, which raises its largest value from 240 to 448 and removes infinities ([[fp8-formats-for-deep-learning]] §3.1). E5M2 follows IEEE conventions and "can be viewed as IEEE half precision with fewer mantissa bits": 2 instead of fp16's 10 (§3).

**Worked examples** (checked with PyTorch 2.12.0 casts).
1. *Precision.* The value 1.00012 lies between 1 and the next fp16 value 1 + 2^-10 = 1.000977, and is nearer to 1, so fp16 stores 1.0; bf16 also stores 1.0; fp32 stores 1.00012004 ([[numerical-nondeterminism-llm-inference]] Table 1).
2. *Range.* A gradient entry of 2^-26 ≈ 1.49e-8 is smaller than half of fp16's smallest subnormal (2^-25), so fp16 stores 0. bf16 stores it exactly, because its smallest normal value is 2^-126.
3. *Overflow in an exponential.* fp16 overflows when a result exceeds 65,504. Since ln 65,504 = 11.09, `exp(12)` computed in fp16 is infinity, while `exp(11)` ≈ 59,874 is finite and is stored in fp16 as 59,872. In bf16 the threshold is ln(3.39e38) = 88.7. A softmax computed from logits above 11 without subtracting the maximum logit therefore overflows in fp16.

**Figure.** [figures/precision-range.html#formats](figures/precision-range.html#formats) lets the reader enter any value and see the stored value, the rounding error, and whether it underflows or overflows in each of the five formats.

**Implication for a general-purpose model.** Range errors produce zeros and infinities that change updates. Precision errors are small per operation but depend on the order of operations (§6), which is the source of the evaluation and RL effects later in this chapter.

## §2 Mixed-precision training: which tensors stay in higher precision

**Definition.** Mixed-precision training stores most activations and runs most matrix multiplications in a 16-bit or 8-bit format, and keeps a smaller set of tensors and operations in a wider format so that the result matches full-precision training.

**Problem.** Micikevicius et al. set the target as matching FP32 accuracy without changing hyperparameters while roughly halving training memory ([[mixed-precision]] Abstract, §3.1).

**Mechanism** ([[mixed-precision]] §3).
1. *FP32 master weights.* The optimizer updates an FP32 copy; an FP16 copy is used in forward and backward passes (§3.1).
2. *Loss scaling* for FP16 gradients (§3 of this chapter).
3. *Arithmetic precision.* Dot products accumulate into an FP32 value before conversion to FP16; "large reductions (sums across elements of a vector) should be carried out in FP32", mainly batch-normalization statistics and softmax; point-wise operations can use either dtype (§3.3).

The paper gives two reasons for master weights (§3.1). First, updates smaller than 2^-24 become zero in FP16; about 5% of weight-gradient exponents in the Mandarin speech model are below −24 (Fig. 2b). Second, when a weight is at least 2048 times larger than its update, FP16 addition can round the update away. Updating FP16 weights directly cost 80% relative accuracy on that model (Fig. 2a).

**Worked example: update rounding in 16-bit weights** (course calculation; status: derived, checked with PyTorch casts).
- fp16: 1 + 2^-12 rounds to 1.0, because the spacing at 1 is 2^-10 and 2^-12 is less than half of it. The ratio 1 / 2^-12 = 4096 exceeds the 2048 threshold of [[mixed-precision]] §3.1.
- bf16 has 7 mantissa bits, so the corresponding threshold is 2^8 = 256. A weight of 0.02 is stored in bf16 as 0.02001953125, and the spacing at that magnitude is 2^-13 ≈ 1.22e-4. An Adam-type update whose per-element size is close to the learning rate (ch-01), at a learning rate of 1e-5 (for comparison, DeepSeek-V3's SFT learning rate decays from 5e-6 to 1e-6, [[deepseek-v3-recipe]], so its per-element updates are smaller still), is below half the spacing (6.1e-5), so `w + u` returns the same bf16 value and the update is lost. An update of 3e-4 changes the stored value by 2.44e-4 instead of 3e-4, a 19% error. In fp32 both updates are kept.
- Consequence: bf16 removes the need for loss scaling (§3) but does not remove the need for full-precision master weights.

**Current practice, per source.**
- *PyTorch autocast (framework policy)* ([[pytorch-amp-autocast-gradscaler]] amp.md L194-272). On CUDA, `linear`, `matmul`, `mm`, `bmm`, `addmm`, and convolutions run in the lower-precision dtype; `softmax`, `log_softmax`, `cross_entropy`, `nll_loss`, `layer_norm`, `group_norm`, `sum`, `prod`, `cumsum`, `exp`, `log`, `pow`, and `rsqrt` run in float32. Unlisted operations run in the dtype of their inputs (L186-188). `scaled_dot_product_attention` and `rms_norm` are not in the CUDA lists, so fused attention and RMSNorm kernels called through them run in the dtype of their inputs, which is bf16 when the preceding operations and parameters are bf16. The float32 list is one framework's default, not a rule every training stack follows.
- *OLMo 2 7B stage 1 (released config)* ([[olmo-2-official-configs]]). `precision: amp_bf16` with `fsdp.precision: mixed`, which the code defines as bf16 parameters and buffers during computation and `reduce_dtype=torch.float32` for gradient reduction (config.py L836-840, L1339-1344).
- *DeepSeek-V3 FP8 framework* ([[deepseek-v3]] §3.3.1, §3.3.3). The three GEMMs (general matrix multiplications) of each Linear run in FP8; the embedding module, output head, MoE gating, normalization operators, and attention operators stay in BF16 or FP32; master weights and gradients "(used for batch size accumulation)" stay in FP32; AdamW moments are stored in BF16 "without incurring observable performance degradation".

```python
# github.com/allenai/OLMo@090253d olmo/config.py L1339-1344
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
```

**Conditions and limits.** The three sources keep different sets in higher precision: Micikevicius et al. (CNNs and RNNs, 2017) keep reductions in FP32; PyTorch keeps a named list; DeepSeek-V3 keeps attention and normalization in BF16 or FP32 and optimizer moments in BF16. None of the sources reports a per-operation ablation of downstream capability for LLMs (Open question). The DeepSeek-V3 statement about BF16 moments gives no numbers.

**Implication for a general-purpose model.** Gradients and master weights are the tensors that carry small updates; low learning rates in SFT, RL, and annealing make updates smaller relative to weights, so these tensors need the wider format in the stages that fine-tune an already capable model (derived from the worked example above).

## §3 Loss scaling for fp16, and why bf16 does not need it

**Definition.** Loss scaling multiplies the loss by a factor S before backpropagation so that small gradients fall inside fp16's range, and divides the gradients by S before they are used.

**Problem.** fp16's normalized exponent range is [−14, 15], while activation gradients are dominated by small magnitudes. For Multibox SSD, gradient values in [2^-27, 2^-24) had to be preserved; without scaling, the model diverged ([[mixed-precision]] §3.2, Fig. 3).

**Mechanism** ([[mixed-precision]] §3.2; [[pytorch-amp-autocast-gradscaler]]).
1. Compute the loss L and multiply it by S.
2. Backpropagate; by the chain rule every gradient is multiplied by S.
3. Unscale the weight gradients by dividing by S "right after the backward pass but before gradient clipping or any other gradient-related computations", so the clipping threshold and weight decay need no change.
4. If any gradient is infinite or NaN, do not apply it, because it would "irreversibly damage the weights"; the paper names skipping the update as one option (§3.2).
5. Dynamic scaling adjusts S. PyTorch 2.12.0 `GradScaler` defaults: initial S = 2^16; on inf or NaN, S is multiplied by 0.5 and the step is skipped; after 2000 consecutive steps without inf or NaN, S is multiplied by 2.0 (grad_scaler.py L123-131, L96-101). Micikevicius et al. used constant factors from 8 to 32K and list automatic selection as future work (§3.2, §5).

**Formula.** `ĝ = S · ∇_θ L` in fp16, `g = ĝ / S` in fp32. An entry is kept when `|g| · S > 2^-25` (half of fp16's smallest subnormal) and the step is valid when `S · max_i |g_i| ≤ 65,504`.
- θ are the parameters, L the loss, ∇_θ L the true gradient, ĝ the scaled fp16 gradient, g the recovered gradient, i indexes gradient entries.

**Worked example** (checked with PyTorch casts; the defaults of [figures/precision-range.html#loss-scaling](figures/precision-range.html#loss-scaling), g = 10^-7.83 ≈ 1.48e-8, largest gradient 1.0, S = 2^16, give the same outcome as the 1.49e-8 entry below).
- Entry g = 1.49e-8 (about 2^-26): fp16 without scaling stores 0.
- With S = 2^16: S·g ≈ 2^-10 ≈ 9.8e-4, a normal fp16 value; dividing by S in fp32 recovers 1.49e-8.
- If the largest gradient entry in the same step is 1.0, then S·1.0 = 65,536 > 65,504, fp16 gives infinity, `GradScaler` skips the step and sets S = 2^15. The next step computes S·1.0 = 32,768, which is finite.
- The figure lets the reader move g, the largest gradient, and S, and shows which entries survive and whether the step is skipped.

**Evidence** ([[mixed-precision]] Table 2, §4.5). Faster R-CNN mAP: 69.1% in FP32, 68.6% in mixed precision without scaling, 69.7% with scaling. Multibox SSD: 76.9% in FP32, divergence without scaling, 77.1% with a factor of 8. bigLSTM on the 1B-word dataset: FP16 without scaling diverges after 300K iterations; a factor of 128 matches FP32. The six ILSVRC CNNs in Table 1, including ResNet-50 (75.92% FP32, 76.04% mixed), needed no scaling.

**Why bf16 does not need it.** bf16 keeps fp32's 8 exponent bits, so its smallest normal value is 1.18e-38 and the entry 2^-26 above is stored exactly (§1). The cost is 7 mantissa bits, which affects precision, not range. PyTorch's documentation describes bfloat16 mixed precision on CPU as using autocast only, without `GradScaler` ([[pytorch-amp-autocast-gradscaler]] amp.md L34-39).

**Conditions and limits.** The loss-scaling evidence comes from CNNs, RNNs, and LSTMs in 2017; no source in this chapter measures loss scaling on a Transformer LLM. For FP8, the FP8-formats authors state that skipping updates on overflow is "not a good choice", because overflows are more frequent with the narrower range; FP8 uses per-tensor or finer scaling with saturation instead ([[fp8-formats-for-deep-learning]] §2).

**Implication for a general-purpose model.** In fp16 runs, the number of skipped steps is a training-health metric: a skipped step is a batch whose data did not update the model.

## §4 FP8 training: formats, scaling granularity, and what the validation shows

**Definition.** FP8 training quantizes the inputs of matrix multiplications to an 8-bit format (E4M3 or E5M2), multiplies them, and writes outputs in a wider format.

**Problem.** E4M3 spans 18 binades and E5M2 32 ([[fp8-formats-for-deep-learning]] §3.1), and in low-precision training "overflows and underflows are common challenges due to the limited dynamic range of the FP8 format" ([[deepseek-v3]] §3.3.2). Values are therefore multiplied by a scale before casting, and a single outlier sets the scale for every value that shares it (§3.3.2).

**Mechanism.**
1. Choose a group of values (a whole tensor, a 1×128 tile, or a 128×128 block).
2. Compute `s_b = F_max / max_{i∈b} |x_i|`, with F_max = 448 for E4M3 ([[fp8-formats-for-deep-learning]] §2).
3. Store `q_i = round_E4M3(s_b · x_i)`; overflowing values are saturated.
4. Multiply in FP8; the output is written in BF16 or FP32 and divided by the scales.
- b is a scaling group, x_i a value in the group, s_b its scale, q_i the stored 8-bit value, and `q_i / s_b` the recovered value.

**Choices reported.**
- *Format split.* The FP8-formats paper recommends E4M3 for weights and activations and E5M2 for gradients ([[fp8-formats-for-deep-learning]] §3). DeepSeek-V3 uses E4M3 on all tensors and attributes this to fine-grained scaling ([[deepseek-v3]] §3.3.2).
- *Granularity.* DeepSeek-V3 scales activations per 1×128 tile and weights per 128×128 block; scales are computed online from the current maximum rather than from a history of earlier iterations (delayed scaling) (§3.3.2).
- *Accumulation.* FP8 GEMM accumulation on H800 retains about 14 bits; for inner dimension K = 4096, the maximum relative error was nearly 2%; DeepSeek-V3 copies partial results to FP32 registers every 128 elements (§3.3.2).
- *Exceptions.* Inputs of the Linear after attention use a custom E5M6 format with power-of-2 scales (§3.3.3).

**Worked example: one scale per tensor versus per group** (course calculation with PyTorch `float8_e4m3fn` casts; interactive in [figures/precision-range.html#block-scaling](figures/precision-range.html#block-scaling)). Group A = [0.004, −0.001, 0.0025, 0.0006]; group B = [1000, −3.2, 0.75, 12].
- One scale for the tensor: s = 448 / 1000 = 0.448. Then 0.001 · 0.448 = 4.48e-4, which is below half of E4M3's smallest subnormal (2^-10 ≈ 9.8e-4), so it is stored as 0; 0.0006 is also stored as 0. The value 0.0025 becomes 1.12e-3, rounds to 2^-9 ≈ 1.95e-3, and is recovered as 0.00436 (74% error).
- One scale per group: s_A = 448 / 0.004 = 112,000. The value −0.001 becomes −112, which E4M3 stores exactly; 0.0025 becomes 280, rounds to 288, and is recovered as 0.00257 (2.9% error); 0.0006 becomes 67.2, rounds to 64, and is recovered as 0.000571 (4.8% error). No value is lost.

**Evidence and what it establishes.**

| Comparison | Setting | Metric | Result | Locus |
|---|---|---|---|---|
| FP8 vs BF16 | MoE ≈16B params, 1.33T tokens | training loss | relative error < 0.25% | [[deepseek-v3]] App. B.1 |
| FP8 vs BF16 | MoE ≈230B params, ≈0.9T tokens | training loss | relative error < 0.25% | App. B.1 |
| Block-wise quantization of activation gradients | MoE ≈16B, ≈300B tokens | training | diverged | App. B.2 |
| Simulated FP8 vs 16-bit | GPT 126M to 175B | training perplexity | 175B: 6.68 FP8 vs 6.65 bfloat16 baseline (baseline reported at 75% of training) | [[fp8-formats-for-deep-learning]] Table 4 |

- Result (single study per row). Loss parity within 0.25% at two scales is established for DeepSeek-V3's framework; the authors call this "well within the acceptable range of training randomness" without reporting seed variance (§3.3).
- Not established by these sources: parity on downstream benchmarks, on unseen evaluation suites, on long-context or agentic tasks, or on how the base model responds to SFT and RL. Neither source reports a downstream task comparison between its FP8 and 16-bit GPT-style or MoE language models (Open question).
- The DeepSeek-V3 report states that FP8 GEMMs "theoretically" double computational speed; it reports no measured end-to-end speedup (§3.3.1).

**Implication for a general-purpose model.** Average validation loss sums over all tokens. A precision change is evaluated for general capability only when the FP8 and 16-bit checkpoints are both scored on the ch-00 development and unseen suites with the same evaluation configuration.

## §5 Precision and tokens per parameter

**Definition.** *Tokens per parameter* is D/N, training tokens divided by parameters. *Post-training quantization (PTQ)* rounds a trained model's weights to fewer bits without further training. δ_PTQ is the increase in loss caused by PTQ.

**Problem.** [[scaling-laws-for-precision]] cites D/N ≈ 2000 for Llama-3-8B (§2.2), about 100 times the ratio of about 20 tokens per parameter associated with compute-optimal training (ch-08a). If quantization cost grows with D/N, the training budget and the deployment precision cannot be chosen independently.

**Setup** ([[scaling-laws-for-precision]] §2.3). OLMo-style models of 30M-220M non-embedding parameters trained on 1.5B-26B tokens of Dolma V1.7; 465 runs in 3-16 bit precision; validation up to 1.7B parameters and 26B tokens. PTQ uses GPTQ, replicated with two other methods (§3).

**Formula** (Finding 1, Eq. 2).

`δ_PTQ(N, D, P_post) = C_T · (D^γ_D / N^γ_N) · e^(−P_post / γ_post)`

- N parameters; D training tokens; P_post the post-training weight precision in bits; C_T, γ_D, γ_N, γ_post positive fitted constants. δ_PTQ rises with D, falls with N, and rises exponentially as P_post decreases (§3.1).

**Worked example** (course calculation; status: derived from App. K constants, which the authors say "are unlikely to be useful" outside their setup). γ_D = 0.5068. At fixed N and P_post, raising D/N from 20 to 200 multiplies δ_PTQ by 10^0.5068 = 3.21; raising it from 20 to 2000 multiplies it by 100^0.5068 = 10.3. The loss before quantization falls as D grows, so the question is which effect is larger; for D/N ≫ 10^3 the authors find the quantization effect "nontrivial around 5-bits, and dominant below that" (App. E.1).

**Further results.**
- Finding 2 (§4.2): training precision acts on an effective parameter count, `N_eff = N(1 − e^(−P_w/γ_w))(1 − e^(−P_a/γ_a))(1 − e^(−P_kv/γ_kv))`, where P_w, P_a, P_kv are the training precisions of weights, activations, and KV cache.
- Finding 3 (§4.3): with N, D, and precision optimized jointly under cost ∝ N·D·P, fits on integer-type quantization give 7-8 bits as compute-optimal; with N fixed, the optimal precision grows with log C.
- §5: models trained in lower precision degrade less under PTQ (R² = 0.90 for the unified fit).

**Conditions and limits** (§6). One architecture; loss only, "without downstream model evaluations"; models at most 1.7B parameters; experiments up to D/N ≈ 10^3. Status: Result (single study). Whether the D/N effect on quantized loss carries over to downstream capability of large over-trained models is an Open question.

**Implication for a general-purpose model.** When a model is trained at high D/N and will be served quantized, the capability to protect is the quantized model's; measure δ_PTQ on held-out loss and on the ch-00 suites for the actual D/N and bit width before fixing the deployment precision.

## §6 Nondeterminism in evaluation

**Definitions** ([[thinkingmachines-defeating-nondeterminism]]). *Run-to-run nondeterminism*: the same kernel on the same inputs gives different outputs. *Batch invariance*: a request's output does not depend on the batch size or on how its sequence is split. Floating-point addition is not associative: `(0.1 + 1e20) − 1e20` gives 0, while `0.1 + (1e20 − 1e20)` gives 0.1.

**Problem.** Greedy decoding (temperature 0) is expected to give one answer per prompt. In practice the answer changes with serving load and hardware, so a benchmark score is a function of the evaluation configuration as well as of the model.

**Mechanism.**
1. A GPU forward pass of an LLM uses no atomic adds, so the same kernel on the same batch gives bitwise-equal results ([[thinkingmachines-defeating-nondeterminism]], "When are atomic adds needed?").
2. Kernels for RMSNorm, matrix multiplication, and attention choose their reduction strategy by shape: split reductions, different tensor-core instructions, or a KV split that depends on how many query tokens are processed ("How do we make kernels batch-invariant?").
3. Server load changes the batch size, so the reduction order and the rounding for one request change.
4. When two candidate tokens have close probabilities, a rounding difference flips the greedy choice; the rest of the response then differs. In one BF16 pair, the top two tokens are "know" 49.75% and "have" 43.91% in one configuration and "have" 46.65% and "know" 46.64% in another ([[numerical-nondeterminism-llm-inference]] Fig. 3).
5. BF16's 7 mantissa bits give larger per-operation rounding than FP16's 10 or FP32's 23; the authors attribute the larger variation of top-1 probabilities in BF16 to this (§3.2, Fig. 4; Interpretation).

**Evidence.**
- Qwen3-235B-A22B-Instruct-2507, 1000 completions at temperature 0 of one prompt: 80 distinct completions; all agree for 102 tokens, then 992 continue "Queens, New York" and 8 "New York City". With batch-invariant kernels, all 1000 are identical ([[thinkingmachines-defeating-nondeterminism]], "How nondeterministic are completions?").
- Four 7B-8B models, five benchmarks, 12 configurations (L40S or A100; 2 or 4 GPUs; batch 8, 16, or 32), vLLM ([[numerical-nondeterminism-llm-inference]] §3.1):

| Model | Benchmark | Std of greedy accuracy over 12 configs: BF16 / FP16 / FP32 | Locus |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | AIME'24 | 9.15% / 5.74% / 0 | Table 3 |
| DeepSeek-R1-Distill-Llama-8B | AIME'24 | 4.60% / 6.00% / 5.8e-17 | Table 3 |
| Qwen2.5-7B-Instruct | AIME'24 | 1.71% / 1.45e-17 / 1.45e-17 | Table 3 |
| DeepSeek-R1-Distill-Qwen-7B | MATH500 | 1.04% / 1.12% / 0.12% | Table 3 |

- Output length on AIME'24 for DeepSeek-R1-Distill-Qwen-7B has a standard deviation of 9,189.53 tokens in BF16 and 0 in FP32 (Table 4). On MATH500, outputs diverge across configurations for 96.6% of examples in BF16, 73.0% in FP16, and 2.2% in FP32 (Fig. 5).
- Sampling is also affected: the standard deviation of AIME'24 Pass@1 (16 samples) across 6 configurations is 1.7151 in BF16 and 0.8273 in FP16 for DeepSeek-R1-Distill-Qwen-7B; with 64 samples the ordering reverses (0.3749 BF16, 0.7377 FP32), which the authors attribute to AIME'24's 30 problems (Table 5, §3.3).

**Mitigations and costs.**
- *Batch-invariant kernels*: on Qwen-3-8B, 1000 sequences took 26 s with default vLLM, 55 s with unoptimized deterministic vLLM, and 42 s with an improved attention kernel; batch-invariant matmul is about 20% slower than cuBLAS ([[thinkingmachines-defeating-nondeterminism]], "Performance"). Batch invariance does not cover changes of tensor-parallel size or GPU type ([[numerical-nondeterminism-llm-inference]] §2.2).
- *LayerCast*: BF16 weight storage with FP32 computation; divergence below 3.4% across configurations; 34% less memory than full FP32, which the authors say doubles memory and inference time relative to BF16 (§4).

**Worked example** (course calculation). AIME'24 has 30 problems, so one problem is 3.33 points ([[numerical-nondeterminism-llm-inference]] §3.3). A standard deviation of 9.15 points across configurations corresponds to about 2.7 problems. Suppose checkpoint A scores 40.0 (12 of 30) on 4 A100s at batch 32, and checkpoint B scores 46.7 (14 of 30) on 2 L40S at batch 8, both greedy in BF16. The 6.7-point difference is smaller than the 9.15-point standard deviation across configurations measured for DeepSeek-R1-Distill-Qwen-7B, so the comparison cannot attribute it to the checkpoints. Evaluating both under the same configuration makes each score repeatable, but it does not remove the dependence: a different fixed configuration can reorder the two checkpoints, because the 9.15-point spread is a spread across fixed configurations.

**Conditions and limits.** The evidence covers 7B-8B models (and Qwen3-32B in the appendix of [[numerical-nondeterminism-llm-inference]]) with outputs up to 32,768 tokens; long reasoning outputs show the largest effects. Status: Replicated for the existence of configuration-dependent greedy outputs (two independent sources above); Result (single study) for the per-benchmark magnitudes.

**Implication for a general-purpose model.** Reasoning and long-output tasks are the capability axes where BF16 configuration noise is largest, so claims about breadth that rest on these axes need either FP32-computation evaluation, batch-invariant serving, or multi-sample averages with the configuration reported (ch-51).

## §7 Sampler–learner mismatch in RL

**Definition.** In hybrid RL frameworks, rollouts are generated by an inference engine with probabilities π_sampler(a; θ), and gradients are computed by a training backend with probabilities π_learner(a; θ), where θ are the shared weights and a a token ([[rollout-training-mismatch-tis]] Summary).

**Problem and formula.** The REINFORCE gradient that is computed is

`E_{a ∼ π_sampler(θ)} [ R(a) · ∇_θ log π_learner(a; θ) ]`

- R(a) is the reward of the sampled response. This equals the on-policy gradient only when π_sampler = π_learner. The logged reward is an expectation under π_sampler, not under π_learner (Fig. 6).

**Numerical sources of the mismatch.**
1. *Kernels and precision.* In MiniMax-M1 RL, training-mode and inference-mode token probabilities differed because of "a precision mismatch between the training and inference kernels"; a layer-by-layer analysis located the error in high-magnitude activations of the LM head ([[minimax-m1]] §3.2).
2. *Parallelism.* On DAPO-Math prompts, a vLLM TP1 sampler with an FSDP SP1 learner gave 1 response with maximum token-probability mismatch above 0.5; a TP2 sampler gave 2; adding Ulysses SP8 in the learner raised the count "from two to double digits"; same-parallelism TP2 or TP4 gave a small number, which the authors report without exact counts ([[rollout-training-mismatch-tis]] Fig. 7-8).
3. *Response length.* Responses capped at 20K tokens have higher maximum mismatch than 4K caps with similar mean mismatch; the maximum mismatch within the first 4K tokens of a 20K response often exceeds that of an independent 4K response (Fig. 9-10).
4. *Batch composition.* Non-batch-invariant kernels make a token's probability depend on the other sequences in the batch (§6).
5. *Quantized rollouts.* The authors describe FP8 and INT8 rollouts as cases where the gap is large and BF16 rollouts as having a smaller gap than INT8 (Fig. 4 paragraph; BF16 paragraph after Fig. 5).
- Patching vLLM to return the probabilities used for sampling and casting its LM head to fp32 did not remove the mismatch on DAPO-32B (Fig. 1).

**Evidence of effect.**
- DAPO on Qwen2.5-32B: maximum token-probability difference 1.0 (some tokens with π_sampler = 1 and π_learner = 0); GSM8K PPO on Qwen2.5-0.5B: about 0.4 ([[rollout-training-mismatch-tis]] Fig. 1-2). With INT8 rollouts on DAPO-32B, entropy fell below 0.2, responses became abnormally long, and the k1 KL estimate was frequently negative; truncated importance sampling reversed these (Fig. 5). With BF16 rollouts and no correction, the logged reward was higher but AIME accuracy lower than with correction (Fig. 6).
- MiniMax-M1 (456B total, 45.9B activated): the mismatch "prevented reward growth"; computing the LM head in FP32 raised the Pearson correlation between training and inference probabilities from 0.987319 to 0.997135 (Fig. 3), after which reward increased ([[minimax-m1]] §3.2). The issue did not appear in smaller dense softmax-attention models.
- Bigmath RLVR from Qwen 2.5-VL instruct 8B, rollouts up to 4096 tokens: without importance weighting, reward collapsed with a KL spike near step 318; with importance weighting, sampler–trainer KL stayed around 0.001; with bitwise-identical sampler and trainer, KL was 0 and training proceeded ([[thinkingmachines-defeating-nondeterminism]], "True on-policy RL").
- Status: Replicated that uncorrected mismatch degraded RL training (three independent sources: lower accuracy with entropy collapse, no reward growth, reward collapse); each numerical magnitude is Result (single study).

**Worked example: why the correction is truncated** ([[rollout-training-mismatch-tis]] callout). With importance ratio r = π_learner/π_sampler = 16 for a token, an untruncated importance weight multiplies that token's contribution to gradient variance by r² = 256. Truncating the weight at C = 2 multiplies it by 4; at C = 8, by 64. A token with π_learner = 0 and π_sampler = 1 receives weight 0. The truncated estimator and its PPO form are taught in ch-54.

**Negative advantages** (negative as gradient: a negative advantage explicitly lowers the likelihood of the sampled tokens). The TIS authors' explanation of entropy collapse is that for rollouts with negative advantage and π_learner/π_sampler < 1, a decrease in π_learner may not appear in π_sampler, so updates keep pushing π_learner down; they state that the exact mechanism is open ([[rollout-training-mismatch-tis]], mechanism paragraph; Interpretation). ch-43a covers negative gradients.

**Conditions and limits.** The mismatch was small for DeepSeek-R1-Distill-Qwen-1.5B with BF16 rollouts, and correction gave no gain there ([[rollout-training-mismatch-tis]] Fig. 3). No sampler backend was consistently better (Fig. 11).

**Implication for a general-purpose model.** Entropy collapse and abnormal response length narrow output diversity and change pass@k (ch-43), and a logged reward that disagrees with held-out accuracy removes the training signal's link to capability. Sampler–learner KL, maximum mismatch, and mean mismatch are RL health metrics alongside reward.

## §8 Instability symptoms and separating numerical from data causes

**Symptoms with a numerical mechanism.**
1. *Non-finite gradients.* fp16 products above 65,504 become infinity; with loss scaling the step is skipped (§3).
2. *Attention logit growth.* In models of 2.4M-1.2B parameters at high learning rates, query and key norms grow; runs whose maximum attention logit exceeded 1e4 diverged, and a 4.8B run at learning rate 1e-2 diverged as predicted from smaller models; QK-layernorm mitigates it ([[small-scale-proxies-instabilities]] §3.1.1, §3.3). These runs used bfloat16 on TPUs (§2.1).
3. *Output logit divergence.* Output logits become very negative toward the end of training in models without weight decay; z-loss, which adds 1e-4 · log²Z with Z the softmax normalizer, resolves it (§3.1.2).
4. *Gradient RMS near the AdamW ε.* Gradient RMS of the first MLP layer decreases with model size and learning rate and is around 1e-8 at the largest scale and learning rate tested; for a 4.8B model at learning rate 0.3, ε = 1e-15 lowered loss and ε = 1e-6 diverged (§3.4, Fig. 11-12). MiniMax-M1 reports RL gradients from 1e-18 to 1e-5, most below 1e-14, and uses ε = 1e-15 ([[minimax-m1]] §3.2). OLMo 2 lowered ε from 1e-5 to 1e-8, which "lowers and stabilizes the norm of the gradient early in training" ([[olmo-2]] §3.4.1).
5. *Implementation-dependent backward.* OLMo 2's FlashAttention z-loss and a PyTorch z-loss gave the same forward value but different backward results, and the z-loss curves diverged; the authors "suspect the root cause lies in differences in precision", saw no effect on cross-entropy or downstream tasks, and still re-trained from the point of divergence ([[olmo-2]] §3.3.3, Fig. 8).

**Worked example: ε and dtype** (course calculation). The Adam update per element is m̂ / (√v̂ + ε) (ch-01). For a constant gradient g, √v̂ ≈ |g| after bias correction, so the update size relative to ε = 0 is |g| / (|g| + ε). With g = 1e-8 and ε = 1e-8 the ratio is 0.5; with ε = 1e-15 it is 0.9999999. With g = 1e-14 (the magnitude MiniMax-M1 reports) and ε = 1e-8 it is 1e-6, so those parameters do not move; with ε = 1e-15 it is 0.91. ε = 1e-15 is stored as 0 in fp16 (smallest subnormal 5.96e-8) and as 9.99e-16 in bf16, so the dtype of the optimizer computation limits which ε values exist.

**Symptoms with a data cause.** OLMo 2 found training batches at loss and gradient-norm spikes often contained long repeated n-gram sequences, but the relation was not deterministic: the same sequence spiked for a larger model and not a smaller one, spiked under one data order and not after reshuffling, and appeared in batches that did not spike. Removing documents with 32 or more repetitions of an n-gram of 1-13 tokens, and masking such sequences in the loss, reduced spikes but did not remove them and did not change the slow growth of gradient norm ([[olmo-2]] §3.1, Fig. 3).

**Metric: spike score** ([[olmo-2]] §3.2). The percentage of values in a time series that are at least seven standard deviations from the rolling average of the last 1,000 values. Example: 80 flagged gradient-norm values in 20,000 steps give a spike score of 0.40. OLMo 2 reports gradient-norm spike scores of 0.40 → 0.03 for its initialization change (§3.2), 0.108 → 0.069 for reordered norm plus QK-norm (§3.3.2, Fig. 7), and 0.16 versus 0.092 with and without weight decay on embeddings (§3.4.2, Fig. 10).

**Diagnostic procedure** (course procedure assembled from the sources above; no source tests the full sequence; Interpretation).
1. Check for non-finite loss or gradient norm and, in fp16, for skipped steps and the current loss scale.
2. Inspect the batch at the spike for repeated n-grams and other degenerate sequences ([[olmo-2]] §3.1).
3. Replay the step from the preceding checkpoint with the same batch and kernels. If the spike does not reproduce, the cause involves state that was not restored or a nondeterministic kernel; FlashAttention backward is the common LLM operation that needs atomic adds or recomputation for determinism ([[thinkingmachines-defeating-nondeterminism]], "When are atomic adds needed?").
4. Replay with the logits, loss, and normalization computed in fp32. A spike that disappears only in fp32 supports a numerical cause; a spike that persists in fp32 supports a data or optimization cause.
5. Read the mechanism-specific metrics: maximum attention logit, mean log Z of the output softmax, and gradient RMS relative to ε ([[small-scale-proxies-instabilities]] §3.1, §3.4).
6. Change a loss or kernel implementation only at a restart point with a comparison run, as OLMo 2 did (§3.3.3).

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 671B total / 37B act. | pretrain-stable | FP8 scope and format | Fprop, Dgrad, Wgrad GEMMs in FP8; E4M3 on all tensors | arXiv:2412.19437v2 §3.3.1-3.3.2 ([[deepseek-v3]]) | verified 2026-09-15 | App. B.1: relative loss error < 0.25% vs BF16 at ≈16B/1.33T and ≈230B/≈0.9T tokens; no downstream comparison |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | scaling granularity | activations 1×128 tiles; weights 128×128 blocks; online max-abs | §3.3.2 | verified 2026-09-15 | App. B.2: 128×128 blocks for activation gradients diverged at ≈16B, ≈300B tokens |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | kept in BF16 or FP32 | embedding, output head, MoE gating, normalization, attention | §3.3.1 | verified 2026-09-15 | no per-component ablation reported |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | optimizer and gradient dtypes | AdamW moments BF16; master weights FP32; gradients FP32 | §3.3.3 | verified 2026-09-15 | "without incurring observable performance degradation"; no numbers |
| DeepSeek-V3 | 671B / 37B | pretrain-stable | FP32 accumulation interval | N_C = 128 elements | §3.3.2 | verified 2026-09-15 | K = 4096 random matrices: max relative error nearly 2% without promotion; no table |
| OLMo 2 7B (stage 1 config) | 7B | pretrain-stable | autocast dtype; FSDP precision | amp_bf16; mixed (params and buffers bf16, gradient reduction fp32) | github.com/allenai/OLMo@090253d configs/official-1124/OLMo2-7B-stage1.yaml L84-88; olmo/config.py L1339-1344 ([[olmo-2-official-configs]]) | verified 2026-09-15 | no ablation reported |
| OLMo 2 7B (stage 1 config) | 7B | pretrain-stable | AdamW ε; grad clip | 1e-8; max_grad_norm 1.0 | yaml L48, L90; arXiv:2501.00656v3 §3.4.1 | verified 2026-09-15 | §3.4.1 Fig. 9: ε 1e-8 vs 1e-5 gives lower, more stable early gradient norm; no downstream numbers |
| OLMo 2 7B | 7B | pretrain-stable | z-loss coefficient | report: 1e-4 · log²Z | arXiv:2501.00656v3 §3.3.3 | conflict | yaml L39 sets auxiliary_loss_multiplier 1e-5, but train.py at 090253d does not pass it and the loss functions default to 1e-4 (L130, L162); the report value matches the code path |
| Wortsman et al. default | 2.4M-1.2B | pretrain-stable | dtype; z-loss; ε | bfloat16 on TPUs; 1e-4; 1e-8 | arXiv:2309.14322v2 §2.1 ([[small-scale-proxies-instabilities]]) | verified 2026-09-14 | z-loss: Fig. 3-4; ε: see next row |
| Wortsman et al. intervention | 4.8B | pretrain-stable | AdamW ε at LR 0.3 | 1e-15 | §3.4, Fig. 12 | verified 2026-09-14 | lower loss than 1e-8; 1e-6 diverged; one run per value |
| MiniMax-M1 | 456B total / 45.9B act. | RL | LM head precision | FP32 | arXiv:2506.13585v1 §3.2 ([[minimax-m1]]) | verified 2026-09-15 | Fig. 3: train–inference probability correlation 0.987319 → 0.997135; reward growth resumed (no curve numbers) |
| MiniMax-M1 | 456B / 45.9B | RL | AdamW β1, β2, ε | 0.9, 0.95, 1e-15 | §3.2 | verified 2026-09-15 | gradient magnitudes 1e-18 to 1e-5 (most < 1e-14); VeRL default (0.9, 0.999, 1e-8) "can result in" non-convergence; no ablation table |
| DAPO run from Qwen2.5-32B (TIS blog) | 32B | RL | TIS cap C; sampler / learner parallelism | 8; vLLM TP2 / FSDP Ulysses SP8 | github.com/yaof20/verl@fd02f7b recipe/flash_rl (card ledger, [[rollout-training-mismatch-tis]]) | verified 2026-09-14 | Fig. 1: with vs without TIS over 250 steps; no sweep over C |
| DeepSeek-R1-Distill-Qwen-7B and three other 7B-8B models (evaluation study) | 7B-8B | eval-gate | runtime configurations; max output; sampling | L40S/A100 × 2/4 GPUs × batch 8/16/32; 32,768 tokens (reasoning), 2,048 (non-reasoning); T 0.7, top-p 0.95 | arXiv:2506.09501v2 §3.1 ([[numerical-nondeterminism-llm-inference]]) | verified 2026-09-15 | Table 3: BF16 Std@Acc up to 9.15%; FP32 ≈ 0 |
| PyTorch 2.12.0 GradScaler (framework default, not a model run) | n/a | pretrain-stable | init scale; backoff; growth; growth interval | 2^16; 0.5; 2.0; 2000 | torch/amp/grad_scaler.py L123-131 (git 7661cd9) ([[pytorch-amp-autocast-gradscaler]]) | verified 2026-09-15 | no ablation reported; Micikevicius et al. used constant factors 8-32K ([[mixed-precision]] §3.2) |

**Starting point for a small general-purpose run.** Every value below comes from a verified row above, with its original conditions. Train under bf16 autocast with parameters and buffers in bf16 during computation and gradient reduction in fp32, as in the OLMo 2 7B stage-1 config (7B dense model, FSDP); keep master weights and gradients in fp32, as DeepSeek-V3 did (671B MoE, FP8 framework; §2 gives the rounding reason). Use AdamW ε = 1e-8 and gradient clipping at 1.0 as in that config, and log gradient RMS: if it approaches ε, the 4.8B study of Wortsman et al. (bfloat16 on TPUs, C4) found ε = 1e-15 better at a high learning rate, and MiniMax-M1 used 1e-15 for RL on a 456B MoE model. If only fp16 is available, start from PyTorch GradScaler defaults (initial scale 2^16, backoff 0.5, growth 2.0 every 2000 clean steps). For RL with a separate inference engine, compute the LM head in FP32 as MiniMax-M1 did, and if a truncated correction is used, the DAPO-32B runs used C = 8. For evaluation, record GPU type, GPU count, batch size, and dtype as in the 12-configuration design of Yuan et al.

## Generalization lens

**(a) What preserves or increases breadth.**
- Numerical choices do not add capability; they decide whether updates and measurements are correct. Full-precision master weights recovered FP32 accuracy where FP16 weight updates lost 80% relative accuracy ([[mixed-precision]] §3.1, Fig. 2a).
- Removing sampler–learner mismatch let RL continue: MiniMax-M1 reward increased after the FP32 LM head fix ([[minimax-m1]] §3.2); bitwise-identical sampling and training ran with KL 0 ([[thinkingmachines-defeating-nondeterminism]]).
- Training in lower precision made models more robust to post-training quantization ([[scaling-laws-for-precision]] §5), which matters when the served model is quantized.

**(b) What causes narrowing, forgetting, or false breadth claims.**
- Over-trained models lose more loss under PTQ: δ_PTQ ∝ D^0.5068 at fixed N in the fitted setting ([[scaling-laws-for-precision]] Eq. 2, App. K).
- INT8 rollouts without correction produced entropy below 0.2 and abnormally long responses on DAPO-32B, a reduction in output diversity ([[rollout-training-mismatch-tis]] Fig. 5).
- Without correction, logged RL reward was higher and AIME accuracy lower than with correction, so the reward curve overstated progress (Fig. 6).
- FP8 or other precision changes accepted on loss parity alone leave downstream parity unmeasured ([[deepseek-v3]] App. B.1).
- In bf16 weights, fine-tuning updates below half the spacing are erased (§2 worked example), which can make a low-learning-rate stage appear to have no effect.

**(c) How to measure it for this stage.**
- For a precision change in training: the ch-00 development and unseen suites on both checkpoints, plus per-domain loss, under one evaluation configuration.
- For a quantized deployment: δ_PTQ on held-out loss and on the same suites at the deployed bit width.
- For evaluation reproducibility: Std@Acc across at least two runtime configurations, the divergence rate, and FP32-computation or batch-invariant reruns for greedy comparisons ([[numerical-nondeterminism-llm-inference]] §3-4).
- For RL: sampler–learner KL, maximum and mean token-probability mismatch per response, entropy, response length, and held-out accuracy alongside logged reward ([[rollout-training-mismatch-tis]] Fig. 5-11).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Clipping fp16 gradients before unscaling | clipping fires on every step or on none; effective threshold is off by the factor S | call `unscale_` before `clip_grad_norm_` and log the unscaled norm ([[mixed-precision]] §3.2) |
| Applying optimizer updates to bf16 or fp16 weights | loss flat at low learning rate; parameters bitwise unchanged after a step | count parameters whose stored value changed; compare with an fp32 master copy (§2) |
| Choosing an ε the optimizer dtype cannot store | ε = 1e-15 becomes 0 in fp16; inf or NaN updates for parameters with zero second moment | compare ε with half of the dtype's smallest subnormal (§1 table, §8 worked example) |
| Treating one framework's fp32 op list as universal | a fused RMSNorm or attention kernel runs in bf16 although "norms are fp32" was assumed | read the op's actual dtype under autocast ([[pytorch-amp-autocast-gradscaler]]) |
| Accepting FP8 (or any lower precision) on loss curves only | loss within 0.25%; no evaluation comparison on record | score both checkpoints on development and unseen suites (§4) |
| One scale per tensor with outliers | nonzero values stored as 0 after FP8 casting | fraction of zeros per tensor or block after quantization (§4 worked example) |
| Quantizing a high-D/N checkpoint to ≤5 bits without measurement | larger quantized-model loss increase than for a checkpoint trained on fewer tokens | measure δ_PTQ and suite scores at the deployed bit width (§5) |
| Comparing greedy scores from different GPU counts, types, or batch sizes in BF16 | differences change sign on rerun; response lengths differ by thousands of tokens | fixed recorded configuration plus FP32 computation, or n-sample averages ([[numerical-nondeterminism-llm-inference]]) |
| Using sampler log-probabilities as learner log-probabilities | reward collapse; KL spike; negative k1 KL estimates; rising logged reward with flat held-out accuracy | log both log-probabilities, max and mean mismatch, and sampler–learner KL (§7) |
| Different sharding in sampler and learner | more responses with maximum mismatch above 0.5 | match tensor parallelism between sampler and learner ([[rollout-training-mismatch-tis]] Fig. 7-8) |
| Switching a loss or kernel implementation mid-run | curves diverge after the switch although forward values match | switch only at a restart with a comparison run ([[olmo-2]] §3.3.3) |
| Attributing every loss spike to numerics | spikes recur on batches with repeated sequences and move with data order | inspect spike batches; replay the step in fp32 (§8) |

## Check your understanding

1. bf16 and fp16 both use 16 bits. Explain why fp16 training needs loss scaling and bf16 training does not, and why bf16 training still keeps fp32 master weights.
2. A fine-tuning run at learning rate 1e-5 shows no change in any evaluation score. Using the spacing of bf16, explain a numerical cause and the measurement that would confirm or rule it out.
3. DeepSeek-V3 reports FP8 training loss within 0.25% of BF16. Explain which claims about general capability this supports and which it does not, and design the comparison that would test the rest.
4. In the §4 worked example, one outlier of 1000 causes −0.001 to be stored as 0. Explain the mechanism, and why DeepSeek-V3 can use E4M3 for all tensors while the FP8-formats paper recommends E5M2 for gradients.
5. Explain why greedy decoding of the same checkpoint gives different AIME'24 accuracies on 2 and 4 GPUs, why the effect is larger for long reasoning outputs, and why fixing one configuration does not remove the problem.
6. An RL run logs rising reward while held-out accuracy stays flat and entropy falls. Give two numerical causes from §7, the metrics that would distinguish them, and what each fix changes in the gradient estimator.
7. Explain why the post-training-quantization degradation of [[scaling-laws-for-precision]] grows with D/N, and what this implies for choosing a training token budget when the deployed model will run at 4 bits.
8. A loss spike occurs at step 40,000 of a bf16 run. Describe the sequence of checks that separates a data cause from a numerical cause, and what result at each step would change your conclusion.

## Connections

- **Previous:** ch-01 — Optimizers for LLM Training: AdamW, Update Size, and Retention of Prior Ability. The AdamW update and ε used in §2 and §8 are defined there.
- **Next:** ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization. QK-norm and normalization placement from §8 are treated there with their stability evidence.
- **Chapters that extend one section:** §1-§3 and §8 → ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures; §2 gradient-reduction dtype → ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen; §5 → ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability; §6 → ch-47 — Evaluation Harness and Suite Design for General Capability, and ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions; §7 → ch-54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments, ch-43 — Entropy, Output Diversity, and KL Control in RL, ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages, and ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL.
- **Measurement contract:** ch-00 — What General Capability Means and How It Is Measured (development and unseen suites, §6 measurement error).

## Sources

- [[mixed-precision]] — FP32 master weights, loss scaling, FP32 reductions, and the FP16 results (arXiv:1710.03740v3 §3-4). The card has not yet been re-verified; values here were read from the primary text and are in the chapter excerpt of the same name.
- [[fp8-formats-for-deep-learning]] — E4M3 and E5M2 limits (Table 1), scaling rule and overflow handling (§2), simulated FP8 training results (Table 4). Chapter excerpt.
- [[pytorch-amp-autocast-gradscaler]] — CUDA autocast op lists and GradScaler defaults at PyTorch v2.12.0. Chapter excerpt.
- [[deepseek-v3]] — FP8 framework, fine-grained quantization, accumulation precision, BF16 optimizer moments, FP8-vs-BF16 loss ablations (§3.3, App. B).
- [[deepseek-v3-recipe]] — model size, token count, and SFT learning-rate schedule of DeepSeek-V3.
- [[scaling-laws-for-precision]] — PTQ degradation versus D/N, effective parameter count, compute-optimal precision, stated limitations (arXiv:2411.04330v2). Chapter excerpt.
- [[numerical-nondeterminism-llm-inference]] — configuration-dependent greedy and sampled scores, divergence rates, LayerCast (arXiv:2506.09501v2). Chapter excerpt.
- [[thinkingmachines-defeating-nondeterminism]] — batch invariance as the cause of inference nondeterminism, completion counts, kernel costs, bitwise on-policy RL. Chapter excerpt; the planned library card was not present on 2026-09-15.
- [[rollout-training-mismatch-tis]] — sampler–learner mismatch measurements, parallelism and length factors, INT8 effects, truncated importance sampling.
- [[minimax-m1]] — LM-head precision as the source of train–inference mismatch, FP32 fix, RL AdamW ε (arXiv:2506.13585v1 §3.2). Chapter excerpt.
- [[small-scale-proxies-instabilities]] — attention logit growth, output logit divergence, z-loss, ε versus gradient RMS.
- [[olmo-2]] — data-caused spikes and repeated n-grams, spike score, z-loss implementation divergence, ε change (arXiv:2501.00656v3 §3; card not yet verified, loci read from the primary text).
- [[olmo-2-official-configs]] — OLMo 2 7B stage-1 precision, FSDP, ε, and z-loss settings in released config and code at commit 090253d. Chapter excerpt.
