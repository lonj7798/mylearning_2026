# Chapter 26: Quantization and Compression

<!-- scope: FP32 → FP16 → INT8 → INT4, PTQ (GPTQ/AWQ), QAT, outlier features, SmoothQuant, knowledge distillation, Pareto frontier
     deps: [[ch-25]]
     see-also: [[ch-27]]
-->

## Overview

[[ch-25]] established that autoregressive decoding is memory-bandwidth-bound: the GPU spends most of its time loading weights and KV cache from HBM, not doing arithmetic. Attention variants (GQA, MLA) compress the KV cache. This chapter attacks the other half of the memory wall — the **model weights themselves**.

A Llama 2 70B model in FP16 occupies 140 GB. That exceeds the memory of a single A100 (80 GB) or H100 (80 GB). Serving it requires tensor parallelism across multiple GPUs, multiplying hardware cost and inter-GPU communication latency. But the same model in INT4 fits in 35 GB — comfortably on a single GPU. The question is what you lose in the process.

Quantization is the art of representing high-precision floating-point weights with fewer bits while preserving model quality. It sounds like a straightforward compression problem, but LLM weight distributions have pathological properties — outlier features with magnitudes 100x larger than typical values — that make naive approaches catastrophically degrade quality. The methods in this chapter (GPTQ, AWQ, SmoothQuant, QAT) exist because round-to-nearest quantization breaks on real LLM weights.

Beyond quantization, knowledge distillation offers a complementary path: instead of compressing weights, train a smaller model to replicate a larger one's behavior. Phi-4 ([[phi-4|report]]) demonstrated that a 14B student can surpass its GPT-4o teacher on STEM benchmarks. DeepSeek-R1 ([[deepseek-r1|report]]) showed that distilling reasoning capabilities from a 671B model into Qwen-32B achieves 72.6% on AIME 2024 versus 47.0% with RL-only training.

The chapter culminates at the Pareto frontier: the tradeoff curve between accuracy and inference speed across quantization levels and model sizes. Understanding this frontier is how you make the practical decision of which model to deploy.

---

## 1. Number Formats: What Precision Actually Means

Before discussing quantization methods, you need to understand what gets lost when you reduce precision. A floating-point number has three components: sign bit, exponent bits (controlling dynamic range), and mantissa bits (controlling precision within that range).

$$x = (-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times (1 + \text{mantissa})$$

[See [figures/precision-formats.html](figures/precision-formats.html) for an interactive comparison of all formats.]

### The Format Landscape

| Format | Bits | Exponent | Mantissa | Dynamic Range | Use Case |
|--------|------|----------|----------|---------------|----------|
| FP32 | 32 | 8 | 23 | ~1e-38 to 1e+38 | Training (master weights) |
| BF16 | 16 | 8 | 7 | Same as FP32 | Training (forward/backward) |
| FP16 | 16 | 5 | 10 | ~6e-5 to 65,504 | Inference default |
| FP8 (E4M3) | 8 | 4 | 3 | ~1e-7 to 448 | Training (DeepSeek-V3) |
| FP8 (E5M2) | 8 | 5 | 2 | ~1e-7 to 57,344 | Training (gradients) |
| INT8 | 8 | 0 | 7 | -128 to 127 | Post-training quantization |
| INT4 | 4 | 0 | 3 | -8 to 7 | Aggressive PTQ |

**BF16 vs FP16:** BF16 keeps FP32's 8-bit exponent, so it covers the same dynamic range. FP16 has only 5 exponent bits — its maximum value is 65,504, meaning large gradient values cause overflow. This is why BF16 has become the default training format: it handles the same magnitude of values as FP32, just with less precision per value.

**FP8 in practice:** DeepSeek-V3 ([[deepseek-v3|report]]) pioneered FP8 mixed-precision *training* (not just inference). Their approach uses E4M3 format with tile-wise quantization (1x128 blocks for activations, 128x128 blocks for weights), fine-grained scaling factors, and high-precision accumulation every 128 elements. The result: 2x computational speedup over BF16 with less than 0.25% relative loss error. Critically, sensitive components — embeddings, output heads, gating modules, normalization layers, and attention — remain in FP32/FP16. This selective precision is a recurring theme: not all weights are equally important.

**The integer formats (INT8, INT4)** have zero exponent bits — they are pure fixed-point. Every value is evenly spaced across the representable range. This is maximally efficient for hardware (integer multiply-accumulate is cheaper and faster than floating-point), but it means a single outlier value can force the entire range to accommodate it, wasting precision on the vast majority of well-behaved weights.

### Memory Arithmetic

The memory savings are proportional to the bit reduction:

$$\text{Model size (bytes)} = \text{Parameters} \times \frac{\text{bits}}{8}$$

| Model | FP32 | FP16/BF16 | INT8 | INT4 |
|-------|------|-----------|------|------|
| Llama 2 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| Llama 2 70B | 280 GB | 140 GB | 70 GB | 35 GB |
| DeepSeek-V3 (37B active) | 148 GB | 74 GB | 37 GB | 18.5 GB |

At INT4, a 70B model fits on a single consumer GPU (RTX 4090 has 24 GB). This is the entire reason the open-weight LLM ecosystem on consumer hardware exists — without INT4 quantization, running 70B models locally would require multi-GPU setups costing $10,000+.

---

## 2. Uniform Quantization: The Baseline

The simplest quantization scheme maps a continuous range of floating-point values to evenly-spaced integer levels.

### Symmetric Quantization

For $b$-bit signed integers, the range $[-2^{b-1}, 2^{b-1}-1]$ is mapped to the weight range $[-\alpha, \alpha]$ where $\alpha = \max(|w|)$:

$$w_q = \text{round}\!\left(\frac{w}{\Delta}\right), \qquad \Delta = \frac{\alpha}{2^{b-1} - 1}$$

$$\hat{w} = w_q \times \Delta \qquad \text{(dequantize)}$$

### Asymmetric Quantization

When the weight distribution is not centered on zero, asymmetric quantization uses a zero-point offset:

$$w_q = \text{round}\!\left(\frac{w - w_{\min}}{\Delta}\right), \qquad \Delta = \frac{w_{\max} - w_{\min}}{2^b - 1}$$

### Per-Tensor vs Per-Channel vs Group Quantization

The granularity of $\Delta$ (the scale factor) dramatically affects quality:

- **Per-tensor:** One scale factor for the entire weight matrix. Cheapest but worst quality — a single outlier channel distorts every other channel's precision.
- **Per-channel:** One scale factor per output channel (row of the weight matrix). Much better quality, standard for INT8.
- **Group quantization:** One scale factor per group of $g$ consecutive weights within a channel. GPTQ and AWQ typically use $g = 128$. The overhead is one FP16 scale per 128 INT4 weights = 0.125 bits/weight, so "INT4 with group size 128" is actually 4.125 bits/weight.

The tradeoff: finer granularity means more scale factors stored (small memory overhead) and more dequantize operations (small compute overhead), but substantially better preservation of the original weight distribution.

---

## 3. The Outlier Problem: Why Naive Quantization Fails

Round-to-nearest quantization works well for weights with roughly Gaussian distributions. LLM weights are *not* Gaussian. Starting around 6.7B parameters, transformer models develop **outlier features** — individual hidden dimensions where activations (and corresponding weights) are 10-100x larger than the mean.

### The Empirical Pattern

Dettmers et al. (2022) ("LLM.int8()") discovered that in models above ~6B parameters:

- A small fraction of hidden dimensions (~0.1-1%) consistently produce activation magnitudes 10-100x larger than the rest
- These outliers appear in the *same* feature dimensions across different inputs
- They emerge during training and become more extreme as models scale
- They are systematically important: zeroing them destroys model quality; zeroing random dimensions of the same proportion has negligible effect

### Why Outliers Break Quantization

Consider quantizing a weight vector to INT8 where 99% of values are in $[-0.5, 0.5]$ but one outlier is at $50.0$:

$$\Delta = \frac{50.0}{127} \approx 0.394$$

The 99% of weights in $[-0.5, 0.5]$ get mapped to the integer range $[-1, 1]$ — only 3 distinct values to represent a rich distribution. Effectively, 99% of the weight information is destroyed to accommodate 1% of outlier values.

### LLM.int8(): Mixed-Precision Decomposition

The simplest solution: keep outlier dimensions in FP16 and quantize everything else to INT8. During matrix multiplication:

$$Y = XW^\top \approx X_{\text{outlier}} W_{\text{outlier}}^\top + \text{dequant}(X_{\text{normal}}^{(\text{int8})} \cdot W_{\text{normal}}^{(\text{int8})})$$

This incurs minimal overhead (the outlier dimensions are <1% of the total) while preventing catastrophic accuracy loss. However, it requires custom kernels to handle the mixed-precision split and does not achieve the full memory savings of pure INT8.

---

## 4. SmoothQuant: Migrating Difficulty from Activations to Weights

Xiao et al. (2023) observed a crucial asymmetry: **weights are easy to quantize; activations are hard.** Weight distributions are relatively smooth and well-behaved. Activation distributions have the outlier spikes described above. But the matrix multiply $Y = XW$ allows you to mathematically redistribute the quantization difficulty.

### The Core Idea

Introduce a per-channel scaling factor $s$ that divides activations and multiplies weights:

$$Y = XW = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) = \hat{X} \hat{W}$$

Choose $s_j$ to equalize the quantization difficulty between the $j$-th activation channel and the corresponding weight channel:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

where $\alpha \in [0, 1]$ controls the migration strength. At $\alpha = 0.5$, outlier magnitudes are geometrically split between activations and weights. In practice, $\alpha = 0.5$ works well for most models, though some layers benefit from $\alpha = 0.75$ (shifting more difficulty to weights, which are more tolerant).

### Why This Works

The key insight: a per-channel scale applied to activations is equivalent to a per-channel scale applied to weights — but weights are static and can be pre-processed offline. After smoothing, both activations and weights have similar dynamic ranges, making both amenable to INT8 quantization. The combined W8A8 (8-bit weights, 8-bit activations) quantization achieves near-lossless quality while enabling INT8 matrix multiply hardware (tensor cores in INT8 mode deliver ~2x the throughput of FP16).

SmoothQuant enabled the first practical W8A8 quantization of 175B-class models with negligible quality loss, and the per-channel smoothing factors are computed once from a small calibration set.

---

## 5. Post-Training Quantization (PTQ): GPTQ and AWQ

PTQ methods quantize a pre-trained model without retraining. They require only a small calibration dataset (typically 128-512 examples) and complete in minutes to hours, not days. This makes them the practical default for deploying open-weight models.

### GPTQ: Optimal Brain Quantization, One Layer at a Time

Frantar et al. (2023) adapted Optimal Brain Quantization (OBQ) to scale to billion-parameter models. The core algorithm processes one layer at a time:

**The objective:** For weight matrix $W$ and calibration inputs $X$, find quantized weights $\hat{W}$ that minimize:

$$\|WX - \hat{W}X\|_2^2$$

This is a layer-wise reconstruction error — the quantized layer should produce the same output as the original on calibration data.

**The algorithm (simplified):**

1. Compute the Hessian $H = 2XX^\top$ for the layer (measures weight sensitivity)
2. Process weights column by column:
   a. Quantize column $i$ using round-to-nearest
   b. Compute the quantization error $\delta_i = w_i - \hat{w}_i$
   c. **Compensate** by adjusting all remaining unquantized columns using the Hessian: $w_{j>i} \mathrel{-}= \delta_i \cdot H_{ij} / H_{ii}$

Step 2c is the critical innovation. When quantizing one weight introduces error, the algorithm *redistributes* that error across the remaining weights to minimize the overall layer output error. This is why GPTQ substantially outperforms naive round-to-nearest: it uses second-order information (the Hessian) to find the best set of integer weights jointly, not independently.

**Practical performance:** GPTQ can quantize a 175B-parameter model to INT4 in approximately 4 GPU-hours using a single A100. The resulting model typically loses 0.5-2 perplexity points compared to FP16 — small enough for most applications.

### AWQ: Activation-Aware Weight Quantization

Lin et al. (2024) observed that not all weights are equally important, and the important ones can be identified by looking at activations, not weights:

**Key insight:** Weights corresponding to large activation magnitudes matter more for model quality. A 1% salient weight channel (identified by activation magnitude) can account for >50% of the quantization error.

**The method:**

1. Run calibration data through the model; record activation magnitudes per channel
2. For each weight channel $j$, compute an importance score proportional to $\text{mean}(|X_j|)$
3. Apply a per-channel scaling factor $s_j > 1$ to important weight channels before quantization, and $s_j^{-1}$ to the corresponding activation channel to preserve mathematical equivalence
4. Quantize the scaled weights using standard round-to-nearest

The scaling protects important weights by expanding their range before quantization — they get more integer levels allocated to them. The inverse scaling on activations is folded into the preceding layer's output projection, so the running model performs no extra computation.

**AWQ vs GPTQ:** AWQ is simpler (no Hessian computation), faster (minutes vs hours for 70B models), and often achieves comparable or slightly better quality. AWQ also generalizes better to instruction-tuned models and diverse tasks because it doesn't overfit to a specific calibration set the way Hessian-based methods can. Both are widely supported in inference frameworks (vLLM, TensorRT-LLM, llama.cpp).

### When PTQ Breaks Down

PTQ methods struggle below 4 bits per weight (INT3 and below). At this extreme compression, the quantization grid is too coarse for even the best error-compensation strategies to preserve quality. The typical failure mode:

- INT8: near-lossless (<0.1 perplexity increase)
- INT4 with GPTQ/AWQ: small degradation (0.5-2 perplexity points)
- INT3: significant degradation (5-15 perplexity points), often with task-specific failures
- INT2: catastrophic — the model produces incoherent outputs

For sub-4-bit deployment, quantization-aware training (Section 6) or knowledge distillation (Section 7) typically outperforms PTQ.

---

## 6. Quantization-Aware Training (QAT)

QAT integrates quantization into the training loop itself. Instead of quantizing a trained model post-hoc, the model *learns* to be quantized — adapting its weight distribution to minimize the error introduced by quantization.

### The Straight-Through Estimator (STE)

The fundamental challenge: quantization (rounding) is non-differentiable. You cannot backpropagate through a round() function because its gradient is zero almost everywhere and undefined at integer boundaries.

The STE workaround: during the forward pass, apply quantization; during the backward pass, **pretend quantization didn't happen** and pass gradients straight through:

$$\text{Forward:} \quad \hat{w} = \text{dequant}(\text{quant}(w))$$

$$\text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}} \quad \text{(STE: gradient passes through quant/dequant)}$$

The model sees quantized weights in every forward pass and learns to compensate — it pushes weights toward values that are representable in the target precision, avoids placing critical information on decision boundaries that quantization will snap to the wrong side, and learns redundant representations that are robust to per-weight rounding errors.

### QAT vs PTQ: When Is It Worth the Cost?

QAT requires a full (or partial) training run with quantization simulation. The cost is typically 5-20% of original pre-training compute. The benefit is substantially better quality at low bit widths:

| Precision | PTQ Quality | QAT Quality | Gap |
|-----------|-------------|-------------|-----|
| INT8 | Near-lossless | Near-lossless | Negligible — PTQ is sufficient |
| INT4 | -0.5 to -2 ppl | -0.1 to -0.5 ppl | Moderate — QAT justified for quality-critical apps |
| INT3 | -5 to -15 ppl | -1 to -3 ppl | Large — QAT essential if deploying at INT3 |
| INT2 | Catastrophic | -5 to -10 ppl | QAT enables what PTQ cannot |

**The practical rule:** Use PTQ (GPTQ/AWQ) for INT8 and INT4 deployment — the quality loss is small and the quantization cost is minutes. Use QAT only when pushing below INT4 or when every fraction of a perplexity point matters.

### FP8 Training as Implicit QAT

DeepSeek-V3's FP8 training ([[deepseek-v3|report]]) is conceptually a form of QAT. By training the model in FP8 precision from the start (for compute-heavy GEMM operations), the model learns weight distributions that are naturally amenable to low-precision representation. The key details:

- **Tile-wise quantization:** Activations use 1x128 tiles, weights use 128x128 blocks. Each tile has its own FP8 scaling factor.
- **High-precision accumulation:** Partial sums are accumulated in FP32 every 128 multiply-accumulate operations, preventing numerical drift.
- **Selective precision:** Embeddings, output heads, gating, normalization, and attention remain in higher precision. Only the dense matrix multiplies — the bulk of computation — run in FP8.
- **Result:** <0.25% relative loss error versus BF16 training at 2x computational throughput. This is the tightest QAT-like result at scale.

---

## 7. Knowledge Distillation: Training Smaller Models

Quantization compresses weights; distillation compresses *knowledge*. A large "teacher" model's behavior is transferred to a smaller "student" model through training.

### The Distillation Loss

Standard training minimizes cross-entropy against hard labels (one-hot targets). Distillation adds a second objective — matching the teacher's soft probability distribution:

$$\mathcal{L}_{\text{distill}} = (1-\alpha)\,\mathcal{L}_{\text{CE}}(y, p_s) + \alpha \, T^2 \, \text{KL}(p_t^{(T)} \| p_s^{(T)})$$

where $p_t^{(T)}$ and $p_s^{(T)}$ are teacher and student logits divided by temperature $T$ before softmax, and $\alpha$ balances the two losses. The temperature softens the distributions — at high $T$, the teacher reveals which wrong answers it considers "almost right," providing richer gradient signal than hard labels.

### Why Distillation Works

The teacher's soft probabilities encode structural knowledge that hard labels discard:

- **Similarity structure:** The teacher might assign 10% probability to a synonym of the correct answer. This tells the student that those tokens are semantically related — information absent from the one-hot label.
- **Negative knowledge:** Low but nonzero probabilities on certain tokens encode "this is plausible but wrong in this context" — harder to learn from one-hot labels alone.
- **Dark knowledge:** The relative probabilities across the vocabulary encode the teacher's internal representation of the input, providing a denser training signal per example.

### Case Study: Phi-4 — Student Surpasses Teacher

Phi-4 ([[phi-4|report]]) is the most striking distillation result in the LLM era. A 14B-parameter model achieves:

- **GPQA (graduate STEM):** 56.1% vs GPT-4o's 50.6%
- **MATH:** 80.4% vs GPT-4o's 74.6%

The mechanism is not pure distillation in the classic sense — Phi-4 uses 40% synthetic training data generated by GPT-4o, effectively distilling the teacher's capabilities into training data rather than through logit matching. The synthetic data is replayed for 13.8 epochs, an extreme departure from the "see each token once" convention that works because synthetic examples have high information density.

**Why the student can surpass the teacher:** The student model has the advantage of training exclusively on the highest-quality subset of the teacher's knowledge. GPT-4o must be a generalist; Phi-4 concentrates on STEM reasoning. This is analogous to a human expert outperforming a polymath on their specialty — less total knowledge but higher density in the target domain.

### Case Study: DeepSeek-R1 Distillation

DeepSeek-R1 ([[deepseek-r1|report]]) provides quantitative evidence that distillation outperforms RL on smaller models:

| Model | Method | AIME 2024 |
|-------|--------|-----------|
| Qwen-32B | RL only | 47.0% |
| Qwen-32B | R1 distillation | 72.6% |
| Qwen-14B | R1 distillation | 69.7% |
| Qwen-7B | R1 distillation | 55.5% |
| Qwen-1.5B | R1 distillation | 28.9% |

The distillation uses 800K samples curated from R1's chain-of-thought outputs. No RL is applied to the distilled models — pure SFT is sufficient when the training data captures the teacher's reasoning traces. The 1.5B distilled model outperforms GPT-4o (9.3%) on AIME 2024, demonstrating that even very small models can acquire strong reasoning through distillation.

### Distillation vs Quantization: Complementary, Not Competing

Distillation and quantization operate on different axes:

- **Distillation** reduces the number of parameters (architectural compression)
- **Quantization** reduces the bits per parameter (numerical compression)

They compose: you can distill a 70B model into a 7B model, then quantize the 7B model to INT4, achieving a combined ~80x memory reduction. The Pareto frontier in Section 8 shows how to navigate this two-dimensional tradeoff space.

---

## 8. The Pareto Frontier: Accuracy vs Speed

[See [figures/pareto-frontier.html](figures/pareto-frontier.html) for an interactive Pareto frontier plot.]

Every deployment decision is a point in the accuracy-speed space. The Pareto frontier is the curve of optimal tradeoffs — points where you cannot improve accuracy without sacrificing speed, or vice versa.

### The Two Axes of Compression

**Axis 1: Model size** (distillation axis)
- 70B → 14B → 7B → 3B → 1.5B
- Each step loses some capability, especially on hard reasoning tasks
- But smaller models are faster even at the same precision

**Axis 2: Precision** (quantization axis)
- FP16 → INT8 → INT4 → INT3
- Each step reduces memory and increases throughput (more tokens/second)
- But lower precision degrades quality, especially for hard tasks

### Empirical Pareto Observations

Several patterns emerge from the research:

**1. A quantized large model usually beats a smaller model at full precision for the same memory budget.**

A 70B model at INT4 (35 GB) typically outperforms a 13B model at FP16 (26 GB) on most benchmarks, despite using more memory. The larger model has more knowledge encoded in its structure; quantization merely reduces the precision of each weight.

**2. The quality gap from quantization is task-dependent.**

INT4 quantization loses relatively little on knowledge-heavy tasks (MMLU, factual QA) where the model's stored knowledge matters more than precision of computation. It loses more on reasoning-heavy tasks (MATH, code generation) where computational precision matters.

**3. INT4 is the practical sweet spot for PTQ.**

Below INT4, quality degrades rapidly without QAT. Above INT4 (INT8), the memory savings may not be sufficient to change the deployment scenario (still need the same number of GPUs). INT4 hits the inflection point: maximum memory reduction with acceptable quality loss.

**4. Distillation + quantization is more efficient than either alone.**

The Phi-4 example: a 14B distilled model at INT4 (~7 GB) achieves STEM performance competitive with GPT-4o while fitting on a consumer GPU. Neither distillation alone (14B at FP16 = 28 GB, still needs a large GPU) nor quantization alone (70B at INT4 = 35 GB, less strong on STEM than Phi-4) achieves this operating point.

### Serving Cost Implications

The practical consequence connects back to [[ch-25]]: quantization directly determines serving cost.

| Config | Memory | GPUs (A100-80G) | Relative Cost |
|--------|--------|-----------------|---------------|
| Llama 2 70B FP16 | 140 GB | 2 | 1.0x |
| Llama 2 70B INT8 | 70 GB | 1 | ~0.5x |
| Llama 2 70B INT4 | 35 GB | 1 | ~0.5x (faster tok/s) |
| Llama 2 70B INT4 + GQA | 35 GB + small KV | 1 | ~0.5x (higher throughput) |

Moving from 2 GPUs to 1 GPU is not merely a 2x cost saving — it eliminates inter-GPU communication latency, simplifies deployment, and increases per-request throughput. This is why INT4 quantization with GPTQ/AWQ has become the standard deployment format for open-weight models.

---

## Core Insights from the Literature

### Insight 1: Outlier features are the fundamental obstacle to LLM quantization
**Source:** Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)

The discovery that LLM activations develop systematic outlier features — specific hidden dimensions with magnitudes 10-100x the mean — reframed quantization from a generic compression problem into an LLM-specific challenge. These outliers emerge during training at scale (>6B parameters) and are structurally important: zeroing them destroys quality while zeroing random dimensions has negligible effect. Every subsequent quantization method (SmoothQuant, GPTQ, AWQ) is fundamentally a strategy for handling these outliers — either by separating them (LLM.int8()), redistributing them (SmoothQuant), or protecting them (AWQ). **Guideline:** Never quantize an LLM without accounting for outlier features. Naive round-to-nearest quantization is adequate for CNNs but catastrophic for large transformers.

### Insight 2: Quantization difficulty can be migrated between activations and weights
**Source:** Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023)

The mathematical identity $Y = (X \text{diag}(s)^{-1})(\text{diag}(s) W)$ enables moving quantization difficulty from activations (hard, due to outliers) to weights (easy, smoother distributions). This is a deeper insight than it appears: it shows that the difficulty of quantization is not intrinsic to the computation but depends on how you partition the representational burden across the operands of each matmul. **Guideline:** W8A8 quantization with SmoothQuant achieves near-lossless quality and ~2x throughput on INT8 tensor cores. For INT8 deployment, SmoothQuant is the method of choice because it quantizes both weights and activations.

### Insight 3: Second-order information rescues weight quantization at low bit widths
**Source:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2023)

GPTQ's Hessian-based error compensation — adjusting unquantized weights to absorb the error from quantizing each column — enables INT4 quantization with only 0.5-2 perplexity points degradation on 175B models. Without this compensation, round-to-nearest at INT4 would be unusable. The Hessian captures which weights are most sensitive to perturbation and how errors propagate through the layer. **Guideline:** For INT4 deployment, use GPTQ or AWQ (not naive RTN). The quality difference is the gap between a usable and a broken model. AWQ is simpler and often equivalent; GPTQ gives slightly better results on some models but takes longer.

### Insight 4: A smaller model distilled from a larger one can surpass its teacher
**Source:** Microsoft, "Phi-4 Technical Report" ([[phi-4|report]])

Phi-4 (14B) surpasses GPT-4o on GPQA (56.1% vs 50.6%) and MATH (80.4% vs 74.6%) despite being orders of magnitude smaller. The mechanism is domain-focused synthetic data: 40% of training tokens are synthetic, replayed for 13.8 epochs. The student surpasses the teacher because it concentrates the teacher's knowledge into a narrower domain, achieving higher density of relevant knowledge per parameter. **Guideline:** When the target domain is well-defined (STEM, code, a specific language), distillation from a strong generalist teacher can produce a specialist student that exceeds the teacher on its specialty. This is often more cost-effective than training a large model from scratch.

### Insight 5: FP8 training is viable at frontier scale with careful quantization design
**Source:** DeepSeek AI, "DeepSeek-V3 Technical Report" ([[deepseek-v3|report]])

DeepSeek-V3 trained a 671B MoE model in FP8 for the compute-heavy GEMM operations, achieving 2x throughput over BF16 with <0.25% relative loss error. The enabling design choices: tile-wise quantization granularity (1x128 for activations, 128x128 for weights), FP32 accumulation every 128 operations, and selective precision (keeping attention, normalization, and gating in FP16/FP32). **Guideline:** FP8 training is no longer experimental — it is production-proven at the largest scales. The savings are substantial enough to halve training costs, which for frontier models is tens of millions of dollars. But it requires fine-grained quantization design, not just a dtype switch.

---

## Key Takeaways

1. **Quantization is an LLM-specific challenge because of outlier features.** Above ~6B parameters, transformers develop hidden dimensions with activation magnitudes 10-100x the mean. Naive quantization destroys information in the 99% of well-behaved dimensions to accommodate the 1% of outliers. Every practical quantization method is fundamentally a strategy for handling this asymmetry.

2. **INT4 with GPTQ or AWQ is the deployment sweet spot.** It halves memory relative to INT8 (enabling single-GPU serving for 70B models) with 0.5-2 perplexity points of degradation — acceptable for most applications. Below INT4, quality degrades rapidly without QAT.

3. **SmoothQuant enables W8A8 quantization by migrating difficulty from activations to weights.** The per-channel scaling identity exploits the fact that weight distributions are smoother than activation distributions. This is the method of choice for INT8 deployment where both weights and activations are quantized.

4. **PTQ (minutes) vs QAT (days) is a cost-quality tradeoff.** Use PTQ for INT8 and INT4. Use QAT only when pushing below INT4 or when the marginal quality improvement justifies 5-20% of pre-training compute. FP8 training (DeepSeek-V3) is an implicit form of QAT that halves training cost.

5. **Knowledge distillation is orthogonal to quantization and composes with it.** Distillation reduces parameters; quantization reduces bits per parameter. The most efficient deployment combines both: distill 70B → 7B, quantize to INT4 → ~80x memory reduction.

6. **A quantized large model typically beats a smaller model at full precision for the same memory budget.** The 70B-INT4 vs 13B-FP16 comparison favors the larger model because quantization loses less information than removing 80% of the parameters.

7. **The Pareto frontier is two-dimensional (model size x precision) and task-dependent.** Reasoning-heavy tasks lose more from quantization than knowledge-heavy tasks. The optimal operating point depends on your target task distribution, latency budget, and hardware constraints.

---

## References

- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022) — outlier features, mixed-precision decomposition
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023) — activation-to-weight difficulty migration
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2023) — Hessian-based INT4 quantization
- Lin et al., "AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression" (2024) — activation-aware scaling for PTQ
- [[phi-4|Microsoft, "Phi-4 Technical Report" (2024) (report)]] — student surpasses teacher via synthetic data distillation
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]] — FP8 mixed-precision training at 671B scale
- [[deepseek-r1|DeepSeek AI, "DeepSeek-R1 Technical Report" (2025) (report)]] — distillation of reasoning capabilities
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) — foundational distillation framework
- [[raschka-llm-architecture-comparison|Raschka, "The Big LLM Architecture Comparison" (2026) (blog)]] — FP8 training trends across model families
