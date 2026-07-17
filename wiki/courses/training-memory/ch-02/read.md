<!-- chapter: ch-02
     track: ledger
     title: Optimizer States, Precision, and the Loss-Head Spike
     deps: [[ch-01]]
     sources: [[mixed-precision-training]], [[fp8-training]], [[liger-fused-ce]]
-->

# Chapter 2 — Optimizer States, Precision, and the Loss-Head Spike

> **Core insight.** The 12 bytes/param Adam tax is not an accident of implementation — it is a direct consequence of fp16's 5-bit exponent: without an fp32 master weight, ~5% of gradient updates underflow to zero each step, and the model quietly diverges. BF16 extends the exponent to 8 bits (same as fp32) to sidestep the dynamic-range trap, but the fp32 master copy remains mandatory even in bf16 mixed-precision because optimizer accumulation still needs fp32 numerical fidelity. The loss-head logit tensor is a transient spike that lives entirely outside this static floor and can exceed it by a large margin at long sequence length and large vocabulary.

> **Guideline.** Budget 18 bytes/param as the static training floor for mixed-precision AdamW (2 bf16 working weights + 4 fp32 master + 4 fp32 gradients + 8 fp32 optimizer states); on A100 and older, stop there. On H100+, FP8 training (Transformer Engine, E4M3/E5M2) compresses that floor by ~39% for GPT-175B class models, but requires Hopper tensor cores — do not attempt FP8 on A100. Always deploy a fused/chunked cross-entropy kernel (Liger) to eliminate the transient B·T×V logit spike; it is the largest single-step memory event in high-vocab training and costs nothing in accuracy.

---

## 1. Adam's 12 Bytes/Param: Why the Math Forces It

The [[ch-01]] ledger assigned 12 bytes/param to the Adam optimizer bucket. Here is the exact accounting and why every byte is non-negotiable under mixed precision.

**The three-tensor layout of mixed-precision AdamW:**

| Tensor | Dtype | Bytes/param |
|--------|-------|-------------|
| Working weights (compute copy) | bf16 / fp16 | 2 |
| fp32 master weights (optimizer copy) | fp32 | 4 |
| fp32 gradients (accumulated) | fp32 | 4 |
| First-order moment m (momentum) | fp32 | 4 |
| Second-order moment v (variance) | fp32 | 4 |
| **Static floor total** | | **18** |

The "Adam costs 12 B/param" figure from [[ch-01]] refers to the fp32 master weights (4) + m (4) + v (4) = 12 that live only in the optimizer; the working weight copy (2) and gradient copy (4) account for the remaining 6 to reach 18. The [[ultrascale-playbook]] and [[transformer-math-101]] both quote this 18 B/param floor, with the note that it becomes 20 B/param under fp32 gradient accumulation.

**Why fp32 master weights are mandatory — the two failure modes ([[mixed-precision-training]]):**

1. **Gradient underflow.** "Any value whose magnitude is smaller than 2^−24 becomes zero in FP16." Empirically, ~5% of weight gradient magnitudes in a representative model fall below this threshold each step. Each such gradient is silently zeroed; those parameters stop learning. At steady state across all steps, the model diverges.

2. **Mantissa cancellation (weight update rule).** Adam applies: `w ← w − lr · m̂ / (√v̂ + ε)`. When `|w| / |update| > 2048`, the 10 mantissa bits of fp16 cannot represent the difference — `w + Δw` rounds back to `w` identically and the update is discarded. With an fp32 master weight, the accumulation has 23 mantissa bits; no such cancellation.

The empirical cost of ignoring this: the Mandarin speech model in the paper showed "80% relative accuracy loss" training with fp16-only weights versus a bf16 master. The fp32 master is not a conservative engineering choice — it is the minimum precision that keeps training numerically stable.

---

## 2. BF16 vs FP16: The Exponent Is the Variable

Understanding why BF16 replaced FP16 as the default training dtype requires seeing what each format trades off.

**Format comparison:**

| Format | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32 | 1 | 8 bits | 23 bits | ±3.4×10³⁸ | ~7 decimal digits |
| BF16 | 1 | 8 bits | 7 bits | ±3.4×10³⁸ | ~2 decimal digits |
| FP16 | 1 | 5 bits | 10 bits | ±65,504 | ~3 decimal digits |

BF16 is a truncated fp32: same exponent width (8 bits), fewer mantissa bits (7 vs 23). FP16 trades exponent bits for mantissa bits compared to fp32.

**The practical consequence** is that fp16's 5-bit exponent limits its representable range to ±65,504, while BF16's 8-bit exponent covers the same range as fp32 (±3.4×10³⁸). Gradient magnitudes during training span many orders of magnitude and routinely exceed fp16's range. BF16 has the same exponent as fp32, so no dynamic-range problem arises — the fp32 master weight is still kept for optimizer accumulation fidelity, but the separate loss-scaling machinery of fp16 is eliminated.

**Loss scaling for FP16 ([[mixed-precision-training]]):**

Because FP16 gradients overflow or underflow during backward without intervention, the standard remedy is loss scaling:

```
# Forward pass
loss_scaled = loss * scale_factor   # e.g. scale_factor = 8 to 32768

# Backward pass (chain rule scales all gradients by scale_factor)
loss_scaled.backward()

# Before optimizer step: unscale and clip
for param in model.parameters():
    param.grad /= scale_factor
clip_grad_norm_(model.parameters(), max_norm)

optimizer.step()
```

The scale factor shifts the entire gradient histogram upward in magnitude, pulling underflowing values (which would be zero in FP16) into FP16's representable range. The chain rule guarantees this is mathematically equivalent to scaling the loss — no rewrite of the computation graph needed. After backward, divide by the same factor before the clip and optimizer step. Without this, for SSD object detection: "67% of gradient values are zero in FP16 without scaling; with 8× scaling, training matched fp32 accuracy." The bigLSTM required 128× scaling. Dynamic loss scaling adjusts the factor during training — if overflow is detected (NaN/Inf in gradients), halve the scale; if no overflow for N steps, double it.

**Modern default is BF16 + fp32 masters, no loss scaling** ([[mixed-precision-training]]): "BF16 has the same exponent width as fp32 (8 bits vs FP16's 5), so BF16 eliminates the dynamic-range problem; this paper's loss-scaling technique is less critical for BF16."

---

## 3. FP8 on Hopper: Compressing the 18 B/Param Floor

FP8 training is the next step down in precision, targeting H100/Hopper hardware. The memory argument is straightforward: if both weights and gradients can be stored and computed in 1-byte formats, the static floor shrinks dramatically.

**Two FP8 formats ([[fp8-training]]):**

- **E4M3**: 1 sign + 4 exponent + 3 mantissa bits; range ±448; higher precision. Used for **weights and activations** (forward pass) because these need precision, not range.
- **E5M2**: 1 sign + 5 exponent + 2 mantissa bits; range ±57,344; wider range. Used for **gradients** (backward pass) because gradients need to represent a wide magnitude spectrum without overflow.

The asymmetric assignment is intentional: weights are numerically stable and their values are bounded, so precision matters more than range (E4M3). Gradients vary wildly in magnitude across layers and steps, so range matters more than precision (E5M2).

**FP8 optimizer memory layout (6 B/param total vs 18 B/param for BF16) ([[fp8-training]]):**

| Tensor | FP8 layout | Bytes/param |
|--------|------------|-------------|
| Master weights | FP16 with scaling | 2 |
| Gradients | FP8 (E5M2) | 1 |
| First-order moment m | FP8 | 1 |
| Second-order moment v | FP16 | 2 |
| **Total optimizer** | | **6** |

Versus BF16 AdamW's 18 B/param: this is a **2.6× optimizer-state reduction** (from ~16 B to ~6 B in the optimizer alone, though different sources count the boundary differently).

**Per-tensor dynamic scaling** handles FP8's narrow range: each tensor gets a dynamic scale factor μ. If overflow fraction exceeds 0.001%: μ → μ/2. If no overflow for 1,000 training steps: μ → μ×2. For distributed all-reduce across GPUs, the global minimum scale `s_g' = min(s'₁, ..., s'_n)` is used — this eliminates per-tensor synchronization overhead that would otherwise dominate at scale.

**Overall memory reduction vs BF16 mixed precision ([[fp8-training]]):**

| Model | Memory reduction |
|-------|-----------------|
| GPT-7B | 29% |
| GPT-13B | 28% |
| GPT-175B | **39%** |

The larger reduction at 175B reflects that optimizer states dominate the memory budget more completely at large parameter count (relative to activations which are fixed by sequence/batch).

**Throughput payoff on H100 ([[fp8-training]]):** GPT-175B runs **75% faster** than BF16 Megatron-LM, and 37% faster than NVIDIA Transformer Engine's own implementation. GPT-7B: 38% faster. FP8 also enables longer sequences: GPT-175B can train at seq=4,096 in FP8 where BF16 is limited to seq=2,048 on the same H100 cluster.

**Why A100 cannot use FP8.** The FP8 tensor core is a Hopper-architecture feature — it maps directly to hardware-accelerated 8-bit matrix multiplication units that do not exist in Ampere (A100). FP8 is not a software trick that degrades gracefully on older hardware; without the hardware tensor cores, the format has no computational advantage and the precision loss is pure downside. On A100: use BF16 + fp32 masters.

---

## 4. The Loss-Head Logit Spike

The static 18 B/param floor accounts for weights, gradients, and optimizer states. But there is a transient peak that dwarfs it at the end of each forward pass: the logit tensor.

**Where it comes from.** The final layer in any language model is a linear projection from hidden dimension h to vocabulary size V:

```
# Standard PyTorch pattern (what NOT to do at scale):
logits = hidden_states @ lm_head.weight.T   # shape: [B, T, V]
loss = F.cross_entropy(
    logits.view(B*T, V),                     # must exist in full
    labels.view(B*T)
)
```

The logits tensor is `B × T × V` elements. In fp32, each element is 4 bytes; in bf16, 2 bytes. This tensor is materialized in full before the cross-entropy loss can be computed.

**The size of the spike ([[liger-fused-ce]]):**

For seq=16,384 tokens and vocab=32,000 in BF16:
```
16,384 × 32,000 × 2 bytes = 1.05 GB
```

This 1+ GB tensor appears transiently at every forward pass — a "spike" because it exists only between the linear projection and the loss reduce, then is freed. But at step boundaries where both forward-pass activations and the logit tensor coexist, this is the peak memory event. It sits **entirely outside** the 18 B/param static floor — it is an activation-layer transient, not a parameter-layer constant. The [[ch-09]] capstone case (seq=32k, vocab=248k) makes this dramatically worse:

```
32,768 × 248,000 × 2 bytes = ~16 GB
```

A single logit tensor at 16 GB would be the dominant memory consumer on an 80 GB H100, before counting any other activations.

**The OOM failure mode.** Memory profilers show jobs that pass step 0 OOM on step 1 or 2, precisely because the logit spike plus optimizer-state materialization (see [[ultrascale-playbook]]'s "peak memory warning") coincides with the maximum allocation. A job that appears to have 4 GB of headroom can OOM due to this transient.

**Monitoring signal: grad_norm and NaN.** Precision trouble — fp16 overflow, fp8 scale misconfiguration, or loss-scaling bugs — manifests first as NaN or Inf gradients. The standard monitoring loop:

```python
# After loss.backward() and before optimizer.step():
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0
)
if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    # Skip optimizer.step(); reduce loss scale (fp16) or investigate
    scaler.update()   # GradScaler will halve the scale
    continue
```

A NaN in `grad_norm` is the earliest observable symptom; waiting for loss divergence means many wasted steps.

---

## 5. Liger-Kernel: Eliminating the Spike via Fused Chunked CE

Liger's fused cross-entropy kernel solves the materialization problem by fusing the linear projection with the loss computation and processing tokens in chunks ([[liger-fused-ce]]).

**The three in-kernel strategies:**

```
# Pseudocode for LigerFusedLinearCrossEntropyLoss (Triton kernel)

for chunk in split(hidden_states, chunk_size=65536):   # ≤65,536 tokens/chunk
    # 1. Incremental projection: project only this chunk
    logit_chunk = chunk @ lm_head.weight.T             # shape: [chunk, V]
    
    # 2. Compute loss contribution from this chunk
    loss_chunk = cross_entropy(logit_chunk, labels[chunk])
    loss_accum += loss_chunk
    
    # 3. In-place gradient accumulation:
    #    grad_weight accumulates across chunks, no full logit buffer needed
    grad_weight += chunk.T @ d_logit_chunk
    d_hidden[chunk] = d_logit_chunk @ lm_head.weight

# Never materializes the full [B*T, V] logit tensor
```

The platform-specific chunk size on CUDA is typically `65,536 ÷ 2 = 32,768` tokens per chunk. For the earlier example (seq=16,384, vocab=32,000):

```
Peak per-chunk allocation: 2,048 × 32,000 × 2 bytes = 131 MB
```

versus 1,050 MB for the full materialization — an **8× reduction in the peak transient allocation** from this one operation.

**Reported memory reductions ([[liger-fused-ce]]):**

| Training mode | Overall memory reduction |
|---------------|--------------------------|
| Pretraining / SFT | ~60% |
| Alignment (DPO, ORPO, CPO) | up to 80% |

The larger reduction for alignment is because DPO and ORPO compute the forward pass twice (over chosen and rejected sequences) over the same vocab head — the spike doubles in vanilla implementations, but Liger chunks both.

**Exact semantics guaranteed:** No approximation is made. The chunked accumulation is mathematically identical to computing CE over the full logit matrix; the backward pass is integrated inside the Triton kernel and returns correct gradients for both hidden states and `lm_head.weight`. Replacing `nn.Linear + CrossEntropyLoss` with `LigerFusedLinearCrossEntropyLoss` is a drop-in substitution with no accuracy impact.

**API:**

```python
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

# Replace the standard PyTorch combination:
# loss_fn = nn.CrossEntropyLoss()
# logits = lm_head(hidden_states)
# loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

# With:
loss_fn = LigerFusedLinearCrossEntropyLoss()
loss = loss_fn(lm_head.weight, hidden_states, labels)
# lm_head bias optional; backward() works normally
```

Apply this early in any training run. It is the highest-leverage single-operator swap available for memory-constrained training with large vocabularies.

---

## 6. How the Pieces Fit Together: The Full Precision Stack

A complete picture of precision choices at each point in the training loop:

```
┌─────────────────────────────────────────────────────────────────┐
│  FORWARD PASS                                                    │
│  Input tokens → Embedding (bf16) → Transformer layers (bf16)   │
│  → Hidden states (bf16) → [Liger fused CE, chunked]            │
│                            ↑ no full logit tensor               │
│  Loss scalar (fp32) ←──────┘                                    │
├─────────────────────────────────────────────────────────────────┤
│  BACKWARD PASS                                                   │
│  Gradients in bf16 (or fp16+scaling, or fp8 E5M2)             │
│  Accumulated into fp32 gradient buffers                         │
├─────────────────────────────────────────────────────────────────┤
│  OPTIMIZER STEP (Adam)                                          │
│  fp32 gradients → update fp32 m, fp32 v                        │
│  → compute Δw in fp32                                           │
│  → apply to fp32 master weights                                 │
│  → cast fp32 masters back to bf16 working weights              │
└─────────────────────────────────────────────────────────────────┘
```

The fp32 master weights live in this final step and are the reason the optimizer costs 4 extra bytes/param beyond what computation requires. They are cast down to bf16 only for the next forward pass.

---

## 7. Cross-Precision Comparison Table

| Aspect | FP16 mixed-prec | BF16 mixed-prec | FP8 (H100 only) |
|--------|-----------------|-----------------|-----------------|
| Working weight dtype | fp16 (2 B) | bf16 (2 B) | fp8 E4M3 (1 B) |
| Master weight dtype | fp32 (4 B) | fp32 (4 B) | fp16 (2 B) |
| Gradient dtype | fp16 (scaled) | bf16 | fp8 E5M2 (1 B) |
| Optimizer moment dtype | fp32+fp32 (8 B) | fp32+fp32 (8 B) | fp8+fp16 (~3 B) |
| Static floor (B/param) | ~18 | ~18 | ~6 |
| Loss scaling required | Yes (dynamic) | No | Per-tensor μ |
| Hardware requirement | Any CUDA | Any CUDA | H100+ (Hopper) |
| Throughput gain vs fp32 | ~2× | ~2× | +75% vs bf16 |
| Memory reduction vs fp32 | ~50% | ~50% | ~89% |

**Invariant across all three**: an fp32 (or equivalent-precision) copy of the weight lives somewhere in the optimizer update path. This is forced by the substrate — the update `w ← w − lr·m̂/(√v̂+ε)` must be computed at sufficient precision to avoid mantissa cancellation, regardless of what precision the forward pass uses. Even FP8 training keeps fp16 master weights (2 B instead of 4 B), not fp8.

**Variant**: what precision is used for gradients and moments, and how the dynamic-range problem is solved (dynamic loss scaling in fp16, eliminated in bf16, per-tensor μ in fp8). These are free design choices given the hardware capability.

---

## Core Insights from the Literature

**From [[mixed-precision-training]] (Micikevicius et al., ICLR 2018):** The fp32 master weight is necessary because of two distinct numerical phenomena — gradient underflow below fp16's floor of 2^−24, and mantissa cancellation when the weight-to-update ratio exceeds 2,048. Both cause silent zero-updates that compound across steps; loss scaling fixes the underflow case but not the cancellation case, which is why the master weight is still required even with scaling.

**From [[fp8-training]] (Peng et al., 2023):** The asymmetric FP8 format assignment (E4M3 for forward, E5M2 for backward) reflects the different numerical requirements of weights/activations versus gradients. Weights are bounded and need mantissa precision; gradients span orders of magnitude and need exponent range. The 75% throughput gain on GPT-175B at 39% memory reduction represents the compound effect of compute acceleration (8-bit matmuls) plus freed bandwidth (smaller tensors in HBM).

**From [[liger-fused-ce]] (Liu et al., 2024):** The B·T×V logit materialization is a pathological spike that is orthogonal to the static memory floor — it can OOM a model that otherwise fits comfortably. The chunked fused kernel is an exact computation (no approximation), is a drop-in replacement, and reduces this spike from 1+ GB to 131 MB for a typical configuration. The 80% alignment-training reduction is especially high because DPO/ORPO double the spike by passing two sequences through the same head.

**From [[transformer-math-101]] and [[ultrascale-playbook]]:** The 18 B/param floor (or 16 B in the playbook's counting that uses bf16 gradients) is the starting estimate, but the activation formula `L·seq·bs·h·(34 + 5·n_heads·seq/h)` grows quadratically in sequence length and can exceed the static floor by 10-100× at long context. The logit spike is an additional transient on top of activations, making it the true worst-case peak memory event per step.

---

## Key Takeaways

- **18 bytes/param** is the static mixed-precision AdamW floor (2 bf16 working weights + 4 fp32 master + 4 fp32 gradients + 8 fp32 optimizer). [[ch-01]]'s "Rule of 16" uses 2 B bf16 gradients to get 16 B; 18 B is the conservative floor with fp32 gradients.
- **The fp32 master weight is non-negotiable** in any precision regime below fp32 — the substrate (Adam's subtraction) forces it because mantissa cancellation will silently zero updates otherwise.
- **BF16 vs FP16 axis**: BF16 eliminates loss scaling (same exponent range as fp32); FP16 requires dynamic loss scaling or training diverges. On modern hardware (A100+), BF16 is the default.
- **FP8 is H100-only**: 39% memory reduction and 75% throughput gain for GPT-175B, but gated by Hopper tensor cores. On A100: use BF16.
- **The logit spike** (`B·T×V × dtype_size`) is the largest transient in a training step and sits outside the static floor. At seq=32k, vocab=248k in BF16, it is ~16 GB — larger than most model parameter footprints below 30B. Use Liger's fused CE to chunk it.
- **Monitoring**: NaN/Inf in `grad_norm` is the first observable signal of precision failure (fp16 scale misconfiguration, fp8 μ runaway, or numerical instability). Monitor every step.
- **Next chapter** [[ch-03]] covers activation memory and gradient checkpointing — the other side of the memory equation, where the quadratic seq-length term dominates.

---

## References

- Paulius Micikevicius et al. "Mixed Precision Training." ICLR 2018. https://arxiv.org/abs/1710.03740 ([[mixed-precision-training]])
- Houwen Peng et al. "FP8-LM: Training FP8 Large Language Models." arXiv:2310.18313, 2023. https://arxiv.org/abs/2310.18313 ([[fp8-training]])
- Austin Liu et al. "Liger Kernel: Efficient Triton Kernels for LLM Training." arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel ([[liger-fused-ce]])
- Quentin Anthony et al. "Transformer Math 101." EleutherAI Blog, 2023. https://blog.eleuther.ai/transformer-math/ ([[transformer-math-101]])
- HuggingFace / nanotron team. "The Ultra-Scale Playbook." 2025. https://nanotron-ultrascale-playbook.static.hf.space/ ([[ultrascale-playbook]])

---

## Questions

1. **Mantissa cancellation:** The fp32 master weight prevents mantissa cancellation when `|w| / |Δw| > 2048`. Given that Adam normalizes updates by `√v̂` (making them roughly unit-scale), under what training conditions is the ratio `|w| / |Δw|` most likely to exceed 2048 — early training, late training, or never? Walk through the numerics.

2. **Exponent arithmetic:** FP16 has a 5-bit exponent (biased representation: range 2^−14 to 2^15, i.e. ±65,504). BF16 has an 8-bit exponent (same as fp32: range 2^−126 to 2^127). The underflow threshold for FP16 is 2^−24 (subnormal floor). At what gradient magnitude does fp16 start losing information relative to bf16, and why is this the underflow threshold rather than 2^−14?

3. **Loss scaling derivation:** In the gradient histogram evidence, "67% of SSD gradient values are zero in fp16 without scaling; 8× scaling recovers full accuracy." If 8× scaling is applied, the smallest representable fp16 gradient shifts from ~2^−24 to ~2^−24 × 8 = 2^−21. What does this tell you about where most of the near-zero SSD gradients are concentrated in the exponent range?

4. **FP8 format assignment:** The paper assigns E4M3 (high precision, narrow range) to weights/activations and E5M2 (wide range, low precision) to gradients. The [[liger-fused-ce]] excerpt says the logit tensor spike reaches 1.05 GB at seq=16k, vocab=32k in BF16. If Transformer Engine uses E4M3 for activations (1 byte vs 2 bytes bf16), what does the logit spike become in FP8 forward pass — and does this change the argument for Liger?

5. **Chunking arithmetic:** The Liger kernel uses `chunk_size ≤ 65,536 ÷ 2 = 32,768` tokens on CUDA. For the [[ch-09]] capstone case (seq=32,768, vocab=248,000, BF16), compute: (a) the full logit tensor size, (b) the per-chunk peak allocation at chunk_size=32,768, and (c) the memory reduction factor. Would a single A100-80GB handle the full spike without Liger?

6. **From the [[fp8-training]] excerpt:** GPT-175B at 39% memory reduction and 75% throughput gain. Selective recomputation + sequence parallelism at 530B (Korthikanti 2022, covered in [[ch-03]]) gives 29% throughput gain from 42.1% to 54.2% MFU. These are additive techniques on H100. If both are applied simultaneously to a 175B model on H100, what is the expected combined effect on peak memory, and which constraint (static floor vs activation spike vs logit spike) becomes the binding one?
