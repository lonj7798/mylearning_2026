---
chapter: ch-22
course: model-quantization
phase: read
excerpt_of: "Side-by-side comparison of the five capstone method options"
created_at: "2026-05-21"
---

# Excerpt: Capstone Method Comparison

**Sources:** [[raw-data/kivi]], [[raw-data/kvquant]], [[raw-data/gear]], [[raw-data/turboquant]], [[raw-data/deepseek-v3-fp8]], [[raw-data/nvfp4-training]]

---

## At a glance

| Method | Surface | Bit target | Calibration | Hardware floor | Algorithmic difficulty | Reproduction risk |
|--------|---------|-----------|-------------|---------------|----------------------|-------------------|
| **KIVI** | KV cache | 2-bit | none (tuning-free) | any GPU | low | medium |
| **KVQuant** | KV cache | 2-3 bit | small (per-channel codebook fit) | any GPU | medium-high (4 jointly-applied techniques) | medium-high |
| **GEAR** | KV cache | 4-bit + low-rank residual | none for SVD path | any GPU | high (streaming SVD) | high |
| **TurboQuant** | KV cache | 2.5-3.5 bit | none (data-oblivious) | any GPU | high (QJL theory) | medium-high |
| **NVFP4 inference** | weights + activations | 4 bit | per-layer sensitivity | Blackwell for native throughput | medium | high (hardware-dependent) |

---

## KIVI — the recommended starting point

**Why pick this:** Simplest method, no calibration, paper text is unambiguous, debugging surface is small.

**Core mechanism (one paragraph):** The K and V caches have different outlier structures. K cache outliers are *channel-wise* (a few channels persistently 10–100× larger than others, induced by RoPE + residual-stream outliers). V cache shows no consistent channel pattern. So K should be quantized per-channel (per-channel scale, group along token axis), V per-token (per-token scale, group along channel axis). Both at INT2 with group size 32. No fine-tuning needed.

**Implementation skeleton:**
```python
def kivi_quant(K, V, group_size=32, bits=2):
    # K: (T, H, D) — quantize per-channel (D axis), group along T
    # V: (T, H, D) — quantize per-token (T axis), group along D
    K_q, K_scales = _quant_axis(K, axis_quant=2, axis_group=0, group_size=group_size, bits=bits)
    V_q, V_scales = _quant_axis(V, axis_quant=0, axis_group=2, group_size=group_size, bits=bits)
    return (K_q, K_scales), (V_q, V_scales)
```

**Headline reproduction risk:** Axis confusion. The phrase "K per-channel" can be misread as "one scale per (channel, token) pair" — which is FP16, not INT2. The correct reading: one scale per (channel, token-group) pair, INT2 element. Test with a small synthetic tensor before integrating into the model.

**Target reproduction:** Llama-2-7B at INT2 KV: Wikitext-2 PPL gap <0.2 vs FP16; LongBench average within 1 point.

---

## KVQuant — the multi-component method

**Why pick this:** Tests whether you can implement four jointly-applied techniques where ablating any one degrades the result.

**Core mechanism (one paragraph):** Sub-4-bit KV cache requires four techniques applied together: (1) per-channel K quantization on the *pre-RoPE* representation — RoPE mixes channels and destroys per-channel structure; (2) sensitivity-weighted *non-uniform* per-channel codes — each K channel gets its own quantile-fit 2/3-bit codebook; (3) dense-and-sparse decomposition — top 1% outlier elements kept in FP16 sparse; (4) Q-norm quantization for the Query so QK^T is computed in low-bit.

**Implementation skeleton:**
```python
def kvquant(K_pre_rope, V, codebooks, sparse_indices_K, sparse_indices_V, bits=3):
    K_dense, K_sparse = _split_dense_sparse(K_pre_rope, sparse_indices_K)
    V_dense, V_sparse = _split_dense_sparse(V, sparse_indices_V)
    K_q = _per_channel_nonuniform_quant(K_dense, codebooks)
    V_q = _per_token_uniform_quant(V_dense, bits)
    # K_sparse, V_sparse stay FP16
    return (K_q, K_sparse), (V_q, V_sparse)

# At attention time:
# K = dequant(K_q, codebooks) + scatter(K_sparse, sparse_indices_K)
# K_post_rope = apply_rope(K, position_ids)
# Q_q = quant_q(Q)  # for Q-norm quantization
# attn_logits = Q_q @ K_post_rope.transpose(-2, -1) / sqrt(d)
```

**Headline reproduction risk:** Order-of-operations. The paper specifies: quantize → split sparse → apply RoPE on dequant. Reversing the last two (apply RoPE first, then sparse-decompose) gives a different (worse) sparse pattern. The codebook fitting (per-channel k-means) has under-training failure modes; train for ≥20 iterations.

**Target reproduction:** 3-bit KV on Llama-7B: <0.1 PPL degradation on Wikitext-2.

---

## GEAR — the streaming-numerics challenge

**Why pick this:** The streaming SVD update is hard to get right. If you want to flex your numerical-linear-algebra muscle.

**Core mechanism (one paragraph):** Quantization residual (the difference between FP16 KV and its low-bit quantized version) has approximately low-rank structure across the (tokens × channels) matrix per head. GEAR represents this residual as a rank-r SVD outer product `A B^T` (r=2–4) plus a sparse correction matrix `S` for the top-1% entries the rank-r SVD can't capture. The full reconstruction: `KV ≈ Q + L + S` where Q is uniform 4-bit, L is the low-rank residual, S is sparse FP16. Streaming SVD updates A, B as new tokens arrive; periodic full refresh every ~256 tokens.

**Headline reproduction risk:** Streaming-SVD orthogonality drift. When projecting a new token's residual onto existing B columns, you must re-orthogonalise B. Forgetting this gives incremental updates that look correct until you compare to a from-scratch SVD on the accumulated data — at which point the drift is obvious. Test: run for 1000 tokens *with periodic refresh disabled*, compare A, B to from-scratch SVD. They should match within ε = 1e-4 in Frobenius norm.

**Target reproduction:** 4-bit KV near-lossless on WikiText-2 + long-context tasks.

---

## TurboQuant — the theoretically deepest

**Why pick this:** Closest to information-theoretic optimal. Data-oblivious; no calibration. The most novel of the five.

**Core mechanism (one paragraph):** Random Hadamard rotation `v' = R · v` (fixed at startup, *not data-dependent*) makes per-coordinate distributions analytically known (a tightly concentrated Beta distribution by measure concentration in high dimensions). Apply the analytically optimal scalar quantizer for that Beta distribution to each coordinate of `v'`. For inner-product unbiasedness (needed for attention scores), compute the residual `e = v' − Q(v')`, apply a JL sketch `Π` of width m, and store only `sign(Π · e)`. The asymmetric inner-product estimator combines the quantized leg and the QJL residual leg.

**Headline reproduction risk:** The QJL asymmetric estimator's α constant. Inner-product reconstruction is `⟨u, v⟩ ≈ ⟨Q(u'), Q(v')⟩ + α · ⟨sign(Π·e_u), Π·e_v⟩` for a specific α that depends on JL width m. Reference the QJL paper (Zandieh 2024) for the formula; the TurboQuant paper assumes you've internalised it. Getting α wrong gives biased inner-product estimates that look like the JL sketch is broken.

**Target reproduction:** 3.5-bit KV: quality-neutral on LongBench/NIAH/RULER on Gemma and Mistral.

---

## NVFP4 inference — the production-frontier option

**Why pick this:** Frontier deployment format. Connects quantization to training-format research. *Only* reproducible at full throughput on Blackwell hardware.

**Core mechanism (one paragraph):** Two-level scaling: each tensor has an outer FP32 scale; within the tensor, 16-element FP4 blocks share an inner E4M3 (FP8) block scale. The FP32 outer scale absorbs global tensor range; the FP8 block scale fixes local dispersion; the 4-bit element carries the actual data. For *inference* (without the training-side stochastic rounding / 2-D consistent quantization complexity), the recipe reduces to: (1) compute per-tensor FP32 scale from FP16 weight amax; (2) compute per-block FP8 scales within the tensor; (3) round each element to the nearest FP4 value after dividing by both scales; (4) apply selective high-precision exceptions per [[fp4-inference-diagnosis]] (MLP up/down at FP8, embedding/head/RMSNorm at BF16, optionally early blocks at FP8).

**Headline reproduction risk:** Native throughput requires Blackwell (Tensor Core SM 10.x). On non-Blackwell hardware you can emulate FP4 storage + per-tensor / per-block scaling math, but the FP4 GEMM kernel doesn't exist — every matmul falls back to FP8 or FP16 with a dequant prelude, losing the throughput claim entirely. Quality is still reproducible (the math is portable); throughput is not. State this in the memo.

**Target reproduction:** With the [[fp4-inference-diagnosis]] exception policy, target PPL gap <0.3 vs FP16 on Llama-3-8B.

---

## Scoring rubric for picking

If you have **3 days**: KIVI.
If you have **5 days and want technical depth**: KVQuant or TurboQuant.
If you have **a week and like numerical analysis**: GEAR.
If you have **Blackwell + a week**: NVFP4 inference.
If you have **a month**: any of the above + the W4A4 + KV-quant stretch goal.

The recommendation is "KIVI for the first attempt; pick the harder one if you finish in <3 days." The capstone is about producing a defensible memo, not about picking the flashiest method.

---

## Connections

- [[ch-22]] §method-options — chapter section.
- [[kivi]] / [[kvquant]] / [[gear]] / [[turboquant]] / [[deepseek-v3-fp8]] / [[nvfp4-training]] — papers.
- [[qjl]] — required reference for TurboQuant's asymmetric estimator.
- [[fp4-inference-diagnosis]] — required reference for NVFP4 exception policy.
